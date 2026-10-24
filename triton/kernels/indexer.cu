// tlaloc_indexer_topk: the selection of query-selective sparse attention (INDEXER_TOPK,
// io.tlaloc.ir.IndexerTopkAttrs) in two kernels, in place of XLA's form (the whole page
// window gathered, a [rows, heads, chunks] product and two sorts of every row's scores).
//
//   ScoreKernel   grid (ceil(C / kTile), Tb): a block reads the pooled keys of kTile chunks
//                 of one table (the key at each chunk's last position) into shared memory,
//                 and for each row of the table lane j scores chunk j of the tile:
//                 scale * sum over heads of relu(q[h] . key), f32 products and sums.
//                 Chunks not complete (past seqLens / chunk) score -inf.
//   SelectKernel  a block per row: the top_k largest of the row's complete chunks by a
//                 radix select over the scores' order-preserving 32-bit keys (four passes
//                 of 8 bits), then the chunks above the k-th largest kept, and of those equal
//                 to it the lowest in chunk order (ties keep the lower chunk, as the
//                 reference's stable order does); every complete chunk when there are at
//                 most top_k; and the incomplete chunk at the end.
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace indexer {

constexpr int kThreads = 256;
constexpr int kTile = 32;          // chunks per score block: one per lane
constexpr int kMaxHeads = 8;
constexpr int kMaxDim = 256;
constexpr int kSelectThreads = 1024;

struct Shape {
  int R = 0, H = 0, D = 0, bs = 0, Tb = 0, M = 0, C = 0, chunk = 0, topK = 0;
  float scale = 1.f;
};

// Grid (ceil(C / kTile), Tb), kThreads threads. Shared memory: the tile's keys
// [kTile][D + 1] and each warp's row of queries [H][D].
__global__ void __launch_bounds__(kThreads) ScoreKernel(
    const float* __restrict__ q, const float* __restrict__ keys, const int* __restrict__ tables,
    const int* __restrict__ lens, float* __restrict__ scores, Shape sh)
{
  extern __shared__ float smem[];
  const int tile = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int D = sh.D, KS = D + 1, Qr = sh.R / sh.Tb;
  float* ks = smem;                                  // [kTile][D + 1]
  float* qs = smem + kTile * KS + warp * sh.H * D;   // this warp's [H][D]
  const int c0 = tile * kTile;
  for (int e = tid; e < kTile * D; e += kThreads) {
    const int j = e / D, d = e % D, c = c0 + j;
    float v = 0.f;
    if (c < sh.C) {
      const int pos = c * sh.chunk + sh.chunk - 1;
      const size_t slot = static_cast<size_t>(tables[t * sh.M + pos / sh.bs]) * sh.bs + pos % sh.bs;
      v = keys[slot * D + d];
    }
    ks[j * KS + d] = v;
  }
  __syncthreads();
  for (int i = warp; i < Qr; i += kThreads / 32) {
    const int row = t * Qr + i;
    for (int e = lane; e < sh.H * D; e += 32) qs[e] = q[static_cast<size_t>(row) * sh.H * D + e];
    __syncwarp();
    const int c = c0 + lane;
    const int complete = lens[row] / sh.chunk;
    if (c < sh.C) {
      float total = 0.f;
      for (int h = 0; h < sh.H; ++h) {
        float dot = 0.f;
        const float* qh = qs + h * D;
        const float* k = ks + lane * KS;
        for (int d = 0; d < D; ++d) dot = fmaf(qh[d], k[d], dot);
        total += fmaxf(dot, 0.f);
      }
      scores[static_cast<size_t>(row) * sh.C + c] = c < complete ? total * sh.scale : -INFINITY;
    }
    __syncwarp();
  }
}

// A float's order as an unsigned integer: larger floats, larger keys.
__device__ __forceinline__ uint32_t OrderKey(float f)
{
  const uint32_t b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// A block per row, kSelectThreads threads.
__global__ void __launch_bounds__(kSelectThreads) SelectKernel(
    const float* __restrict__ scores, const int* __restrict__ lens, int* __restrict__ keep, Shape sh)
{
  __shared__ unsigned int hist[256];
  __shared__ uint32_t s_prefix;
  __shared__ int s_need;
  __shared__ int s_warp[kSelectThreads / 32];
  __shared__ int s_base;
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const float* x = scores + static_cast<size_t>(row) * sh.C;
  int* out = keep + static_cast<size_t>(row) * sh.C;
  const int len = lens[row];
  const int complete = min(len / sh.chunk, sh.C);
  const bool tail = len % sh.chunk != 0 && complete < sh.C;
  if (complete <= sh.topK) {
    for (int c = tid; c < sh.C; c += kSelectThreads) out[c] = c < complete || (tail && c == complete) ? 1 : 0;
    return;
  }
  // The k-th largest key: four passes of 8 bits from the top.
  uint32_t prefix = 0, mask = 0;
  int need = sh.topK;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int i = tid; i < 256; i += kSelectThreads) hist[i] = 0;
    __syncthreads();
    for (int c = tid; c < complete; c += kSelectThreads) {
      const uint32_t k = OrderKey(x[c]);
      if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 0xFF], 1u);
    }
    __syncthreads();
    if (tid == 0) {
      int above = 0, b = 255;
      for (; b > 0; --b) {
        if (above + static_cast<int>(hist[b]) >= need) break;
        above += hist[b];
      }
      s_prefix = prefix | (static_cast<uint32_t>(b) << shift);
      s_need = need - above;
    }
    __syncthreads();
    prefix = s_prefix;
    need = s_need;
    mask |= 0xFFu << shift;
    __syncthreads();
  }
  // Keys above the threshold kept; of those equal to it, the first `need` in chunk order.
  if (tid == 0) s_base = 0;
  __syncthreads();
  for (int c0 = 0; c0 < sh.C; c0 += kSelectThreads) {
    const int c = c0 + tid;
    uint32_t k = 0;
    bool equal = false;
    if (c < complete) {
      k = OrderKey(x[c]);
      equal = k == prefix;
    }
    const unsigned ballot = __ballot_sync(0xffffffffu, equal);
    if (lane == 0) s_warp[warp] = __popc(ballot);
    __syncthreads();
    int before = s_base;
    for (int w = 0; w < warp; ++w) before += s_warp[w];
    before += __popc(ballot & ((1u << lane) - 1u));
    if (c < sh.C) {
      int v = 0;
      if (c < complete) v = k > prefix || (equal && before < need) ? 1 : 0;
      else if (tail && c == complete) v = 1;
      out[c] = v;
    }
    __syncthreads();
    if (tid == 0) {
      int sum = 0;
      for (int w = 0; w < kSelectThreads / 32; ++w) sum += s_warp[w];
      s_base += sum;
    }
    __syncthreads();
  }
}

inline size_t ScoreSharedBytes(const Shape& sh)
{
  return sizeof(float) * (static_cast<size_t>(kTile) * (sh.D + 1) + static_cast<size_t>(kThreads / 32) * sh.H * sh.D);
}

inline cudaError_t Prepare()
{
  return cudaFuncSetAttribute(ScoreKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
}

inline cudaError_t Launch(const float* q, const float* keys, const int* tables, const int* lens, int* keep, float* scores,
                          const Shape& sh, cudaStream_t stream)
{
  ScoreKernel<<<dim3((sh.C + kTile - 1) / kTile, sh.Tb), kThreads, ScoreSharedBytes(sh), stream>>>(q, keys, tables, lens, scores, sh);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return e;
  SelectKernel<<<sh.R, kSelectThreads, 0, stream>>>(scores, lens, keep, sh);
  return cudaGetLastError();
}

}  // namespace indexer
