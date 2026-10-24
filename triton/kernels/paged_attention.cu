// Paged attention for decode, speculative verify and prefill steps, as an XLA typed-FFI
// custom call (`tlaloc_paged_attention`), registered with a PJRT GPU plugin by
// TlalocRegisterKernels (tlaloc_kernels.cu).
//
// Operands, as PAGED_ATTENTION's (io.tlaloc.ir.PagedAttentionAttrs):
//   0 query       f32  [R, H, D]
//   1 keyCache    f8e4m3fn | bf16 | f32  [P, bs, Hkv, D]
//   2 valueCache  same as keyCache
//   3 blockTables i32  [Tb, M]     the R rows are Tb groups of R / Tb consecutive rows
//   4 seqLens     i32  [R]         row r attends positions [0, seqLens[r])
//   5 keep        i32  [R, C]      optional: row r attends position p only if keep[r, p / (M bs / C)] != 0
//                                  (query-selective sparse attention's chunks)
// Results:
//   0 out         f32  [R, H, D]
//   1 scratch     f32  [S, R, H, D + 2]   per context slice: max, sum, weighted values
//                                          (unused by the prefill form: [1, 1, 1, 1])
// Attributes: scale (f32); prefill_f16 (s32, optional).
//
// The XLA form gathers each row's whole context bucket out of the pool and
// writes it before attending; here each block of threads reads the pages of
// one slice of S once, for every row sharing the table and every query head
// of one KV head (an online softmax over tiles of positions), and a second
// kernel combines the slices. The arithmetic is f32 throughout, or, for e4m3fn
// pools with D 256, f16 tensor-core products of split (high and low) f16
// operands, accumulated in f32. More than kMaxQueries queries per table and KV
// head (prefill rows) take the prefill form: blocks of query tiles, each walking
// the whole context. With the attribute prefill_f16 (s32, nonzero) they take its
// f16 form instead: single f16 queries and probabilities, at twice the rate.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <string>

#include "kernel_setup.cuh"
#include "xla/ffi/api/c_api.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_extension.h"

namespace {

constexpr int kThreads = 256;
// The most query vectors per table and KV head (rows per table x group).
constexpr int kMaxQueries = 64;

__device__ __forceinline__ float Widen(__nv_fp8_e4m3 x) { return static_cast<float>(x); }
__device__ __forceinline__ float Widen(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float Widen(float x) { return x; }

struct Shape {
  int R, H, Hkv, D, bs, M, Tb, S, slice;
  float scale;
  // The keep mask [R, keepChunks] (null: every position), positions per chunk.
  const int* keep;
  int keepChunks, chunk;
};

// Whether row [row] attends position [pos] under the keep mask.
__device__ __forceinline__ bool Kept(const Shape& sh, int row, int pos)
{
  return sh.keep == nullptr || __ldg(sh.keep + static_cast<size_t>(row) * sh.keepChunks + pos / sh.chunk) != 0;
}

// Sixteen consecutive values of a pool row, widened to f32.
__device__ __forceinline__ void Load16(const __nv_fp8_e4m3* p, float* out)
{
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  const __nv_fp8_e4m3* x = reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
#pragma unroll
  for (int c = 0; c < 16; ++c) out[c] = static_cast<float>(x[c]);
}
__device__ __forceinline__ void Load16(const __nv_bfloat16* p, float* out)
{
  const uint4 a = reinterpret_cast<const uint4*>(p)[0], b = reinterpret_cast<const uint4*>(p)[1];
  const __nv_bfloat16* x = reinterpret_cast<const __nv_bfloat16*>(&a);
  const __nv_bfloat16* y = reinterpret_cast<const __nv_bfloat16*>(&b);
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    out[c] = __bfloat162float(x[c]);
    out[8 + c] = __bfloat162float(y[c]);
  }
}
__device__ __forceinline__ void Load16(const float* p, float* out)
{
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const float4 v = reinterpret_cast<const float4*>(p)[c];
    out[4 * c] = v.x;
    out[4 * c + 1] = v.y;
    out[4 * c + 2] = v.z;
    out[4 * c + 3] = v.w;
  }
}

// The stride of a position's row of weights in shared memory: NQ rounded up to
// a multiple of 4 (float4 reads), and not a multiple of 8 (at most 4-way bank
// conflicts when a warp reads one query across positions).
__host__ __device__ inline int WeightStride(int nq)
{
  int s = (nq + 3) / 4 * 4;
  if (s % 8 == 0) s += 4;
  return s;
}

// Where the slice's pool offsets start in shared memory, in floats: after
// max(q, p), m and l, 16-byte aligned.
__host__ __device__ inline int BaseOffset(int nq, int d)
{
  const int qp = nq * d > kThreads * WeightStride(nq) ? nq * d : kThreads * WeightStride(nq);
  return (qp + 2 * nq + 3) / 4 * 4;
}

// Grid (S, Hkv, Tb), kThreads threads, two blocks per SM. Each block runs one
// slice of kThreads positions for the NQ (<= MAXQ) = rows-per-table x group
// queries of one table and KV head:
//   scores   thread j takes position j of the slice: it reads the key row in
//            16-value vectors (the next one in flight while the current one is
//            used) and dots it with every query (float4 broadcasts from
//            shared memory);
//   softmax  warp w takes queries w, w + 8, ... over the slice;
//   values   thread d owns head dimension d of every query and streams the
//            slice's values, coalesced, into its NQ accumulators, reading
//            four queries' weights per float4.
// Dynamic shared memory: q [NQ][D] (scaled) and then, in the same space,
// p [kThreads][WeightStride(NQ)]; m, l [NQ]; the pool offsets [kThreads].
template <typename KV, int MAXQ>
__global__ void __launch_bounds__(kThreads, 2) SliceKernel(
    const float* __restrict__ q, const KV* __restrict__ kc, const KV* __restrict__ vc,
    const int* __restrict__ tables, const int* __restrict__ lens, float* __restrict__ scratch, Shape sh)
{
  const int s = blockIdx.x, hk = blockIdx.y, t = blockIdx.z, tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5, warps = kThreads / 32;
  const int G = sh.H / sh.Hkv, Qr = sh.R / sh.Tb, NQ = Qr * G, D = sh.D, PS = WeightStride(NQ);
  extern __shared__ __align__(16) float smem[];
  // The weights reuse the queries' space once every score is computed.
  float* qs = smem;
  float* ps = smem;
  float* ms = smem + max(NQ * D, kThreads * PS);
  float* ls = ms + NQ;
  // The slice's pool offsets per position (the block table read once).
  size_t* base = reinterpret_cast<size_t*>(smem + BaseOffset(NQ, D));

  // Query i of the block: row t * Qr + i / G, head hk * G + i % G.
  int longest = 0;
  for (int j = 0; j < Qr; ++j) longest = max(longest, lens[t * Qr + j]);
  const int start = s * kThreads, n = min(kThreads, longest - start);

  for (int e = tid; e < NQ * D; e += kThreads) {
    const int i = e / D, d = e % D;
    const int row = t * Qr + i / G, head = hk * G + i % G;
    qs[e] = q[(static_cast<size_t>(row) * sh.H + head) * D + d] * sh.scale;
  }
  __syncthreads();

  const size_t rowStride = static_cast<size_t>(sh.Hkv) * D;  // one position of the pool
  if (tid < n) {
    const int pos = start + tid;
    base[tid] = (static_cast<size_t>(tables[t * sh.M + pos / sh.bs]) * sh.bs + pos % sh.bs) * rowStride + hk * D;
  }
  __syncthreads();
  auto at = [&](int j) { return base[j]; };
  if (n > 0) {
    // Scores, -inf past a row's length.
    float sc[MAXQ];
#pragma unroll
    for (int i = 0; i < MAXQ; ++i) sc[i] = 0.f;
    if (tid < n) {
      const KV* key = kc + at(tid);
      float cur[16], nxt[16];
      Load16(key, cur);
      for (int d0 = 0; d0 < D; d0 += 16) {
        if (d0 + 16 < D) Load16(key + d0 + 16, nxt);
#pragma unroll
        for (int i = 0; i < MAXQ; ++i) {
          if (i < NQ) {
            const float4* qi = reinterpret_cast<const float4*>(qs + i * D + d0);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const float4 x = qi[c];
              sc[i] = fmaf(x.x, cur[4 * c], sc[i]);
              sc[i] = fmaf(x.y, cur[4 * c + 1], sc[i]);
              sc[i] = fmaf(x.z, cur[4 * c + 2], sc[i]);
              sc[i] = fmaf(x.w, cur[4 * c + 3], sc[i]);
            }
          }
        }
#pragma unroll
        for (int c = 0; c < 16; ++c) cur[c] = nxt[c];
      }
    }
    __syncthreads();  // every thread is done with the queries
    float* mine = ps + tid * PS;
#pragma unroll
    for (int i = 0; i < MAXQ; ++i) {
      if (i < NQ) mine[i] = tid < n && start + tid < lens[t * Qr + i / G] && Kept(sh, t * Qr + i / G, start + tid) ? sc[i] : -INFINITY;
    }
    __syncthreads();
    // Softmax over the slice, per query.
    for (int i = warp; i < NQ; i += warps) {
      float mx = -INFINITY;
      for (int j = lane; j < kThreads; j += 32) mx = fmaxf(mx, ps[j * PS + i]);
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
      float sum = 0.f;
      for (int j = lane; j < kThreads; j += 32) {
        const float x = ps[j * PS + i];
        const float w = x == -INFINITY ? 0.f : expf(x - mx);
        ps[j * PS + i] = w;
        sum += w;
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
      if (lane == 0) {
        ms[i] = mx;
        ls[i] = sum;
      }
    }
    __syncthreads();
  } else {
    for (int i = tid; i < NQ; i += kThreads) {
      ms[i] = -INFINITY;
      ls[i] = 0.f;
    }
    __syncthreads();
  }
  // Values.
  if (tid < D) {
    float acc[MAXQ];
#pragma unroll
    for (int i = 0; i < MAXQ; ++i) acc[i] = 0.f;
    constexpr int U = 16;
    for (int j0 = 0; j0 < n; j0 += U) {
      float v[U];
#pragma unroll
      for (int u = 0; u < U; ++u) v[u] = j0 + u < n ? Widen(vc[at(j0 + u) + tid]) : 0.f;
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const float4* w = reinterpret_cast<const float4*>(ps + (j0 + u) * PS);
#pragma unroll
        for (int i4 = 0; i4 < MAXQ / 4; ++i4) {
          if (4 * i4 < NQ) {
            const float4 x = w[i4];
            acc[4 * i4] = fmaf(x.x, v[u], acc[4 * i4]);
            acc[4 * i4 + 1] = fmaf(x.y, v[u], acc[4 * i4 + 1]);
            acc[4 * i4 + 2] = fmaf(x.z, v[u], acc[4 * i4 + 2]);
            acc[4 * i4 + 3] = fmaf(x.w, v[u], acc[4 * i4 + 3]);
          }
        }
      }
    }
    // This slice's (max, sum, weighted values) per query: [S, R, H, D + 2].
#pragma unroll
    for (int i = 0; i < MAXQ; ++i) {
      if (i < NQ) {
        const int row = t * Qr + i / G, head = hk * G + i % G;
        float* out = scratch + ((static_cast<size_t>(s) * sh.R + row) * sh.H + head) * (D + 2);
        out[2 + tid] = acc[i];
        if (tid == 0) {
          out[0] = ms[i];
          out[1] = ls[i];
        }
      }
    }
  }
}

// The tensor-core slice kernel, for e4m3fn pools with D = 256 and NQ <= 64
// queries: the same grid, slices and scratch as SliceKernel, the slice taken
// in four chunks of 64 positions with an online softmax. Q.K^T and P.V run as
// mma.sync m16n8k16 in f16 with f32 sums; the e4m3 keys and values are exact in
// f16, and Q and P are each split into a high and a low f16 part (two mmas),
// which keeps about 22 bits of them.
namespace tc {

constexpr int kD = 256, kChunk = 32, kMaxQ = 48;
constexpr int kMaxMT = kMaxQ / 16;
constexpr int kSStride = kChunk + 4;   // f32 scores per query row
constexpr int kPStride = kChunk + 8;   // f16 probabilities per query row
constexpr int kKStride = kD;           // bytes per key or value row

// The byte of row p, column o of a key or value chunk in shared memory: the
// 16-byte pieces of a row are permuted by p % 8, so that the 8 rows a fragment
// reads at one column are in 8 different banks.
__device__ __forceinline__ int Sw(int p, int o) { return p * kKStride + (((o >> 4) ^ (p & 7)) << 4) + (o & 15); }
constexpr int kWarps = kThreads / 32;
constexpr int kDimsPerWarp = kD / kWarps;  // 32: each warp's share of Q.K^T's sum
// Slices of kThreads positions a block takes, under one online softmax: its
// result goes to the first slice of the group, and the others are marked empty.
constexpr int kSlicesPerBlock = 4;
constexpr int kBlockPositions = kSlicesPerBlock * kThreads;

// Shared memory for MQ (a multiple of 16) query rows: the planes below, in order.
// The queries are not here: each warp keeps its share of them in registers.
struct Layout {
  size_t kc, vc, sc, phi, plo, base, m, l, alpha, len, bytes;
  __host__ __device__ explicit Layout(int mq, bool det = false)
  {
    size_t o = 0;
    auto take = [&](size_t n) { const size_t at = o; o = (o + n + 15) / 16 * 16; return at; };
    kc = take(2 * kChunk * kKStride);  // two buffers: the chunk in use and the next one in flight
    vc = take(2 * kChunk * kKStride);
    sc = take((det ? 8 : 4) * mq * kSStride);  // f32 partial sums, or 64-bit fixed point
    phi = take(2 * mq * kPStride);
    plo = take(2 * mq * kPStride);
    base = take(sizeof(uint32_t) * kBlockPositions);
    m = take(4 * mq);
    l = take(4 * mq);
    alpha = take(4 * mq);
    len = take(4 * mq);
    bytes = o;
  }
};

// Partial scores in shared memory: signed 64-bit integers in units of 2^-24, held as unsigned
// (two's complement; the sum wraps the same way), up to |s| < 2^39.
constexpr float kScoreUnit = 16777216.f;  // 2^24
__device__ __forceinline__ unsigned long long ToScoreUnits(float x)
{
  return static_cast<unsigned long long>(__float2ll_rn(x * kScoreUnit));
}
__device__ __forceinline__ float FromScoreUnits(unsigned long long v)
{
  return __ll2float_rn(static_cast<long long>(v)) * (1.f / kScoreUnit);
}

__device__ __forceinline__ uint32_t E4m3x2ToF16x2(uint16_t two)
{
  uint32_t r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(two));
  return r;
}

__device__ __forceinline__ void MmaF16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1)
{
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t PackHalf2(__half lo, __half hi)
{
  return static_cast<uint32_t>(__half_as_ushort(lo)) | (static_cast<uint32_t>(__half_as_ushort(hi)) << 16);
}

// Grid (S, Hkv, Tb), kThreads threads, MQ = 16 ceil(NQ / 16) <= kMaxQ:
//   scores  warp w sums Q.K^T over dims [32 w, 32 w + 32) for every query and
//           position of the chunk (its query fragments in registers, the high and
//           the low f16 part of each), adding its partial sums into the chunk's
//           scores in shared memory (f32 atomics; with DET, 64-bit integers in
//           units of 2^-24, kScoreUnit, whose sums do not depend on the order the
//           warps add in, so a score is the same in every run);
//   softmax warp per query, lane per position, online over the chunks;
//   values  warp w takes (query tile, 8-dim tile) pairs w, w + 8, ...: P.V into
//           registers, rescaled by each chunk's correction.
template <int MT, bool DET>
__global__ void __launch_bounds__(kThreads, MT <= 2 ? 2 : 1) SliceKernelTc(
    const float* __restrict__ q, const uint8_t* __restrict__ kpool, const uint8_t* __restrict__ vpool,
    const int* __restrict__ tables, const int* __restrict__ lens, float* __restrict__ scratch, Shape sh)
{
  extern __shared__ __align__(16) unsigned char raw[];
  const int s = blockIdx.x, hk = blockIdx.y, t = blockIdx.z, tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int G = sh.H / sh.Hkv, Qr = sh.R / sh.Tb, NQ = Qr * G;
  constexpr int MQ = MT * 16;
  const Layout ly(MQ, DET);
  uint8_t* kcAll = raw + ly.kc;
  uint8_t* vcAll = raw + ly.vc;
  float* sc = reinterpret_cast<float*>(raw + ly.sc);
  unsigned long long* sci = reinterpret_cast<unsigned long long*>(raw + ly.sc);
  __half* phi = reinterpret_cast<__half*>(raw + ly.phi);
  __half* plo = reinterpret_cast<__half*>(raw + ly.plo);
  uint32_t* base = reinterpret_cast<uint32_t*>(raw + ly.base);  // pool row of each position
  float* sm_m = reinterpret_cast<float*>(raw + ly.m);
  float* sm_l = reinterpret_cast<float*>(raw + ly.l);
  float* sm_alpha = reinterpret_cast<float*>(raw + ly.alpha);
  int* sm_len = reinterpret_cast<int*>(raw + ly.len);

  int longest = 0;
  for (int j = 0; j < Qr; ++j) longest = max(longest, lens[t * Qr + j]);
  const int start = s * kBlockPositions, n = min(kBlockPositions, longest - start);
  const int r = lane >> 2, c2 = 2 * (lane & 3);

  // This warp's query fragments: query tile mt, k-steps of its dims, high and low parts.
  uint32_t qa[MT][kDimsPerWarp / 16][2][4];
  {
    auto qv = [&](int i, int d) -> float {
      if (i >= NQ) return 0.f;
      return q[(static_cast<size_t>(t * Qr + i / G) * sh.H + hk * G + i % G) * kD + d] * sh.scale;
    };
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
      for (int ks = 0; ks < kDimsPerWarp / 16; ++ks) {
        const int d = warp * kDimsPerWarp + ks * 16 + c2;
        const int rows[4] = {mt * 16 + r, mt * 16 + r + 8, mt * 16 + r, mt * 16 + r + 8};
        const int cols[4] = {d, d, d + 8, d + 8};
#pragma unroll
        for (int f = 0; f < 4; ++f) {
          const float x0 = qv(rows[f], cols[f]), x1 = qv(rows[f], cols[f] + 1);
          const __half h0 = __float2half_rn(x0), h1 = __float2half_rn(x1);
          qa[mt][ks][0][f] = PackHalf2(h0, h1);
          qa[mt][ks][1][f] = PackHalf2(__float2half_rn(x0 - __half2float(h0)), __float2half_rn(x1 - __half2float(h1)));
        }
      }
    }
  }
  for (int i = tid; i < MQ; i += kThreads) {
    sm_m[i] = -INFINITY;
    sm_l[i] = 0.f;
    sm_len[i] = i < NQ ? lens[t * Qr + i / G] : 0;
  }
  const size_t rowStride = static_cast<size_t>(sh.Hkv) * kD;
  for (int i = tid; i < n; i += kThreads) {
    const int pos = start + i;
    base[i] = static_cast<uint32_t>(tables[t * sh.M + pos / sh.bs]) * sh.bs + pos % sh.bs;
  }
  __syncthreads();

  constexpr int kTiles = MT * (kD / 8) / kWarps;  // value tiles per warp
  float acc[kTiles][4];
#pragma unroll
  for (int i = 0; i < kTiles; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

  auto fetch = [&](int c0, int buf) {
    const int cn = min(kChunk, n - c0);
    for (int e = tid; e < kChunk * (kD / 16); e += kThreads) {
      const int p = e / (kD / 16), o = (e % (kD / 16)) * 16;
      const size_t at = p < cn ? base[c0 + p] * rowStride + hk * kD + o : 0;
      __pipeline_memcpy_async(kcAll + buf * kChunk * kKStride + Sw(p, o), kpool + at, 16, p < cn ? 0 : 16);
      __pipeline_memcpy_async(vcAll + buf * kChunk * kKStride + Sw(p, o), vpool + at, 16, p < cn ? 0 : 16);
    }
    __pipeline_commit();
  };
  if (n > 0) fetch(0, 0);
  for (int c0 = 0, buf = 0; c0 < n; c0 += kChunk, buf ^= 1) {
    const int cn = min(kChunk, n - c0);
    for (int i = tid; i < MQ * kSStride; i += kThreads) {
      if constexpr (DET) sci[i] = 0ull; else sc[i] = 0.f;
    }
    if (c0 + kChunk < n) {
      fetch(c0 + kChunk, buf ^ 1);
      __pipeline_wait_prior(1);
    } else {
      __pipeline_wait_prior(0);
    }
    __syncthreads();
    const uint8_t* kc = kcAll + buf * kChunk * kKStride;
    const uint8_t* vcb = vcAll + buf * kChunk * kKStride;
    // Partial scores over this warp's dims, every query tile and 8-position tile.
#pragma unroll
    for (int nt = 0; nt < kChunk / 8; ++nt) {
      const int p = nt * 8 + r;
      float c[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) c[mt][0] = c[mt][1] = c[mt][2] = c[mt][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < kDimsPerWarp / 16; ++ks) {
        const int d = warp * kDimsPerWarp + ks * 16 + c2;
        const uint32_t b0 = E4m3x2ToF16x2(*reinterpret_cast<const uint16_t*>(kc + Sw(p, d)));
        const uint32_t b1 = E4m3x2ToF16x2(*reinterpret_cast<const uint16_t*>(kc + Sw(p, d + 8)));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          MmaF16(c[mt], qa[mt][ks][0], b0, b1);
          MmaF16(c[mt], qa[mt][ks][1], b0, b1);
        }
      }
      const int pc = nt * 8 + c2;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int qr = mt * 16 + r;
        if constexpr (DET) {
          atomicAdd(sci + qr * kSStride + pc, ToScoreUnits(c[mt][0]));
          atomicAdd(sci + qr * kSStride + pc + 1, ToScoreUnits(c[mt][1]));
          atomicAdd(sci + (qr + 8) * kSStride + pc, ToScoreUnits(c[mt][2]));
          atomicAdd(sci + (qr + 8) * kSStride + pc + 1, ToScoreUnits(c[mt][3]));
        } else {
          atomicAdd(sc + qr * kSStride + pc, c[mt][0]);
          atomicAdd(sc + qr * kSStride + pc + 1, c[mt][1]);
          atomicAdd(sc + (qr + 8) * kSStride + pc, c[mt][2]);
          atomicAdd(sc + (qr + 8) * kSStride + pc + 1, c[mt][3]);
        }
      }
    }
    __syncthreads();
    // Online softmax per query: warp w takes queries w, w + 8, ...; lane j position j;
    // -inf past a query's length.
    for (int qi = warp; qi < MQ; qi += kWarps) {
      const int pos = start + c0 + lane;
      const float sv = DET ? FromScoreUnits(sci[qi * kSStride + lane]) : sc[qi * kSStride + lane];
      const float s0 = lane < cn && pos < sm_len[qi] && Kept(sh, t * Qr + qi / G, pos) ? sv : -INFINITY;
      float mx = s0;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
      const float mOld = sm_m[qi], mNew = fmaxf(mOld, mx);
      const float p0 = s0 == -INFINITY ? 0.f : expf(s0 - mNew);
      float sum = p0;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
      const __half h0 = __float2half_rn(p0);
      phi[qi * kPStride + lane] = h0;
      plo[qi * kPStride + lane] = __float2half_rn(p0 - __half2float(h0));
      if (lane == 0) {
        const float a = mOld == -INFINITY ? (mNew == -INFINITY ? 1.f : 0.f) : expf(mOld - mNew);
        sm_alpha[qi] = a;
        sm_l[qi] = sm_l[qi] * a + sum;
        sm_m[qi] = mNew;
      }
    }
    __syncthreads();
    // O = O * alpha + P . V over this warp's tiles.
#pragma unroll
    for (int i = 0; i < kTiles; ++i) {
      const int tile = warp + i * kWarps;
      const int mt = tile / (kD / 8), nt = tile % (kD / 8);
      const int qr = mt * 16 + r, dcol = nt * 8 + r;
      const float a0 = sm_alpha[qr], a1 = sm_alpha[qr + 8];
      acc[i][0] *= a0;
      acc[i][1] *= a0;
      acc[i][2] *= a1;
      acc[i][3] *= a1;
#pragma unroll
      for (int ks = 0; ks < kChunk / 16; ++ks) {
        const int pk = ks * 16 + c2;
        const uint16_t v01 = static_cast<uint16_t>(vcb[Sw(pk, dcol)] | (vcb[Sw(pk + 1, dcol)] << 8));
        const uint16_t v89 = static_cast<uint16_t>(vcb[Sw(pk + 8, dcol)] | (vcb[Sw(pk + 9, dcol)] << 8));
        const uint32_t b0 = E4m3x2ToF16x2(v01), b1 = E4m3x2ToF16x2(v89);
#pragma unroll
        for (int part = 0; part < 2; ++part) {
          const __half* ps = part == 0 ? phi : plo;
          uint32_t a[4];
          a[0] = *reinterpret_cast<const uint32_t*>(ps + qr * kPStride + pk);
          a[1] = *reinterpret_cast<const uint32_t*>(ps + (qr + 8) * kPStride + pk);
          a[2] = *reinterpret_cast<const uint32_t*>(ps + qr * kPStride + pk + 8);
          a[3] = *reinterpret_cast<const uint32_t*>(ps + (qr + 8) * kPStride + pk + 8);
          MmaF16(acc[i], a, b0, b1);
        }
      }
    }
    __syncthreads();
  }
  // The group's (max, sum, weighted values) per query in its first slice of
  // [S, R, H, D + 2]; its other slices empty (max -inf).
#pragma unroll
  for (int i = 0; i < kTiles; ++i) {
    const int tile = warp + i * kWarps;
    const int mt = tile / (kD / 8), nt = tile % (kD / 8);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int qi = mt * 16 + r + 8 * h;
      if (qi >= NQ) continue;
      const int row = t * Qr + qi / G, head = hk * G + qi % G;
      const size_t slice = static_cast<size_t>(sh.R) * sh.H * (kD + 2);
      float* out = scratch + (static_cast<size_t>(s) * kSlicesPerBlock * slice) + (static_cast<size_t>(row) * sh.H + head) * (kD + 2);
      const int d = nt * 8 + c2;
      out[2 + d] = acc[i][2 * h];
      out[2 + d + 1] = acc[i][2 * h + 1];
      if (nt == 0 && c2 == 0) {
        out[0] = n > 0 ? sm_m[qi] : -INFINITY;
        out[1] = n > 0 ? sm_l[qi] : 0.f;
        for (int k = 1; k < kSlicesPerBlock && s * kSlicesPerBlock + k < sh.S; ++k) out[k * slice] = -INFINITY;
      }
    }
  }
}

// Launches the tensor-core slice kernel for NQ queries per table and KV head.
inline cudaError_t LaunchTc(const void* q, const void* k, const void* v, const void* tables, const void* lens,
                            void* scratch, const Shape& sh, int NQ, cudaStream_t stream)
{
  const int mt = (NQ + 15) / 16;
  const bool det = tlaloc_kernels::Deterministic();
  auto run = [&](auto kernel) {
    const int smem = static_cast<int>(Layout(mt * 16, det).bytes);
    cudaError_t e = tlaloc_kernels::AllowMaxSharedMemory(kernel);
    if (e != cudaSuccess) return e;
    kernel<<<dim3((sh.S + kSlicesPerBlock - 1) / kSlicesPerBlock, sh.Hkv, sh.Tb), kThreads, smem, stream>>>(
        static_cast<const float*>(q), static_cast<const uint8_t*>(k), static_cast<const uint8_t*>(v),
        static_cast<const int*>(tables), static_cast<const int*>(lens), static_cast<float*>(scratch), sh);
    return cudaGetLastError();
  };
  if (det) {
    if (mt == 1) return run(SliceKernelTc<1, true>);
    if (mt == 2) return run(SliceKernelTc<2, true>);
    return run(SliceKernelTc<3, true>);
  }
  if (mt == 1) return run(SliceKernelTc<1, false>);
  if (mt == 2) return run(SliceKernelTc<2, false>);
  return run(SliceKernelTc<3, false>);
}

// Prefill rows: many tokens of one sequence. A block takes a tile of 64 / G
// tokens' query rows for one KV head (at most kPrefillRows) and walks the context
// they read in chunks of 32 positions under one online softmax, writing
// normalized outputs to out [R, H, D]. Warp w owns query rows 16 (w % 4) on and
// dims 128 (w / 4) on: it keeps its share of the queries in registers, adds
// Q.K^T with the warp owning the other dims through shared memory, and keeps its
// rows' softmax and its 16 x 128 share of the output in registers. Values are
// transposed in shared memory, so the value fragments are two adjacent bytes.
constexpr int kPrefillRows = 64;
constexpr int kVtStride = kChunk + 8;  // bytes per transposed value row (dim)

struct PrefillLayout {
  size_t kc, vc, vt, sc, len, bytes;
  __host__ __device__ PrefillLayout()
  {
    size_t o = 0;
    auto take = [&](size_t n) { const size_t at = o; o = (o + n + 15) / 16 * 16; return at; };
    kc = take(2 * kChunk * kKStride);
    vc = take(2 * kChunk * kKStride);
    vt = take(kD * kVtStride);
    sc = take(sizeof(float) * kWarps * 32 * 16);
    len = take(sizeof(int) * kPrefillRows);
    bytes = o;
  }
};

//
// SPLIT: decode and verify rows (at most kPrefillRows query vectors per table and
// KV head), grid (ceil(S / kSlicesPerBlock), Hkv, Tb): a block takes all the rows
// of table blockIdx.z and the kBlockPositions positions from blockIdx.x
// kBlockPositions on, and writes SliceKernelTc's (max, sum, values) per query to
// the scratch (the group's first slice; the others empty) for CombineKernel.
// Slices per block of the split form below: more than SliceKernelTc takes,
// because each block loads its 64 query rows once (12: the 5-row MTP head pass
// of Qwen3.6-35B-A3B at 32K, 1.23 to 0.79 ms against SliceKernelTc).
constexpr int kSplitSlices = 12;
constexpr int kSplitPositions = kSplitSlices * kThreads;

template <bool SPLIT>
__global__ void __launch_bounds__(kThreads, 1) PrefillKernelTc(
    const float* __restrict__ q, const uint8_t* __restrict__ kpool, const uint8_t* __restrict__ vpool,
    const int* __restrict__ tables, const int* __restrict__ lens, float* __restrict__ outp, Shape sh, int TQ)
{
  extern __shared__ __align__(16) unsigned char raw[];
  const int tileIdx = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5, rg = warp & 3, half = warp >> 2;
  const int G = sh.H / sh.Hkv, Qr = sh.R / sh.Tb;
  const int tilesPerTable = SPLIT ? 1 : (Qr + TQ - 1) / TQ;
  const int t = SPLIT ? blockIdx.z : tileIdx / tilesPerTable;
  const int row0 = SPLIT ? t * Qr : t * Qr + (tileIdx % tilesPerTable) * TQ;
  const int NQ = SPLIT ? Qr * G : min(TQ, t * Qr + Qr - row0) * G;
  // Warps whose 16 rows hold no query skip the products.
  const bool live = rg * 16 < NQ;
  const PrefillLayout ly;
  uint8_t* kcAll = raw + ly.kc;
  uint8_t* vcAll = raw + ly.vc;
  uint8_t* vt = raw + ly.vt;
  float* sc = reinterpret_cast<float*>(raw + ly.sc);
  int* sm_len = reinterpret_cast<int*>(raw + ly.len);
  const int r = lane >> 2, c2 = 2 * (lane & 3);
  const int d0 = half * (kD / 2);

  if (tid < kPrefillRows) sm_len[tid] = tid < NQ ? lens[row0 + tid / G] : 0;
  __syncthreads();
  int n = 0;
  for (int j = 0; j < kPrefillRows; ++j) n = max(n, sm_len[j]);
  const int qr0 = rg * 16 + r, qr1 = qr0 + 8;
  const int len0 = sm_len[qr0], len1 = sm_len[qr1];
  // The positions this block reads.
  const int begin = SPLIT ? blockIdx.x * kSplitPositions : 0;
  const int end = SPLIT ? min(n, begin + kSplitPositions) : n;

  uint32_t qa[kD / 32][2][4];
  {
    auto qv = [&](int i, int d) -> float {
      if (i >= NQ) return 0.f;
      return q[(static_cast<size_t>(row0 + i / G) * sh.H + hk * G + i % G) * kD + d] * sh.scale;
    };
#pragma unroll
    for (int ks = 0; ks < kD / 32; ++ks) {
      const int d = d0 + ks * 16 + c2;
      const int rr[4] = {qr0, qr1, qr0, qr1};
      const int cc[4] = {d, d, d + 8, d + 8};
#pragma unroll
      for (int f = 0; f < 4; ++f) {
        const float x0 = qv(rr[f], cc[f]), x1 = qv(rr[f], cc[f] + 1);
        const __half h0 = __float2half_rn(x0), h1 = __float2half_rn(x1);
        qa[ks][0][f] = PackHalf2(h0, h1);
        qa[ks][1][f] = PackHalf2(__float2half_rn(x0 - __half2float(h0)), __float2half_rn(x1 - __half2float(h1)));
      }
    }
  }

  float acc[kD / 16][4];
#pragma unroll
  for (int i = 0; i < kD / 16; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

  const size_t rowStride = static_cast<size_t>(sh.Hkv) * kD;
  auto fetch = [&](int c0, int buf) {
    const int cn = min(kChunk, end - c0);
    for (int e = tid; e < kChunk * (kD / 16); e += kThreads) {
      const int p = e / (kD / 16), o = (e % (kD / 16)) * 16;
      size_t at = 0;
      if (p < cn) {
        const int pos = c0 + p;
        at = (static_cast<size_t>(tables[t * sh.M + pos / sh.bs]) * sh.bs + pos % sh.bs) * rowStride + hk * kD + o;
      }
      __pipeline_memcpy_async(kcAll + buf * kChunk * kKStride + Sw(p, o), kpool + at, 16, p < cn ? 0 : 16);
      __pipeline_memcpy_async(vcAll + buf * kChunk * kKStride + Sw(p, o), vpool + at, 16, p < cn ? 0 : 16);
    }
    __pipeline_commit();
  };
  if (begin < end) fetch(begin, 0);
  for (int c0 = begin, buf = 0; c0 < end; c0 += kChunk, buf ^= 1) {
    if (c0 + kChunk < end) {
      fetch(c0 + kChunk, buf ^ 1);
      __pipeline_wait_prior(1);
    } else {
      __pipeline_wait_prior(0);
    }
    __syncthreads();
    const uint8_t* kc = kcAll + buf * kChunk * kKStride;
    const uint8_t* vc = vcAll + buf * kChunk * kKStride;
    // Values transposed: vt[d][p], four positions at a time.
    for (int e = tid; e < kD * (kChunk / 4); e += kThreads) {
      const int d = e % kD, p = (e / kD) * 4;
      const uint32_t w = vc[Sw(p, d)] | (vc[Sw(p + 1, d)] << 8) | (vc[Sw(p + 2, d)] << 16) | (vc[Sw(p + 3, d)] << 24);
      *reinterpret_cast<uint32_t*>(vt + d * kVtStride + p) = w;
    }
    // This warp's share of Q.K^T for its 16 rows and the chunk's 32 positions.
    float s[kChunk / 8][4];
#pragma unroll
    for (int nt = 0; nt < kChunk / 8; ++nt) {
      s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
      if (!live) continue;
      const int p = nt * 8 + r;
#pragma unroll
      for (int ks = 0; ks < kD / 32; ++ks) {
        const int d = d0 + ks * 16 + c2;
        const uint32_t b0 = E4m3x2ToF16x2(*reinterpret_cast<const uint16_t*>(kc + Sw(p, d)));
        const uint32_t b1 = E4m3x2ToF16x2(*reinterpret_cast<const uint16_t*>(kc + Sw(p, d + 8)));
        MmaF16(s[nt], qa[ks][0], b0, b1);
        MmaF16(s[nt], qa[ks][1], b0, b1);
      }
    }
    float* mine = sc + (warp * 16) * 32 + lane;
#pragma unroll
    for (int i = 0; i < 16; ++i) mine[i * 32] = s[i / 4][i % 4];
    __syncthreads();
    const float* other = sc + ((warp ^ 4) * 16) * 32 + lane;
#pragma unroll
    for (int i = 0; i < 16; ++i) s[i / 4][i % 4] += other[i * 32];

    // Online softmax over the chunk, for rows qr0 (s[.][0..1]) and qr1 (s[.][2..3]).
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < kChunk / 8; ++nt) {
      const int pos = c0 + nt * 8 + c2;
      if (pos >= len0) s[nt][0] = -INFINITY;
      if (pos + 1 >= len0) s[nt][1] = -INFINITY;
      if (pos >= len1) s[nt][2] = -INFINITY;
      if (pos + 1 >= len1) s[nt][3] = -INFINITY;
      if (sh.keep != nullptr) {
        // pos is even and a chunk is even, so pos and pos + 1 share one.
        if (pos < len0 && !Kept(sh, row0 + qr0 / G, pos)) s[nt][0] = s[nt][1] = -INFINITY;
        if (pos < len1 && !Kept(sh, row0 + qr1 / G, pos)) s[nt][2] = s[nt][3] = -INFINITY;
      }
      mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
      mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
    }
    const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
    const float a0 = m0 == -INFINITY ? 1.f : expf(m0 - mn0), a1 = m1 == -INFINITY ? 1.f : expf(m1 - mn1);
    m0 = mn0;
    m1 = mn1;
    float sum0 = 0.f, sum1 = 0.f;
    uint32_t pa[kChunk / 16][2][4];
#pragma unroll
    for (int nt = 0; nt < kChunk / 8; ++nt) {
      float p[4];
#pragma unroll
      for (int f = 0; f < 4; ++f) {
        const float mref = f < 2 ? mn0 : mn1;
        p[f] = s[nt][f] == -INFINITY ? 0.f : expf(s[nt][f] - mref);
      }
      sum0 += p[0] + p[1];
      sum1 += p[2] + p[3];
      const __half h[4] = {__float2half_rn(p[0]), __float2half_rn(p[1]), __float2half_rn(p[2]), __float2half_rn(p[3])};
      const int ks = nt / 2, f0 = (nt % 2) * 2;
      pa[ks][0][f0] = PackHalf2(h[0], h[1]);
      pa[ks][0][f0 + 1] = PackHalf2(h[2], h[3]);
      pa[ks][1][f0] = PackHalf2(__float2half_rn(p[0] - __half2float(h[0])), __float2half_rn(p[1] - __half2float(h[1])));
      pa[ks][1][f0 + 1] = PackHalf2(__float2half_rn(p[2] - __half2float(h[2])), __float2half_rn(p[3] - __half2float(h[3])));
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      sum0 += __shfl_xor_sync(0xffffffffu, sum0, o);
      sum1 += __shfl_xor_sync(0xffffffffu, sum1, o);
    }
    l0 = l0 * a0 + sum0;
    l1 = l1 * a1 + sum1;

    // P.V for this warp's 128 dims.
#pragma unroll
    for (int nt = 0; nt < kD / 16 && live; ++nt) {
      acc[nt][0] *= a0;
      acc[nt][1] *= a0;
      acc[nt][2] *= a1;
      acc[nt][3] *= a1;
      const uint8_t* vrow = vt + (d0 + nt * 8 + r) * kVtStride + c2;
#pragma unroll
      for (int ks = 0; ks < kChunk / 16; ++ks) {
        const uint32_t b0 = E4m3x2ToF16x2(*reinterpret_cast<const uint16_t*>(vrow + ks * 16));
        const uint32_t b1 = E4m3x2ToF16x2(*reinterpret_cast<const uint16_t*>(vrow + ks * 16 + 8));
        MmaF16(acc[nt], pa[ks][0], b0, b1);
        MmaF16(acc[nt], pa[ks][1], b0, b1);
      }
    }
    __syncthreads();
  }
  if constexpr (SPLIT) {
    const size_t slice = static_cast<size_t>(sh.R) * sh.H * (kD + 2);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int qi = h == 0 ? qr0 : qr1;
      if (qi >= NQ) continue;
      float* o = outp + static_cast<size_t>(blockIdx.x) * kSplitSlices * slice +
                 (static_cast<size_t>(row0 + qi / G) * sh.H + hk * G + qi % G) * (kD + 2);
#pragma unroll
      for (int nt = 0; nt < kD / 16; ++nt) {
        o[2 + d0 + nt * 8 + c2] = acc[nt][2 * h];
        o[2 + d0 + nt * 8 + c2 + 1] = acc[nt][2 * h + 1];
      }
      if (half == 0 && c2 == 0) {
        const float m = h == 0 ? m0 : m1;
        o[0] = m;
        o[1] = h == 0 ? l0 : l1;
        for (int k = 1; k < kSplitSlices && blockIdx.x * kSplitSlices + k < sh.S; ++k) o[k * slice] = -INFINITY;
      }
    }
    return;
  }
  const float inv0 = l0 > 0.f ? 1.f / l0 : 0.f, inv1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
  for (int nt = 0; nt < kD / 16; ++nt) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int qi = h == 0 ? qr0 : qr1;
      if (qi >= NQ) continue;
      const float inv = h == 0 ? inv0 : inv1;
      float* o = outp + (static_cast<size_t>(row0 + qi / G) * sh.H + hk * G + qi % G) * kD + d0 + nt * 8 + c2;
      *reinterpret_cast<float2*>(o) = make_float2(acc[nt][2 * h] * inv, acc[nt][2 * h + 1] * inv);
    }
  }
}

// Prefill attention: tiles of 64 / G tokens per block (G <= 64).
inline cudaError_t LaunchPrefillTc(const void* q, const void* k, const void* v, const void* tables, const void* lens,
                                   void* out, const Shape& sh, cudaStream_t stream)
{
  const int G = sh.H / sh.Hkv, Qr = sh.R / sh.Tb;
  if (G > kPrefillRows || sh.D != kD) return cudaErrorInvalidValue;
  const int TQ = kPrefillRows / G;
  const int tiles = sh.Tb * ((Qr + TQ - 1) / TQ);
  cudaError_t e = tlaloc_kernels::AllowMaxSharedMemory(PrefillKernelTc<false>);
  if (e != cudaSuccess) return e;
  PrefillKernelTc<false><<<dim3(tiles, sh.Hkv), kThreads, PrefillLayout().bytes, stream>>>(
      static_cast<const float*>(q), static_cast<const uint8_t*>(k), static_cast<const uint8_t*>(v),
      static_cast<const int*>(tables), static_cast<const int*>(lens), static_cast<float*>(out), sh, TQ);
  return cudaGetLastError();
}

// Decode and verify rows through PrefillKernelTc's layout, the context split in
// blocks of kBlockPositions: partial results in the scratch, for CombineKernel.
inline cudaError_t LaunchSplitTc(const void* q, const void* k, const void* v, const void* tables, const void* lens,
                                 void* scratch, const Shape& sh, cudaStream_t stream)
{
  cudaError_t e = tlaloc_kernels::AllowMaxSharedMemory(PrefillKernelTc<true>);
  if (e != cudaSuccess) return e;
  PrefillKernelTc<true><<<dim3((sh.S + kSplitSlices - 1) / kSplitSlices, sh.Hkv, sh.Tb), kThreads,
                          PrefillLayout().bytes, stream>>>(
      static_cast<const float*>(q), static_cast<const uint8_t*>(k), static_cast<const uint8_t*>(v),
      static_cast<const int*>(tables), static_cast<const int*>(lens), static_cast<float*>(scratch), sh, 0);
  return cudaGetLastError();
}

// Prefill rows with f16 queries and probabilities (FlashAttention's precision;
// the keys and values are exact in f16): a block of 8 warps takes a tile of 128 / G
// tokens' query rows for one KV head, and warp w owns rows 16 w to 16 w + 15 over
// all 256 dims: its queries (64 registers) and its 16 x 256 share of the output
// (128) stay in registers, and it runs Q.K^T, the online softmax and P.V of each
// 32-position chunk on its own, reading key and value fragments with ldmatrix.
// The block meets once per chunk c, then issues chunk c + 2's e4m3 rows, widens
// chunk c + 1's to f16 and attends over chunk c. Scores are in base 2 (log2(e) is
// in the query scale), and the output is rescaled only where a row's max moved.
// PrefillKernelTc keeps about 22 bits of Q and P at less than half the rate.
constexpr int kF16Warps = 8;
constexpr int kF16Threads = 32 * kF16Warps;
constexpr int kF16Rows = 16 * kF16Warps;
constexpr int kF16Stride = kD + 8;  // f16 per widened row: the 8 rows of an ldmatrix in 8 bank groups

struct F16Layout {
  size_t kc, vc, kh, vh, len, bytes;
  __host__ __device__ F16Layout()
  {
    size_t o = 0;
    auto take = [&](size_t n) { const size_t at = o; o = (o + n + 15) / 16 * 16; return at; };
    kc = take(2 * kChunk * kKStride);  // e4m3 rows as read, two buffers
    vc = take(2 * kChunk * kKStride);
    kh = take(2 * 2 * kChunk * kF16Stride);  // widened, two buffers
    vh = take(2 * 2 * kChunk * kF16Stride);
    len = take(sizeof(int) * kF16Rows);
    bytes = o;
  }
};

__device__ __forceinline__ void Ldmatrix4(uint32_t* r, const __half* at, bool trans)
{
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(at));
  if (trans)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
  else
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__global__ void __launch_bounds__(kF16Threads, 1) PrefillKernelF16(
    const float* __restrict__ q, const uint8_t* __restrict__ kpool, const uint8_t* __restrict__ vpool,
    const int* __restrict__ tables, const int* __restrict__ lens, float* __restrict__ outp, Shape sh, int TQ)
{
  extern __shared__ __align__(16) unsigned char raw[];
  const int tileIdx = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int G = sh.H / sh.Hkv, Qr = sh.R / sh.Tb;
  const int tilesPerTable = (Qr + TQ - 1) / TQ;
  const int t = tileIdx / tilesPerTable;
  const int row0 = t * Qr + (tileIdx % tilesPerTable) * TQ;
  const int NQ = min(TQ, t * Qr + Qr - row0) * G;
  const F16Layout ly;
  uint8_t* kcAll = raw + ly.kc;
  uint8_t* vcAll = raw + ly.vc;
  __half* khAll = reinterpret_cast<__half*>(raw + ly.kh);
  __half* vhAll = reinterpret_cast<__half*>(raw + ly.vh);
  int* sm_len = reinterpret_cast<int*>(raw + ly.len);
  const int r = lane >> 2, c2 = 2 * (lane & 3);
  // ldmatrix row addresses: lane l gives row l % 8 of matrix l / 8.
  const int lm = lane >> 3, lr = lane & 7;

  if (tid < kF16Rows) sm_len[tid] = tid < NQ ? lens[row0 + tid / G] : 0;
  __syncthreads();
  int n = 0;
  for (int j = 0; j < kF16Rows; ++j) n = max(n, sm_len[j]);
  const int qr0 = warp * 16 + r, qr1 = qr0 + 8;
  const int len0 = sm_len[qr0], len1 = sm_len[qr1];
  // The positions this warp's rows read; a warp whose rows hold no query skips the products.
  int mine = 0;
  for (int j = warp * 16; j < warp * 16 + 16; ++j) mine = max(mine, sm_len[j]);

  // Scores in base 2: the queries scaled by log2(e) too, and exp2 for exp.
  const float qscale = sh.scale * 1.4426950408889634f;
  uint32_t qa[kD / 16][4];
  {
    auto qv = [&](int i, int d) -> float {
      if (i >= NQ) return 0.f;
      return q[(static_cast<size_t>(row0 + i / G) * sh.H + hk * G + i % G) * kD + d] * qscale;
    };
#pragma unroll
    for (int ks = 0; ks < kD / 16; ++ks) {
      const int d = ks * 16 + c2;
      const int rr[4] = {qr0, qr1, qr0, qr1};
      const int cc[4] = {d, d, d + 8, d + 8};
#pragma unroll
      for (int f = 0; f < 4; ++f) qa[ks][f] = PackHalf2(__float2half_rn(qv(rr[f], cc[f])), __float2half_rn(qv(rr[f], cc[f] + 1)));
    }
  }

  float acc[kD / 8][4];
#pragma unroll
  for (int i = 0; i < kD / 8; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

  const size_t rowStride = static_cast<size_t>(sh.Hkv) * kD;
  auto fetch = [&](int c0, int buf) {
    const int cn = min(kChunk, n - c0);
    for (int e = tid; e < kChunk * (kD / 16); e += kF16Threads) {
      const int p = e / (kD / 16), o = (e % (kD / 16)) * 16;
      size_t at = 0;
      if (p < cn) {
        const int pos = c0 + p;
        at = (static_cast<size_t>(tables[t * sh.M + pos / sh.bs]) * sh.bs + pos % sh.bs) * rowStride + hk * kD + o;
      }
      __pipeline_memcpy_async(kcAll + buf * kChunk * kKStride + p * kKStride + o, kpool + at, 16, p < cn ? 0 : 16);
      __pipeline_memcpy_async(vcAll + buf * kChunk * kKStride + p * kKStride + o, vpool + at, 16, p < cn ? 0 : 16);
    }
  };
  // Chunk c's e4m3 rows (landed) to f16, 16 values a thread at a time.
  auto widen = [&](int buf) {
    for (int e = tid; e < 2 * kChunk * (kD / 16); e += kF16Threads) {
      const bool isV = e >= kChunk * (kD / 16);
      const int i = isV ? e - kChunk * (kD / 16) : e;
      const int p = i / (kD / 16), o = (i % (kD / 16)) * 16;
      const uint4 w = *reinterpret_cast<const uint4*>((isV ? vcAll : kcAll) + buf * kChunk * kKStride + p * kKStride + o);
      const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
      uint32_t h[8];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        h[2 * j] = E4m3x2ToF16x2(static_cast<uint16_t>(ws[j] & 0xffffu));
        h[2 * j + 1] = E4m3x2ToF16x2(static_cast<uint16_t>(ws[j] >> 16));
      }
      uint4* dst = reinterpret_cast<uint4*>((isV ? vhAll : khAll) + buf * kChunk * kF16Stride + p * kF16Stride + o);
      dst[0] = make_uint4(h[0], h[1], h[2], h[3]);
      dst[1] = make_uint4(h[4], h[5], h[6], h[7]);
    }
  };
  // Chunk 0 widened and chunk 1 in flight before the loop.
  if (n > 0) {
    fetch(0, 0);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    if (kChunk < n) fetch(kChunk, 1);
    __pipeline_commit();
    widen(0);
  }
  for (int c0 = 0, buf = 0; c0 < n; c0 += kChunk, buf ^= 1) {
    // Chunk c is widened, chunk c + 1 has landed, every warp is done with chunk c - 1.
    __pipeline_wait_prior(0);
    __syncthreads();
    if (c0 + 2 * kChunk < n) fetch(c0 + 2 * kChunk, buf);
    __pipeline_commit();
    if (c0 + kChunk < n) widen(buf ^ 1);
    if (c0 >= mine) continue;
    if (sh.keep != nullptr && kChunk % sh.chunk == 0) {
      // A chunk none of this warp's rows keeps adds nothing to them.
      const int first = (warp * 16) / G, last = (min(warp * 16 + 16, NQ) - 1) / G, per = kChunk / sh.chunk;
      bool any = false;
      for (int e = lane; e < (last - first + 1) * per; e += 32) {
        any = any || __ldg(sh.keep + static_cast<size_t>(row0 + first + e / per) * sh.keepChunks + c0 / sh.chunk + e % per) != 0;
      }
      if (!__any_sync(0xffffffffu, any)) continue;
    }
    const __half* kh = khAll + buf * kChunk * kF16Stride;
    const __half* vh = vhAll + buf * kChunk * kF16Stride;
    // Q.K^T for the warp's 16 rows and the chunk's 32 positions: four independent sums;
    // an ldmatrix.x4 gives an 8-position tile's key fragments for two 16-dim steps.
    float s[kChunk / 8][4];
#pragma unroll
    for (int nt = 0; nt < kChunk / 8; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < kD / 16; ks += 2) {
#pragma unroll
      for (int nt = 0; nt < kChunk / 8; ++nt) {
        uint32_t b[4];
        Ldmatrix4(b, kh + (nt * 8 + lr) * kF16Stride + ks * 16 + lm * 8, false);
        MmaF16(s[nt], qa[ks], b[0], b[1]);
        MmaF16(s[nt], qa[ks + 1], b[2], b[3]);
      }
    }
    // Online softmax over the chunk, for rows qr0 (s[.][0..1]) and qr1 (s[.][2..3]).
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < kChunk / 8; ++nt) {
      const int pos = c0 + nt * 8 + c2;
      if (pos >= len0) s[nt][0] = -INFINITY;
      if (pos + 1 >= len0) s[nt][1] = -INFINITY;
      if (pos >= len1) s[nt][2] = -INFINITY;
      if (pos + 1 >= len1) s[nt][3] = -INFINITY;
      if (sh.keep != nullptr) {
        // pos is even and a chunk is even, so pos and pos + 1 share one.
        if (pos < len0 && !Kept(sh, row0 + qr0 / G, pos)) s[nt][0] = s[nt][1] = -INFINITY;
        if (pos < len1 && !Kept(sh, row0 + qr1 / G, pos)) s[nt][2] = s[nt][3] = -INFINITY;
      }
      mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
      mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
    }
    const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
    const float a0 = m0 == -INFINITY ? 1.f : exp2f(m0 - mn0), a1 = m1 == -INFINITY ? 1.f : exp2f(m1 - mn1);
    m0 = mn0;
    m1 = mn1;
    float sum0 = 0.f, sum1 = 0.f;
    uint32_t pa[kChunk / 16][4];
#pragma unroll
    for (int nt = 0; nt < kChunk / 8; ++nt) {
      float p[4];
#pragma unroll
      for (int f = 0; f < 4; ++f) {
        const float mref = f < 2 ? mn0 : mn1;
        p[f] = s[nt][f] == -INFINITY ? 0.f : exp2f(s[nt][f] - mref);
      }
      sum0 += p[0] + p[1];
      sum1 += p[2] + p[3];
      const int ks = nt / 2, f0 = (nt % 2) * 2;
      pa[ks][f0] = PackHalf2(__float2half_rn(p[0]), __float2half_rn(p[1]));
      pa[ks][f0 + 1] = PackHalf2(__float2half_rn(p[2]), __float2half_rn(p[3]));
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      sum0 += __shfl_xor_sync(0xffffffffu, sum0, o);
      sum1 += __shfl_xor_sync(0xffffffffu, sum1, o);
    }
    l0 = l0 * a0 + sum0;
    l1 = l1 * a1 + sum1;
    // The output rescaled where a row's max moved (most chunks of a long context
    // leave every row's max where it was).
    if (!__all_sync(0xffffffffu, a0 == 1.f && a1 == 1.f)) {
#pragma unroll
      for (int nt = 0; nt < kD / 8; ++nt) {
        acc[nt][0] *= a0;
        acc[nt][1] *= a0;
        acc[nt][2] *= a1;
        acc[nt][3] *= a1;
      }
    }
    // P.V over all 256 dims: an ldmatrix.x4.trans gives 16 positions' value
    // fragments for two 8-dim tiles.
#pragma unroll
    for (int nt = 0; nt < kD / 8; nt += 2) {
#pragma unroll
      for (int ks = 0; ks < kChunk / 16; ++ks) {
        uint32_t b[4];
        Ldmatrix4(b, vh + (ks * 16 + (lm & 1) * 8 + lr) * kF16Stride + nt * 8 + (lm >> 1) * 8, true);
        MmaF16(acc[nt], pa[ks], b[0], b[1]);
        MmaF16(acc[nt + 1], pa[ks], b[2], b[3]);
      }
    }
  }
  const float inv0 = l0 > 0.f ? 1.f / l0 : 0.f, inv1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
  for (int nt = 0; nt < kD / 8; ++nt) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int qi = h == 0 ? qr0 : qr1;
      if (qi >= NQ) continue;
      const float inv = h == 0 ? inv0 : inv1;
      float* o = outp + (static_cast<size_t>(row0 + qi / G) * sh.H + hk * G + qi % G) * kD + nt * 8 + c2;
      *reinterpret_cast<float2*>(o) = make_float2(acc[nt][2 * h] * inv, acc[nt][2 * h + 1] * inv);
    }
  }
}

// Prefill attention with f16 queries and probabilities: tiles of 64 / G tokens per block.
inline cudaError_t LaunchPrefillF16(const void* q, const void* k, const void* v, const void* tables, const void* lens,
                                    void* out, const Shape& sh, cudaStream_t stream)
{
  const int G = sh.H / sh.Hkv, Qr = sh.R / sh.Tb;
  if (G > kF16Rows || sh.D != kD) return cudaErrorInvalidValue;
  const int TQ = kF16Rows / G;
  const int tiles = sh.Tb * ((Qr + TQ - 1) / TQ);
  cudaError_t e = tlaloc_kernels::AllowMaxSharedMemory(PrefillKernelF16);
  if (e != cudaSuccess) return e;
  PrefillKernelF16<<<dim3(tiles, sh.Hkv), kF16Threads, F16Layout().bytes, stream>>>(
      static_cast<const float*>(q), static_cast<const uint8_t*>(k), static_cast<const uint8_t*>(v),
      static_cast<const int*>(tables), static_cast<const int*>(lens), static_cast<float*>(out), sh, TQ);
  return cudaGetLastError();
}

}  // namespace tc

// Grid R * H, kThreads threads: the slices of each (row, head) combined. The
// block finds the max over the slices together, and each slice's weight
// exp(m_s - max) (0 for an empty slice) goes to shared memory ([S] floats); thread
// d then reads dim d of the non-empty slices only. Sums in a fixed order.
__global__ void __launch_bounds__(kThreads) CombineKernel(const float* __restrict__ scratch, float* __restrict__ out, Shape sh)
{
  extern __shared__ float w[];  // [S]
  __shared__ float red[kThreads / 32];
  const int rh = blockIdx.x, D = sh.D, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const size_t slice = static_cast<size_t>(sh.R) * sh.H * (D + 2);
  const float* base = scratch + static_cast<size_t>(rh) * (D + 2);
  auto blockReduce = [&](float v, bool isMax) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float x = __shfl_xor_sync(0xffffffffu, v, o);
      v = isMax ? fmaxf(v, x) : v + x;
    }
    __syncthreads();  // red is free
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float r = red[0];
    for (int k = 1; k < kThreads / 32; ++k) r = isMax ? fmaxf(r, red[k]) : r + red[k];
    return r;
  };
  float mx = -INFINITY;
  for (int s = tid; s < sh.S; s += kThreads) mx = fmaxf(mx, base[s * slice]);
  mx = blockReduce(mx, true);
  float sum = 0.f;
  for (int s = tid; s < sh.S; s += kThreads) {
    const float m = base[s * slice];
    const float ws = m == -INFINITY ? 0.f : expf(m - mx);
    w[s] = ws;
    if (ws != 0.f) sum += base[s * slice + 1] * ws;
  }
  sum = mx == -INFINITY ? 0.f : blockReduce(sum, false);
  __syncthreads();  // w is written
  for (int d = tid; d < D; d += kThreads) {
    float a = 0.f;
    if (mx != -INFINITY) {
      for (int s = 0; s < sh.S; ++s) {
        if (w[s] != 0.f) a += base[s * slice + 2 + d] * w[s];
      }
      a /= sum;
    }
    out[static_cast<size_t>(rh) * D + d] = a;
  }
}

size_t SharedBytes(const Shape& sh)
{
  const int NQ = (sh.R / sh.Tb) * (sh.H / sh.Hkv);
  return sizeof(float) * BaseOffset(NQ, sh.D) + sizeof(size_t) * kThreads;
}

template <typename KV, int MAXQ>
cudaError_t LaunchSlices(const void* q, const void* k, const void* v, const void* tables, const void* lens,
                         void* scratch, const Shape& sh, cudaStream_t stream)
{
  const size_t smem = SharedBytes(sh);
  cudaError_t e = tlaloc_kernels::AllowMaxSharedMemory(SliceKernel<KV, MAXQ>);
  if (e != cudaSuccess) return e;
  SliceKernel<KV, MAXQ><<<dim3(sh.S, sh.Hkv, sh.Tb), kThreads, smem, stream>>>(
      static_cast<const float*>(q), static_cast<const KV*>(k), static_cast<const KV*>(v),
      static_cast<const int*>(tables), static_cast<const int*>(lens), static_cast<float*>(scratch), sh);
  return cudaGetLastError();
}

// Every kernel Launch may use, given the dynamic shared memory it may ask for
// (XLA's INITIALIZE stage, before any CUDA graph is captured).
template <typename KV>
cudaError_t PrepareAttention()
{
  cudaError_t e = cudaSuccess;
  for (auto k : {SliceKernel<KV, 8>, SliceKernel<KV, 16>, SliceKernel<KV, 24>, SliceKernel<KV, 32>, SliceKernel<KV, 40>,
                 SliceKernel<KV, 64>}) {
    if (e == cudaSuccess) e = tlaloc_kernels::AllowMaxSharedMemory(k);
  }
  return e;
}

inline cudaError_t PrepareAttentionKernels()
{
  cudaError_t e = PrepareAttention<__nv_fp8_e4m3>();
  if (e == cudaSuccess) e = PrepareAttention<__nv_bfloat16>();
  if (e == cudaSuccess) e = PrepareAttention<float>();
  for (auto k : {tc::SliceKernelTc<1, false>, tc::SliceKernelTc<2, false>, tc::SliceKernelTc<3, false>,
                 tc::SliceKernelTc<1, true>, tc::SliceKernelTc<2, true>, tc::SliceKernelTc<3, true>}) {
    if (e == cudaSuccess) e = tlaloc_kernels::AllowMaxSharedMemory(k);
  }
  if (e == cudaSuccess) e = tlaloc_kernels::AllowMaxSharedMemory(tc::PrefillKernelTc<false>);
  if (e == cudaSuccess) e = tlaloc_kernels::AllowMaxSharedMemory(tc::PrefillKernelTc<true>);
  if (e == cudaSuccess) e = tlaloc_kernels::AllowMaxSharedMemory(tc::PrefillKernelF16);
  return e;
}

// Queries per table and KV head above which e4m3fn pools take the tensor-core
// kernel: all of them (decode rows too: 228 against 133 GB/s at Qwen3.6-35B-A3B's
// shapes); $TLALOC_ATTN_TC_MIN raises it, for measuring the CUDA-core kernel.
inline int TcMinQueries()
{
  static const int v = [] {
    const char* e = std::getenv("TLALOC_ATTN_TC_MIN");
    return e != nullptr ? std::atoi(e) : 0;
  }();
  return v;
}

// $TLALOC_ATTN_PREFILL_EXACT=1: prefill rows take PrefillKernelTc (about 22 bits
// of Q and P) even where the program asks for the f16 form.
inline bool PrefillExact()
{
  static const bool exact = [] {
    const char* e = std::getenv("TLALOC_ATTN_PREFILL_EXACT");
    return e != nullptr && std::strcmp(e, "1") == 0;
  }();
  return exact;
}

// Queries per table and KV head above which e4m3fn D 256 rows take the split
// form of the prefill kernel instead of SliceKernelTc (32: the 5-row MTP head
// pass of Qwen3.6-35B-A3B, 40 queries); $TLALOC_ATTN_SPLIT_MIN.
inline int SplitMinQueries()
{
  static const int v = [] {
    const char* e = std::getenv("TLALOC_ATTN_SPLIT_MIN");
    return e != nullptr ? std::atoi(e) : 32;
  }();
  return v;
}

template <typename KV>
cudaError_t Launch(const void* q, const void* k, const void* v, const void* tables, const void* lens,
                   void* out, void* scratch, const Shape& sh, cudaStream_t stream)
{
  const int NQ = (sh.R / sh.Tb) * (sh.H / sh.Hkv);
  cudaError_t e;
  if (std::is_same<KV, __nv_fp8_e4m3>::value && sh.D == tc::kD && NQ > SplitMinQueries() && NQ <= tc::kPrefillRows) {
    e = tc::LaunchSplitTc(q, k, v, tables, lens, scratch, sh, stream);
  } else if (std::is_same<KV, __nv_fp8_e4m3>::value && sh.D == tc::kD && NQ > TcMinQueries() && NQ <= tc::kMaxQ) {
    e = tc::LaunchTc(q, k, v, tables, lens, scratch, sh, NQ, stream);
  } else if (NQ <= 8) e = LaunchSlices<KV, 8>(q, k, v, tables, lens, scratch, sh, stream);
  else if (NQ <= 16) e = LaunchSlices<KV, 16>(q, k, v, tables, lens, scratch, sh, stream);
  else if (NQ <= 24) e = LaunchSlices<KV, 24>(q, k, v, tables, lens, scratch, sh, stream);
  else if (NQ <= 32) e = LaunchSlices<KV, 32>(q, k, v, tables, lens, scratch, sh, stream);
  else if (NQ <= 40) e = LaunchSlices<KV, 40>(q, k, v, tables, lens, scratch, sh, stream);
  else e = LaunchSlices<KV, 64>(q, k, v, tables, lens, scratch, sh, stream);
  if (e != cudaSuccess) return e;
  CombineKernel<<<sh.R * sh.H, kThreads, sizeof(float) * sh.S, stream>>>(static_cast<const float*>(scratch), static_cast<float*>(out), sh);
  return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// The typed-FFI handler.

XLA_FFI_Error* Fail(const XLA_FFI_Api* api, XLA_FFI_Error_Code code, const std::string& msg)
{
  XLA_FFI_Error_Create_Args a;
  std::memset(&a, 0, sizeof(a));
  a.struct_size = XLA_FFI_Error_Create_Args_STRUCT_SIZE;
  a.message = msg.c_str();
  a.errc = code;
  return api->XLA_FFI_Error_Create(&a);
}

XLA_FFI_Error* PagedAttentionHandler(XLA_FFI_CallFrame* frame)
{
  // XLA asks for the handler's metadata (the FFI version it was built for) first.
  for (XLA_FFI_Extension_Base* ext = frame->extension_start; ext != nullptr; ext = ext->next) {
    if (ext->type == XLA_FFI_Extension_Metadata) {
      XLA_FFI_Metadata* md = reinterpret_cast<XLA_FFI_Metadata_Extension*>(ext)->metadata;
      md->api_version.major_version = XLA_FFI_API_MAJOR;
      md->api_version.minor_version = XLA_FFI_API_MINOR;
      md->traits = XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE;
      return nullptr;
    }
  }
  const XLA_FFI_Api* api = frame->api;
  if (frame->stage == XLA_FFI_ExecutionStage_INITIALIZE) {
    const cudaError_t e = PrepareAttentionKernels();
    return e == cudaSuccess ? nullptr : Fail(api, XLA_FFI_Error_Code_INTERNAL, std::string("tlaloc_paged_attention: ") + cudaGetErrorString(e));
  }
  if (frame->stage != XLA_FFI_ExecutionStage_EXECUTE) return nullptr;
  auto invalid = [&](const std::string& m) {
    return Fail(api, XLA_FFI_Error_Code_INVALID_ARGUMENT, "tlaloc_paged_attention: " + m);
  };
  if ((frame->args.size != 5 && frame->args.size != 6) || frame->rets.size != 2) {
    return invalid("takes 5 operands (6 with a keep mask) and returns 2 results");
  }
  XLA_FFI_Buffer* b[7];
  for (int i = 0; i < 5; ++i) b[i] = static_cast<XLA_FFI_Buffer*>(frame->args.args[i]);
  for (int i = 0; i < 2; ++i) b[5 + i] = static_cast<XLA_FFI_Buffer*>(frame->rets.rets[i]);
  const XLA_FFI_Buffer* keep = frame->args.size == 6 ? static_cast<XLA_FFI_Buffer*>(frame->args.args[5]) : nullptr;
  float scale = 0.f;
  bool has_scale = false, prefill_f16 = false;
  for (int64_t i = 0; i < frame->attrs.size; ++i) {
    const XLA_FFI_ByteSpan* name = frame->attrs.names[i];
    if (frame->attrs.types[i] != XLA_FFI_AttrType_SCALAR) continue;
    const XLA_FFI_Scalar* sc = static_cast<XLA_FFI_Scalar*>(frame->attrs.attrs[i]);
    if (name->len == 5 && std::memcmp(name->ptr, "scale", 5) == 0 && sc->dtype == XLA_FFI_DataType_F32) {
      scale = *static_cast<const float*>(sc->value);
      has_scale = true;
    }
    if (name->len == 11 && std::memcmp(name->ptr, "prefill_f16", 11) == 0 && sc->dtype == XLA_FFI_DataType_S32) {
      prefill_f16 = *static_cast<const int32_t*>(sc->value) != 0 && !PrefillExact();
    }
  }
  if (!has_scale) return invalid("needs an f32 'scale' attribute");
  const XLA_FFI_Buffer *q = b[0], *k = b[1], *t = b[3], *s = b[6];
  if (q->rank != 3 || k->rank != 4 || t->rank != 2 || s->rank != 4 || q->dtype != XLA_FFI_DataType_F32) {
    return invalid("unexpected operand ranks or query dtype");
  }
  Shape sh;
  sh.R = static_cast<int>(q->dims[0]);
  sh.H = static_cast<int>(q->dims[1]);
  sh.D = static_cast<int>(q->dims[2]);
  sh.bs = static_cast<int>(k->dims[1]);
  sh.Hkv = static_cast<int>(k->dims[2]);
  sh.Tb = static_cast<int>(t->dims[0]);
  sh.M = static_cast<int>(t->dims[1]);
  sh.S = static_cast<int>(s->dims[0]);
  sh.slice = (sh.M * sh.bs + sh.S - 1) / sh.S;
  sh.scale = scale;
  sh.keep = nullptr;
  sh.keepChunks = 0;
  sh.chunk = 1;
  if (keep != nullptr) {
    if (keep->rank != 2 || keep->dtype != XLA_FFI_DataType_S32 || keep->dims[0] != sh.R || keep->dims[1] < 1 ||
        (static_cast<int64_t>(sh.M) * sh.bs) % keep->dims[1] != 0) {
      return invalid("keep must be s32 [R, C] with C dividing the context");
    }
    sh.keep = static_cast<const int*>(keep->data);
    sh.keepChunks = static_cast<int>(keep->dims[1]);
    sh.chunk = sh.M * sh.bs / sh.keepChunks;
    if (sh.chunk % 2 != 0) return invalid("a keep chunk must be an even number of positions");
  }
  if (sh.D > kThreads || sh.D % 16 != 0 || sh.R % sh.Tb != 0 || sh.H % sh.Hkv != 0) return invalid("unsupported shape");
  const int NQ = (sh.R / sh.Tb) * (sh.H / sh.Hkv);
  // More queries per table and KV head than the slice kernels hold: prefill rows,
  // for e4m3fn pools with D 256, written directly to the output.
  const bool prefill = NQ > kMaxQueries;
  if (prefill ? (k->dtype != XLA_FFI_DataType_F8E4M3FN || sh.D != tc::kD || sh.H / sh.Hkv > tc::kPrefillRows)
              : sh.M * sh.bs > sh.S * kThreads) {
    return invalid("unsupported shape");
  }
  XLA_FFI_Stream_Get_Args sa;
  std::memset(&sa, 0, sizeof(sa));
  sa.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
  sa.ctx = frame->ctx;
  if (XLA_FFI_Error* err = api->XLA_FFI_Stream_Get(&sa)) return err;
  cudaStream_t stream = static_cast<cudaStream_t>(sa.stream);
  cudaError_t e;
  if (prefill) {
    e = prefill_f16 ? tc::LaunchPrefillF16(q->data, k->data, b[2]->data, t->data, b[4]->data, b[5]->data, sh, stream)
                    : tc::LaunchPrefillTc(q->data, k->data, b[2]->data, t->data, b[4]->data, b[5]->data, sh, stream);
    if (e != cudaSuccess) {
      return Fail(api, XLA_FFI_Error_Code_INTERNAL, std::string("tlaloc_paged_attention: ") + cudaGetErrorString(e));
    }
    return nullptr;
  }
  switch (k->dtype) {
    case XLA_FFI_DataType_F8E4M3FN:
      e = Launch<__nv_fp8_e4m3>(q->data, k->data, b[2]->data, t->data, b[4]->data, b[5]->data, s->data, sh, stream);
      break;
    case XLA_FFI_DataType_BF16:
      e = Launch<__nv_bfloat16>(q->data, k->data, b[2]->data, t->data, b[4]->data, b[5]->data, s->data, sh, stream);
      break;
    case XLA_FFI_DataType_F32:
      e = Launch<float>(q->data, k->data, b[2]->data, t->data, b[4]->data, b[5]->data, s->data, sh, stream);
      break;
    default:
      return invalid("pools must be f8e4m3fn, bf16 or f32");
  }
  if (e != cudaSuccess) {
    return Fail(api, XLA_FFI_Error_Code_INTERNAL, std::string("tlaloc_paged_attention: ") + cudaGetErrorString(e));
  }
  return nullptr;
}

}  // namespace
