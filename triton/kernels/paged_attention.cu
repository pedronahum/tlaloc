// Paged attention for decode and speculative verify steps, as an XLA typed-FFI
// custom call (`tlaloc_paged_attention`), registered with a PJRT GPU plugin by
// TlalocRegisterKernels (tlaloc_kernels.cu).
//
// Operands, as PAGED_ATTENTION's (io.tlaloc.ir.PagedAttentionAttrs):
//   0 query       f32  [R, H, D]
//   1 keyCache    f8e4m3fn | bf16 | f32  [P, bs, Hkv, D]
//   2 valueCache  same as keyCache
//   3 blockTables i32  [Tb, M]     the R rows are Tb groups of R / Tb consecutive rows
//   4 seqLens     i32  [R]         row r attends positions [0, seqLens[r])
// Results:
//   0 out         f32  [R, H, D]
//   1 scratch     f32  [S, R, H, D + 2]   per context slice: max, sum, weighted values
// Attribute: scale (f32).
//
// The XLA form gathers each row's whole context bucket out of the pool and
// writes it before attending; here each block of threads reads the pages of
// one slice of S once, for every row sharing the table and every query head
// of one KV head (an online softmax over tiles of positions), and a second
// kernel combines the slices. The arithmetic is f32 throughout.

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
};

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
      if (i < NQ) mine[i] = tid < n && start + tid < lens[t * Qr + i / G] ? sc[i] : -INFINITY;
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
constexpr int kQStride = kD + 8;       // f16 per query row (hi and lo planes)
constexpr int kSStride = kChunk + 4;   // f32 scores per query row
constexpr int kPStride = kChunk + 8;   // f16 probabilities per query row
constexpr int kKStride = kD;           // bytes per key or value row (padding measured slower)

// Shared memory for MQ (a multiple of 16) query rows: the planes below, in order.
struct Layout {
  size_t qhi, qlo, kc, vc, sc, phi, plo, base, m, l, alpha, len, bytes;
  __host__ __device__ explicit Layout(int mq)
  {
    size_t o = 0;
    auto take = [&](size_t n) { const size_t at = o; o = (o + n + 15) / 16 * 16; return at; };
    qhi = take(2 * mq * kQStride);
    qlo = take(2 * mq * kQStride);
    kc = take(2 * kChunk * kKStride);  // two buffers: the chunk in use and the next one in flight
    vc = take(2 * kChunk * kKStride);
    sc = take(4 * mq * kSStride);
    phi = take(2 * mq * kPStride);
    plo = take(2 * mq * kPStride);
    base = take(sizeof(size_t) * kThreads);
    m = take(4 * mq);
    l = take(4 * mq);
    alpha = take(4 * mq);
    len = take(4 * mq);
    bytes = o;
  }
};

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

__global__ void __launch_bounds__(kThreads) SliceKernelTc(
    const float* __restrict__ q, const uint8_t* __restrict__ kpool, const uint8_t* __restrict__ vpool,
    const int* __restrict__ tables, const int* __restrict__ lens, float* __restrict__ scratch, Shape sh)
{
  extern __shared__ __align__(16) unsigned char raw[];
  const int s = blockIdx.x, hk = blockIdx.y, t = blockIdx.z, tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int G = sh.H / sh.Hkv, Qr = sh.R / sh.Tb, NQ = Qr * G, MT = (NQ + 15) / 16;
  const Layout ly(MT * 16);
  struct {
    __half *qhi, *qlo, *phi, *plo;
    uint8_t *kc, *vc;
    float *sc, *m, *l, *alpha;
    size_t* base;
    int* len;
  } sm{reinterpret_cast<__half*>(raw + ly.qhi), reinterpret_cast<__half*>(raw + ly.qlo),
       reinterpret_cast<__half*>(raw + ly.phi), reinterpret_cast<__half*>(raw + ly.plo), raw + ly.kc, raw + ly.vc,
       reinterpret_cast<float*>(raw + ly.sc), reinterpret_cast<float*>(raw + ly.m), reinterpret_cast<float*>(raw + ly.l),
       reinterpret_cast<float*>(raw + ly.alpha), reinterpret_cast<size_t*>(raw + ly.base), reinterpret_cast<int*>(raw + ly.len)};
  int longest = 0;
  for (int j = 0; j < Qr; ++j) longest = max(longest, lens[t * Qr + j]);
  const int start = s * kThreads, n = min(kThreads, longest - start);

  // Queries, scaled and split into f16 high and low parts; rows past NQ zero.
  for (int e = tid; e < MT * 16 * kD; e += kThreads) {
    const int i = e / kD, d = e % kD;
    float x = 0.f;
    if (i < NQ) x = q[(static_cast<size_t>(t * Qr + i / G) * sh.H + hk * G + i % G) * kD + d] * sh.scale;
    const __half hi = __float2half_rn(x);
    sm.qhi[i * kQStride + d] = hi;
    sm.qlo[i * kQStride + d] = __float2half_rn(x - __half2float(hi));
  }
  for (int i = tid; i < MT * 16; i += kThreads) {
    sm.m[i] = -INFINITY;
    sm.l[i] = 0.f;
    sm.len[i] = i < NQ ? lens[t * Qr + i / G] : 0;
  }
  const size_t rowStride = static_cast<size_t>(sh.Hkv) * kD;
  if (tid < n) {
    const int pos = start + tid;
    sm.base[tid] = (static_cast<size_t>(tables[t * sh.M + pos / sh.bs]) * sh.bs + pos % sh.bs) * rowStride + hk * kD;
  }
  __syncthreads();

  // This warp's output tiles: (m-tile, 8-dim n-tile) pairs w, w + 8, ... of MT x 32.
  constexpr int kMaxTiles = kMaxQ / 16 * (kD / 8) / (kThreads / 32);  // 12
  float acc[kMaxTiles][4];
#pragma unroll
  for (int i = 0; i < kMaxTiles; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
  const int r = lane >> 2, c2 = 2 * (lane & 3);

  // A chunk's key and value rows into buffer `buf`, 16 bytes per copy, in flight
  // until waited for; positions past the slice's live ones are zero-filled.
  auto fetch = [&](int c0, int buf) {
    const int cn = min(kChunk, n - c0);
    for (int e = tid; e < kChunk * (kD / 16); e += kThreads) {
      const int p = e / (kD / 16), o = (e % (kD / 16)) * 16;
      const size_t at = p < cn ? sm.base[c0 + p] + o : 0;
      __pipeline_memcpy_async(sm.kc + (buf * kChunk + p) * kKStride + o, kpool + at, 16, p < cn ? 0 : 16);
      __pipeline_memcpy_async(sm.vc + (buf * kChunk + p) * kKStride + o, vpool + at, 16, p < cn ? 0 : 16);
    }
    __pipeline_commit();
  };
  if (n > 0) fetch(0, 0);
  for (int c0 = 0, buf = 0; c0 < n; c0 += kChunk, buf ^= 1) {
    const int cn = min(kChunk, n - c0);
    if (c0 + kChunk < n) {
      fetch(c0 + kChunk, buf ^ 1);
      __pipeline_wait_prior(1);
    } else {
      __pipeline_wait_prior(0);
    }
    __syncthreads();
    const uint8_t* kc = sm.kc + buf * kChunk * kKStride;
    const uint8_t* vcb = sm.vc + buf * kChunk * kKStride;
    // Scores: MT x 8 tiles of [16 queries x 8 positions] over the warps.
    for (int tile = warp; tile < MT * (kChunk / 8); tile += kThreads / 32) {
      const int mt = tile / (kChunk / 8), nt = tile % (kChunk / 8);
      float c[4] = {0.f, 0.f, 0.f, 0.f};
      const int qr = mt * 16 + r, p = nt * 8 + r;
#pragma unroll 4
      for (int ks = 0; ks < kD / 16; ++ks) {
        const int d = ks * 16 + c2;
        const uint32_t b0 = E4m3x2ToF16x2(*reinterpret_cast<const uint16_t*>(kc + p * kKStride + d));
        const uint32_t b1 = E4m3x2ToF16x2(*reinterpret_cast<const uint16_t*>(kc + p * kKStride + d + 8));
        for (int part = 0; part < 2; ++part) {
          const __half* qs = part == 0 ? sm.qhi : sm.qlo;
          uint32_t a[4];
          a[0] = *reinterpret_cast<const uint32_t*>(qs + qr * kQStride + d);
          a[1] = *reinterpret_cast<const uint32_t*>(qs + (qr + 8) * kQStride + d);
          a[2] = *reinterpret_cast<const uint32_t*>(qs + qr * kQStride + d + 8);
          a[3] = *reinterpret_cast<const uint32_t*>(qs + (qr + 8) * kQStride + d + 8);
          MmaF16(c, a, b0, b1);
        }
      }
      // c: {query qr, positions nt*8 + c2, +1}, {query qr + 8, same}; -inf past a query's length.
      const int pc = nt * 8 + c2;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int qi = qr + 8 * h;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int pos = start + c0 + pc + j;
          sm.sc[qi * kSStride + pc + j] = pc + j < cn && pos < sm.len[qi] ? c[2 * h + j] : -INFINITY;
        }
      }
    }
    __syncthreads();
    // Online softmax per query: warp w takes queries w, w + 8, ...; lane j position j.
    for (int qi = warp; qi < MT * 16; qi += kThreads / 32) {
      const float s0 = sm.sc[qi * kSStride + lane];
      float mx = s0;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
      const float mOld = sm.m[qi], mNew = fmaxf(mOld, mx);
      const float p0 = s0 == -INFINITY ? 0.f : expf(s0 - mNew);
      float sum = p0;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
      const __half h0 = __float2half_rn(p0);
      sm.phi[qi * kPStride + lane] = h0;
      sm.plo[qi * kPStride + lane] = __float2half_rn(p0 - __half2float(h0));
      if (lane == 0) {
        const float a = mOld == -INFINITY ? 0.f : expf(mOld - mNew);
        sm.alpha[qi] = a;
        sm.l[qi] = sm.l[qi] * a + sum;
        sm.m[qi] = mNew;
      }
    }
    __syncthreads();
    // O = O * alpha + P . V over this warp's tiles.
#pragma unroll
    for (int i = 0; i < kMaxTiles; ++i) {
      const int tile = warp + i * (kThreads / 32);
      if (tile >= MT * (kD / 8)) break;
      const int mt = tile / (kD / 8), nt = tile % (kD / 8);
      const int qr = mt * 16 + r, dcol = nt * 8 + r;
      const float a0 = sm.alpha[qr], a1 = sm.alpha[qr + 8];
      acc[i][0] *= a0;
      acc[i][1] *= a0;
      acc[i][2] *= a1;
      acc[i][3] *= a1;
#pragma unroll
      for (int ks = 0; ks < kChunk / 16; ++ks) {
        const int pk = ks * 16 + c2;
        // B: positions pk, pk + 1 (and + 8) at dim dcol.
        const uint16_t v01 = static_cast<uint16_t>(vcb[pk * kKStride + dcol] | (vcb[(pk + 1) * kKStride + dcol] << 8));
        const uint16_t v89 = static_cast<uint16_t>(vcb[(pk + 8) * kKStride + dcol] | (vcb[(pk + 9) * kKStride + dcol] << 8));
        const uint32_t b0 = E4m3x2ToF16x2(v01), b1 = E4m3x2ToF16x2(v89);
        for (int part = 0; part < 2; ++part) {
          const __half* ps = part == 0 ? sm.phi : sm.plo;
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
  // This slice's (max, sum, weighted values) per query: [S, R, H, D + 2].
#pragma unroll
  for (int i = 0; i < kMaxTiles; ++i) {
    const int tile = warp + i * (kThreads / 32);
    if (tile >= MT * (kD / 8)) break;
    const int mt = tile / (kD / 8), nt = tile % (kD / 8);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int qi = mt * 16 + r + 8 * h;
      if (qi >= NQ) continue;
      const int row = t * Qr + qi / G, head = hk * G + qi % G;
      float* out = scratch + ((static_cast<size_t>(s) * sh.R + row) * sh.H + head) * (kD + 2);
      const int d = nt * 8 + c2;
      out[2 + d] = acc[i][2 * h];
      out[2 + d + 1] = acc[i][2 * h + 1];
      if (nt == 0 && c2 == 0) {
        out[0] = n > 0 ? sm.m[qi] : -INFINITY;
        out[1] = n > 0 ? sm.l[qi] : 0.f;
      }
    }
  }
}

}  // namespace tc

// Grid R * H, kThreads threads: the slices of each (row, head) combined.
__global__ void __launch_bounds__(kThreads) CombineKernel(const float* __restrict__ scratch, float* __restrict__ out, Shape sh)
{
  const int rh = blockIdx.x, D = sh.D;
  const size_t slice = static_cast<size_t>(sh.R) * sh.H * (D + 2);
  float mx = -INFINITY;
  for (int s = 0; s < sh.S; ++s) mx = fmaxf(mx, scratch[s * slice + rh * (D + 2)]);
  float sum = 0.f;
  if (mx != -INFINITY) {
    for (int s = 0; s < sh.S; ++s) {
      const float* p = scratch + s * slice + rh * (D + 2);
      if (p[0] != -INFINITY) sum += p[1] * expf(p[0] - mx);
    }
  }
  for (int d = threadIdx.x; d < D; d += kThreads) {
    float a = 0.f;
    if (mx != -INFINITY) {
      for (int s = 0; s < sh.S; ++s) {
        const float* p = scratch + s * slice + rh * (D + 2);
        if (p[0] != -INFINITY) a += p[2 + d] * expf(p[0] - mx);
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
  cudaError_t e = cudaFuncSetAttribute(SliceKernel<KV, MAXQ>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  if (e != cudaSuccess) return e;
  SliceKernel<KV, MAXQ><<<dim3(sh.S, sh.Hkv, sh.Tb), kThreads, smem, stream>>>(
      static_cast<const float*>(q), static_cast<const KV*>(k), static_cast<const KV*>(v),
      static_cast<const int*>(tables), static_cast<const int*>(lens), static_cast<float*>(scratch), sh);
  return cudaGetLastError();
}

// Queries per table and KV head above which e4m3fn pools take the tensor-core
// kernel ($TLALOC_ATTN_TC_MIN overrides, for measuring).
inline int TcMinQueries()
{
  static const int v = [] {
    const char* e = std::getenv("TLALOC_ATTN_TC_MIN");
    return e != nullptr ? std::atoi(e) : 8;
  }();
  return v;
}

template <typename KV>
cudaError_t Launch(const void* q, const void* k, const void* v, const void* tables, const void* lens,
                   void* out, void* scratch, const Shape& sh, cudaStream_t stream)
{
  const int NQ = (sh.R / sh.Tb) * (sh.H / sh.Hkv);
  cudaError_t e;
  if (std::is_same<KV, __nv_fp8_e4m3>::value && sh.D == tc::kD && NQ > TcMinQueries() && NQ <= tc::kMaxQ) {
    const int smem = static_cast<int>(tc::Layout((NQ + 15) / 16 * 16).bytes);
    e = cudaFuncSetAttribute(tc::SliceKernelTc, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess) return e;
    tc::SliceKernelTc<<<dim3(sh.S, sh.Hkv, sh.Tb), kThreads, smem, stream>>>(
        static_cast<const float*>(q), static_cast<const uint8_t*>(k), static_cast<const uint8_t*>(v),
        static_cast<const int*>(tables), static_cast<const int*>(lens), static_cast<float*>(scratch), sh);
    e = cudaGetLastError();
  } else if (NQ <= 8) e = LaunchSlices<KV, 8>(q, k, v, tables, lens, scratch, sh, stream);
  else if (NQ <= 16) e = LaunchSlices<KV, 16>(q, k, v, tables, lens, scratch, sh, stream);
  else if (NQ <= 24) e = LaunchSlices<KV, 24>(q, k, v, tables, lens, scratch, sh, stream);
  else if (NQ <= 32) e = LaunchSlices<KV, 32>(q, k, v, tables, lens, scratch, sh, stream);
  else if (NQ <= 40) e = LaunchSlices<KV, 40>(q, k, v, tables, lens, scratch, sh, stream);
  else e = LaunchSlices<KV, 64>(q, k, v, tables, lens, scratch, sh, stream);
  if (e != cudaSuccess) return e;
  CombineKernel<<<sh.R * sh.H, kThreads, 0, stream>>>(static_cast<const float*>(scratch), static_cast<float*>(out), sh);
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
      md->traits = 0;
      return nullptr;
    }
  }
  const XLA_FFI_Api* api = frame->api;
  if (frame->stage != XLA_FFI_ExecutionStage_EXECUTE) return nullptr;
  auto invalid = [&](const std::string& m) {
    return Fail(api, XLA_FFI_Error_Code_INVALID_ARGUMENT, "tlaloc_paged_attention: " + m);
  };
  if (frame->args.size != 5 || frame->rets.size != 2) return invalid("takes 5 operands and returns 2 results");
  XLA_FFI_Buffer* b[7];
  for (int i = 0; i < 5; ++i) b[i] = static_cast<XLA_FFI_Buffer*>(frame->args.args[i]);
  for (int i = 0; i < 2; ++i) b[5 + i] = static_cast<XLA_FFI_Buffer*>(frame->rets.rets[i]);
  float scale = 0.f;
  bool has_scale = false;
  for (int64_t i = 0; i < frame->attrs.size; ++i) {
    const XLA_FFI_ByteSpan* name = frame->attrs.names[i];
    if (frame->attrs.types[i] == XLA_FFI_AttrType_SCALAR && name->len == 5 && std::memcmp(name->ptr, "scale", 5) == 0) {
      const XLA_FFI_Scalar* sc = static_cast<XLA_FFI_Scalar*>(frame->attrs.attrs[i]);
      if (sc->dtype == XLA_FFI_DataType_F32) {
        scale = *static_cast<const float*>(sc->value);
        has_scale = true;
      }
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
  const int NQ = (sh.R / sh.Tb) * (sh.H / sh.Hkv);
  if (sh.D > kThreads || sh.D % 16 != 0 || sh.R % sh.Tb != 0 || sh.H % sh.Hkv != 0 || NQ > kMaxQueries ||
      sh.M * sh.bs > sh.S * kThreads) {
    return invalid("unsupported shape");
  }
  XLA_FFI_Stream_Get_Args sa;
  std::memset(&sa, 0, sizeof(sa));
  sa.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
  sa.ctx = frame->ctx;
  if (XLA_FFI_Error* err = api->XLA_FFI_Stream_Get(&sa)) return err;
  cudaStream_t stream = static_cast<cudaStream_t>(sa.stream);
  cudaError_t e;
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
