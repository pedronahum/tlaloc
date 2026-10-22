// tlaloc_sample: one token per row of logits, greedy or sampled with temperature, top-k and
// top-p, and the speculative acceptance of a greedy draft. Every quantity that decides the
// token is an integer or a float computed in a fixed order, so the result does not depend on
// the thread count or the batch (io.tlaloc.ir.SampleTokensAttrs.reference computes the same):
//
//   temperature <= 0: the argmax (the lowest id among equal logits).
//   Otherwise, for row logits x, m = max x:
//     W(i) = floor(2^20 * 2^(((x_i - m) / t) * log2 e)), 0 below 2^-20 (Exp2Fixed: an exp2
//            polynomial in round-to-nearest f32 steps); the token's integer weight.
//     order: x descending, then id ascending. Kept: the first k (k <= 0: all), then the
//            shortest prefix of those whose weights sum to at least p of theirs
//            (cum 2^24 >= floor(p 2^24) sum), of weight > 0. M: the kept weights' sum.
//     u0, u1: 24-bit uniforms from splitmix64 of (seed, position) (Uniforms).
//     draft d >= 0 (a greedy draft is a point mass): accepted iff d is kept and
//            u0 M < W(d) 2^24; the token is then d. Otherwise the token is drawn from the
//            kept ids other than d: r = (u1 M') >> 24 with M' their weight, the first id
//            in id order whose running sum passes r.
//
// One block per row. The kept set's boundary is found by radix select over a 64-bit key
// (the logit's order-preserving bits, then the id reversed), counting for top-k and summing
// weights for top-p; histograms are integer sums.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

#include "kernel_setup.cuh"

namespace sample {

constexpr int kThreads = 1024;
constexpr int kBits = 8;  // radix digit
constexpr int kBins = 1 << kBits;

// 2^z for z in [-21, 0], times 2^20, floored: the f32 steps io.tlaloc.ir.SampleTokensAttrs mirrors.
__host__ __device__ inline uint32_t Exp2Fixed(float z)
{
  if (!(z > -21.f)) return 0u;
  const float fi = floorf(z);
  const float f = z - fi;  // [0, 1), exact
  // 2^f on [0, 1): a degree-6 fit (relative error below 2e-7).
  float p = 1.5403530393381606e-4f;
#if defined(__CUDA_ARCH__)
  p = __fadd_rn(__fmul_rn(p, f), 1.3333558146428443e-3f);
  p = __fadd_rn(__fmul_rn(p, f), 9.6181291076284772e-3f);
  p = __fadd_rn(__fmul_rn(p, f), 5.5504108664821580e-2f);
  p = __fadd_rn(__fmul_rn(p, f), 2.4022650695910071e-1f);
  p = __fadd_rn(__fmul_rn(p, f), 6.9314718055994531e-1f);
  p = __fadd_rn(__fmul_rn(p, f), 1.f);
#else
  // Separate roundings on the host too (no contraction into fused multiply-adds).
  auto step = [](float a, float b, float c) { volatile float m = a * b; volatile float s = m + c; return static_cast<float>(s); };
  p = step(p, f, 1.3333558146428443e-3f);
  p = step(p, f, 9.6181291076284772e-3f);
  p = step(p, f, 5.5504108664821580e-2f);
  p = step(p, f, 2.4022650695910071e-1f);
  p = step(p, f, 6.9314718055994531e-1f);
  p = step(p, f, 1.f);
#endif
  const int e = 20 + static_cast<int>(fi);  // -1 .. 20
  if (e < 0) return 0u;
  return static_cast<uint32_t>(ldexpf(p, e));  // exact scaling, then floor
}

__host__ __device__ inline uint64_t SplitMix64(uint64_t x)
{
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// Two 24-bit uniforms of (seed, position).
__host__ __device__ inline void Uniforms(int32_t seed, int32_t position, uint32_t* u0, uint32_t* u1)
{
  const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(seed)) << 32) | static_cast<uint32_t>(position);
  *u0 = static_cast<uint32_t>(SplitMix64(key) >> 40);
  *u1 = static_cast<uint32_t>(SplitMix64(key ^ 0x5851F42D4C957F2Dull) >> 40);
}

// The order-preserving bits of a float (larger float, larger key).
__host__ __device__ inline uint32_t Ordered(float x)
{
  uint32_t b;
#if defined(__CUDA_ARCH__)
  b = __float_as_uint(x);
#else
  memcpy(&b, &x, 4);
#endif
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// Key of id i: larger is earlier in the order (x descending, id ascending).
__device__ __forceinline__ uint64_t Key(float x, int i, int V)
{
  return (static_cast<uint64_t>(Ordered(x)) << 32) | static_cast<uint32_t>(V - 1 - i);
}

struct Row {
  const float* x;
  int V;
  float t;
  float m;
};

// Block-wide sums (integer, so in any order the same).
__device__ __forceinline__ unsigned long long BlockSum(unsigned long long v, unsigned long long* red)
{
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int w = threadIdx.x >> 5;
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[w] = v;
  __syncthreads();
  if (threadIdx.x < 32) {
    unsigned long long s = threadIdx.x < kThreads / 32 ? red[threadIdx.x] : 0ull;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (threadIdx.x == 0) red[0] = s;
  }
  __syncthreads();
  const unsigned long long out = red[0];
  __syncthreads();
  return out;
}

__device__ __forceinline__ uint32_t WeightAt(const Row& r, int i)
{
  return Exp2Fixed(__fmul_rn(__fdiv_rn(__fsub_rn(r.x[i], r.m), r.t), 1.4426950408889634f));
}

// The largest key b such that the keys >= b number at least `count` (byCount), or their weights
// sum to at least `mass` (otherwise), among keys >= lo. Radix select, 8 bits a pass from the top.
__device__ uint64_t SelectBoundary(const Row& r, uint64_t lo, bool byCount, unsigned long long need,
                                   unsigned long long* hist)
{
  uint64_t prefix = 0, mask = 0;
  for (int shift = 64 - kBits; shift >= 0; shift -= kBits) {
    for (int b = threadIdx.x; b < kBins; b += kThreads) hist[b] = 0ull;
    __syncthreads();
    for (int i = threadIdx.x; i < r.V; i += kThreads) {
      const uint64_t k = Key(r.x[i], i, r.V);
      if (k < lo || (k & mask) != prefix) continue;
      const unsigned long long add = byCount ? 1ull : WeightAt(r, i);
      if (add) atomicAdd(&hist[(k >> shift) & (kBins - 1)], add);
    }
    __syncthreads();
    // From the top bin down: the bin where the running total reaches `need`.
    __shared__ int digit;
    __shared__ unsigned long long above;
    if (threadIdx.x == 0) {
      unsigned long long run = 0;
      int d = 0;
      for (int b = kBins - 1; b >= 0; --b) {
        if (run + hist[b] >= need) { d = b; break; }
        run += hist[b];
      }
      digit = d;
      above = run;
    }
    __syncthreads();
    need -= above;
    prefix |= static_cast<uint64_t>(digit) << shift;
    mask |= static_cast<uint64_t>(kBins - 1) << shift;
    __syncthreads();
  }
  return prefix;
}

// The fast path: the tokens of weight > 0 (those within about 14.6 t of the max logit, the top
// of the order, ties included) number at most kCap. They are sorted in shared memory, and the
// kept set, the acceptance and the draw are taken over them: the same result as the general
// path, since every other token has weight 0.
constexpr int kCap = 4096;
constexpr size_t kFastBytes = kCap * (sizeof(uint64_t) + sizeof(uint32_t));

// Sorts the first n (a power of two, <= kCap) entries of (key, w) by key, descending.
__device__ void BitonicDesc(uint64_t* key, uint32_t* w, int n)
{
  for (int size = 2; size <= n; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = threadIdx.x; i < n; i += kThreads) {
        const int j = i ^ stride;
        if (j > i) {
          const bool desc = (i & size) == 0;
          const uint64_t a = key[i], b = key[j];
          if ((a < b) == desc) {
            key[i] = b; key[j] = a;
            const uint32_t t = w[i]; w[i] = w[j]; w[j] = t;
          }
        }
      }
      __syncthreads();
    }
  }
}

// The first index i < n where the running sum of w[0..i] reaches `need` (>=: atLeast; >: else),
// or n. Running sums over chunks of kThreads, block scans; red holds kThreads / 32 words.
__device__ int FirstReaching(const uint32_t* w, int n, unsigned long long need, bool atLeast, unsigned long long* red)
{
  __shared__ unsigned long long base;
  __shared__ int found;
  if (threadIdx.x == 0) { base = 0; found = n; }
  __syncthreads();
  for (int i0 = 0; i0 < n; i0 += kThreads) {
    const int i = i0 + threadIdx.x;
    const unsigned long long v = i < n ? w[i] : 0ull;
    unsigned long long s = v;
    for (int o = 1; o < 32; o <<= 1) {
      const unsigned long long y = __shfl_up_sync(0xffffffffu, s, o);
      if ((threadIdx.x & 31) >= o) s += y;
    }
    if ((threadIdx.x & 31) == 31) red[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
      unsigned long long t = threadIdx.x < kThreads / 32 ? red[threadIdx.x] : 0ull;
      for (int o = 1; o < 32; o <<= 1) {
        const unsigned long long y = __shfl_up_sync(0xffffffffu, t, o);
        if (threadIdx.x >= o) t += y;
      }
      if (threadIdx.x < kThreads / 32) red[threadIdx.x] = t;
    }
    __syncthreads();
    const unsigned long long cum = base + (threadIdx.x >= 32 ? red[(threadIdx.x >> 5) - 1] : 0ull) + s;
    const bool hit = i < n && (atLeast ? cum >= need : cum > need) && (atLeast ? cum - v < need : cum - v <= need);
    if (hit) found = i;  // one thread at most: the running sum crosses once
    __syncthreads();
    if (found < n) break;
    if (threadIdx.x == 0) base += red[kThreads / 32 - 1];
    __syncthreads();
  }
  const int f = found;
  __syncthreads();
  return f;
}

// Grid (N), kThreads threads. logits [N, V]; draft, topK, seed, position i32 [N]; temperature,
// topP f32 [N]; token i32 [N].
__global__ void __launch_bounds__(kThreads) SampleKernel(
    const float* __restrict__ logits, int V, const int* __restrict__ draft, const float* __restrict__ temperature,
    const float* __restrict__ topP, const int* __restrict__ topK, const int* __restrict__ seed,
    const int* __restrict__ position, int* __restrict__ token)
{
  __shared__ unsigned long long hist[kBins];
  __shared__ unsigned long long red[kThreads / 32];
  __shared__ float fred[kThreads / 32];
  __shared__ int ired[kThreads / 32];
  const int n = blockIdx.x;
  const float* x = logits + static_cast<size_t>(n) * V;
  // The argmax (the lowest id among equal values), and the max.
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int i = threadIdx.x; i < V; i += kThreads) {
    const float v = x[i];
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
  }
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
  }
  if ((threadIdx.x & 31) == 0) { fred[threadIdx.x >> 5] = bv; ired[threadIdx.x >> 5] = bi; }
  __syncthreads();
  __shared__ float maxv;
  __shared__ int argmax;
  if (threadIdx.x == 0) {
    float v = fred[0];
    int a = ired[0];
    for (int w = 1; w < kThreads / 32; ++w) {
      if (fred[w] > v || (fred[w] == v && ired[w] < a)) { v = fred[w]; a = ired[w]; }
    }
    maxv = v;
    argmax = a;
  }
  __syncthreads();
  const float t = temperature[n];
  if (!(t > 0.f)) {
    if (threadIdx.x == 0) token[n] = argmax;
    return;
  }
  const Row r{x, V, t, maxv};
  const int k = topK[n];
  const int d = draft[n];
  const float p = topP[n];
  const unsigned long long p24 = p >= 1.f ? (1ull << 24) : static_cast<unsigned long long>(fmaxf(p, 0.f) * 16777216.f);
  uint32_t u0, u1;
  Uniforms(seed[n], position[n], &u0, &u1);
  {
    extern __shared__ __align__(16) unsigned char fast[];
    uint64_t* ckey = reinterpret_cast<uint64_t*>(fast);
    uint32_t* cw = reinterpret_cast<uint32_t*>(ckey + kCap);
    __shared__ int count;
    if (threadIdx.x == 0) count = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < V; i += kThreads) {
      const uint32_t w = WeightAt(r, i);
      if (w == 0) continue;
      const int at = atomicAdd(&count, 1);
      if (at < kCap) { ckey[at] = Key(x[i], i, V); cw[at] = w; }
    }
    __syncthreads();
    const int C = count;
    if (C <= kCap) {
      int n2 = 1;
      while (n2 < C) n2 <<= 1;
      for (int i = C + threadIdx.x; i < n2; i += kThreads) { ckey[i] = 0; cw[i] = 0; }
      __syncthreads();
      BitonicDesc(ckey, cw, n2);
      const int kk = k > 0 && k < C ? k : C;
      unsigned long long sumK = 0;
      for (int i = threadIdx.x; i < kk; i += kThreads) sumK += cw[i];
      sumK = BlockSum(sumK, red);
      int kept = kk;
      if (p24 < (1ull << 24)) {
        unsigned long long need = (p24 * sumK + (1ull << 24) - 1) >> 24;
        if (need == 0) need = 1;
        kept = FirstReaching(cw, kk, need, true, red) + 1;
        if (kept > kk) kept = kk;
      }
      unsigned long long M = 0;
      for (int i = threadIdx.x; i < kept; i += kThreads) M += cw[i];
      M = BlockSum(M, red);
      // The draft's weight if kept (its key is in the first `kept`).
      __shared__ unsigned long long wdS;
      if (threadIdx.x == 0) wdS = 0;
      __syncthreads();
      if (d >= 0 && d < V) {
        const uint64_t kd = Key(x[d], d, V);
        for (int i = threadIdx.x; i < kept; i += kThreads) if (ckey[i] == kd) wdS = cw[i];
      }
      __syncthreads();
      const unsigned long long wd = wdS;
      if (M == 0) { if (threadIdx.x == 0) token[n] = argmax; return; }
      if (wd > 0 && static_cast<unsigned long long>(u0) * M < (wd << 24)) { if (threadIdx.x == 0) token[n] = d; return; }
      const unsigned long long M2 = M - wd;
      if (M2 == 0) { if (threadIdx.x == 0) token[n] = argmax; return; }
      const unsigned long long target = (static_cast<unsigned long long>(u1) * M2) >> 24;
      // The kept tokens in id order: the low word of a key is V - 1 - id, so sorting by it
      // descending is ids ascending; it is stored plus 1, so padding (0) sorts last. The
      // draft's weight goes to 0.
      int m2 = 1;
      while (m2 < kept) m2 <<= 1;
      for (int i = threadIdx.x; i < m2; i += kThreads) {
        if (i < kept) {
          const uint32_t low = static_cast<uint32_t>(ckey[i]);
          if (d >= 0 && static_cast<int>(V - 1 - low) == d) cw[i] = 0;
          ckey[i] = static_cast<uint64_t>(low) + 1;
        } else {
          ckey[i] = 0;
          cw[i] = 0;
        }
      }
      __syncthreads();
      BitonicDesc(ckey, cw, m2);
      const int at = FirstReaching(cw, kept, target, false, red);
      if (threadIdx.x == 0) token[n] = at < kept ? V - static_cast<int>(ckey[at]) : argmax;
      return;
    }
  }
  // More than kCap tokens of weight > 0: the general path over the whole row.
  // Top-k boundary, then top-p within it.
  uint64_t lo = 0;
  if (k > 0 && k < V) lo = SelectBoundary(r, 0, true, static_cast<unsigned long long>(k), hist);
  unsigned long long sumK = 0;
  for (int i = threadIdx.x; i < V; i += kThreads) {
    if (Key(x[i], i, V) >= lo) sumK += WeightAt(r, i);
  }
  sumK = BlockSum(sumK, red);
  // Mass needed: ceil(p24 sumK / 2^24) (cum 2^24 >= p24 sumK); at least 1.
  unsigned long long need = (p24 * sumK + (1ull << 24) - 1) >> 24;
  if (need == 0) need = 1;
  uint64_t bound = lo;
  if (p24 < (1ull << 24)) bound = SelectBoundary(r, lo, false, need, hist);
  // Kept: key >= bound and weight > 0. M, the draft's weight.
  unsigned long long M = 0;
  for (int i = threadIdx.x; i < V; i += kThreads) {
    if (Key(x[i], i, V) >= bound) M += WeightAt(r, i);
  }
  M = BlockSum(M, red);
  const bool dKept = d >= 0 && d < V && Key(x[d], d, V) >= bound;
  const unsigned long long wd = dKept ? WeightAt(r, d) : 0ull;
  if (M == 0) {
    if (threadIdx.x == 0) token[n] = argmax;
    return;
  }
  if (dKept && static_cast<unsigned long long>(u0) * M < (wd << 24)) {
    if (threadIdx.x == 0) token[n] = d;
    return;
  }
  const unsigned long long M2 = M - wd;
  if (M2 == 0) {
    if (threadIdx.x == 0) token[n] = argmax;
    return;
  }
  const unsigned long long target = (static_cast<unsigned long long>(u1) * M2) >> 24;  // < M2
  // The first id in id order whose running sum (kept, not d) passes target: blocks of kThreads
  // ids, a block scan of their weights.
  __shared__ unsigned long long base;
  __shared__ int found;
  if (threadIdx.x == 0) { base = 0; found = -1; }
  __syncthreads();
  for (int i0 = 0; i0 < V; i0 += kThreads) {
    const int i = i0 + threadIdx.x;
    unsigned long long w = 0;
    if (i < V && i != d && Key(x[i], i, V) >= bound) w = WeightAt(r, i);
    // Inclusive scan within the block (warp scans, then the warps' totals).
    unsigned long long s = w;
    for (int o = 1; o < 32; o <<= 1) {
      const unsigned long long y = __shfl_up_sync(0xffffffffu, s, o);
      if ((threadIdx.x & 31) >= o) s += y;
    }
    if ((threadIdx.x & 31) == 31) red[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
      unsigned long long v = threadIdx.x < kThreads / 32 ? red[threadIdx.x] : 0ull;
      for (int o = 1; o < 32; o <<= 1) {
        const unsigned long long y = __shfl_up_sync(0xffffffffu, v, o);
        if (threadIdx.x >= o) v += y;
      }
      if (threadIdx.x < kThreads / 32) red[threadIdx.x] = v;  // inclusive warp totals
    }
    __syncthreads();
    const unsigned long long before = base + (threadIdx.x >= 32 ? red[(threadIdx.x >> 5) - 1] : 0ull);
    const unsigned long long cum = before + s;
    // The id where the running sum first exceeds target: cum > target >= cum - w.
    if (w > 0 && cum > target && before + (s - w) <= target) found = i;
    __syncthreads();
    if (found >= 0) break;
    if (threadIdx.x == 0) base += red[kThreads / 32 - 1];
    __syncthreads();
  }
  if (threadIdx.x == 0) token[n] = found >= 0 ? found : argmax;
}

// The fast path's shared memory (kFastBytes, over 48 KB): at XLA's INITIALIZE stage.
inline cudaError_t Prepare() { return tlaloc_kernels::AllowMaxSharedMemory(SampleKernel); }

inline cudaError_t Launch(const float* logits, int N, int V, const int* draft, const float* temperature, const float* topP,
                          const int* topK, const int* seed, const int* position, int* token, cudaStream_t stream)
{
  if (N == 0) return cudaSuccess;
  SampleKernel<<<N, kThreads, kFastBytes, stream>>>(logits, V, draft, temperature, topP, topK, seed, position, token);
  return cudaGetLastError();
}

}  // namespace sample
