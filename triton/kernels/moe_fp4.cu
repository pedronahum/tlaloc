// A mixture of SwiGLU experts with NVFP4 weights, reading each expert used by
// any row once per group of up to 8 of its rows (fp4_gemm.cu's packing and
// tensor-core tiles):
//
//   lists    the (row, slot) pairs routed to each expert, in pair order
//   gateUp   per expert used, per 16-row tile of [gate | up]:
//            gu[pair] = bf16(x[row]) . gateUp[e]^T  (f32), times the row scales
//   act      h[pair] = bf16(silu(g) * u)
//   down     part[pair] = h[pair] . down[e]^T, times the row scales
//   combine  y[row] = sum over slots j of w[row, j] * part[(row, j)], in slot order
//
// The routing (top-k experts and their weights per row) is the caller's.
// Packed weights, per expert, are fp4::PackFp4's layout of the expert's
// [2I, H] (gate rows, then up rows) and [H, I] matrices, experts concatenated.

#include <cuda_pipeline.h>

#include "fp4_gemm.cu"

namespace moe {

struct Shape {
  int R, H, I, E, K;  // rows, hidden, intermediate, experts, top-k
  bool actFp4 = false;  // W4A4: the activations quantized to NVFP4 (below)
};

// One block of E threads: for each expert, its pairs (row * K + slot) in
// order, at lists[offsets[e] .. offsets[e + 1]).
__global__ void ListsKernel(const int* __restrict__ top, int* __restrict__ offsets, int* __restrict__ lists, Shape sh)
{
  extern __shared__ int counts[];  // [E + 1]
  const int P = sh.R * sh.K;
  for (int e = threadIdx.x; e < sh.E; e += blockDim.x) {
    int n = 0;
    for (int p = 0; p < P; ++p) n += top[p] == e;
    counts[e] = n;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int s = 0;
    for (int e = 0; e < sh.E; ++e) {
      const int n = counts[e];
      offsets[e] = s;
      counts[e] = s;
      s += n;
    }
    offsets[sh.E] = s;
  }
  __syncthreads();
  for (int e = threadIdx.x; e < sh.E; e += blockDim.x) {
    int at = counts[e];
    for (int p = 0; p < P; ++p) {
      if (top[p] == e) lists[at++] = p;
    }
  }
}

// Grid (E, ceil(tiles / fp4::kWarps)). For expert blockIdx.x, its pairs in
// groups of 8: the group's input rows (x rows, or h of the pairs) to shared
// memory as bf16, then each warp one 16-row tile of the expert's matrix over
// all of `in` (a multiple of 64), written to out[pair][n] times the row scale.
// IN_PAIRS: the input of pair p is row p of `in` (h), else row p / K (x).
template <bool IN_PAIRS, typename IN>
__global__ void __launch_bounds__(fp4::kWarps * 32) ExpertGemmKernel(
    const IN* __restrict__ in, const uint4* __restrict__ codes, const uint32_t* __restrict__ scales,
    const float* __restrict__ scale2, const int* __restrict__ offsets, const int* __restrict__ lists,
    float* __restrict__ out, int N, int Kin, Shape sh)
{
  const int e = blockIdx.x;
  const int begin = offsets[e], count = offsets[e + 1] - begin;
  if (count == 0) return;
  extern __shared__ __align__(16) __nv_bfloat16 xs[];  // [8][Kin + 8]
  __shared__ uint32_t pairs[256];
  __shared__ int rowsOf[8];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) pairs[i] = fp4::E2m1Bf16(i & 15) | (fp4::E2m1Bf16(i >> 4) << 16);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int tiles = (N + 15) / 16, tile = blockIdx.y * fp4::kWarps + warp;
  const int G = Kin / 64, stride = Kin + 8;
  const int r = lane >> 2, c = 2 * (lane & 3);
  const size_t expertCodes = static_cast<size_t>(tiles) * G * 32;  // uint4 per expert
  const size_t expertScales = static_cast<size_t>(tiles) * G * 16;  // uint32 per expert
  for (int g0 = 0; g0 < count; g0 += 8) {
    const int n = min(8, count - g0);
    __syncthreads();  // the previous group is done with xs
    if (threadIdx.x < 8) rowsOf[threadIdx.x] = threadIdx.x < n ? lists[begin + g0 + threadIdx.x] : -1;
    __syncthreads();
    for (int q = threadIdx.x; q < 8 * (Kin / 2); q += blockDim.x) {
      const int i = q / (Kin / 2), k = 2 * (q % (Kin / 2));
      float2 v = make_float2(0.f, 0.f);
      const int p = rowsOf[i];
      if (p >= 0) {
        const size_t row = IN_PAIRS ? static_cast<size_t>(p) : static_cast<size_t>(p / sh.K);
        if constexpr (sizeof(IN) == 4) {
          v = *reinterpret_cast<const float2*>(reinterpret_cast<const float*>(in) + row * Kin + k);
        } else {
          v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(reinterpret_cast<const __nv_bfloat16*>(in) + row * Kin + k));
        }
      }
      *reinterpret_cast<__nv_bfloat162*>(xs + i * stride + k) = __floats2bfloat162_rn(v.x, v.y);
    }
    __syncthreads();
    if (tile >= tiles) continue;
    const uint4* cp = codes + e * expertCodes + static_cast<size_t>(tile) * G * 32 + lane;
    const uint32_t* sp = scales + e * expertScales + static_cast<size_t>(tile) * G * 16;
    float acc[4] = {};
    uint4 w = cp[0];
    uint32_t sa = sp[r], sb = sp[r + 8];
    for (int g = 0; g < G; ++g) {
      uint4 wn = w;
      uint32_t san = sa, sbn = sb;
      if (g + 1 < G) {
        wn = cp[(g + 1) * 32];
        san = sp[(g + 1) * 16 + r];
        sbn = sp[(g + 1) * 16 + r + 8];
      }
      const uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const __nv_bfloat162 lo = fp4::Scale2(sa, s), hi = fp4::Scale2(sb, s);
        const uint32_t q = words[s];
        uint32_t a[4];
        a[0] = fp4::Scaled(pairs, q & 0xff, lo);
        a[1] = fp4::Scaled(pairs, (q >> 8) & 0xff, hi);
        a[2] = fp4::Scaled(pairs, (q >> 16) & 0xff, lo);
        a[3] = fp4::Scaled(pairs, q >> 24, hi);
        const __nv_bfloat16* xr = xs + r * stride + g * 64 + s * 16 + c;
        fp4::Mma(acc, a, *reinterpret_cast<const uint32_t*>(xr), *reinterpret_cast<const uint32_t*>(xr + 8));
      }
      w = wn;
      sa = san;
      sb = sbn;
    }
    // acc: {feature f0, rows c, c+1}, {feature f0 + 8, rows c, c+1} of the group.
    const int f0 = tile * 16 + r, f1 = f0 + 8;
    const float* s2 = scale2 + static_cast<size_t>(e) * N;
    const int pc0 = c < n ? rowsOf[c] : -1, pc1 = c + 1 < n ? rowsOf[c + 1] : -1;
    if (f0 < N) {
      if (pc0 >= 0) out[static_cast<size_t>(pc0) * N + f0] = acc[0] * s2[f0];
      if (pc1 >= 0) out[static_cast<size_t>(pc1) * N + f0] = acc[1] * s2[f0];
    }
    if (f1 < N) {
      if (pc0 >= 0) out[static_cast<size_t>(pc0) * N + f1] = acc[2] * s2[f1];
      if (pc1 >= 0) out[static_cast<size_t>(pc1) * N + f1] = acc[3] * s2[f1];
    }
  }
}

// For many pairs per expert (prefill), lists, gate/up with the activation, and
// down in three kernels instead of ListsKernel, two ExpertGemmKernel and ActKernel.
constexpr int kWideRows = 64, kWideK = 128, kWideStride = kWideK + 8;
// Pairs per expert, on average, from which the wide kernels pay.
constexpr int kWideMinPairsPerExpert = 8;

// One block of 1024 threads: ListsKernel's lists, a pair's place within its
// expert's list taken by an atomic (the order there does not change a result).
__global__ void __launch_bounds__(1024) WideListsKernel(const int* __restrict__ top, int* __restrict__ offsets,
                                                        int* __restrict__ lists, Shape sh)
{
  extern __shared__ int counts[];  // [E]
  const int P = sh.R * sh.K;
  for (int e = threadIdx.x; e < sh.E; e += blockDim.x) counts[e] = 0;
  __syncthreads();
  for (int p = threadIdx.x; p < P; p += blockDim.x) atomicAdd(&counts[top[p]], 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    int s = 0;
    for (int e = 0; e < sh.E; ++e) {
      const int n = counts[e];
      offsets[e] = s;
      counts[e] = s;
      s += n;
    }
    offsets[sh.E] = s;
  }
  __syncthreads();
  for (int p = threadIdx.x; p < P; p += blockDim.x) lists[atomicAdd(&counts[top[p]], 1)] = p;
}

// xb = bf16(x), [R, H].
__global__ void ToBf16Kernel(const float* __restrict__ x, __nv_bfloat16* __restrict__ xb, int n2)
{
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= n2) return;
  const float2 v = reinterpret_cast<const float2*>(x)[q];
  reinterpret_cast<__nv_bfloat162*>(xb)[q] = __floats2bfloat162_rn(v.x, v.y);
}

// Grid (E, tiles / (2 fp4::kWarps)). For expert blockIdx.x, its pairs in groups
// of kWideRows, their bf16 inputs staged kWideK columns at a time (two buffers,
// cp.async); each warp multiplies two 16-row tiles of the expert's matrix, each
// unpacked once per 64 columns, with the group's kWideRows / 8 column tiles.
// GLU: the input of pair p is row p / K of `in`, the warp's tiles are gate rows
// and the up rows I on, and it writes h[p] = bf16(silu(g) * u) to `out` (bf16,
// [P, I]), with N = 2I. Otherwise the input of pair p is row p of `in`, the
// tiles adjacent, and out[p][n] (f32) is the product times the row scale.
template <bool GLU>
__global__ void __launch_bounds__(fp4::kWarps * 32) ExpertGemmWideKernel(
    const __nv_bfloat16* __restrict__ in, const uint4* __restrict__ codes, const uint32_t* __restrict__ scales,
    const float* __restrict__ scale2, const int* __restrict__ offsets, const int* __restrict__ lists,
    void* __restrict__ outp, int N, int Kin, Shape sh)
{
  constexpr int NT = kWideRows / 8;
  const int e = blockIdx.x;
  const int begin = offsets[e], count = offsets[e + 1] - begin;
  if (count == 0) return;
  extern __shared__ __align__(16) __nv_bfloat16 xs[];  // [2][kWideRows][kWideStride]
  __shared__ uint32_t pairs[256];
  __shared__ int rowsOf[kWideRows];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) pairs[i] = fp4::E2m1Bf16(i & 15) | (fp4::E2m1Bf16(i >> 4) << 16);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int tiles = N / 16, G = Kin / 64;
  const int u = blockIdx.y * fp4::kWarps + warp;
  const int tA = GLU ? u : 2 * u, tB = GLU ? u + tiles / 2 : 2 * u + 1;
  const bool active = GLU ? u < tiles / 2 : tB < tiles;
  const int r = lane >> 2, c = 2 * (lane & 3);
  const size_t expertCodes = static_cast<size_t>(tiles) * G * 32, expertScales = static_cast<size_t>(tiles) * G * 16;
  const uint4* cpA = codes + e * expertCodes + static_cast<size_t>(tA) * G * 32 + lane;
  const uint4* cpB = codes + e * expertCodes + static_cast<size_t>(tB) * G * 32 + lane;
  const uint32_t* spA = scales + e * expertScales + static_cast<size_t>(tA) * G * 16;
  const uint32_t* spB = scales + e * expertScales + static_cast<size_t>(tB) * G * 16;
  const int chunks = Kin / kWideK;
  for (int g0 = 0; g0 < count; g0 += kWideRows) {
    const int n = min(kWideRows, count - g0), ntUsed = (n + 7) / 8;
    __syncthreads();  // the previous group is done with rowsOf and xs
    if (threadIdx.x < kWideRows) rowsOf[threadIdx.x] = threadIdx.x < n ? lists[begin + g0 + threadIdx.x] : -1;
    __syncthreads();
    auto stage = [&](int ch, int buf) {
      for (int q = threadIdx.x; q < kWideRows * (kWideK / 8); q += blockDim.x) {
        const int i = q / (kWideK / 8), k = (q % (kWideK / 8)) * 8;
        const int p = rowsOf[i];
        const size_t row = p < 0 ? 0 : GLU ? static_cast<size_t>(p / sh.K) : static_cast<size_t>(p);
        __pipeline_memcpy_async(xs + (buf * kWideRows + i) * kWideStride + k, in + row * Kin + ch * kWideK + k, 16,
                                p < 0 ? 16 : 0);
      }
      __pipeline_commit();
    };
    // The chunk's weights (two tiles, kWideK / 64 column groups) in registers,
    // the next chunk's loaded while this one is multiplied.
    constexpr int GC = kWideK / 64;
    uint4 w[GC][2];
    uint32_t sa[GC][2], sb[GC][2];
    auto loadWeights = [&](int ch, uint4 (&wv)[GC][2], uint32_t (&av)[GC][2], uint32_t (&bv)[GC][2]) {
#pragma unroll
      for (int g = 0; g < GC; ++g) {
        const int gg = ch * GC + g;
        wv[g][0] = cpA[gg * 32];
        wv[g][1] = cpB[gg * 32];
        av[g][0] = spA[gg * 16 + r];
        av[g][1] = spB[gg * 16 + r];
        bv[g][0] = spA[gg * 16 + r + 8];
        bv[g][1] = spB[gg * 16 + r + 8];
      }
    };
    float acc[2][NT][4] = {};
    if (active) loadWeights(0, w, sa, sb);
    stage(0, 0);
    for (int ch = 0; ch < chunks; ++ch) {
      if (ch + 1 < chunks) {
        stage(ch + 1, (ch + 1) & 1);
        __pipeline_wait_prior(1);
      } else {
        __pipeline_wait_prior(0);
      }
      uint4 wn[GC][2];
      uint32_t san[GC][2], sbn[GC][2];
      if (active && ch + 1 < chunks) loadWeights(ch + 1, wn, san, sbn);
      __syncthreads();
      const __nv_bfloat16* xb = xs + (ch & 1) * kWideRows * kWideStride;
      if (active) {
#pragma unroll
        for (int g = 0; g < GC; ++g) {
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            uint32_t a[2][4];
#pragma unroll
            for (int t = 0; t < 2; ++t) {
              const __nv_bfloat162 lo = fp4::Scale2(sa[g][t], s), hi = fp4::Scale2(sb[g][t], s);
              const uint32_t q = s == 0 ? w[g][t].x : s == 1 ? w[g][t].y : s == 2 ? w[g][t].z : w[g][t].w;
              a[t][0] = fp4::Scaled(pairs, q & 0xff, lo);
              a[t][1] = fp4::Scaled(pairs, (q >> 8) & 0xff, hi);
              a[t][2] = fp4::Scaled(pairs, (q >> 16) & 0xff, lo);
              a[t][3] = fp4::Scaled(pairs, q >> 24, hi);
            }
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
              if (nt < ntUsed) {
                const __nv_bfloat16* xr = xb + (nt * 8 + r) * kWideStride + g * 64 + s * 16 + c;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(xr), b1 = *reinterpret_cast<const uint32_t*>(xr + 8);
                fp4::Mma(acc[0][nt], a[0], b0, b1);
                fp4::Mma(acc[1][nt], a[1], b0, b1);
              }
            }
          }
        }
        if (ch + 1 < chunks) {
#pragma unroll
          for (int g = 0; g < GC; ++g) {
#pragma unroll
            for (int t = 0; t < 2; ++t) {
              w[g][t] = wn[g][t];
              sa[g][t] = san[g][t];
              sb[g][t] = sbn[g][t];
            }
          }
        }
      }
      __syncthreads();  // this buffer is free for the chunk after next
    }
    if (!active) continue;
    // acc[t][nt]: {feature r, pairs nt*8 + c, + 1}, {feature r + 8, same pairs} of tile t.
    const float* s2 = scale2 + static_cast<size_t>(e) * N;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int i0 = nt * 8 + c;
      const int pr[2] = {i0 < n ? rowsOf[i0] : -1, i0 + 1 < n ? rowsOf[i0 + 1] : -1};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int p = pr[j & 1];
        if (p < 0) continue;
        const int fa = tA * 16 + r + (j >> 1) * 8, fb = tB * 16 + r + (j >> 1) * 8;
        if constexpr (GLU) {
          const float g = acc[0][nt][j] * s2[fa], v = acc[1][nt][j] * s2[fb];
          static_cast<__nv_bfloat16*>(outp)[static_cast<size_t>(p) * (N / 2) + fa] = __float2bfloat16(g / (1.f + expf(-g)) * v);
        } else {
          float* o = static_cast<float*>(outp) + static_cast<size_t>(p) * N;
          o[fa] = acc[0][nt][j] * s2[fa];
          o[fb] = acc[1][nt][j] * s2[fb];
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// W4A4 (Shape::actFp4): the activations quantized to NVFP4 as well, and the products on
// the block-scaled FP4 tensor-core instruction
//   mma.sync.m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X (e2m1 x e2m1, ue4m3 scales)
// (sm_120a / sm_121a). A row v of K values is quantized as:
//   g = amax(v) / 2688 (6 x 448), the row's f32 scale (rows of zeros: g = 0, all codes 0);
//   per 16 values, s = e4m3_rn_satfinite(amax16 / (6 g)), and each value's code
//   e2m1_rn_even_satfinite(v / (s g)) (code 0 where s is 0).
// The products sum in f32 inside the instruction; the result is multiplied by the weight
// row's scale and the activation row's g. h = silu(g) u stays f32 and is quantized per pair
// before the down projection. MoeExpertsAttrs.ACT_QUANT and the interpreters define the same.
// The down projection adds each pair's result, times its routing weight, into y with f32
// atomics (y fits in L2; no [P, H] partials and no CombineKernel), so a row's experts add in
// the order they finish.
// Qwen3.6-35B-A3B's experts, 2,048 rows, one layer (bench/moe_fp4_bench.cu): 5.24 ms on the
// bf16 path (bound by widening the weights), 3.15 ms here (bound by reading them).

// The e2m1 code of q (round to nearest, ties to the even code; |q| > 6 saturates).
__host__ __device__ inline uint32_t E2m1Code(float q)
{
  const float a = fabsf(q);
  uint32_t c;
  if (a <= 0.25f) c = 0;
  else if (a < 0.75f) c = 1;
  else if (a <= 1.25f) c = 2;
  else if (a < 1.75f) c = 3;
  else if (a <= 2.5f) c = 4;
  else if (a < 3.5f) c = 5;
  else if (a <= 5.f) c = 6;
  else c = 7;
  return (c != 0 && q < 0.f) ? (c | 8u) : c;
}

constexpr int kQuantThreads = 256;

// One block per row of `in` (f32, [rows, K], K a multiple of 16 and at most 16 kQuantThreads):
// codes [rows, K / 2] (low nibble the even column), scales [rows, K / 16], g [rows].
// GLU input rows of the gate/up projection are rows of x; h rows are pairs.
__global__ void __launch_bounds__(kQuantThreads) QuantizeRowsKernel(
    const float* __restrict__ in, uint8_t* __restrict__ codes, uint8_t* __restrict__ scales, float* __restrict__ gOut, int K)
{
  const int row = blockIdx.x, tid = threadIdx.x, groups = K / 16;
  const float* v = in + static_cast<size_t>(row) * K;
  __shared__ float red[kQuantThreads / 32];
  float m = 0.f;
  for (int k = tid; k < K; k += kQuantThreads) m = fmaxf(m, fabsf(v[k]));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((tid & 31) == 0) red[tid >> 5] = m;
  __syncthreads();
  if (tid < 32) {
    m = tid < kQuantThreads / 32 ? red[tid] : 0.f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (tid == 0) red[0] = m;
  }
  __syncthreads();
  const float g = __fdiv_rn(red[0], 2688.f);
  if (tid == 0) gOut[row] = g;
  const float g6 = __fmul_rn(g, 6.f);
  for (int q = tid; q < groups; q += kQuantThreads) {
    const float* x = v + 16 * q;
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) a = fmaxf(a, fabsf(x[i]));
    const __nv_fp8_e4m3 s8(g6 > 0.f ? __fdiv_rn(a, g6) : 0.f);  // round to nearest, saturating
    const uint8_t sb = *reinterpret_cast<const uint8_t*>(&s8);
    scales[static_cast<size_t>(row) * groups + q] = sb;
    const float sg = __fmul_rn(static_cast<float>(s8), g);
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const uint32_t c = sg > 0.f ? E2m1Code(__fdiv_rn(x[i], sg)) : 0u;
      if (i < 8) lo |= c << (4 * i); else hi |= c << (4 * (i - 8));
    }
    uint32_t* o = reinterpret_cast<uint32_t*>(codes + static_cast<size_t>(row) * (K / 2) + 8 * q);
    o[0] = lo;
    o[1] = hi;
  }
}

__device__ __forceinline__ void MmaFp4(float* c, const uint32_t* a, uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb)
{
#if defined(__CUDA_ARCH_FEAT_SM121_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__)
  const uint16_t z = 0;
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%11,%11}, %12, {%11,%11};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "h"(z), "r"(sb));
#else
  __trap();  // built without the sm_121a / sm_120a features
#endif
}

// The m16n8k64 A fragment of one 16-row tile and 64-column group from fp4::PackFp4's
// m16n8k16 layout. Lane j of a quad holds w = the words s = 0..3 of its row r = lane / 4:
// word s byte i is row r (i even) or r + 8 (i odd) at bytes 8s + j (i < 2) or 8s + j + 4 of
// the row's 32. The FP4 fragment wants, in lane q, register 0 = bytes 4q .. 4q + 3 of row r,
// 1 = the same of row r + 8, 2 and 3 the bytes 16 on: byte j of each from quad lane j. Each
// lane sends every other its four bytes (three shuffles), then a 4 x 4 byte transpose.
__device__ __forceinline__ void Fp4Fragment(const uint4& w, uint32_t (&a)[4])
{
  const int q = threadIdx.x & 3;
  // What this lane sends lane t of its quad: bytes 2(t % 2), 2(t % 2) + 1 of words t / 2 and 2 + t / 2.
  auto msg = [&](int t) {
    const uint32_t lo = (t >> 1) ? w.y : w.x, hi = (t >> 1) ? w.w : w.z;
    return __byte_perm(lo, hi, (t & 1) ? 0x7632 : 0x5410);
  };
  // recv[d]: what lane q ^ d sent this lane.
  const uint32_t r0 = msg(q);
  const uint32_t r1 = __shfl_xor_sync(0xffffffffu, msg(q ^ 1), 1);
  const uint32_t r2 = __shfl_xor_sync(0xffffffffu, msg(q ^ 2), 2);
  const uint32_t r3 = __shfl_xor_sync(0xffffffffu, msg(q ^ 3), 3);
  auto from = [&](int j) {  // what lane j sent: recv[q ^ j]
    const int d = q ^ j;
    return d == 0 ? r0 : d == 1 ? r1 : d == 2 ? r2 : r3;
  };
  const uint32_t m0 = from(0), m1 = from(1), m2 = from(2), m3 = from(3);
  const uint32_t x = __byte_perm(m0, m1, 0x5140), y = __byte_perm(m2, m3, 0x5140);
  const uint32_t x2 = __byte_perm(m0, m1, 0x7362), y2 = __byte_perm(m2, m3, 0x7362);
  a[0] = __byte_perm(x, y, 0x5410);
  a[1] = __byte_perm(x, y, 0x7632);
  a[2] = __byte_perm(x2, y2, 0x5410);
  a[3] = __byte_perm(x2, y2, 0x7632);
}

constexpr int kFp4Rows = 64;    // pairs per block tile (8 column tiles)
constexpr int kFp4K = 128;      // columns per pipeline stage (two 64-column groups)
constexpr int kFp4Stages = 3;
constexpr int kFp4Tiles = 16;   // 16-row weight tiles per block, two per warp
// Shared memory per stage: the tiles' packed codes and scales, the pairs' codes (rows padded
// against bank conflicts) and scale words.
constexpr int kFp4WBytes = kFp4Tiles * (kFp4K / 64) * 512, kFp4WsBytes = kFp4Tiles * (kFp4K / 64) * 64;
constexpr int kFp4XStride = kFp4K / 2 + 16;
constexpr int kFp4XBytes = kFp4Rows * kFp4XStride, kFp4XsBytes = kFp4Rows * (kFp4K / 16);
constexpr int kFp4StageBytes = kFp4WBytes + kFp4WsBytes + kFp4XBytes + kFp4XsBytes;
constexpr int kFp4SmemBytes = kFp4Stages * kFp4StageBytes;

// Grid (E, row blocks); 8 warps. A block takes kFp4Tiles 16-row tiles of expert blockIdx.x's
// matrix (GLU: gate tiles 8b .. 8b + 7 and their up tiles, I / 16 on; else tiles 16b .. 16b + 15)
// against its pairs, kFp4Rows at a time, kFp4K columns per stage through a kFp4Stages-deep
// cp.async pipeline. Warp w takes local tiles w and w + 8 (a gate tile and its up tile, or
// tiles 16b + w and 16b + w + 8) against all 8 column tiles. Inputs: NVFP4 rows
// (QuantizeRowsKernel's layout, K = Kin) and their row scales gIn; GLU: pair p reads row p / K
// and writes h[p] = silu(g) u (f32, [P, N / 2]); otherwise pair p reads row p and adds its
// result times the pair's routing weight to row p / K of `out` ([R, N], zeroed first).
template <bool GLU>
__global__ void __launch_bounds__(fp4::kWarps * 32, 1) ExpertGemmFp4Kernel(
    const uint8_t* __restrict__ inCodes, const uint8_t* __restrict__ inScales, const float* __restrict__ gIn,
    const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, const float* __restrict__ scale2,
    const int* __restrict__ offsets, const int* __restrict__ lists, float* __restrict__ outp, int N, int Kin, Shape sh,
    const float* __restrict__ routeWeights)
{
  constexpr int GC = kFp4K / 64;
  static_assert(fp4::kWarps == 8 && kFp4Tiles == 16 && kFp4Rows == 64, "the staging below assumes these");
  const int e = blockIdx.x;
  const int begin = offsets[e], count = offsets[e + 1] - begin;
  if (count == 0) return;
  extern __shared__ __align__(16) uint4 smem4[];
  uint8_t* smem = reinterpret_cast<uint8_t*>(smem4);
  __shared__ int rowsOf[2][kFp4Rows];  // per group parity: input row of each pair (-1: none)
  __shared__ int pairsOf[2][kFp4Rows];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int tiles = N / 16, G = Kin / 64;
  auto tileOf = [&](int j) {
    return GLU ? (j < 8 ? 8 * blockIdx.y + j : tiles / 2 + 8 * blockIdx.y + (j - 8)) : kFp4Tiles * blockIdx.y + j;
  };
  const int r = lane >> 2, c = 2 * (lane & 3);
  const int srow = (lane & 3) == 1 ? r + 8 : r;
  const size_t expertCodes = static_cast<size_t>(tiles) * G * 512, expertScales = static_cast<size_t>(tiles) * G * 64;
  // This thread's staging per stage: of weight tile tid / 16, the 16-byte chunks at wo + 256 i
  // (i = 0..3) of its 1 KB of codes; of pair row tid / 4, 16 bytes of codes; threads 64 .. 191 a
  // 16-byte chunk of the tiles' scales (128 bytes a tile), threads below 64 a row's scale bytes.
  const int wj = tid >> 4, wo = (tid & 15) * 16;
  const uint8_t* wSrc = codes + e * expertCodes + static_cast<size_t>(tileOf(wj)) * G * 512 + wo;
  const int sj = (tid - 64) >> 3, so = ((tid - 64) & 7) * 16;
  const uint8_t* wsSrc = tid >= 64 && tid < 64 + 8 * kFp4Tiles
      ? scales + e * expertScales + static_cast<size_t>(tileOf(sj)) * G * 64 + so
      : nullptr;
  const int xi = tid >> 2, xo = (tid & 3) * 16;
  // The expert's groups of kFp4Rows pairs.
  const int chunks = Kin / kFp4K, groups = (count + kFp4Rows - 1) / kFp4Rows, total = groups * chunks;
  const size_t rowBytes = Kin / 2, rowScales = Kin / 16;

  auto setRows = [&](int grp) {  // before the group's first stage is issued (grp: this block's)
    if (tid < kFp4Rows) {
      const int q = grp * kFp4Rows + tid;
      const int p = q < count ? lists[begin + q] : -1;
      pairsOf[grp & 1][tid] = p;
      rowsOf[grp & 1][tid] = p < 0 ? -1 : GLU ? p / sh.K : p;
    }
  };
  auto issue = [&](int it) {
    if (it < total) {
      const int grp = it / chunks, ch = it - grp * chunks;
      uint8_t* st = smem + (it % kFp4Stages) * kFp4StageBytes;
      uint8_t* w = st;
      uint8_t* ws = st + kFp4WBytes;
      uint8_t* x = ws + kFp4WsBytes;
      uint8_t* xsc = x + kFp4XBytes;
      const size_t at = static_cast<size_t>(ch) * GC * 512;
#pragma unroll
      for (int i = 0; i < 4; ++i) __pipeline_memcpy_async(w + wj * GC * 512 + wo + 256 * i, wSrc + at + 256 * i, 16);
      if (wsSrc != nullptr) __pipeline_memcpy_async(ws + sj * GC * 64 + so, wsSrc + static_cast<size_t>(ch) * GC * 64, 16);
      const int row = rowsOf[grp & 1][xi];
      const size_t rr = row < 0 ? 0 : static_cast<size_t>(row);
      __pipeline_memcpy_async(x + xi * kFp4XStride + xo, inCodes + rr * rowBytes + ch * (kFp4K / 2) + xo, 16, row < 0 ? 16 : 0);
      if (tid < kFp4Rows) {
        const int row2 = rowsOf[grp & 1][tid];
        __pipeline_memcpy_async(xsc + tid * (kFp4K / 16), inScales + (row2 < 0 ? 0 : static_cast<size_t>(row2)) * rowScales + ch * (kFp4K / 16),
                                kFp4K / 16, row2 < 0 ? kFp4K / 16 : 0);
      }
    }
    __pipeline_commit();
  };

  float acc[2][8][4] = {};
  setRows(0);
  if (groups > 1) setRows(1);
  __syncthreads();
#pragma unroll
  for (int s = 0; s < kFp4Stages - 1; ++s) issue(s);
  for (int it = 0; it < total; ++it) {
    __pipeline_wait_prior(kFp4Stages - 2);
    __syncthreads();  // stage it is in; everyone is done with stage it - 1
    issue(it + kFp4Stages - 1);
    const int grp = it / chunks, ch = it - grp * chunks;
    const int nUsed = min(kFp4Rows, count - grp * kFp4Rows);
    const uint8_t* st = smem + (it % kFp4Stages) * kFp4StageBytes;
    const uint8_t* w = st;
    const uint32_t* ws = reinterpret_cast<const uint32_t*>(st + kFp4WBytes);
    const uint8_t* x = st + kFp4WBytes + kFp4WsBytes;
    const uint32_t* xsc = reinterpret_cast<const uint32_t*>(x + kFp4XBytes);
#pragma unroll
    for (int g = 0; g < GC; ++g) {
      uint32_t a[2][4];
      Fp4Fragment(reinterpret_cast<const uint4*>(w + (warp * GC + g) * 512)[lane], a[0]);
      Fp4Fragment(reinterpret_cast<const uint4*>(w + ((warp + 8) * GC + g) * 512)[lane], a[1]);
      const uint32_t sa0 = ws[(warp * GC + g) * 16 + srow], sa1 = ws[((warp + 8) * GC + g) * 16 + srow];
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        if (nt * 8 < nUsed) {
          const uint8_t* xr = x + (nt * 8 + r) * kFp4XStride + g * 32 + 4 * (lane & 3);
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(xr), b1 = *reinterpret_cast<const uint32_t*>(xr + 16);
          const uint32_t sb = xsc[(nt * 8 + r) * GC + g];
          MmaFp4(acc[0][nt], a[0], b0, b1, sa0, sb);
          MmaFp4(acc[1][nt], a[1], b0, b1, sa1, sb);
        }
      }
    }
    if (ch + 1 < chunks) continue;
    // The group's results: acc[m][nt] holds {feature r, pairs 8 nt + c, + 1}, {feature r + 8, ...}.
    const float* s2 = scale2 + static_cast<size_t>(e) * N;
    const int tA = tileOf(warp), tB = tileOf(warp + 8);
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const int i0 = nt * 8 + c;
      const int pr[2] = {i0 < nUsed ? pairsOf[grp & 1][i0] : -1, i0 + 1 < nUsed ? pairsOf[grp & 1][i0 + 1] : -1};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int p = pr[j & 1];
        if (p >= 0) {
          const float gp = gIn[GLU ? p / sh.K : p];
          const int fa = tA * 16 + r + (j >> 1) * 8, fb = tB * 16 + r + (j >> 1) * 8;
          if constexpr (GLU) {
            const float g = acc[0][nt][j] * s2[fa] * gp, v = acc[1][nt][j] * s2[fb] * gp;
            outp[static_cast<size_t>(p) * (N / 2) + fa] = g / (1.f + expf(-g)) * v;
          } else {
            // y[row] += w[pair] (h . down^T).
            const float wp = routeWeights[p] * gp;
            float* o = outp + static_cast<size_t>(p / sh.K) * N;
            atomicAdd(o + fa, acc[0][nt][j] * s2[fa] * wp);
            atomicAdd(o + fb, acc[1][nt][j] * s2[fb] * wp);
          }
        }
        acc[0][nt][j] = 0.f;
        acc[1][nt][j] = 0.f;
      }
    }
    // The rows of the group after next go where this group's were (stages of the next group
    // were already issued with its rows, set at the previous group's end).
    __syncthreads();
    if (grp + 2 < groups) setRows(grp + 2);
  }
}

// h[pair][i] = bf16(silu(g) * u), g and u the pair's gate and up outputs.
__global__ void ActKernel(const float* __restrict__ gu, __nv_bfloat16* __restrict__ h, int pairs, int I)
{
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= pairs * I) return;
  const int p = q / I, i = q % I;
  const float g = gu[static_cast<size_t>(p) * 2 * I + i], u = gu[static_cast<size_t>(p) * 2 * I + I + i];
  h[q] = __float2bfloat16(g / (1.f + expf(-g)) * u);
}

// y[row][d] = sum_j w[row][j] * part[row * K + j][d].
__global__ void CombineKernel(const float* __restrict__ part, const float* __restrict__ weights, float* __restrict__ y, Shape sh)
{
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= sh.R * sh.H) return;
  const int row = q / sh.H, d = q % sh.H;
  float s = 0.f;
  for (int j = 0; j < sh.K; ++j) s += weights[row * sh.K + j] * part[(static_cast<size_t>(row) * sh.K + j) * sh.H + d];
  y[q] = s;
}

// Shapes the W4A4 kernels take: whole chunks of kFp4K columns, whole tiles, and QuantizeRowsKernel's
// rows of at most 16 kQuantThreads values, and the scratch ScratchBytes gives fits their buffers.
inline bool Fp4Shape(const Shape& sh)
{
  const size_t P = static_cast<size_t>(sh.R) * sh.K;
  const size_t xBytes = static_cast<size_t>(sh.R) * sh.H / 2 + ((static_cast<size_t>(sh.R) * sh.H / 16 + 15) / 16) * 16 + 4 * sh.R;
  const size_t hBytes = P * sh.I / 2 + ((P * sh.I / 16 + 15) / 16) * 16 + 4 * P;
  return sh.K >= 2 && sh.H % 256 == 0 && sh.I % 256 == 0 && sh.H <= 16 * kQuantThreads && sh.I <= 16 * kQuantThreads &&
         P * sh.I * 4 + xBytes <= P * 2 * sh.I * 4 && hBytes <= P * sh.I * 2;
}

// Every kernel Launch may use (XLA's INITIALIZE stage).
inline cudaError_t Prepare()
{
  cudaError_t e = tlaloc_kernels::AllowMaxSharedMemory(ExpertGemmKernel<false, float>);
  if (e == cudaSuccess) e = tlaloc_kernels::AllowMaxSharedMemory(ExpertGemmKernel<true, __nv_bfloat16>);
  if (e == cudaSuccess) e = tlaloc_kernels::AllowMaxSharedMemory(ExpertGemmWideKernel<true>);
  if (e == cudaSuccess) e = tlaloc_kernels::AllowMaxSharedMemory(ExpertGemmWideKernel<false>);
  if (e == cudaSuccess) e = tlaloc_kernels::AllowMaxSharedMemory(ExpertGemmFp4Kernel<true>);
  if (e == cudaSuccess) e = tlaloc_kernels::AllowMaxSharedMemory(ExpertGemmFp4Kernel<false>);
  return e;
}

// Scratch, in bytes, for Launch: lists, gate/up outputs, h, down outputs.
inline size_t ScratchBytes(const Shape& sh)
{
  const size_t P = static_cast<size_t>(sh.R) * sh.K;
  return 4 * (sh.E + 1 + P) + 4 * P * 2 * sh.I + 2 * P * sh.I + 4 * P * sh.H + 64;
}

// x [R, H] f32, top [R, K] i32 expert ids, weights [R, K] f32; gate/up codes,
// scales, row scales [E, 2I]; down codes, scales, row scales [E, H]; y [R, H].
inline cudaError_t Launch(
    const float* x, const int* top, const float* weights, const void* guCodes, const void* guScales, const float* guScale2,
    const void* dnCodes, const void* dnScales, const float* dnScale2, float* y, void* scratch, const Shape& sh,
    cudaStream_t stream)
{
  const size_t P = static_cast<size_t>(sh.R) * sh.K;
  char* s = static_cast<char*>(scratch);
  int* offsets = reinterpret_cast<int*>(s);
  int* lists = offsets + sh.E + 1;
  float* gu = reinterpret_cast<float*>(s + ((4 * (sh.E + 1 + P) + 15) / 16) * 16);
  __nv_bfloat16* h = reinterpret_cast<__nv_bfloat16*>(gu + P * 2 * sh.I);
  float* part = reinterpret_cast<float*>(h + P * sh.I + (P * sh.I) % 2);
  const int t1 = (2 * sh.I + 15) / 16, t2 = (sh.H + 15) / 16;
  if (sh.actFp4) {
    if (!Fp4Shape(sh)) return cudaErrorInvalidValue;
    // gu's space: h f32 [P, I], then x's codes, scales and row scales. h's bf16 space: h's.
    float* hf = gu;
    uint8_t* xq = reinterpret_cast<uint8_t*>(gu + P * sh.I);
    uint8_t* xqs = xq + static_cast<size_t>(sh.R) * sh.H / 2;
    float* gx = reinterpret_cast<float*>(xqs + ((static_cast<size_t>(sh.R) * sh.H / 16 + 15) / 16) * 16);
    uint8_t* hq = reinterpret_cast<uint8_t*>(h);
    uint8_t* hqs = hq + P * sh.I / 2;
    float* gh = reinterpret_cast<float*>(hqs + ((P * sh.I / 16 + 15) / 16) * 16);
    const int fp4Bytes = kFp4SmemBytes;
    WideListsKernel<<<1, 1024, 4 * sh.E, stream>>>(top, offsets, lists, sh);
    QuantizeRowsKernel<<<sh.R, kQuantThreads, 0, stream>>>(x, xq, xqs, gx, sh.H);
    ExpertGemmFp4Kernel<true><<<dim3(sh.E, sh.I / 128), fp4::kWarps * 32, fp4Bytes, stream>>>(
        xq, xqs, gx, static_cast<const uint8_t*>(guCodes), static_cast<const uint8_t*>(guScales), guScale2, offsets, lists,
        hf, 2 * sh.I, sh.H, sh, nullptr);
    QuantizeRowsKernel<<<static_cast<int>(P), kQuantThreads, 0, stream>>>(hf, hq, hqs, gh, sh.I);
    cudaMemsetAsync(y, 0, static_cast<size_t>(sh.R) * sh.H * sizeof(float), stream);
    ExpertGemmFp4Kernel<false><<<dim3(sh.E, sh.H / 256), fp4::kWarps * 32, fp4Bytes, stream>>>(
        hq, hqs, gh, static_cast<const uint8_t*>(dnCodes), static_cast<const uint8_t*>(dnScales), dnScale2, offsets,
        lists, y, sh.H, sh.I, sh, weights);
    return cudaGetLastError();
  } else if (P >= static_cast<size_t>(kWideMinPairsPerExpert) * sh.E && sh.I % 16 == 0 && sh.H % kWideK == 0 && sh.I % kWideK == 0) {
    // x as bf16 where the gate/up outputs would be: R H 2 < P 2I 4 bytes.
    __nv_bfloat16* xb = reinterpret_cast<__nv_bfloat16*>(gu);
    const int wideBytes = 2 * kWideRows * kWideStride * 2;
    WideListsKernel<<<1, 1024, 4 * sh.E, stream>>>(top, offsets, lists, sh);
    ToBf16Kernel<<<(sh.R * sh.H / 2 + 255) / 256, 256, 0, stream>>>(x, xb, sh.R * sh.H / 2);
    ExpertGemmWideKernel<true><<<dim3(sh.E, (sh.I / 16 + fp4::kWarps - 1) / fp4::kWarps), fp4::kWarps * 32, wideBytes, stream>>>(
        xb, static_cast<const uint4*>(guCodes), static_cast<const uint32_t*>(guScales), guScale2, offsets, lists, h,
        2 * sh.I, sh.H, sh);
    ExpertGemmWideKernel<false><<<dim3(sh.E, (t2 / 2 + fp4::kWarps - 1) / fp4::kWarps), fp4::kWarps * 32, wideBytes, stream>>>(
        h, static_cast<const uint4*>(dnCodes), static_cast<const uint32_t*>(dnScales), dnScale2, offsets, lists, part,
        sh.H, sh.I, sh);
  } else {
    ListsKernel<<<1, 256, 4 * (sh.E + 1), stream>>>(top, offsets, lists, sh);
    {
      auto k = ExpertGemmKernel<false, float>;
      const int bytes = 8 * (sh.H + 8) * 2;
      tlaloc_kernels::AllowMaxSharedMemory(k);
      k<<<dim3(sh.E, (t1 + fp4::kWarps - 1) / fp4::kWarps), fp4::kWarps * 32, bytes, stream>>>(
          x, static_cast<const uint4*>(guCodes), static_cast<const uint32_t*>(guScales), guScale2, offsets, lists, gu,
          2 * sh.I, sh.H, sh);
    }
    ActKernel<<<static_cast<int>((P * sh.I + 255) / 256), 256, 0, stream>>>(gu, h, static_cast<int>(P), sh.I);
    {
      auto k = ExpertGemmKernel<true, __nv_bfloat16>;
      const int bytes = 8 * (sh.I + 8) * 2;
      tlaloc_kernels::AllowMaxSharedMemory(k);
      k<<<dim3(sh.E, (t2 + fp4::kWarps - 1) / fp4::kWarps), fp4::kWarps * 32, bytes, stream>>>(
          h, static_cast<const uint4*>(dnCodes), static_cast<const uint32_t*>(dnScales), dnScale2, offsets, lists, part,
          sh.H, sh.I, sh);
    }
  }
  CombineKernel<<<(sh.R * sh.H + 255) / 256, 256, 0, stream>>>(part, weights, y, sh);
  return cudaGetLastError();
}

}  // namespace moe
