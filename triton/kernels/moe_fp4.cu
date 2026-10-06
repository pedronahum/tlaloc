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

// Every kernel Launch may use (XLA's INITIALIZE stage).
inline cudaError_t Prepare()
{
  cudaError_t e = tlaloc_kernels::AllowMaxSharedMemory(ExpertGemmKernel<false, float>);
  if (e == cudaSuccess) e = tlaloc_kernels::AllowMaxSharedMemory(ExpertGemmKernel<true, __nv_bfloat16>);
  if (e == cudaSuccess) e = tlaloc_kernels::AllowMaxSharedMemory(ExpertGemmWideKernel<true>);
  if (e == cudaSuccess) e = tlaloc_kernels::AllowMaxSharedMemory(ExpertGemmWideKernel<false>);
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
  if (P >= static_cast<size_t>(kWideMinPairsPerExpert) * sh.E && sh.I % 16 == 0 && sh.H % kWideK == 0 && sh.I % kWideK == 0) {
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
