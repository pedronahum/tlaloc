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
  ListsKernel<<<1, 256, 4 * (sh.E + 1), stream>>>(top, offsets, lists, sh);
  const int t1 = (2 * sh.I + 15) / 16, t2 = (sh.H + 15) / 16;
  {
    auto k = ExpertGemmKernel<false, float>;
    const int bytes = 8 * (sh.H + 8) * 2;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    k<<<dim3(sh.E, (t1 + fp4::kWarps - 1) / fp4::kWarps), fp4::kWarps * 32, bytes, stream>>>(
        x, static_cast<const uint4*>(guCodes), static_cast<const uint32_t*>(guScales), guScale2, offsets, lists, gu,
        2 * sh.I, sh.H, sh);
  }
  ActKernel<<<static_cast<int>((P * sh.I + 255) / 256), 256, 0, stream>>>(gu, h, static_cast<int>(P), sh.I);
  {
    auto k = ExpertGemmKernel<true, __nv_bfloat16>;
    const int bytes = 8 * (sh.I + 8) * 2;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    k<<<dim3(sh.E, (t2 + fp4::kWarps - 1) / fp4::kWarps), fp4::kWarps * 32, bytes, stream>>>(
        h, static_cast<const uint4*>(dnCodes), static_cast<const uint32_t*>(dnScales), dnScale2, offsets, lists, part,
        sh.H, sh.I, sh);
  }
  CombineKernel<<<(sh.R * sh.H + 255) / 256, 256, 0, stream>>>(part, weights, y, sh);
  return cudaGetLastError();
}

}  // namespace moe
