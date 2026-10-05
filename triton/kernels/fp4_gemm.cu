// y = x W^T for a few rows of x and an NVFP4 weight W, on tensor cores.
//
// W [N, K] is e2m1 codes with one e4m3 scale per 16 consecutive values of a row
// (a group) and one f32 scale for the tensor, as NVIDIA's NVFP4 checkpoints
// store it. The tensor scale is given per output row, so weights stacked along
// N (gate and up) keep their own. The codes and group scales are repacked once, at staging
// (PackFp4), into the order the mma.sync m16n8k16 A fragment reads them, so
// each lane loads 16 contiguous bytes for four k-steps of its 16-row tile.
//
// A code times its group scale has at most 6 significant bits and is exact in
// bf16, so the tile is widened to bf16 exactly, multiplied with x rounded to
// bf16, summed in f32, and the row's tensor scale applied to the sum: what a bf16 dot
// of the widened weight computes.
//
// Packed layout, for tile t (rows 16t .. 16t+15), k-group g (k 64g .. 64g+63),
// lane l (r = l / 4, c = 2 (l % 4)) and k-step s (k base kb = 64g + 16s):
//   codes[(((t G + g) 32 + l) 4 + s) 4 + i], i = 0..3, the bytes holding
//     k = kb+c, kb+c+1 of row r; of row r+8; k = kb+c+8, kb+c+9 of row r; of row r+8
//   scales[((t G + g) 16 + r) 4 + s], the e4m3 scale of row r's group kb / 16
// with G = K / 64; rows past N are zero codes.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <cstring>

#include "kernel_setup.cuh"

namespace fp4 {

constexpr int kWarps = 8;  // tiles per block

// K per block (a split of K): x's chunk in shared memory is 8 NB rows of
// Chunk(NB) bf16, 33 KB, so three blocks fit on an SM.
__host__ __device__ constexpr int Chunk(int nb) { return nb == 1 ? 2048 : 1024; }
// bf16 per row of x in shared memory (no bank conflicts).
__host__ __device__ constexpr int XStride(int nb) { return Chunk(nb) + 8; }

__host__ __device__ inline float E2m1(int code)
{
  const int m = code & 7;
  const float v = m < 2 ? 0.5f * m : (m < 4 ? 1.f + 0.5f * (m - 2) : (m < 6 ? 2.f + (m - 4) : 4.f + 2.f * (m - 6)));
  return (code & 8) ? -v : v;
}

// The bf16 bits of an e2m1 code (exact).
__device__ inline uint32_t E2m1Bf16(int code)
{
  const float f = E2m1(code);
  return __float_as_uint(f) >> 16;
}

__device__ __forceinline__ uint32_t Scaled(const uint32_t* pairs, uint32_t codeByte, __nv_bfloat162 scale)
{
  uint32_t p = pairs[codeByte];
  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&p);
  v = __hmul2(v, scale);
  return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ __nv_bfloat162 Scale2(uint32_t bytes, int s)
{
  __nv_fp8_e4m3 e;
  *reinterpret_cast<uint8_t*>(&e) = static_cast<uint8_t>(bytes >> (8 * s));
  const __nv_bfloat16 b = __float2bfloat16(static_cast<float>(e));  // exact
  return __nv_bfloat162(b, b);
}

__device__ __forceinline__ void Mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1)
{
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Grid (ceil(tiles / kWarps), ceil(K / Chunk(NB))); kWarps * 32 threads. Each warp
// takes one tile of 16 output rows over one chunk of K, for NB (1 or 2) blocks
// of 8 rows of x, and writes its f32 partial sums to part [splits, M, N].
template <int NB>
__global__ void __launch_bounds__(kWarps * 32) Fp4GemmKernel(
    const float* __restrict__ x, const uint4* __restrict__ codes, const uint32_t* __restrict__ scales,
    float* __restrict__ part, int M, int N, int K)
{
  constexpr int kChunk = Chunk(NB), kXStride = XStride(NB);
  extern __shared__ __align__(16) __nv_bfloat16 xs[];  // [8 NB][kXStride]
  // The two bf16 values of each code byte (low nibble: the even k), unscaled.
  // In shared memory: lanes read different entries, which __constant__ memory
  // would serialize.
  __shared__ uint32_t pairs[256];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int i = threadIdx.x; i < 256; i += blockDim.x) pairs[i] = E2m1Bf16(i & 15) | (E2m1Bf16(i >> 4) << 16);
  const int k0 = blockIdx.y * kChunk, kn = min(kChunk, K - k0);
  // x's chunk, rounded to bf16; rows past M are zero.
  for (int e = threadIdx.x; e < 8 * NB * (kn / 2); e += blockDim.x) {
    const int n = e / (kn / 2), k = 2 * (e % (kn / 2));
    float2 v = make_float2(0.f, 0.f);
    if (n < M) v = *reinterpret_cast<const float2*>(x + static_cast<size_t>(n) * K + k0 + k);
    *reinterpret_cast<__nv_bfloat162*>(xs + n * kXStride + k) = __floats2bfloat162_rn(v.x, v.y);
  }
  __syncthreads();
  const int tile = blockIdx.x * kWarps + warp, tiles = (N + 15) / 16;
  if (tile >= tiles) return;
  const int G = K / 64, g0 = k0 / 64, gn = kn / 64;
  const int r = lane >> 2, c = 2 * (lane & 3);
  const uint4* cp = codes + (static_cast<size_t>(tile) * G + g0) * 32 + lane;
  const uint32_t* sp = scales + (static_cast<size_t>(tile) * G + g0) * 16;

  float acc[NB][4] = {};
  uint4 w = cp[0];
  uint32_t sa = sp[r], sb = sp[r + 8];
  for (int g = 0; g < gn; ++g) {
    uint4 wn = w;
    uint32_t san = sa, sbn = sb;
    if (g + 1 < gn) {  // the next group in flight
      wn = cp[(g + 1) * 32];
      san = sp[(g + 1) * 16 + r];
      sbn = sp[(g + 1) * 16 + r + 8];
    }
    const uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const __nv_bfloat162 lo = Scale2(sa, s), hi = Scale2(sb, s);
      const uint32_t q = words[s];
      uint32_t a[4];
      a[0] = Scaled(pairs, q & 0xff, lo);
      a[1] = Scaled(pairs, (q >> 8) & 0xff, hi);
      a[2] = Scaled(pairs, (q >> 16) & 0xff, lo);
      a[3] = Scaled(pairs, q >> 24, hi);
      const int k = g * 64 + s * 16 + c;
#pragma unroll
      for (int nb = 0; nb < NB; ++nb) {
        const __nv_bfloat16* xr = xs + (nb * 8 + r) * kXStride + k;
        Mma(acc[nb], a, *reinterpret_cast<const uint32_t*>(xr), *reinterpret_cast<const uint32_t*>(xr + 8));
      }
    }
    w = wn;
    sa = san;
    sb = sbn;
  }
  // acc[nb]: {row r, x rows c, c+1}, {row r+8, x rows c, c+1} of block nb.
  float* out = part + static_cast<size_t>(blockIdx.y) * M * N;
  const int f0 = tile * 16 + r, f1 = f0 + 8;
#pragma unroll
  for (int nb = 0; nb < NB; ++nb) {
    const int n0 = nb * 8 + c, n1 = n0 + 1;
    if (n0 < M && f0 < N) out[static_cast<size_t>(n0) * N + f0] = acc[nb][0];
    if (n1 < M && f0 < N) out[static_cast<size_t>(n1) * N + f0] = acc[nb][1];
    if (n0 < M && f1 < N) out[static_cast<size_t>(n0) * N + f1] = acc[nb][2];
    if (n1 < M && f1 < N) out[static_cast<size_t>(n1) * N + f1] = acc[nb][3];
  }
}

// y [M, N] = scale2[n] * (sum of the splits' partials), in split order.
__global__ void ReduceKernel(const float* __restrict__ part, float* __restrict__ y, int splits, int N, int MN, const float* __restrict__ scale2)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= MN) return;
  float s = 0.f;
  for (int p = 0; p < splits; ++p) s += part[static_cast<size_t>(p) * MN + i];
  y[i] = s * scale2[i % N];
}

// The splits of K for M rows of x: the partial sums are [Splits(K, M), M, N].
// Every kernel Launch may use (XLA's INITIALIZE stage).
inline cudaError_t Prepare()
{
  cudaError_t e = tlaloc_kernels::AllowMaxSharedMemory(Fp4GemmKernel<1>);
  if (e == cudaSuccess) e = tlaloc_kernels::AllowMaxSharedMemory(Fp4GemmKernel<2>);
  return e;
}

inline int Splits(int K, int M) { const int c = Chunk(M <= 8 ? 1 : 2); return (K + c - 1) / c; }

// x [M, K] f32 (M <= 16, K % 64 == 0), packed codes and scales, the tensor scale of
// each output row [N] (on the device), part [Splits(K, M), M, N].
inline cudaError_t Launch(
    const float* x, const void* codes, const void* scales, const float* scale2, float* y, float* part, int M, int N, int K,
    cudaStream_t stream)
{
  const int tiles = (N + 15) / 16;
  const dim3 grid((tiles + kWarps - 1) / kWarps, Splits(K, M));
  auto run = [&](auto kernel, int nb) {
    const int bytes = 8 * nb * XStride(nb) * static_cast<int>(sizeof(__nv_bfloat16));
    tlaloc_kernels::AllowMaxSharedMemory(kernel);
    kernel<<<grid, kWarps * 32, bytes, stream>>>(
        x, static_cast<const uint4*>(codes), static_cast<const uint32_t*>(scales), part, M, N, K);
  };
  if (M <= 8) {
    run(Fp4GemmKernel<1>, 1);
  } else {
    run(Fp4GemmKernel<2>, 2);
  }
  const int MN = M * N;
  ReduceKernel<<<(MN + 255) / 256, 256, 0, stream>>>(part, y, Splits(K, M), N, MN, scale2);
  return cudaGetLastError();
}

// Repacks checkpoint codes [N, K / 2] and group scales [N, K / 16] (e4m3 bytes)
// into the layout above: codes 16 * 32 * (K / 64) bytes per tile, scales 64 * (K / 64).
inline void PackFp4(const uint8_t* w, const uint8_t* sc, int N, int K, uint8_t* codes, uint8_t* scales)
{
  const int G = K / 64, tiles = (N + 15) / 16;
  auto code = [&](int row, int k) -> uint8_t { return row < N ? w[static_cast<size_t>(row) * (K / 2) + k / 2] : 0; };
  for (int t = 0; t < tiles; ++t) {
    for (int g = 0; g < G; ++g) {
      for (int l = 0; l < 32; ++l) {
        const int r = l / 4, c = 2 * (l % 4);
        for (int s = 0; s < 4; ++s) {
          const int kb = 64 * g + 16 * s;
          uint8_t* o = codes + ((((static_cast<size_t>(t) * G + g) * 32 + l) * 4 + s) * 4);
          o[0] = code(16 * t + r, kb + c);
          o[1] = code(16 * t + r + 8, kb + c);
          o[2] = code(16 * t + r, kb + c + 8);
          o[3] = code(16 * t + r + 8, kb + c + 8);
        }
      }
      for (int r = 0; r < 16; ++r) {
        for (int s = 0; s < 4; ++s) {
          const int row = 16 * t + r;
          scales[((static_cast<size_t>(t) * G + g) * 16 + r) * 4 + s] =
              row < N ? sc[static_cast<size_t>(row) * (K / 16) + (64 * g + 16 * s) / 16] : 0;
        }
      }
    }
  }
}

}  // namespace fp4
