// tlaloc_moe_route: a mixture-of-experts router's top-k, one warp per row, in place of
// the stable descending sort over every expert that XLA runs as a segmented radix sort
// (45 ms of a 2,048-token Qwen3.6-35B-A3B chunk).
//
// Softmax routing: p = softmax(logits); the k largest p, ties to the lower expert (the
// stable sort's order); weights p / sum of the k.
// Sigmoid routing: s = sigmoid(logits); chosen by s + bias, ties to the lower expert;
// weights s / (sum of the k + 1e-20) * routed_scale.
#pragma once

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace route {

constexpr int kWarps = 4;     // rows per block
constexpr int kMaxPer = 32;   // experts per lane: up to 1,024 experts
constexpr int kMaxTopK = 32;

struct Shape {
  int R = 0, E = 0, K = 0;
  bool sigmoid = false;
  float scale = 1.f;
};

// Lane l holds experts l, l + 32, ... in v[j]; taken experts are set to -inf.
template <int PER>
__global__ void __launch_bounds__(kWarps * 32) RouteKernel(
    const float* __restrict__ logits, const float* __restrict__ bias, int* __restrict__ top, float* __restrict__ wts,
    Shape sh)
{
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);
  if (row >= sh.R) return;
  const float* x = logits + static_cast<size_t>(row) * sh.E;
  float sel[PER], val[PER];
  if (!sh.sigmoid) {
    float mx = -FLT_MAX;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int e = lane + 32 * j;
      val[j] = e < sh.E ? x[e] : -FLT_MAX;
      mx = fmaxf(mx, val[j]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int e = lane + 32 * j;
      val[j] = e < sh.E ? expf(val[j] - mx) : 0.f;
      sum += val[j];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int e = lane + 32 * j;
      val[j] = val[j] / sum;
      sel[j] = e < sh.E ? val[j] : -INFINITY;
    }
  } else {
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int e = lane + 32 * j;
      if (e < sh.E) {
        val[j] = 1.f / (1.f + expf(-x[e]));
        sel[j] = val[j] + bias[e];
      } else {
        val[j] = 0.f;
        sel[j] = -INFINITY;
      }
    }
  }
  float chosen[kMaxTopK];
  int ids[kMaxTopK];
  float total = 0.f;
  for (int i = 0; i < sh.K; ++i) {
    // This lane's best, then the warp's: the larger value, else the lower expert.
    float bv = -INFINITY;
    int be = 0x7fffffff;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      if (sel[j] > bv) {  // j ascending: equal values keep the lower expert
        bv = sel[j];
        be = lane + 32 * j;
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oe = __shfl_xor_sync(0xffffffffu, be, o);
      if (ov > bv || (ov == bv && oe < be)) {
        bv = ov;
        be = oe;
      }
    }
    float pv = 0.f;
    if ((be & 31) == lane) {
      const int j = be >> 5;
#pragma unroll
      for (int jj = 0; jj < PER; ++jj) {
        if (jj == j) {
          pv = val[jj];
          sel[jj] = -INFINITY;
        }
      }
    }
    pv = __shfl_sync(0xffffffffu, pv, be & 31);
    chosen[i] = pv;
    ids[i] = be;
    total += pv;
  }
  if (lane == 0) {
    const float den = sh.sigmoid ? total + 1e-20f : total;
    for (int i = 0; i < sh.K; ++i) {
      top[static_cast<size_t>(row) * sh.K + i] = ids[i];
      const float w = chosen[i] / den;
      wts[static_cast<size_t>(row) * sh.K + i] = sh.sigmoid ? w * sh.scale : w;
    }
  }
}

inline cudaError_t Launch(const float* logits, const float* bias, int* top, float* wts, const Shape& sh, cudaStream_t stream)
{
  if (sh.R == 0) return cudaSuccess;
  const dim3 grid((sh.R + kWarps - 1) / kWarps), block(kWarps * 32);
  const int per = (sh.E + 31) / 32;
  if (per <= 4) RouteKernel<4><<<grid, block, 0, stream>>>(logits, bias, top, wts, sh);
  else if (per <= 8) RouteKernel<8><<<grid, block, 0, stream>>>(logits, bias, top, wts, sh);
  else if (per <= 16) RouteKernel<16><<<grid, block, 0, stream>>>(logits, bias, top, wts, sh);
  else RouteKernel<kMaxPer><<<grid, block, 0, stream>>>(logits, bias, top, wts, sh);
  return cudaGetLastError();
}

}  // namespace route
