// GATED_DELTA_RULE (io.tlaloc.ir.GatedDeltaRuleAttrs) for a few tokens per
// sequence, the state held in registers across a row's tokens and written to
// the pool in place: once at the end, or after every token at writeSlots (a
// speculative step's per-token states).
//
// Per live token and value head h (key head h / (Hv / Hk)), with S [Dk, Dv]:
//   S = S * exp(g);  kv = (v - k^T S) * beta;  S = S + k kv^T;  out = q^T S
// A row reads its slot's state, or zero when its first live token is at
// position 0; padding tokens (slot < 0) come first and output zero. The pool
// is f32 or bf16; a bf16 state is widened when read, carried in f32 and
// rounded to nearest even when written.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace gdr {

struct Shape {
  int B, T, Hk, Hv, Dk, Dv, S;
};

constexpr int kCols = 32;  // value columns per block

// Four state values from or to a pool row.
__device__ inline float4 Load4(const float* p) { return *reinterpret_cast<const float4*>(p); }
__device__ inline float4 Load4(const __nv_bfloat16* p)
{
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
  const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
  return make_float4(a.x, a.y, b.x, b.y);
}
__device__ inline void Store4(float* p, float4 x) { *reinterpret_cast<float4*>(p) = x; }
__device__ inline void Store4(__nv_bfloat16* p, float4 x)
{
  const __nv_bfloat162 a = __floats2bfloat162_rn(x.x, x.y), b = __floats2bfloat162_rn(x.z, x.w);
  uint2 u;
  u.x = *reinterpret_cast<const uint32_t*>(&a);
  u.y = *reinterpret_cast<const uint32_t*>(&b);
  *reinterpret_cast<uint2*>(p) = u;
}

// Grid (Dv / kCols, Hv, B), Dk threads (Dk <= 128): thread i holds row i of
// the head's state for the block's kCols columns.
template <typename State>
__global__ void __launch_bounds__(128) GatedDeltaKernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ g,
    const float* __restrict__ beta, State* pool, const int* __restrict__ slots, const int* __restrict__ positions,
    const int* __restrict__ writes, float* __restrict__ out, Shape sh)
{
  const int c0 = blockIdx.x * kCols, h = blockIdx.y, row = blockIdx.z, i = threadIdx.x;
  const int kh = h / (sh.Hv / sh.Hk);
  __shared__ float red[128][kCols + 1];
  __shared__ float kvs[kCols];
  // The row's first live token, its slot and whether it starts from zero.
  int first = sh.T;
  for (int t = 0; t < sh.T; ++t) {
    if (slots[row * sh.T + t] >= 0) {
      first = t;
      break;
    }
  }
  for (int t = 0; t < first; ++t) {
    if (i < kCols) out[((static_cast<size_t>(row) * sh.T + t) * sh.Hv + h) * sh.Dv + c0 + i] = 0.f;
  }
  if (first == sh.T) return;
  const int slot = slots[row * sh.T + first];
  const bool reset = positions[row * sh.T + first] == 0;
  const size_t headState = static_cast<size_t>(sh.Dk) * sh.Dv;
  float s[kCols];
  {
    const State* src = pool + (static_cast<size_t>(slot) * sh.Hv + h) * headState + static_cast<size_t>(i) * sh.Dv + c0;
#pragma unroll
    for (int j = 0; j < kCols; j += 4) {
      const float4 x = reset ? make_float4(0.f, 0.f, 0.f, 0.f) : Load4(src + j);
      s[j] = x.x;
      s[j + 1] = x.y;
      s[j + 2] = x.z;
      s[j + 3] = x.w;
    }
  }
  // A block reads and writes only its own head and columns of the row's slots,
  // and has read its part of the state before writing any.
  for (int t = first; t < sh.T; ++t) {
    const size_t bt = static_cast<size_t>(row) * sh.T + t;
    const float decay = expf(g[bt * sh.Hv + h]);
    const float ki = k[(bt * sh.Hk + kh) * sh.Dk + i];
    const float qi = q[(bt * sh.Hk + kh) * sh.Dk + i];
    // kv = (v - k^T S) * beta, S decayed first.
#pragma unroll
    for (int j = 0; j < kCols; ++j) {
      s[j] *= decay;
      red[i][j] = s[j] * ki;
    }
    __syncthreads();
    if (i < kCols) {
      float acc = 0.f;
      for (int r = 0; r < sh.Dk; ++r) acc += red[r][i];
      kvs[i] = (v[(bt * sh.Hv + h) * sh.Dv + c0 + i] - acc) * beta[bt * sh.Hv + h];
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < kCols; ++j) {
      s[j] = fmaf(ki, kvs[j], s[j]);
      red[i][j] = s[j] * qi;
    }
    __syncthreads();
    if (i < kCols) {
      float acc = 0.f;
      for (int r = 0; r < sh.Dk; ++r) acc += red[r][i];
      out[(bt * sh.Hv + h) * sh.Dv + c0 + i] = acc;
    }
    const int w = writes != nullptr ? writes[bt] : (t == sh.T - 1 ? slot : -1);
    if (w >= 0) {
      State* dst = pool + (static_cast<size_t>(w) * sh.Hv + h) * headState + static_cast<size_t>(i) * sh.Dv + c0;
#pragma unroll
      for (int j = 0; j < kCols; j += 4) Store4(dst + j, make_float4(s[j], s[j + 1], s[j + 2], s[j + 3]));
    }
    __syncthreads();  // red is reused
  }
}

// pool (float or __nv_bfloat16) is updated in place. writes may be null (the
// state goes back to the row's slot).
template <typename State>
inline cudaError_t Launch(
    const float* q, const float* k, const float* v, const float* g, const float* beta, State* pool, const int* slots,
    const int* positions, const int* writes, float* out, const Shape& sh, cudaStream_t stream)
{
  if (sh.Dk > 128 || sh.Dv % kCols != 0) return cudaErrorInvalidValue;
  GatedDeltaKernel<State><<<dim3(sh.Dv / kCols, sh.Hv, sh.B), sh.Dk, 0, stream>>>(q, k, v, g, beta, pool, slots, positions, writes, out, sh);
  return cudaGetLastError();
}

}  // namespace gdr
