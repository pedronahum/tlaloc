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
//
// Committed (io.tlaloc.ir.GatedDeltaRuleAttrs.Commit, commit != null): a row's
// state is read from its base slot commit[row][0] instead, with the first
// commit[row][2] records of that slot applied (S = S * exp(g); S = S + k kv^T,
// the same operations as a token's, so the state is bit for bit the one a
// per-token write would have kept). kFinal then writes the state after the
// last token at the base; kRecords writes the starting state at the target
// commit[row][1] and, per token, its record (k, kv, g) at the target, and no
// state after any token. The records pool may hold more records per slot than
// the call's T (a wider verify step's): records T .. R-1 are left as they are.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace gdr {

struct Shape {
  int B, T, Hk, Hv, Dk, Dv, S;
};

enum Mode { kPlain = 0, kFinal = 1, kRecords = 2 };

// The committed form's operands: per row (base, target, count), and the records
// pool [S, R, Hv, Dk + Dv + 4] (k, kv, g, padding).
struct Commit {
  const int* rows = nullptr;
  float* records = nullptr;
  int R = 0;
  Mode mode = kPlain;
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
    const int* __restrict__ writes, float* __restrict__ out, Shape sh, Commit cm)
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
  const bool committed = cm.mode != kPlain;
  const int slot = committed ? cm.rows[row * 3] : slots[row * sh.T + first];
  const int target = committed ? cm.rows[row * 3 + 1] : -1;
  const bool reset = positions[row * sh.T + first] == 0;
  const size_t headState = static_cast<size_t>(sh.Dk) * sh.Dv;
  const int W = sh.Dk + sh.Dv + 4;
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
  if (committed && !reset) {
    // The base slot's pending records, in order.
    const int count = cm.rows[row * 3 + 2];
    for (int r = 0; r < count; ++r) {
      const float* rec = cm.records + ((static_cast<size_t>(slot) * cm.R + r) * sh.Hv + h) * W;
      const float decay = expf(rec[sh.Dk + sh.Dv]);
      const float ki = rec[i];
      const float* kv = rec + sh.Dk + c0;
#pragma unroll
      for (int j = 0; j < kCols; ++j) {
        s[j] *= decay;
        s[j] = fmaf(ki, kv[j], s[j]);
      }
    }
  }
  float* recOut = nullptr;
  if (cm.mode == kRecords) {
    State* dst = pool + (static_cast<size_t>(target) * sh.Hv + h) * headState + static_cast<size_t>(i) * sh.Dv + c0;
#pragma unroll
    for (int j = 0; j < kCols; j += 4) Store4(dst + j, make_float4(s[j], s[j + 1], s[j + 2], s[j + 3]));
    recOut = cm.records + (static_cast<size_t>(target) * cm.R * sh.Hv + h) * W;
    // A dead token's record is zero (the identity).
    for (int t = 0; t < first; ++t) {
      float* r = recOut + static_cast<size_t>(t) * sh.Hv * W;
      if (blockIdx.x == 0) {
        r[i] = 0.f;
        if (i < 4) r[sh.Dk + sh.Dv + i] = 0.f;
      }
      if (i < kCols) r[sh.Dk + c0 + i] = 0.f;
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
    if (recOut != nullptr) {
      float* r = recOut + static_cast<size_t>(t) * sh.Hv * W;
      if (blockIdx.x == 0) {
        r[i] = ki;
        if (i == 0) r[sh.Dk + sh.Dv] = g[bt * sh.Hv + h];
        if (i > 0 && i < 4) r[sh.Dk + sh.Dv + i] = 0.f;
      }
      if (i < kCols) r[sh.Dk + c0 + i] = kvs[i];
    }
    const int w = recOut != nullptr ? -1 : writes != nullptr ? writes[bt] : (t == sh.T - 1 ? slot : -1);
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
    const int* positions, const int* writes, float* out, const Shape& sh, cudaStream_t stream, Commit cm = {})
{
  if (sh.Dk > 128 || sh.Dv % kCols != 0) return cudaErrorInvalidValue;
  if (cm.mode == kRecords && (writes != nullptr || cm.R < sh.T)) return cudaErrorInvalidValue;
  GatedDeltaKernel<State><<<dim3(sh.Dv / kCols, sh.Hv, sh.B), sh.Dk, 0, stream>>>(q, k, v, g, beta, pool, slots, positions, writes, out, sh, cm);
  return cudaGetLastError();
}

}  // namespace gdr
