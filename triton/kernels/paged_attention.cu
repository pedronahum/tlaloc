// Paged attention for decode and speculative verify steps, as an XLA typed-FFI
// custom call (`tlaloc_paged_attention`), registered with a PJRT GPU plugin by
// TlalocRegisterKernels.
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
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <cmath>
#include <cstring>
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

// Grid (S, Hkv, Tb), kThreads threads. Each block runs one slice of kThreads
// positions for the NQ (<= MAXQ) = rows-per-table x group queries of one table
// and KV head:
//   scores   thread j takes position j of the slice: it reads the key row in
//            16-value vectors and dots it with every query (broadcast from
//            shared memory);
//   softmax  warp w takes queries w, w + 8, ... over the slice;
//   values   thread d owns head dimension d of every query and streams the
//            slice's values, coalesced, into its NQ accumulators.
// Dynamic shared memory: q [NQ][D] (scaled), p [NQ][kThreads], m, l [NQ].
template <typename KV, int MAXQ>
__global__ void __launch_bounds__(kThreads) SliceKernel(
    const float* __restrict__ q, const KV* __restrict__ kc, const KV* __restrict__ vc,
    const int* __restrict__ tables, const int* __restrict__ lens, float* __restrict__ scratch, Shape sh)
{
  const int s = blockIdx.x, hk = blockIdx.y, t = blockIdx.z, tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5, warps = kThreads / 32;
  const int G = sh.H / sh.Hkv, Qr = sh.R / sh.Tb, NQ = Qr * G, D = sh.D;
  extern __shared__ float smem[];
  float* qs = smem;
  float* ps = qs + NQ * D;
  float* ms = ps + NQ * kThreads;
  float* ls = ms + NQ;

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
  auto at = [&](int pos) {
    return (static_cast<size_t>(tables[t * sh.M + pos / sh.bs]) * sh.bs + pos % sh.bs) * rowStride + hk * D;
  };
  if (n > 0) {
    // Scores, -inf past a row's length.
    float sc[MAXQ];
#pragma unroll
    for (int i = 0; i < MAXQ; ++i) sc[i] = 0.f;
    if (tid < n) {
      const KV* key = kc + at(start + tid);
      for (int d0 = 0; d0 < D; d0 += 16) {
        float k16[16];
        Load16(key + d0, k16);
#pragma unroll
        for (int i = 0; i < MAXQ; ++i) {
          if (i < NQ) {
            const float* qi = qs + i * D + d0;
#pragma unroll
            for (int c = 0; c < 16; ++c) sc[i] = fmaf(qi[c], k16[c], sc[i]);
          }
        }
      }
    }
#pragma unroll
    for (int i = 0; i < MAXQ; ++i) {
      if (i < NQ) ps[i * kThreads + tid] = tid < n && start + tid < lens[t * Qr + i / G] ? sc[i] : -INFINITY;
    }
    __syncthreads();
    // Softmax over the slice, per query.
    for (int i = warp; i < NQ; i += warps) {
      float* row = ps + i * kThreads;
      float mx = -INFINITY;
      for (int j = lane; j < kThreads; j += 32) mx = fmaxf(mx, row[j]);
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
      float sum = 0.f;
      for (int j = lane; j < kThreads; j += 32) {
        const float w = row[j] == -INFINITY ? 0.f : expf(row[j] - mx);
        row[j] = w;
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
    for (int j0 = 0; j0 < n; j0 += 8) {
      float v[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) v[u] = j0 + u < n ? Widen(vc[at(start + j0 + u) + tid]) : 0.f;
#pragma unroll
      for (int u = 0; u < 8; ++u) {
#pragma unroll
        for (int i = 0; i < MAXQ; ++i) {
          if (i < NQ) acc[i] = fmaf(ps[i * kThreads + j0 + u], v[u], acc[i]);
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
  return sizeof(float) * (static_cast<size_t>(NQ) * sh.D + static_cast<size_t>(NQ) * kThreads + 2 * NQ);
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

template <typename KV>
cudaError_t Launch(const void* q, const void* k, const void* v, const void* tables, const void* lens,
                   void* out, void* scratch, const Shape& sh, cudaStream_t stream)
{
  const int NQ = (sh.R / sh.Tb) * (sh.H / sh.Hkv);
  cudaError_t e;
  if (NQ <= 8) e = LaunchSlices<KV, 8>(q, k, v, tables, lens, scratch, sh, stream);
  else if (NQ <= 16) e = LaunchSlices<KV, 16>(q, k, v, tables, lens, scratch, sh, stream);
  else if (NQ <= 32) e = LaunchSlices<KV, 32>(q, k, v, tables, lens, scratch, sh, stream);
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

// Registers the kernels with the PJRT GPU plugin `api`. Returns null on success,
// else a message (static storage). Call before compiling a program that names them.
extern "C" __attribute__((visibility("default"))) const char* TlalocRegisterKernels(const PJRT_Api* api)
{
  static std::string error;
  for (PJRT_Extension_Base* ext = api->extension_start; ext != nullptr; ext = ext->next) {
    if (ext->type != PJRT_Extension_Type_Gpu_Custom_Call) continue;
    PJRT_Gpu_Register_Custom_Call_Args a;
    std::memset(&a, 0, sizeof(a));
    a.struct_size = PJRT_Gpu_Register_Custom_Call_Args_STRUCT_SIZE;
    a.function_name = "tlaloc_paged_attention";
    a.function_name_size = std::strlen(a.function_name);
    a.api_version = 1;  // typed FFI
    a.handler_execute = reinterpret_cast<void*>(&PagedAttentionHandler);
    PJRT_Error* err = reinterpret_cast<PJRT_Gpu_Custom_Call*>(ext)->custom_call(&a);
    if (err == nullptr) return nullptr;
    PJRT_Error_Message_Args m;
    std::memset(&m, 0, sizeof(m));
    m.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    m.error = err;
    api->PJRT_Error_Message(&m);
    error = std::string("registering tlaloc_paged_attention: ") + std::string(m.message, m.message_size);
    PJRT_Error_Destroy_Args d;
    std::memset(&d, 0, sizeof(d));
    d.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    d.error = err;
    api->PJRT_Error_Destroy(&d);
    return error.c_str();
  }
  error = "the PJRT plugin has no GPU custom-call extension";
  return error.c_str();
}
