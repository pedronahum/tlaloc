// libtlaloc_kernels.so: the CUDA kernels Tlaloc's graphs call as XLA typed-FFI
// custom calls, and TlalocRegisterKernels, which registers them with a PJRT GPU
// plugin:
//   tlaloc_paged_attention  paged_attention.cu
//   tlaloc_fp4_gemm         fp4_gemm.cu
//   tlaloc_moe_fp4          moe_fp4.cu
//   tlaloc_gated_delta      gated_delta.cu
//   tlaloc_moe_route        moe_route.cu
#include "paged_attention.cu"
#include "moe_fp4.cu"
#include "gated_delta.cu"
#include "moe_route.cu"

namespace {

// tlaloc_fp4_gemm
//   operands  x f32 [M, K] (M <= 16), codes u8 [ceil(N / 16), K / 64, 512],
//             scales u8 [ceil(N / 16), K / 64, 64] (fp4::PackFp4's layout), scale2 f32 [N]
//   results   y f32 [M, N], partial sums f32 [fp4::Splits(K, M), M, N]: K in chunks of 2048
//             for up to 8 rows, 1024 for more
XLA_FFI_Error* Fp4GemmHandler(XLA_FFI_CallFrame* frame)
{
  for (XLA_FFI_Extension_Base* ext = frame->extension_start; ext != nullptr; ext = ext->next) {
    if (ext->type == XLA_FFI_Extension_Metadata) {
      XLA_FFI_Metadata* md = reinterpret_cast<XLA_FFI_Metadata_Extension*>(ext)->metadata;
      md->api_version.major_version = XLA_FFI_API_MAJOR;
      md->api_version.minor_version = XLA_FFI_API_MINOR;
      md->traits = XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE;
      return nullptr;
    }
  }
  const XLA_FFI_Api* api = frame->api;
  if (frame->stage == XLA_FFI_ExecutionStage_INITIALIZE) {
    const cudaError_t e = fp4::Prepare();
    return e == cudaSuccess ? nullptr : Fail(api, XLA_FFI_Error_Code_INTERNAL, std::string("tlaloc_fp4_gemm: ") + cudaGetErrorString(e));
  }
  if (frame->stage != XLA_FFI_ExecutionStage_EXECUTE) return nullptr;
  auto invalid = [&](const std::string& m) {
    return Fail(api, XLA_FFI_Error_Code_INVALID_ARGUMENT, "tlaloc_fp4_gemm: " + m);
  };
  if (frame->args.size != 4 || frame->rets.size != 2) return invalid("takes 4 operands and returns 2 results");
  auto arg = [&](int i) { return static_cast<XLA_FFI_Buffer*>(frame->args.args[i]); };
  auto ret = [&](int i) { return static_cast<XLA_FFI_Buffer*>(frame->rets.rets[i]); };
  const XLA_FFI_Buffer *x = arg(0), *codes = arg(1), *scales = arg(2), *s2 = arg(3), *y = ret(0), *part = ret(1);
  if (x->rank != 2 || x->dtype != XLA_FFI_DataType_F32 || codes->rank != 3 || codes->dtype != XLA_FFI_DataType_U8 ||
      scales->rank != 3 || scales->dtype != XLA_FFI_DataType_U8 || s2->dtype != XLA_FFI_DataType_F32 || y->rank != 2 ||
      part->rank != 3) {
    return invalid("unexpected operand ranks or dtypes");
  }
  const int M = static_cast<int>(x->dims[0]), K = static_cast<int>(x->dims[1]), N = static_cast<int>(y->dims[1]);
  if (M < 1 || M > 16 || K % 64 != 0 || codes->dims[0] != (N + 15) / 16 || codes->dims[1] != K / 64 ||
      codes->dims[2] != 512 || scales->dims[0] != codes->dims[0] || scales->dims[1] != K / 64 || scales->dims[2] != 64 ||
      s2->rank != 1 || s2->dims[0] != N || y->dims[0] != M || part->dims[0] != fp4::Splits(K, M) || part->dims[1] != M || part->dims[2] != N) {
    return invalid("unsupported shape");
  }
  XLA_FFI_Stream_Get_Args sa;
  std::memset(&sa, 0, sizeof(sa));
  sa.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
  sa.ctx = frame->ctx;
  if (XLA_FFI_Error* err = api->XLA_FFI_Stream_Get(&sa)) return err;
  cudaStream_t stream = static_cast<cudaStream_t>(sa.stream);
  const cudaError_t e = fp4::Launch(static_cast<const float*>(x->data), codes->data, scales->data,
                                    static_cast<const float*>(s2->data), static_cast<float*>(y->data),
                                    static_cast<float*>(part->data), M, N, K, stream);
  if (e != cudaSuccess) return Fail(api, XLA_FFI_Error_Code_INTERNAL, std::string("tlaloc_fp4_gemm: ") + cudaGetErrorString(e));
  return nullptr;
}

// tlaloc_moe_fp4
//   operands  x f32 [R, H], top i32 [R, K] (expert ids), weights f32 [R, K],
//             gate/up codes u8 [E, 2I / 16, H / 64, 512], scales u8 [E, 2I / 16, H / 64, 64], row scales f32 [E, 2I],
//             down codes u8 [E, H / 16, I / 64, 512], scales u8 [E, H / 16, I / 64, 64], row scales f32 [E, H]
//             then, with a shared expert (16 operands): its gate/up codes u8 [2I / 16, H / 64, 512],
//             scales u8 [2I / 16, H / 64, 64], row scales f32 [2I], its down codes, scales and row
//             scales [H], and each row's weight for it f32 [R] (y = routed + weight * shared)
//   results   y f32 [R, H], scratch u8 [moe::ScratchBytes]
//   attrs     act_fp4 s32 (optional): 1 quantizes the activations to NVFP4 (W4A4, moe_fp4.cu)
XLA_FFI_Error* MoeFp4Handler(XLA_FFI_CallFrame* frame)
{
  for (XLA_FFI_Extension_Base* ext = frame->extension_start; ext != nullptr; ext = ext->next) {
    if (ext->type == XLA_FFI_Extension_Metadata) {
      XLA_FFI_Metadata* md = reinterpret_cast<XLA_FFI_Metadata_Extension*>(ext)->metadata;
      md->api_version.major_version = XLA_FFI_API_MAJOR;
      md->api_version.minor_version = XLA_FFI_API_MINOR;
      md->traits = XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE;
      return nullptr;
    }
  }
  const XLA_FFI_Api* api = frame->api;
  if (frame->stage == XLA_FFI_ExecutionStage_INITIALIZE) {
    const cudaError_t e = moe::Prepare();
    return e == cudaSuccess ? nullptr : Fail(api, XLA_FFI_Error_Code_INTERNAL, std::string("tlaloc_moe_fp4: ") + cudaGetErrorString(e));
  }
  if (frame->stage != XLA_FFI_ExecutionStage_EXECUTE) return nullptr;
  auto invalid = [&](const std::string& m) { return Fail(api, XLA_FFI_Error_Code_INVALID_ARGUMENT, "tlaloc_moe_fp4: " + m); };
  if ((frame->args.size != 9 && frame->args.size != 16) || frame->rets.size != 2) {
    return invalid("takes 9 operands (16 with a shared expert) and returns 2 results");
  }
  const bool withShared = frame->args.size == 16;
  auto arg = [&](int i) { return static_cast<XLA_FFI_Buffer*>(frame->args.args[i]); };
  auto ret = [&](int i) { return static_cast<XLA_FFI_Buffer*>(frame->rets.rets[i]); };
  const XLA_FFI_Buffer *x = arg(0), *top = arg(1), *w = arg(2), *guC = arg(3), *guS2 = arg(5), *dnC = arg(6), *y = ret(0),
                       *scratch = ret(1);
  if (x->rank != 2 || top->rank != 2 || guC->rank != 4 || dnC->rank != 4 || guS2->rank != 2 ||
      x->dtype != XLA_FFI_DataType_F32 || top->dtype != XLA_FFI_DataType_S32 || w->dtype != XLA_FFI_DataType_F32) {
    return invalid("unexpected operand ranks or dtypes");
  }
  moe::Shape sh;
  for (int64_t i = 0; i < frame->attrs.size; ++i) {
    const XLA_FFI_ByteSpan* name = frame->attrs.names[i];
    if (frame->attrs.types[i] != XLA_FFI_AttrType_SCALAR) continue;
    const XLA_FFI_Scalar* sc = static_cast<XLA_FFI_Scalar*>(frame->attrs.attrs[i]);
    if (name->len == 7 && std::memcmp(name->ptr, "act_fp4", 7) == 0 && sc->dtype == XLA_FFI_DataType_S32) {
      sh.actFp4 = *static_cast<const int32_t*>(sc->value) != 0;
    }
  }
  sh.R = static_cast<int>(x->dims[0]);
  sh.H = static_cast<int>(x->dims[1]);
  sh.K = static_cast<int>(top->dims[1]);
  sh.E = static_cast<int>(guC->dims[0]);
  sh.I = static_cast<int>(guS2->dims[1]) / 2;
  if (sh.H % 64 != 0 || sh.I % 64 != 0 || top->dims[0] != sh.R || guC->dims[1] != (2 * sh.I + 15) / 16 ||
      guC->dims[2] != sh.H / 64 || dnC->dims[1] != (sh.H + 15) / 16 || dnC->dims[2] != sh.I / 64 ||
      scratch->dims[0] < static_cast<int64_t>(moe::ScratchBytes(sh, withShared))) {
    return invalid("unsupported shape");
  }
  moe::SharedExpert sGu, sDn;
  const float* sharedW = nullptr;
  if (withShared) {
    const XLA_FFI_Buffer *c1 = arg(9), *s1 = arg(10), *r1 = arg(11), *c2 = arg(12), *s2b = arg(13), *r2 = arg(14), *wv = arg(15);
    if (c1->rank != 3 || c1->dims[0] != (2 * sh.I + 15) / 16 || c1->dims[1] != sh.H / 64 || s1->rank != 3 ||
        r1->rank != 1 || r1->dims[0] != 2 * sh.I || c2->rank != 3 || c2->dims[0] != (sh.H + 15) / 16 ||
        c2->dims[1] != sh.I / 64 || s2b->rank != 3 || r2->rank != 1 || r2->dims[0] != sh.H || wv->rank != 1 ||
        wv->dims[0] != sh.R || wv->dtype != XLA_FFI_DataType_F32) {
      return invalid("the shared expert's operands do not match the experts' shapes");
    }
    sGu.codes = c1->data; sGu.scales = s1->data; sGu.s2 = static_cast<const float*>(r1->data);
    sDn.codes = c2->data; sDn.scales = s2b->data; sDn.s2 = static_cast<const float*>(r2->data);
    sharedW = static_cast<const float*>(wv->data);
  }
  if (sh.actFp4 && !moe::Fp4Shape(sh)) return invalid("act_fp4 needs top-k >= 2 and H and I in multiples of 256 (and at most 4096)");
  XLA_FFI_Stream_Get_Args sa;
  std::memset(&sa, 0, sizeof(sa));
  sa.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
  sa.ctx = frame->ctx;
  if (XLA_FFI_Error* err = api->XLA_FFI_Stream_Get(&sa)) return err;
  const cudaError_t e = moe::Launch(
      static_cast<const float*>(x->data), static_cast<const int*>(top->data), static_cast<const float*>(w->data),
      guC->data, arg(4)->data, static_cast<const float*>(guS2->data), dnC->data, arg(7)->data,
      static_cast<const float*>(arg(8)->data), static_cast<float*>(y->data), scratch->data, sh,
      static_cast<cudaStream_t>(sa.stream), sharedW, sGu, sDn);
  if (e != cudaSuccess) return Fail(api, XLA_FFI_Error_Code_INTERNAL, std::string("tlaloc_moe_fp4: ") + cudaGetErrorString(e));
  return nullptr;
}

// tlaloc_gated_delta (GATED_DELTA_RULE)
//   operands  q, k f32 [B, T, Hk, Dk], v f32 [B, T, Hv, Dv], g, beta f32 [B, T, Hv],
//             state pool f32 or bf16 [S, Hv, Dk, Dv], tokenSlots, positions i32 [B, T], writeSlots i32 [B, T] (optional)
//   results   out f32 [B, T, Hv, Dv], the pool (aliasing operand 5, updated in place)
XLA_FFI_Error* GatedDeltaHandler(XLA_FFI_CallFrame* frame)
{
  for (XLA_FFI_Extension_Base* ext = frame->extension_start; ext != nullptr; ext = ext->next) {
    if (ext->type == XLA_FFI_Extension_Metadata) {
      XLA_FFI_Metadata* md = reinterpret_cast<XLA_FFI_Metadata_Extension*>(ext)->metadata;
      md->api_version.major_version = XLA_FFI_API_MAJOR;
      md->api_version.minor_version = XLA_FFI_API_MINOR;
      md->traits = XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE;
      return nullptr;
    }
  }
  const XLA_FFI_Api* api = frame->api;
  if (frame->stage != XLA_FFI_ExecutionStage_EXECUTE) return nullptr;
  auto invalid = [&](const std::string& m) { return Fail(api, XLA_FFI_Error_Code_INVALID_ARGUMENT, "tlaloc_gated_delta: " + m); };
  if ((frame->args.size != 8 && frame->args.size != 9) || frame->rets.size != 2) return invalid("takes 8 or 9 operands and returns 2 results");
  auto arg = [&](int i) { return static_cast<XLA_FFI_Buffer*>(frame->args.args[i]); };
  auto ret = [&](int i) { return static_cast<XLA_FFI_Buffer*>(frame->rets.rets[i]); };
  const XLA_FFI_Buffer *q = arg(0), *v = arg(2), *pool = arg(5), *slots = arg(6), *outPool = ret(1);
  const bool bf16 = pool->dtype == XLA_FFI_DataType_BF16;
  if (q->rank != 4 || v->rank != 4 || pool->rank != 4 || q->dtype != XLA_FFI_DataType_F32 ||
      (pool->dtype != XLA_FFI_DataType_F32 && !bf16) || slots->dtype != XLA_FFI_DataType_S32) {
    return invalid("unexpected operand ranks or dtypes");
  }
  if (outPool->data != pool->data) return invalid("the pool result must alias operand 5");
  gdr::Shape sh;
  sh.B = static_cast<int>(q->dims[0]);
  sh.T = static_cast<int>(q->dims[1]);
  sh.Hk = static_cast<int>(q->dims[2]);
  sh.Dk = static_cast<int>(q->dims[3]);
  sh.Hv = static_cast<int>(v->dims[2]);
  sh.Dv = static_cast<int>(v->dims[3]);
  sh.S = static_cast<int>(pool->dims[0]);
  if (sh.Dk > 128 || sh.Dv % gdr::kCols != 0 || sh.Hv % sh.Hk != 0) return invalid("unsupported shape");
  XLA_FFI_Stream_Get_Args sa;
  std::memset(&sa, 0, sizeof(sa));
  sa.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
  sa.ctx = frame->ctx;
  if (XLA_FFI_Error* err = api->XLA_FFI_Stream_Get(&sa)) return err;
  auto launch = [&](auto* statePool) {
    return gdr::Launch(
        static_cast<const float*>(q->data), static_cast<const float*>(arg(1)->data), static_cast<const float*>(v->data),
        static_cast<const float*>(arg(3)->data), static_cast<const float*>(arg(4)->data), statePool,
        static_cast<const int*>(slots->data), static_cast<const int*>(arg(7)->data),
        frame->args.size == 9 ? static_cast<const int*>(arg(8)->data) : nullptr, static_cast<float*>(ret(0)->data), sh,
        static_cast<cudaStream_t>(sa.stream));
  };
  const cudaError_t e = bf16 ? launch(static_cast<__nv_bfloat16*>(pool->data)) : launch(static_cast<float*>(pool->data));
  if (e != cudaSuccess) return Fail(api, XLA_FFI_Error_Code_INTERNAL, std::string("tlaloc_gated_delta: ") + cudaGetErrorString(e));
  return nullptr;
}

// tlaloc_moe_route (a MOE_EXPERTS router's top-k)
//   operands  logits f32 [R, E], bias f32 [E] (sigmoid routing only)
//   results   top i32 [R, K], weights f32 [R, K]
//   attrs     sigmoid s32 (0: softmax), routed_scale f32 (sigmoid routing)
XLA_FFI_Error* MoeRouteHandler(XLA_FFI_CallFrame* frame)
{
  for (XLA_FFI_Extension_Base* ext = frame->extension_start; ext != nullptr; ext = ext->next) {
    if (ext->type == XLA_FFI_Extension_Metadata) {
      XLA_FFI_Metadata* md = reinterpret_cast<XLA_FFI_Metadata_Extension*>(ext)->metadata;
      md->api_version.major_version = XLA_FFI_API_MAJOR;
      md->api_version.minor_version = XLA_FFI_API_MINOR;
      md->traits = XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE;
      return nullptr;
    }
  }
  const XLA_FFI_Api* api = frame->api;
  if (frame->stage != XLA_FFI_ExecutionStage_EXECUTE) return nullptr;
  auto invalid = [&](const std::string& m) { return Fail(api, XLA_FFI_Error_Code_INVALID_ARGUMENT, "tlaloc_moe_route: " + m); };
  route::Shape sh;
  for (int64_t i = 0; i < frame->attrs.size; ++i) {
    const XLA_FFI_ByteSpan* name = frame->attrs.names[i];
    if (frame->attrs.types[i] != XLA_FFI_AttrType_SCALAR) continue;
    const XLA_FFI_Scalar* sc = static_cast<XLA_FFI_Scalar*>(frame->attrs.attrs[i]);
    if (name->len == 7 && std::memcmp(name->ptr, "sigmoid", 7) == 0 && sc->dtype == XLA_FFI_DataType_S32) {
      sh.sigmoid = *static_cast<const int32_t*>(sc->value) != 0;
    }
    if (name->len == 12 && std::memcmp(name->ptr, "routed_scale", 12) == 0 && sc->dtype == XLA_FFI_DataType_F32) {
      sh.scale = *static_cast<const float*>(sc->value);
    }
  }
  if (frame->args.size != (sh.sigmoid ? 2 : 1) || frame->rets.size != 2) {
    return invalid("takes the logits (and a bias for sigmoid routing) and returns 2 results");
  }
  auto arg = [&](int i) { return static_cast<XLA_FFI_Buffer*>(frame->args.args[i]); };
  auto ret = [&](int i) { return static_cast<XLA_FFI_Buffer*>(frame->rets.rets[i]); };
  const XLA_FFI_Buffer *x = arg(0), *top = ret(0), *w = ret(1);
  if (x->rank != 2 || x->dtype != XLA_FFI_DataType_F32 || top->rank != 2 || top->dtype != XLA_FFI_DataType_S32 ||
      w->rank != 2 || w->dtype != XLA_FFI_DataType_F32) {
    return invalid("unexpected operand ranks or dtypes");
  }
  sh.R = static_cast<int>(x->dims[0]);
  sh.E = static_cast<int>(x->dims[1]);
  sh.K = static_cast<int>(top->dims[1]);
  const XLA_FFI_Buffer* bias = sh.sigmoid ? arg(1) : nullptr;
  if (top->dims[0] != sh.R || w->dims[0] != sh.R || w->dims[1] != sh.K || sh.K < 1 || sh.K > route::kMaxTopK ||
      sh.K > sh.E || sh.E > 32 * route::kMaxPer ||
      (bias != nullptr && (bias->rank != 1 || bias->dims[0] != sh.E || bias->dtype != XLA_FFI_DataType_F32))) {
    return invalid("unsupported shape");
  }
  XLA_FFI_Stream_Get_Args sa;
  std::memset(&sa, 0, sizeof(sa));
  sa.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
  sa.ctx = frame->ctx;
  if (XLA_FFI_Error* err = api->XLA_FFI_Stream_Get(&sa)) return err;
  const cudaError_t e = route::Launch(
      static_cast<const float*>(x->data), bias != nullptr ? static_cast<const float*>(bias->data) : nullptr,
      static_cast<int*>(top->data), static_cast<float*>(w->data), sh, static_cast<cudaStream_t>(sa.stream));
  if (e != cudaSuccess) return Fail(api, XLA_FFI_Error_Code_INTERNAL, std::string("tlaloc_moe_route: ") + cudaGetErrorString(e));
  return nullptr;
}

}  // namespace

// Registers the kernels with the PJRT GPU plugin `api`. Returns null on success,
// else a message (static storage). Call before compiling a program that names them.
extern "C" __attribute__((visibility("default"))) const char* TlalocRegisterKernels(const PJRT_Api* api)
{
  static std::string error;
  PJRT_Gpu_Custom_Call* ext = nullptr;
  for (PJRT_Extension_Base* e = api->extension_start; e != nullptr; e = e->next) {
    if (e->type == PJRT_Extension_Type_Gpu_Custom_Call) ext = reinterpret_cast<PJRT_Gpu_Custom_Call*>(e);
  }
  if (ext == nullptr) {
    error = "the PJRT plugin has no GPU custom-call extension";
    return error.c_str();
  }
  const struct {
    const char* name;
    XLA_FFI_Error* (*handler)(XLA_FFI_CallFrame*);
  } kernels[] = {{"tlaloc_paged_attention", &PagedAttentionHandler}, {"tlaloc_fp4_gemm", &Fp4GemmHandler},
                 {"tlaloc_moe_fp4", &MoeFp4Handler},
                 {"tlaloc_gated_delta", &GatedDeltaHandler}, {"tlaloc_moe_route", &MoeRouteHandler}};
  for (const auto& k : kernels) {
    PJRT_Gpu_Register_Custom_Call_Args a;
    std::memset(&a, 0, sizeof(a));
    a.struct_size = PJRT_Gpu_Register_Custom_Call_Args_STRUCT_SIZE;
    a.function_name = k.name;
    a.function_name_size = std::strlen(k.name);
    a.api_version = 1;  // typed FFI
    // Also at INITIALIZE: the handlers set up their kernels there, outside any CUDA graph capture.
    a.handler_initialize = reinterpret_cast<void*>(k.handler);
    a.handler_execute = reinterpret_cast<void*>(k.handler);
    PJRT_Error* err = ext->custom_call(&a);
    if (err == nullptr) continue;
    PJRT_Error_Message_Args m;
    std::memset(&m, 0, sizeof(m));
    m.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    m.error = err;
    api->PJRT_Error_Message(&m);
    error = std::string("registering ") + k.name + ": " + std::string(m.message, m.message_size);
    PJRT_Error_Destroy_Args d;
    std::memset(&d, 0, sizeof(d));
    d.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    d.error = err;
    api->PJRT_Error_Destroy(&d);
    return error.c_str();
  }
  return nullptr;
}
