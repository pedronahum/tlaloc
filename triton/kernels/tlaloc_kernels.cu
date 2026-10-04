// libtlaloc_kernels.so: the CUDA kernels Tlaloc's graphs call as XLA typed-FFI
// custom calls, and TlalocRegisterKernels, which registers them with a PJRT GPU
// plugin:
//   tlaloc_paged_attention  paged_attention.cu
//   tlaloc_fp4_gemm         fp4_gemm.cu
//   tlaloc_moe_fp4          moe_fp4.cu
#include "paged_attention.cu"
#include "moe_fp4.cu"

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
      md->traits = 0;
      return nullptr;
    }
  }
  const XLA_FFI_Api* api = frame->api;
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
//   results   y f32 [R, H], scratch u8 [moe::ScratchBytes]
XLA_FFI_Error* MoeFp4Handler(XLA_FFI_CallFrame* frame)
{
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
  auto invalid = [&](const std::string& m) { return Fail(api, XLA_FFI_Error_Code_INVALID_ARGUMENT, "tlaloc_moe_fp4: " + m); };
  if (frame->args.size != 9 || frame->rets.size != 2) return invalid("takes 9 operands and returns 2 results");
  auto arg = [&](int i) { return static_cast<XLA_FFI_Buffer*>(frame->args.args[i]); };
  auto ret = [&](int i) { return static_cast<XLA_FFI_Buffer*>(frame->rets.rets[i]); };
  const XLA_FFI_Buffer *x = arg(0), *top = arg(1), *w = arg(2), *guC = arg(3), *guS2 = arg(5), *dnC = arg(6), *y = ret(0),
                       *scratch = ret(1);
  if (x->rank != 2 || top->rank != 2 || guC->rank != 4 || dnC->rank != 4 || guS2->rank != 2 ||
      x->dtype != XLA_FFI_DataType_F32 || top->dtype != XLA_FFI_DataType_S32 || w->dtype != XLA_FFI_DataType_F32) {
    return invalid("unexpected operand ranks or dtypes");
  }
  moe::Shape sh;
  sh.R = static_cast<int>(x->dims[0]);
  sh.H = static_cast<int>(x->dims[1]);
  sh.K = static_cast<int>(top->dims[1]);
  sh.E = static_cast<int>(guC->dims[0]);
  sh.I = static_cast<int>(guS2->dims[1]) / 2;
  if (sh.H % 64 != 0 || sh.I % 64 != 0 || top->dims[0] != sh.R || guC->dims[1] != (2 * sh.I + 15) / 16 ||
      guC->dims[2] != sh.H / 64 || dnC->dims[1] != (sh.H + 15) / 16 || dnC->dims[2] != sh.I / 64 ||
      scratch->dims[0] < static_cast<int64_t>(moe::ScratchBytes(sh))) {
    return invalid("unsupported shape");
  }
  XLA_FFI_Stream_Get_Args sa;
  std::memset(&sa, 0, sizeof(sa));
  sa.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
  sa.ctx = frame->ctx;
  if (XLA_FFI_Error* err = api->XLA_FFI_Stream_Get(&sa)) return err;
  const cudaError_t e = moe::Launch(
      static_cast<const float*>(x->data), static_cast<const int*>(top->data), static_cast<const float*>(w->data),
      guC->data, arg(4)->data, static_cast<const float*>(guS2->data), dnC->data, arg(7)->data,
      static_cast<const float*>(arg(8)->data), static_cast<float*>(y->data), scratch->data, sh,
      static_cast<cudaStream_t>(sa.stream));
  if (e != cudaSuccess) return Fail(api, XLA_FFI_Error_Code_INTERNAL, std::string("tlaloc_moe_fp4: ") + cudaGetErrorString(e));
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
                 {"tlaloc_moe_fp4", &MoeFp4Handler}};
  for (const auto& k : kernels) {
    PJRT_Gpu_Register_Custom_Call_Args a;
    std::memset(&a, 0, sizeof(a));
    a.struct_size = PJRT_Gpu_Register_Custom_Call_Args_STRUCT_SIZE;
    a.function_name = k.name;
    a.function_name_size = std::strlen(k.name);
    a.api_version = 1;  // typed FFI
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
