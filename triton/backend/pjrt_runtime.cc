// Copyright 2026 Pedro N. Rodriguez
// SPDX-License-Identifier: Apache-2.0

#include "pjrt_runtime.h"

#include <dlfcn.h>

#include <cstddef>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>

namespace tlaloc_triton {

namespace {

// Minimal serialized xla.CompileOptionsProto: executable_build_options
// (field 3) holding num_replicas = 1 (field 4) and num_partitions = 1
// (field 5). XLA's PJRT compile path otherwise reads both as 0 and fails.
const char kCompileOptions[] = {0x1A, 0x04, 0x20, 0x01, 0x28, 0x01};

// The highest PJRT_Api entry this file calls. A plugin whose table ends
// before it cannot be used.
constexpr size_t kApiBytesNeeded =
    offsetof(PJRT_Api, PJRT_Buffer_ToHostBuffer) + sizeof(void*);

std::string
TakeError(const PJRT_Api* api, PJRT_Error* error)
{
  if (error == nullptr) return "";
  PJRT_Error_Message_Args msg{};
  msg.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  msg.error = error;
  api->PJRT_Error_Message(&msg);
  std::string text(msg.message, msg.message_size);
  PJRT_Error_Destroy_Args destroy{};
  destroy.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  destroy.error = error;
  api->PJRT_Error_Destroy(&destroy);
  return text.empty() ? "PJRT returned an error without a message" : text;
}

std::string
AwaitAndDestroy(const PJRT_Api* api, PJRT_Event* event)
{
  if (event == nullptr) return "";
  PJRT_Event_Await_Args await{};
  await.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  await.event = event;
  std::string err = TakeError(api, api->PJRT_Event_Await(&await));
  PJRT_Event_Destroy_Args destroy{};
  destroy.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
  destroy.event = event;
  std::string derr = TakeError(api, api->PJRT_Event_Destroy(&destroy));
  return err.empty() ? derr : err;
}

void
DestroyBuffer(const PJRT_Api* api, PJRT_Buffer* buffer)
{
  if (buffer == nullptr) return;
  PJRT_Buffer_Destroy_Args args{};
  args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
  args.buffer = buffer;
  TakeError(api, api->PJRT_Buffer_Destroy(&args));
}

std::mutex g_plugins_mu;
std::map<std::string, std::unique_ptr<PjrtPlugin>> g_plugins;

}  // namespace

PJRT_Buffer_Type
ToPjrtType(DType t)
{
  switch (t) {
    case DType::F32: return PJRT_Buffer_Type_F32;
    case DType::F64: return PJRT_Buffer_Type_F64;
    case DType::F16: return PJRT_Buffer_Type_F16;
    case DType::BF16: return PJRT_Buffer_Type_BF16;
    case DType::I8: return PJRT_Buffer_Type_S8;
    case DType::I32: return PJRT_Buffer_Type_S32;
    case DType::I64: return PJRT_Buffer_Type_S64;
    case DType::U8: return PJRT_Buffer_Type_U8;
    case DType::BOOL: return PJRT_Buffer_Type_PRED;
    default: return PJRT_Buffer_Type_INVALID;
  }
}

DType
FromPjrtType(PJRT_Buffer_Type t)
{
  switch (t) {
    case PJRT_Buffer_Type_F32: return DType::F32;
    case PJRT_Buffer_Type_F64: return DType::F64;
    case PJRT_Buffer_Type_F16: return DType::F16;
    case PJRT_Buffer_Type_BF16: return DType::BF16;
    case PJRT_Buffer_Type_S8: return DType::I8;
    case PJRT_Buffer_Type_S32: return DType::I32;
    case PJRT_Buffer_Type_S64: return DType::I64;
    case PJRT_Buffer_Type_U8: return DType::U8;
    case PJRT_Buffer_Type_PRED: return DType::BOOL;
    default: return DType::UNSUPPORTED;
  }
}

std::string
LoadPjrtPlugin(const std::string& path, const PjrtPlugin** out)
{
  std::lock_guard<std::mutex> lock(g_plugins_mu);
  auto it = g_plugins.find(path);
  if (it != g_plugins.end()) {
    *out = it->second.get();
    return "";
  }
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    const char* why = dlerror();
    return "cannot load the PJRT plugin " + path + ": " + (why ? why : "dlopen failed");
  }
  using GetApiFn = const PJRT_Api* (*)();
  auto get_api = reinterpret_cast<GetApiFn>(dlsym(handle, "GetPjrtApi"));
  if (get_api == nullptr) {
    return path + " does not export GetPjrtApi, so it is not a PJRT plugin";
  }
  const PJRT_Api* api = get_api();
  if (api == nullptr) {
    return "GetPjrtApi in " + path + " returned no API table";
  }
  const int major = api->pjrt_api_version.major_version;
  const int minor = api->pjrt_api_version.minor_version;
  if (major != PJRT_API_MAJOR) {
    std::ostringstream m;
    m << "the PJRT plugin " << path << " implements PJRT C API " << major << "."
      << minor << "; this backend is built against " << PJRT_API_MAJOR << "."
      << PJRT_API_MINOR << " and cannot call a plugin with a different major version";
    return m.str();
  }
  if (api->struct_size < kApiBytesNeeded) {
    std::ostringstream m;
    m << "the PJRT plugin " << path << " (C API " << major << "." << minor
      << ") has a " << api->struct_size << "-byte function table; this backend needs at least "
      << kApiBytesNeeded << " bytes, up to PJRT_Buffer_ToHostBuffer";
    return m.str();
  }
  if (api->PJRT_Plugin_Initialize != nullptr) {
    PJRT_Plugin_Initialize_Args init{};
    init.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
    std::string err = TakeError(api, api->PJRT_Plugin_Initialize(&init));
    if (!err.empty()) return "PJRT_Plugin_Initialize failed for " + path + ": " + err;
  }
  auto plugin = std::make_unique<PjrtPlugin>();
  plugin->path = path;
  plugin->api = api;
  plugin->major = major;
  plugin->minor = minor;
  *out = plugin.get();
  g_plugins.emplace(path, std::move(plugin));
  return "";
}

std::string
PjrtClient::Create(
    const PjrtPlugin* plugin, const ClientOptions& options,
    std::unique_ptr<PjrtClient>* out)
{
  if (!(options.memory_fraction > 0.0f && options.memory_fraction <= 1.0f)) {
    return "memory fraction must be in (0, 1], got " + std::to_string(options.memory_fraction);
  }
  const PJRT_Api* api = plugin->api;
  const char kFraction[] = "memory_fraction";
  const char kPreallocate[] = "preallocate";
  PJRT_NamedValue named[2]{};
  named[0].struct_size = PJRT_NamedValue_STRUCT_SIZE;
  named[0].name = kFraction;
  named[0].name_size = sizeof(kFraction) - 1;
  named[0].type = PJRT_NamedValue_kFloat;
  named[0].float_value = options.memory_fraction;
  named[0].value_size = 1;
  named[1].struct_size = PJRT_NamedValue_STRUCT_SIZE;
  named[1].name = kPreallocate;
  named[1].name_size = sizeof(kPreallocate) - 1;
  named[1].type = PJRT_NamedValue_kBool;
  named[1].bool_value = options.preallocate;
  named[1].value_size = 1;

  PJRT_Client_Create_Args create{};
  create.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  create.create_options = named;
  create.num_options = 2;
  std::string err = TakeError(api, api->PJRT_Client_Create(&create));
  if (!err.empty()) return "PJRT_Client_Create failed: " + err;

  std::unique_ptr<PjrtClient> client(new PjrtClient());
  client->plugin_ = plugin;
  client->client_ = create.client;
  client->options_ = options;

  PJRT_Client_PlatformName_Args name{};
  name.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
  name.client = create.client;
  err = TakeError(api, api->PJRT_Client_PlatformName(&name));
  if (!err.empty()) return "PJRT_Client_PlatformName failed: " + err;
  client->platform_.assign(name.platform_name, name.platform_name_size);

  PJRT_Client_AddressableDevices_Args devices{};
  devices.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  devices.client = create.client;
  err = TakeError(api, api->PJRT_Client_AddressableDevices(&devices));
  if (!err.empty()) return "PJRT_Client_AddressableDevices failed: " + err;
  if (devices.num_addressable_devices == 0) {
    return "the PJRT client on platform " + client->platform_ + " has no addressable devices";
  }
  client->devices_.assign(
      devices.addressable_devices,
      devices.addressable_devices + devices.num_addressable_devices);
  *out = std::move(client);
  return "";
}

PjrtClient::~PjrtClient()
{
  if (client_ == nullptr) return;
  PJRT_Client_Destroy_Args args{};
  args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  args.client = client_;
  TakeError(plugin_->api, plugin_->api->PJRT_Client_Destroy(&args));
}

std::string
PjrtClient::Compile(const std::string& mlir, std::unique_ptr<PjrtExecutable>* out)
{
  const PJRT_Api* api = plugin_->api;
  std::string code = mlir;
  static const char kFormat[] = "mlir";
  PJRT_Program program{};
  program.struct_size = PJRT_Program_STRUCT_SIZE;
  program.code = &code[0];
  program.code_size = code.size();
  program.format = kFormat;
  program.format_size = sizeof(kFormat) - 1;

  PJRT_Client_Compile_Args compile{};
  compile.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
  compile.client = client_;
  compile.program = &program;
  compile.compile_options = kCompileOptions;
  compile.compile_options_size = sizeof(kCompileOptions);
  std::string err = TakeError(api, api->PJRT_Client_Compile(&compile));
  if (!err.empty()) return "PJRT_Client_Compile failed: " + err;

  std::unique_ptr<PjrtExecutable> exe(new PjrtExecutable());
  exe->client_ = this;
  exe->exe_ = compile.executable;

  PJRT_LoadedExecutable_GetExecutable_Args get{};
  get.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
  get.loaded_executable = compile.executable;
  err = TakeError(api, api->PJRT_LoadedExecutable_GetExecutable(&get));
  if (!err.empty()) return "PJRT_LoadedExecutable_GetExecutable failed: " + err;
  PJRT_Executable_NumOutputs_Args num{};
  num.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
  num.executable = get.executable;
  err = TakeError(api, api->PJRT_Executable_NumOutputs(&num));
  PJRT_Executable_Destroy_Args destroy{};
  destroy.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
  destroy.executable = get.executable;
  TakeError(api, api->PJRT_Executable_Destroy(&destroy));
  if (!err.empty()) return "PJRT_Executable_NumOutputs failed: " + err;
  exe->num_outputs_ = num.num_outputs;
  *out = std::move(exe);
  return "";
}

PjrtExecutable::~PjrtExecutable()
{
  if (exe_ == nullptr) return;
  const PJRT_Api* api = client_->plugin_->api;
  PJRT_LoadedExecutable_Destroy_Args args{};
  args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
  args.executable = exe_;
  TakeError(api, api->PJRT_LoadedExecutable_Destroy(&args));
}

std::string
PjrtExecutable::Execute(
    const std::vector<HostInput>& inputs, std::unique_ptr<PjrtResults>* out)
{
  const PJRT_Api* api = client_->plugin_->api;
  PJRT_Device* device = client_->devices_[0];

  // Input buffers are destroyed on every path out of this function.
  struct Inputs {
    const PJRT_Api* api;
    std::vector<PJRT_Buffer*> buffers;
    ~Inputs()
    {
      for (PJRT_Buffer* b : buffers) DestroyBuffer(api, b);
    }
  } in{api, {}};

  for (size_t i = 0; i < inputs.size(); ++i) {
    const HostInput& h = inputs[i];
    PJRT_Client_BufferFromHostBuffer_Args up{};
    up.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    up.client = client_->client_;
    up.data = h.data;
    up.type = ToPjrtType(h.dtype);
    up.dims = h.dims.empty() ? nullptr : h.dims.data();
    up.num_dims = h.dims.size();
    up.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    up.device = device;
    std::string err = TakeError(api, api->PJRT_Client_BufferFromHostBuffer(&up));
    if (!err.empty()) {
      return "copying input " + std::to_string(i) + " to the device failed: " + err;
    }
    in.buffers.push_back(up.buffer);
    err = AwaitAndDestroy(api, up.done_with_host_buffer);
    if (!err.empty()) {
      return "copying input " + std::to_string(i) + " to the device failed: " + err;
    }
  }

  std::unique_ptr<PjrtResults> results(new PjrtResults());
  results->api_ = api;
  results->buffers_.assign(num_outputs_, nullptr);

  PJRT_ExecuteOptions options{};
  options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
  PJRT_Buffer* const* argument_list = in.buffers.data();
  PJRT_Buffer** output_list = results->buffers_.data();
  PJRT_Event* complete = nullptr;

  PJRT_LoadedExecutable_Execute_Args run{};
  run.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
  run.executable = exe_;
  run.options = &options;
  run.argument_lists = &argument_list;
  run.num_devices = 1;
  run.num_args = in.buffers.size();
  run.output_lists = &output_list;
  run.device_complete_events = &complete;
  std::string err = TakeError(api, api->PJRT_LoadedExecutable_Execute(&run));
  if (!err.empty()) return "PJRT_LoadedExecutable_Execute failed: " + err;
  err = AwaitAndDestroy(api, complete);
  if (!err.empty()) return "execution failed: " + err;
  *out = std::move(results);
  return "";
}

PjrtResults::~PjrtResults()
{
  for (PJRT_Buffer* b : buffers_) DestroyBuffer(api_, b);
}

std::string
PjrtResults::Describe(size_t i, DType* dtype, std::vector<int64_t>* dims) const
{
  PJRT_Buffer_ElementType_Args type{};
  type.struct_size = PJRT_Buffer_ElementType_Args_STRUCT_SIZE;
  type.buffer = buffers_[i];
  std::string err = TakeError(api_, api_->PJRT_Buffer_ElementType(&type));
  if (!err.empty()) return "PJRT_Buffer_ElementType failed: " + err;
  *dtype = FromPjrtType(type.type);

  PJRT_Buffer_Dimensions_Args shape{};
  shape.struct_size = PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
  shape.buffer = buffers_[i];
  err = TakeError(api_, api_->PJRT_Buffer_Dimensions(&shape));
  if (!err.empty()) return "PJRT_Buffer_Dimensions failed: " + err;
  dims->assign(shape.dims, shape.dims + shape.num_dims);
  return "";
}

std::string
PjrtResults::CopyToHost(size_t i, void* dst, size_t byte_size) const
{
  // Ask for the size first: PJRT fills dst_size when dst is null.
  PJRT_Buffer_ToHostBuffer_Args probe{};
  probe.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
  probe.src = buffers_[i];
  std::string err = TakeError(api_, api_->PJRT_Buffer_ToHostBuffer(&probe));
  if (!err.empty()) return "PJRT_Buffer_ToHostBuffer (size query) failed: " + err;
  if (probe.dst_size != byte_size) {
    return "output " + std::to_string(i) + " is " + std::to_string(probe.dst_size) +
           " bytes on the host, expected " + std::to_string(byte_size);
  }
  if (byte_size == 0) return "";
  PJRT_Buffer_ToHostBuffer_Args copy{};
  copy.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
  copy.src = buffers_[i];
  copy.dst = dst;
  copy.dst_size = byte_size;
  err = TakeError(api_, api_->PJRT_Buffer_ToHostBuffer(&copy));
  if (!err.empty()) return "PJRT_Buffer_ToHostBuffer failed: " + err;
  return AwaitAndDestroy(api_, copy.event);
}

}  // namespace tlaloc_triton
