// Copyright 2026 Pedro N. Rodriguez
// SPDX-License-Identifier: Apache-2.0

#include "pjrt_runtime.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

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

// The memory under a view belongs to the caller; the plugin calls this when
// the view goes away. (The XLA GPU plugin calls it unconditionally, so it
// must not be null.)
void
ViewReleased(void* /*device_buffer_ptr*/, void* /*user_arg*/)
{
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

void
DropFileCache(const std::string& path)
{
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return;
  ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  ::close(fd);
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
    const PjrtPlugin* plugin, const ClientOptions& options, int device,
    std::unique_ptr<PjrtClient>* out)
{
  if (!(options.memory_fraction > 0.0f && options.memory_fraction <= 1.0f)) {
    return "memory fraction must be in (0, 1], got " + std::to_string(options.memory_fraction);
  }
  if (device < 0) return "GPU " + std::to_string(device) + " is not a device ordinal";
  const PJRT_Api* api = plugin->api;
  const char kFraction[] = "memory_fraction";
  const char kPreallocate[] = "preallocate";
  const char kVisible[] = "visible_devices";
  const int64_t visible[1] = {device};
  PJRT_NamedValue named[3]{};
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
  // One client per GPU: the plugin only sees `device`.
  named[2].struct_size = PJRT_NamedValue_STRUCT_SIZE;
  named[2].name = kVisible;
  named[2].name_size = sizeof(kVisible) - 1;
  named[2].type = PJRT_NamedValue_kInt64List;
  named[2].int64_array_value = visible;
  named[2].value_size = 1;

  const std::string gpu = "GPU " + std::to_string(device);
  PJRT_Client_Create_Args create{};
  create.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  create.create_options = named;
  create.num_options = 3;
  std::string err = TakeError(api, api->PJRT_Client_Create(&create));
  if (!err.empty()) return "PJRT_Client_Create for " + gpu + " failed: " + err;

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
  if (devices.num_addressable_devices != 1) {
    return "the PJRT client for " + gpu + " on platform " + client->platform_ + " has " +
           std::to_string(devices.num_addressable_devices) +
           " addressable devices; it was created to see exactly that one";
  }
  client->devices_.assign(
      devices.addressable_devices,
      devices.addressable_devices + devices.num_addressable_devices);
  PJRT_Device_LocalHardwareId_Args hw{};
  hw.struct_size = PJRT_Device_LocalHardwareId_Args_STRUCT_SIZE;
  hw.device = client->devices_[0];
  err = TakeError(api, api->PJRT_Device_LocalHardwareId(&hw));
  if (!err.empty()) return "PJRT_Device_LocalHardwareId failed: " + err;
  if (hw.local_hardware_id != device) {
    return "the PJRT client created for " + gpu + " runs on GPU " +
           std::to_string(hw.local_hardware_id) + " instead: the plugin has no " + gpu +
           " or does not honour visible_devices";
  }
  client->ordinal_ = device;
  *out = std::move(client);
  return "";
}

bool
PjrtClient::SupportsDeviceViews() const
{
  const PJRT_Api* api = plugin_->api;
  constexpr size_t kNeed =
      offsetof(PJRT_Api, PJRT_Client_CreateViewOfDeviceBuffer) + sizeof(void*);
  return api->struct_size >= kNeed && api->PJRT_Client_CreateViewOfDeviceBuffer != nullptr &&
         api->PJRT_Buffer_OpaqueDeviceMemoryDataPointer != nullptr &&
         api->PJRT_Buffer_OnDeviceSizeInBytes != nullptr &&
         api->PJRT_Buffer_IncreaseExternalReferenceCount != nullptr &&
         api->PJRT_Buffer_DecreaseExternalReferenceCount != nullptr;
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

namespace {

// Copies a host tensor to `device`; on success `*out` owns the new buffer.
std::string
UploadToDevice(
    const PJRT_Api* api, PJRT_Client* client, PJRT_Device* device, const HostInput& h,
    PJRT_Buffer** out)
{
  PJRT_Client_BufferFromHostBuffer_Args up{};
  up.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  up.client = client;
  up.data = h.data;
  up.type = ToPjrtType(h.dtype);
  up.dims = h.dims.empty() ? nullptr : h.dims.data();
  up.num_dims = h.dims.size();
  up.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
  up.device = device;
  std::string err = TakeError(api, api->PJRT_Client_BufferFromHostBuffer(&up));
  if (!err.empty()) return err;
  err = AwaitAndDestroy(api, up.done_with_host_buffer);
  if (!err.empty()) {
    DestroyBuffer(api, up.buffer);
    return err;
  }
  *out = up.buffer;
  return "";
}

}  // namespace

PjrtBuffer::~PjrtBuffer()
{
  DestroyBuffer(api_, buffer_);
}

std::string
PjrtBuffer::Upload(PjrtClient* client, const HostInput& host, std::unique_ptr<PjrtBuffer>* out)
{
  const PJRT_Api* api = client->plugin_->api;
  PJRT_Buffer* buffer = nullptr;
  std::string err = UploadToDevice(api, client->client_, client->devices_[0], host, &buffer);
  if (!err.empty()) return "copying to the device failed: " + err;
  std::unique_ptr<PjrtBuffer> b(new PjrtBuffer());
  b->api_ = api;
  b->buffer_ = buffer;
  *out = std::move(b);
  return "";
}

std::string
PjrtExecutable::Execute(
    const std::vector<ExecuteArg>& args, std::unique_ptr<PjrtResults>* out)
{
  const PJRT_Api* api = client_->plugin_->api;
  PJRT_Device* device = client_->devices_[0];

  // Buffers uploaded for this call are destroyed on every path out of this
  // function; device arguments are borrowed.
  struct Uploaded {
    const PJRT_Api* api;
    std::vector<PJRT_Buffer*> buffers;
    ~Uploaded()
    {
      for (PJRT_Buffer* b : buffers) DestroyBuffer(api, b);
    }
  } uploaded{api, {}};

  std::vector<PJRT_Buffer*> arguments;
  std::vector<int64_t> views;  // argument indices that alias caller memory
  arguments.reserve(args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i].device != nullptr) {
      arguments.push_back(args[i].device->buffer_);
      continue;
    }
    if (args[i].device_ptr != nullptr) {
      const HostInput& h = args[i].host;
      PJRT_Client_CreateViewOfDeviceBuffer_Args view{};
      view.struct_size = PJRT_Client_CreateViewOfDeviceBuffer_Args_STRUCT_SIZE;
      view.client = client_->client_;
      view.device_buffer_ptr = const_cast<void*>(args[i].device_ptr);
      view.dims = h.dims.empty() ? nullptr : h.dims.data();
      view.num_dims = h.dims.size();
      view.element_type = ToPjrtType(h.dtype);
      view.device = device;
      view.on_delete_callback = &ViewReleased;
      std::string err = TakeError(api, api->PJRT_Client_CreateViewOfDeviceBuffer(&view));
      if (!err.empty()) {
        return "wrapping input " + std::to_string(i) + " in GPU memory failed: " + err;
      }
      uploaded.buffers.push_back(view.buffer);  // destroying a view frees nothing
      arguments.push_back(view.buffer);
      views.push_back(static_cast<int64_t>(i));
      continue;
    }
    PJRT_Buffer* buffer = nullptr;
    std::string err = UploadToDevice(api, client_->client_, device, args[i].host, &buffer);
    if (!err.empty()) {
      return "copying input " + std::to_string(i) + " to the device failed: " + err;
    }
    uploaded.buffers.push_back(buffer);
    arguments.push_back(buffer);
  }

  std::unique_ptr<PjrtResults> results(new PjrtResults());
  results->api_ = api;
  results->buffers_.assign(num_outputs_, nullptr);

  PJRT_ExecuteOptions options{};
  options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
  // Memory the caller owns is read, never written.
  options.non_donatable_input_indices = views.empty() ? nullptr : views.data();
  options.num_non_donatable_input_indices = views.size();
  PJRT_Buffer* const* argument_list = arguments.data();
  PJRT_Buffer** output_list = results->buffers_.data();
  PJRT_Event* complete = nullptr;

  PJRT_LoadedExecutable_Execute_Args run{};
  run.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
  run.executable = exe_;
  run.options = &options;
  run.argument_lists = &argument_list;
  run.num_devices = 1;
  run.num_args = arguments.size();
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

std::unique_ptr<PjrtBuffer>
PjrtResults::Release(size_t i)
{
  std::unique_ptr<PjrtBuffer> b(new PjrtBuffer());
  b->api_ = api_;
  b->buffer_ = buffers_[i];
  buffers_[i] = nullptr;
  return b;
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

std::string
PjrtResults::WithDevicePointer(
    size_t i, size_t byte_size, bool* dense,
    const std::function<std::string(const void*)>& use) const
{
  *dense = false;
  PJRT_Buffer_OnDeviceSizeInBytes_Args size{};
  size.struct_size = PJRT_Buffer_OnDeviceSizeInBytes_Args_STRUCT_SIZE;
  size.buffer = buffers_[i];
  std::string err = TakeError(api_, api_->PJRT_Buffer_OnDeviceSizeInBytes(&size));
  if (!err.empty()) return "PJRT_Buffer_OnDeviceSizeInBytes failed: " + err;
  if (size.on_device_size_in_bytes != byte_size) return "";
  *dense = true;

  // The address is only guaranteed while an external reference is held.
  PJRT_Buffer_IncreaseExternalReferenceCount_Args inc{};
  inc.struct_size = PJRT_Buffer_IncreaseExternalReferenceCount_Args_STRUCT_SIZE;
  inc.buffer = buffers_[i];
  err = TakeError(api_, api_->PJRT_Buffer_IncreaseExternalReferenceCount(&inc));
  if (!err.empty()) return "PJRT_Buffer_IncreaseExternalReferenceCount failed: " + err;
  PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args ptr{};
  ptr.struct_size = PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args_STRUCT_SIZE;
  ptr.buffer = buffers_[i];
  err = TakeError(api_, api_->PJRT_Buffer_OpaqueDeviceMemoryDataPointer(&ptr));
  if (err.empty()) {
    err = use(ptr.device_memory_ptr);
  } else {
    err = "PJRT_Buffer_OpaqueDeviceMemoryDataPointer failed: " + err;
  }
  PJRT_Buffer_DecreaseExternalReferenceCount_Args dec{};
  dec.struct_size = PJRT_Buffer_DecreaseExternalReferenceCount_Args_STRUCT_SIZE;
  dec.buffer = buffers_[i];
  std::string derr = TakeError(api_, api_->PJRT_Buffer_DecreaseExternalReferenceCount(&dec));
  if (err.empty() && !derr.empty()) err = "PJRT_Buffer_DecreaseExternalReferenceCount failed: " + derr;
  return err;
}

}  // namespace tlaloc_triton
