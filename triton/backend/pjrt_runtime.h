// Copyright 2026 Pedro N. Rodriguez
// SPDX-License-Identifier: Apache-2.0
//
// A small C++ layer over the PJRT C API: load a plugin, create a client with
// allocator options, compile StableHLO text, move host bytes in and out, and
// execute. Every call returns an error string; an empty string is success.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "stablehlo_text.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace tlaloc_triton {

// A loaded plugin. Plugins are never unloaded (XLA plugins do not support
// dlclose), so a PjrtPlugin lives until the process exits.
struct PjrtPlugin {
  std::string path;
  const PJRT_Api* api = nullptr;
  int major = 0;
  int minor = 0;
};

// Loads `path` (once per path per process), checks the API table size and
// major version, and calls PJRT_Plugin_Initialize.
std::string LoadPjrtPlugin(const std::string& path, const PjrtPlugin** out);

// Allocator options for the XLA GPU plugin. Without them the plugin reserves
// 75% of device memory at client creation; on a machine whose GPU memory is
// the system RAM that has taken the whole machine down.
struct ClientOptions {
  float memory_fraction = 0.3f;
  bool preallocate = false;
};

// Whether a device pointer can be handed to PJRT as an argument without a
// copy: XLA's GPU entry parameters must be 16-byte aligned.
constexpr size_t kDeviceArgumentAlignment = 16;

class PjrtExecutable;
class PjrtBuffer;

class PjrtClient {
 public:
  // A client that sees one device: the GPU whose CUDA ordinal is `device`
  // (the plugin's "visible_devices" option). Refused by name when the plugin
  // has no such device, or when the device it gives back is another one.
  static std::string Create(
      const PjrtPlugin* plugin, const ClientOptions& options, int device,
      std::unique_ptr<PjrtClient>* out);
  ~PjrtClient();

  const PjrtPlugin* plugin() const { return plugin_; }
  const std::string& platform() const { return platform_; }
  size_t device_count() const { return devices_.size(); }
  const ClientOptions& options() const { return options_; }
  // The CUDA ordinal of the one device this client runs on.
  int device_ordinal() const { return ordinal_; }
  // True when the plugin can wrap device memory it does not own
  // (PJRT_Client_CreateViewOfDeviceBuffer) and hand out the device address
  // of a result (PJRT_Buffer_OpaqueDeviceMemoryDataPointer).
  bool SupportsDeviceViews() const;

  // Compiles MLIR text whose entry function is public and named main.
  std::string Compile(const std::string& mlir, std::unique_ptr<PjrtExecutable>* out);

 private:
  friend class PjrtExecutable;
  friend class PjrtBuffer;
  PjrtClient() = default;
  const PjrtPlugin* plugin_ = nullptr;
  PJRT_Client* client_ = nullptr;
  std::vector<PJRT_Device*> devices_;
  std::string platform_;
  ClientOptions options_;
  int ordinal_ = 0;
};

// A host tensor handed to Execute or Upload. `data` must stay valid for the
// call.
struct HostInput {
  const void* data = nullptr;
  size_t byte_size = 0;
  DType dtype = DType::UNSUPPORTED;
  std::vector<int64_t> dims;
};

// A buffer on device 0 that outlives one execution: a weight uploaded at
// model load, or state carried from one request to the next. Destroyed with
// the object.
class PjrtBuffer {
 public:
  ~PjrtBuffer();
  // Copies `host` to device 0 and waits until the host bytes are no longer
  // needed.
  static std::string Upload(
      PjrtClient* client, const HostInput& host, std::unique_ptr<PjrtBuffer>* out);

 private:
  friend class PjrtExecutable;
  friend class PjrtResults;
  const PJRT_Api* api_ = nullptr;
  PJRT_Buffer* buffer_ = nullptr;
};

// One argument of an execution, in order of preference:
//   `device`      a PJRT buffer the caller owns (a weight, state),
//   `device_ptr`  memory on the client's GPU that the caller owns, described
//                 by `host.dtype` and `host.dims`, read in place through a
//                 PJRT view (no copy) and never donated,
//   `host`        a host tensor, copied to the device for this call.
struct ExecuteArg {
  HostInput host;
  const PjrtBuffer* device = nullptr;
  const void* device_ptr = nullptr;
};

// Device results of one execution. Destroys its buffers when it goes away.
class PjrtResults {
 public:
  ~PjrtResults();
  size_t size() const { return buffers_.size(); }
  // Element type and dimensions of output `i`.
  std::string Describe(size_t i, DType* dtype, std::vector<int64_t>* dims) const;
  // Copies output `i` into `dst`, which must be exactly `byte_size` bytes.
  std::string CopyToHost(size_t i, void* dst, size_t byte_size) const;
  // Calls `use` with the device address of output `i` when the output is
  // stored densely, row-major, in exactly `byte_size` bytes; the address is
  // valid only during the call. Sets `*dense` false and does not call `use`
  // when the storage differs (a padded or tiled layout).
  std::string WithDevicePointer(
      size_t i, size_t byte_size, bool* dense,
      const std::function<std::string(const void*)>& use) const;
  // Takes output `i` out of the results, to keep it on the device.
  std::unique_ptr<PjrtBuffer> Release(size_t i);

 private:
  friend class PjrtExecutable;
  const PJRT_Api* api_ = nullptr;
  std::vector<PJRT_Buffer*> buffers_;
};

class PjrtExecutable {
 public:
  ~PjrtExecutable();
  size_t num_outputs() const { return num_outputs_; }
  // Uploads the host arguments to device 0, runs, and waits for completion.
  std::string Execute(const std::vector<ExecuteArg>& args, std::unique_ptr<PjrtResults>* out);

 private:
  friend class PjrtClient;
  PjrtClient* client_ = nullptr;
  PJRT_LoadedExecutable* exe_ = nullptr;
  size_t num_outputs_ = 0;
};

PJRT_Buffer_Type ToPjrtType(DType t);
DType FromPjrtType(PJRT_Buffer_Type t);

}  // namespace tlaloc_triton
