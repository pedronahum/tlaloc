// Copyright 2026 Pedro N. Rodriguez
// SPDX-License-Identifier: Apache-2.0
//
// A small C++ layer over the PJRT C API: load a plugin, create a client with
// allocator options, compile StableHLO text, move host bytes in and out, and
// execute. Every call returns an error string; an empty string is success.

#pragma once

#include <cstdint>
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

class PjrtExecutable;

class PjrtClient {
 public:
  static std::string Create(
      const PjrtPlugin* plugin, const ClientOptions& options,
      std::unique_ptr<PjrtClient>* out);
  ~PjrtClient();

  const PjrtPlugin* plugin() const { return plugin_; }
  const std::string& platform() const { return platform_; }
  size_t device_count() const { return devices_.size(); }
  const ClientOptions& options() const { return options_; }

  // Compiles MLIR text whose entry function is public and named main.
  std::string Compile(const std::string& mlir, std::unique_ptr<PjrtExecutable>* out);

 private:
  friend class PjrtExecutable;
  PjrtClient() = default;
  const PjrtPlugin* plugin_ = nullptr;
  PJRT_Client* client_ = nullptr;
  std::vector<PJRT_Device*> devices_;
  std::string platform_;
  ClientOptions options_;
};

// A host tensor handed to Execute. `data` must stay valid for the call.
struct HostInput {
  const void* data = nullptr;
  size_t byte_size = 0;
  DType dtype = DType::UNSUPPORTED;
  std::vector<int64_t> dims;
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

 private:
  friend class PjrtExecutable;
  const PJRT_Api* api_ = nullptr;
  std::vector<PJRT_Buffer*> buffers_;
};

class PjrtExecutable {
 public:
  ~PjrtExecutable();
  size_t num_outputs() const { return num_outputs_; }
  // Uploads the inputs to device 0, runs, and waits for completion.
  std::string Execute(const std::vector<HostInput>& inputs, std::unique_ptr<PjrtResults>* out);

 private:
  friend class PjrtClient;
  PjrtClient* client_ = nullptr;
  PJRT_LoadedExecutable* exe_ = nullptr;
  size_t num_outputs_ = 0;
};

PJRT_Buffer_Type ToPjrtType(DType t);
DType FromPjrtType(PJRT_Buffer_Type t);

}  // namespace tlaloc_triton
