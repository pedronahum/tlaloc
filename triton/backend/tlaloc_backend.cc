// Copyright 2026 Pedro N. Rodriguez
// SPDX-License-Identifier: Apache-2.0
//
// Triton backend "tlaloc": serves StableHLO emitted by Tlaloc through the
// PJRT C API.
//
// One PJRT client per plugin path, shared by every model that uses that
// plugin, created with allocator options (memory fraction, no
// preallocation) and destroyed when the last such model unloads. Each model
// compiles its artifact(s) once at load; each artifact is one shape bucket,
// and a request runs on the bucket whose input shapes it matches exactly.

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif

#include "pjrt_runtime.h"
#include "stablehlo_text.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/core/tritonbackend.h"

namespace triton { namespace backend { namespace tlaloc {

using tlaloc_triton::ClientOptions;
using tlaloc_triton::DType;
using tlaloc_triton::FunctionSignature;
using tlaloc_triton::HostInput;
using tlaloc_triton::PjrtClient;
using tlaloc_triton::PjrtExecutable;
using tlaloc_triton::PjrtPlugin;
using tlaloc_triton::PjrtResults;
using tlaloc_triton::ShapeString;

namespace {

TRITONSERVER_Error*
Err(TRITONSERVER_Error_Code code, const std::string& msg)
{
  return TRITONSERVER_ErrorNew(code, msg.c_str());
}

uint64_t
NowNs()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::string
Trim(const std::string& s)
{
  size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  size_t e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

std::string
EnvOr(const char* name, const std::string& fallback)
{
  const char* v = std::getenv(name);
  return (v != nullptr && *v != '\0') ? std::string(v) : fallback;
}

// ---------------------------------------------------------------------------
// Backend state: settings from --backend-config=tlaloc,<key>=<value>.

struct BackendState {
  std::string plugin_path;  // "" = not set at backend level
  ClientOptions options;
};

TRITONSERVER_Error*
ParseBackendConfig(TRITONBACKEND_Backend* backend, BackendState* state)
{
  TRITONSERVER_Message* message = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_BackendConfig(backend, &message));
  const char* buffer = nullptr;
  size_t size = 0;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(message, &buffer, &size));
  common::TritonJson::Value config;
  RETURN_IF_ERROR(config.Parse(buffer, size));

  std::string fraction = EnvOr("TLALOC_PJRT_MEMORY_FRACTION", "");
  std::string preallocate = EnvOr("TLALOC_PJRT_PREALLOCATE", "");
  state->plugin_path = "";

  common::TritonJson::Value cmdline;
  if (config.Find("cmdline", &cmdline)) {
    std::string v;
    if (cmdline.MemberAsString("pjrt-plugin-path", &v) == nullptr) state->plugin_path = v;
    if (cmdline.MemberAsString("memory-fraction", &v) == nullptr) fraction = v;
    if (cmdline.MemberAsString("preallocate", &v) == nullptr) preallocate = v;
  }

  if (!fraction.empty()) {
    char* end = nullptr;
    float f = std::strtof(fraction.c_str(), &end);
    if (end == fraction.c_str() || *end != '\0' || !(f > 0.0f && f <= 1.0f)) {
      return Err(
          TRITONSERVER_ERROR_INVALID_ARG,
          "tlaloc backend: memory-fraction (or TLALOC_PJRT_MEMORY_FRACTION) must be a "
          "number in (0, 1], got '" + fraction + "'");
    }
    state->options.memory_fraction = f;
  }
  if (!preallocate.empty()) {
    if (preallocate != "true" && preallocate != "false") {
      return Err(
          TRITONSERVER_ERROR_INVALID_ARG,
          "tlaloc backend: preallocate (or TLALOC_PJRT_PREALLOCATE) must be true or false, "
          "got '" + preallocate + "'");
    }
    state->options.preallocate = preallocate == "true";
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// Shared clients, one per plugin path.

std::mutex g_clients_mu;
std::map<std::string, std::weak_ptr<PjrtClient>> g_clients;

TRITONSERVER_Error*
AcquireClient(
    const std::string& plugin_path, const ClientOptions& options,
    std::shared_ptr<PjrtClient>* out)
{
  std::lock_guard<std::mutex> lock(g_clients_mu);
  auto it = g_clients.find(plugin_path);
  if (it != g_clients.end()) {
    if (auto live = it->second.lock()) {
      *out = live;
      return nullptr;
    }
  }
  const PjrtPlugin* plugin = nullptr;
  std::string err = tlaloc_triton::LoadPjrtPlugin(plugin_path, &plugin);
  if (!err.empty()) return Err(TRITONSERVER_ERROR_UNAVAILABLE, "tlaloc backend: " + err);
  std::unique_ptr<PjrtClient> client;
  err = PjrtClient::Create(plugin, options, &client);
  if (!err.empty()) return Err(TRITONSERVER_ERROR_UNAVAILABLE, "tlaloc backend: " + err);
  std::ostringstream m;
  m << "tlaloc backend: PJRT client created on platform '" << client->platform() << "' with "
    << client->device_count() << " device(s), PJRT C API " << plugin->major << "."
    << plugin->minor << ", memory_fraction=" << options.memory_fraction
    << ", preallocate=" << (options.preallocate ? "true" : "false") << ", plugin "
    << plugin_path;
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, m.str().c_str());
  std::shared_ptr<PjrtClient> shared(client.release(), [plugin_path](PjrtClient* c) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        ("tlaloc backend: destroying the PJRT client for " + plugin_path).c_str());
    delete c;
  });
  g_clients[plugin_path] = shared;
  *out = shared;
  return nullptr;
}

// ---------------------------------------------------------------------------
// Model state.

struct TensorSpec {
  std::string name;
  DType dtype = DType::UNSUPPORTED;
  std::vector<int64_t> shape;  // what the model sees; -1 = any
};

struct Bucket {
  std::string file;
  FunctionSignature signature;
  std::unique_ptr<PjrtExecutable> executable;
};

// Element count; 1 for a rank-0 tensor. (The backend utilities'
// GetElementCount returns 0 for an empty shape.)
uint64_t
Elements(const std::vector<int64_t>& dims)
{
  uint64_t n = 1;
  for (int64_t d : dims) n *= static_cast<uint64_t>(d);
  return n;
}

// Triton has no rank-0 tensors in a non-batching model, so a rank-0
// argument or result of the entry function is declared as dims [ 1 ].
bool
IsScalarAsOne(const std::vector<int64_t>& config, const std::vector<int64_t>& dims)
{
  return dims.empty() && config.size() == 1 && config[0] == 1;
}

bool
ShapeAccepts(const std::vector<int64_t>& pattern, const std::vector<int64_t>& dims)
{
  if (IsScalarAsOne(pattern, dims)) return true;
  if (pattern.size() != dims.size()) return false;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (pattern[i] != -1 && pattern[i] != dims[i]) return false;
  }
  return true;
}

class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(TRITONBACKEND_Model* model, ModelState** state);

  const std::vector<TensorSpec>& inputs() const { return inputs_; }
  const std::vector<TensorSpec>& outputs() const { return outputs_; }
  const std::vector<Bucket>& buckets() const { return buckets_; }

 private:
  explicit ModelState(TRITONBACKEND_Model* model) : BackendModel(model) {}
  TRITONSERVER_Error* Load(const BackendState& backend);
  TRITONSERVER_Error* ReadSpecs(const char* key, std::vector<TensorSpec>* specs);
  TRITONSERVER_Error* Parameter(const std::string& key, std::string* value);
  std::string Where() const { return "model '" + Name() + "': "; }

  // Declared before the buckets so that executables are destroyed first.
  std::shared_ptr<PjrtClient> client_;
  std::vector<TensorSpec> inputs_;
  std::vector<TensorSpec> outputs_;
  std::vector<Bucket> buckets_;
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* model, ModelState** state)
{
  try {
    *state = new ModelState(model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }
  TRITONBACKEND_Backend* backend = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(model, &backend));
  void* vstate = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  TRITONSERVER_Error* err = (*state)->Load(*reinterpret_cast<BackendState*>(vstate));
  if (err != nullptr) {
    delete *state;
    *state = nullptr;
  }
  return err;
}

TRITONSERVER_Error*
ModelState::Parameter(const std::string& key, std::string* value)
{
  value->clear();
  common::TritonJson::Value params;
  if (!ModelConfig().Find("parameters", &params)) return nullptr;
  common::TritonJson::Value entry;
  if (!params.Find(key.c_str(), &entry)) return nullptr;
  return GetParameterValue(params, key, value);
}

TRITONSERVER_Error*
ModelState::ReadSpecs(const char* key, std::vector<TensorSpec>* specs)
{
  common::TritonJson::Value ios;
  if (!ModelConfig().Find(key, &ios)) return nullptr;
  for (size_t i = 0; i < ios.ArraySize(); ++i) {
    common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
    TensorSpec spec;
    RETURN_IF_ERROR(io.MemberAsString("name", &spec.name));
    std::string dt;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &dt));
    spec.dtype = tlaloc_triton::DTypeFromTriton(dt);
    if (spec.dtype == DType::UNSUPPORTED) {
      return Err(
          TRITONSERVER_ERROR_INVALID_ARG,
          Where() + key + " '" + spec.name + "' has data_type " + dt +
              "; the tlaloc backend serves FP32, FP64, FP16, BF16, INT8, INT32, INT64, "
              "UINT8 and BOOL");
    }
    common::TritonJson::Value reshape;
    std::vector<int64_t> dims;
    if (io.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(ParseShape(reshape, "shape", &dims));
    } else {
      RETURN_IF_ERROR(ParseShape(io, "dims", &dims));
    }
    if (MaxBatchSize() > 0) spec.shape.push_back(-1);
    spec.shape.insert(spec.shape.end(), dims.begin(), dims.end());
    bool optional = false;
    common::TritonJson::Value opt;
    if (io.Find("optional", &opt)) RETURN_IF_ERROR(opt.AsBool(&optional));
    if (optional) {
      return Err(
          TRITONSERVER_ERROR_INVALID_ARG,
          Where() + "input '" + spec.name + "' is optional; the tlaloc backend passes every "
          "input positionally and has no value for a missing one");
    }
    specs->push_back(spec);
  }
  return nullptr;
}

TRITONSERVER_Error*
ModelState::Load(const BackendState& backend)
{
  RETURN_IF_ERROR(ReadSpecs("input", &inputs_));
  RETURN_IF_ERROR(ReadSpecs("output", &outputs_));

  std::string artifact_list, entry, plugin_path;
  RETURN_IF_ERROR(Parameter("artifact", &artifact_list));
  RETURN_IF_ERROR(Parameter("entry", &entry));
  RETURN_IF_ERROR(Parameter("pjrt_plugin_path", &plugin_path));
  entry = Trim(entry);

  if (Trim(artifact_list).empty()) {
    return Err(
        TRITONSERVER_ERROR_INVALID_ARG,
        Where() + "config.pbtxt has no 'artifact' parameter. Add parameters { key: "
        "\"artifact\" value: { string_value: \"model.mlir\" } } naming the StableHLO file, "
        "relative to the model version directory");
  }

  if (Trim(plugin_path).empty()) plugin_path = backend.plugin_path;
  if (Trim(plugin_path).empty()) plugin_path = EnvOr("TLALOC_PJRT_PLUGIN_PATH", "");
  plugin_path = Trim(plugin_path);
  if (plugin_path.empty()) {
    return Err(
        TRITONSERVER_ERROR_INVALID_ARG,
        Where() + "no PJRT plugin configured. Set the model parameter 'pjrt_plugin_path', "
        "the backend config --backend-config=tlaloc,pjrt-plugin-path=<path>, or the "
        "environment variable TLALOC_PJRT_PLUGIN_PATH to a PJRT plugin .so");
  }

  // Parse every bucket before touching the GPU, so that a config mistake
  // costs no client.
  const std::string version_dir = JoinPath({RepositoryPath(), std::to_string(Version())});
  std::stringstream list(artifact_list);
  std::string item;
  std::vector<std::string> texts;
  while (std::getline(list, item, ',')) {
    item = Trim(item);
    if (item.empty()) continue;
    if (item[0] == '/') {
      return Err(
          TRITONSERVER_ERROR_INVALID_ARG,
          Where() + "artifact '" + item + "' is an absolute path; artifacts are named "
          "relative to the model version directory " + version_dir);
    }
    const std::string path = JoinPath({version_dir, item});
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      return Err(
          TRITONSERVER_ERROR_NOT_FOUND, Where() + "cannot read the artifact " + path);
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (tlaloc_triton::IsMlirBytecode(text)) {
      return Err(
          TRITONSERVER_ERROR_INVALID_ARG,
          Where() + path + " is MLIR bytecode. The tlaloc backend reads StableHLO text, "
          "which is what Tlaloc writes; it needs the text to find and rename the entry "
          "function and to check the signature against config.pbtxt");
    }
    std::vector<FunctionSignature> functions;
    std::string perr;
    FunctionSignature sig;
    if (!tlaloc_triton::ListFunctions(text, &functions, &perr) ||
        !tlaloc_triton::SelectEntry(functions, entry, &sig, &perr)) {
      return Err(TRITONSERVER_ERROR_INVALID_ARG, Where() + path + ": " + perr);
    }

    // The signature must agree with config.pbtxt, positionally.
    auto check = [&](const char* what, const std::vector<TensorSpec>& specs,
                     const std::vector<tlaloc_triton::TensorType>& types) -> TRITONSERVER_Error* {
      if (specs.size() != types.size()) {
        return Err(
            TRITONSERVER_ERROR_INVALID_ARG,
            Where() + path + ": @" + sig.name + " has " + std::to_string(types.size()) + " " +
                what + "s but config.pbtxt declares " + std::to_string(specs.size()));
      }
      for (size_t i = 0; i < specs.size(); ++i) {
        const auto& t = types[i];
        std::string at = Where() + path + ": " + what + " " + std::to_string(i) + " ('" +
                         specs[i].name + "') ";
        if (t.dtype == DType::UNSUPPORTED) {
          return Err(
              TRITONSERVER_ERROR_INVALID_ARG,
              at + "has element type " + t.element + ", which the tlaloc backend does not serve");
        }
        for (int64_t d : t.dims) {
          if (d < 0) {
            return Err(
                TRITONSERVER_ERROR_INVALID_ARG,
                at + "is " + t.text + ", which has a dynamic dimension. Export one artifact "
                "per shape and list them all in the 'artifact' parameter");
          }
        }
        if (t.dtype != specs[i].dtype) {
          return Err(
              TRITONSERVER_ERROR_INVALID_ARG,
              at + "is " + t.text + " but config.pbtxt says " +
                  tlaloc_triton::TritonName(specs[i].dtype));
        }
        if (!ShapeAccepts(specs[i].shape, t.dims)) {
          return Err(
              TRITONSERVER_ERROR_INVALID_ARG,
              at + "is " + t.text + " but config.pbtxt allows shape " +
                  ShapeString(specs[i].shape) +
                  (MaxBatchSize() > 0 ? " (the leading -1 is the batch dimension)" : ""));
        }
      }
      return nullptr;
    };
    RETURN_IF_ERROR(check("input", inputs_, sig.args));
    RETURN_IF_ERROR(check("output", outputs_, sig.results));
    for (const Bucket& b : buckets_) {
      bool same = true;
      for (size_t i = 0; i < sig.args.size(); ++i) same &= b.signature.args[i].dims == sig.args[i].dims;
      if (same) {
        return Err(
            TRITONSERVER_ERROR_INVALID_ARG,
            Where() + b.file + " and " + item + " take the same input shapes, so a request "
            "could not choose between them");
      }
    }
    Bucket bucket;
    bucket.file = item;
    bucket.signature = sig;
    buckets_.push_back(std::move(bucket));
    texts.push_back(tlaloc_triton::PrepareForXla(text, sig.name));
  }
  if (buckets_.empty()) {
    return Err(TRITONSERVER_ERROR_INVALID_ARG, Where() + "the 'artifact' parameter names no file");
  }

  RETURN_IF_ERROR(AcquireClient(plugin_path, backend.options, &client_));
  for (size_t i = 0; i < buckets_.size(); ++i) {
    const uint64_t t0 = NowNs();
    std::string err = client_->Compile(texts[i], &buckets_[i].executable);
    if (!err.empty()) {
      return Err(TRITONSERVER_ERROR_INVALID_ARG, Where() + buckets_[i].file + ": " + err);
    }
    if (buckets_[i].executable->num_outputs() != outputs_.size()) {
      return Err(
          TRITONSERVER_ERROR_INTERNAL,
          Where() + buckets_[i].file + " compiled to " +
              std::to_string(buckets_[i].executable->num_outputs()) + " outputs, expected " +
              std::to_string(outputs_.size()));
    }
    std::ostringstream m;
    m << "tlaloc backend: " << Where() << "compiled " << buckets_[i].file << " (@"
      << buckets_[i].signature.name << ") in " << (NowNs() - t0) / 1000000 << " ms";
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, m.str().c_str());
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// Instance state.

class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state, TRITONBACKEND_ModelInstance* instance,
      ModelInstanceState** state);
  ModelState* StateForModel() const { return model_state_; }

  void ProcessRequests(TRITONBACKEND_Request** requests, uint32_t count);

 private:
  ModelInstanceState(ModelState* model_state, TRITONBACKEND_ModelInstance* instance)
      : BackendModelInstance(model_state, instance), model_state_(model_state)
  {
  }
  TRITONSERVER_Error* Run(
      TRITONBACKEND_Request* request, TRITONBACKEND_Response* response,
      uint64_t* compute_start, uint64_t* compute_end, uint64_t* batch);

  ModelState* model_state_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }
  if ((*state)->Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU && (*state)->DeviceId() != 0) {
    std::string name = (*state)->Name();
    delete *state;
    *state = nullptr;
    return Err(
        TRITONSERVER_ERROR_INVALID_ARG,
        "instance '" + name + "': the tlaloc backend runs every instance on PJRT device 0; "
        "set instance_group gpus: [ 0 ] (or use KIND_CPU / KIND_MODEL)");
  }
  return nullptr;
}

TRITONSERVER_Error*
CopyToHost(
    const void* src, TRITONSERVER_MemoryType mt, size_t bytes, char* dst,
    const std::string& what)
{
  if (mt == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
    cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
      return Err(
          TRITONSERVER_ERROR_INTERNAL,
          what + ": copying from GPU memory failed: " + cudaGetErrorString(e));
    }
    return nullptr;
#else
    return Err(
        TRITONSERVER_ERROR_UNSUPPORTED,
        what + " is in GPU memory and this build of the tlaloc backend has no CUDA");
#endif
  }
  std::memcpy(dst, src, bytes);
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::Run(
    TRITONBACKEND_Request* request, TRITONBACKEND_Response* response,
    uint64_t* compute_start, uint64_t* compute_end, uint64_t* batch)
{
  const ModelState& model = *model_state_;
  const auto& specs = model.inputs();

  uint32_t input_count = 0;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(request, &input_count));
  if (input_count != specs.size()) {
    return Err(
        TRITONSERVER_ERROR_INVALID_ARG,
        "expected " + std::to_string(specs.size()) + " inputs, got " +
            std::to_string(input_count));
  }

  std::vector<HostInput> host(specs.size());
  std::vector<std::vector<char>> staging(specs.size());
  std::vector<std::vector<int64_t>> shapes(specs.size());
  for (size_t i = 0; i < specs.size(); ++i) {
    const TensorSpec& spec = specs[i];
    TRITONBACKEND_Input* input = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, spec.name.c_str(), &input));
    TRITONSERVER_DataType dt;
    const int64_t* shape = nullptr;
    uint32_t dims = 0;
    uint64_t byte_size = 0;
    uint32_t buffers = 0;
    RETURN_IF_ERROR(
        TRITONBACKEND_InputProperties(input, nullptr, &dt, &shape, &dims, &byte_size, &buffers));
    DType dtype = tlaloc_triton::DTypeFromTriton(TRITONSERVER_DataTypeString(dt));
    if (dtype != spec.dtype) {
      return Err(
          TRITONSERVER_ERROR_INVALID_ARG,
          "input '" + spec.name + "' is " + TRITONSERVER_DataTypeString(dt) + ", expected " +
              tlaloc_triton::TritonName(spec.dtype));
    }
    shapes[i].assign(shape, shape + dims);
    const uint64_t expected = Elements(shapes[i]) * tlaloc_triton::ByteWidth(dtype);
    if (byte_size != expected) {
      return Err(
          TRITONSERVER_ERROR_INVALID_ARG,
          "input '" + spec.name + "' has " + std::to_string(byte_size) + " bytes, but shape " +
              ShapeString(shapes[i]) + " of " + tlaloc_triton::TritonName(dtype) + " needs " +
              std::to_string(expected));
    }
    host[i].dtype = dtype;
    host[i].dims = shapes[i];
    host[i].byte_size = byte_size;

    const void* ptr = nullptr;
    uint64_t size = 0;
    TRITONSERVER_MemoryType mt = TRITONSERVER_MEMORY_CPU;
    int64_t mid = 0;
    if (buffers == 1) {
      RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(input, 0, &ptr, &size, &mt, &mid));
      if (mt != TRITONSERVER_MEMORY_GPU) {
        host[i].data = ptr;  // host memory: hand it to PJRT as is
        continue;
      }
    }
    staging[i].resize(byte_size);
    size_t offset = 0;
    for (uint32_t b = 0; b < buffers; ++b) {
      mt = TRITONSERVER_MEMORY_CPU;
      mid = 0;
      RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(input, b, &ptr, &size, &mt, &mid));
      if (offset + size > byte_size) {
        return Err(
            TRITONSERVER_ERROR_INVALID_ARG,
            "input '" + spec.name + "' buffers exceed its declared byte size");
      }
      RETURN_IF_ERROR(
          CopyToHost(ptr, mt, size, staging[i].data() + offset, "input '" + spec.name + "'"));
      offset += size;
    }
    host[i].data = staging[i].data();
  }

  // Pick the bucket compiled for exactly these input shapes.
  const Bucket* bucket = nullptr;
  for (const Bucket& b : model.buckets()) {
    bool match = true;
    for (size_t i = 0; i < shapes.size(); ++i) {
      const auto& want = b.signature.args[i].dims;
      match &= want == shapes[i] || IsScalarAsOne(shapes[i], want);
    }
    if (match) {
      bucket = &b;
      break;
    }
  }
  if (bucket == nullptr) {
    std::string have, want;
    for (const auto& s : shapes) have += (have.empty() ? "" : ", ") + ShapeString(s);
    for (const Bucket& b : model.buckets()) {
      std::string one;
      for (const auto& a : b.signature.args) one += (one.empty() ? "" : ", ") + ShapeString(a.dims);
      want += (want.empty() ? "(" : "; (") + one + ")";
    }
    return Err(
        TRITONSERVER_ERROR_INVALID_ARG,
        "no compiled artifact takes input shapes (" + have + "); this model serves " + want);
  }
  // PJRT gets the entry function's own shapes ([1] -> rank 0).
  for (size_t i = 0; i < host.size(); ++i) host[i].dims = bucket->signature.args[i].dims;
  if (model.MaxBatchSize() > 0 && !shapes.empty() && !shapes[0].empty()) {
    *batch = static_cast<uint64_t>(shapes[0][0]);
  }

  *compute_start = NowNs();
  std::unique_ptr<PjrtResults> results;
  std::string err = bucket->executable->Execute(host, &results);
  *compute_end = NowNs();
  if (!err.empty()) {
    return Err(TRITONSERVER_ERROR_INTERNAL, bucket->file + ": " + err);
  }

  std::set<std::string> requested;
  uint32_t requested_count = 0;
  RETURN_IF_ERROR(TRITONBACKEND_RequestOutputCount(request, &requested_count));
  for (uint32_t i = 0; i < requested_count; ++i) {
    const char* name = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_RequestOutputName(request, i, &name));
    requested.insert(name);
  }

  const auto& outs = model.outputs();
  for (size_t j = 0; j < outs.size(); ++j) {
    if (requested_count > 0 && requested.count(outs[j].name) == 0) continue;
    DType dtype;
    std::vector<int64_t> dims;
    err = results->Describe(j, &dtype, &dims);
    if (!err.empty()) return Err(TRITONSERVER_ERROR_INTERNAL, err);
    if (dtype != outs[j].dtype) {
      return Err(
          TRITONSERVER_ERROR_INTERNAL,
          "output '" + outs[j].name + "' came back as " + tlaloc_triton::TritonName(dtype) +
              ", expected " + tlaloc_triton::TritonName(outs[j].dtype));
    }
    const size_t bytes = Elements(dims) * tlaloc_triton::ByteWidth(dtype);
    if (IsScalarAsOne(outs[j].shape, dims)) dims = {1};
    TRITONBACKEND_Output* output = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(
        response, &output, outs[j].name.c_str(),
        TRITONSERVER_StringToDataType(tlaloc_triton::TritonName(dtype)), dims.data(),
        static_cast<uint32_t>(dims.size())));
    void* buffer = nullptr;
    TRITONSERVER_MemoryType mt = TRITONSERVER_MEMORY_CPU;
    int64_t mid = 0;
    RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(output, &buffer, bytes, &mt, &mid));
    if (mt == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
      std::vector<char> tmp(bytes);
      err = results->CopyToHost(j, tmp.data(), bytes);
      if (!err.empty()) return Err(TRITONSERVER_ERROR_INTERNAL, err);
      cudaError_t e = cudaMemcpy(buffer, tmp.data(), bytes, cudaMemcpyHostToDevice);
      if (e != cudaSuccess) {
        return Err(
            TRITONSERVER_ERROR_INTERNAL, "output '" + outs[j].name +
                                             "': copying to GPU memory failed: " +
                                             cudaGetErrorString(e));
      }
#else
      return Err(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "output '" + outs[j].name + "' was given GPU memory and this build has no CUDA");
#endif
    } else {
      err = results->CopyToHost(j, buffer, bytes);
      if (!err.empty()) return Err(TRITONSERVER_ERROR_INTERNAL, err);
    }
  }
  return nullptr;
}

void
ModelInstanceState::ProcessRequests(TRITONBACKEND_Request** requests, uint32_t count)
{
  const uint64_t exec_start = NowNs();
  uint64_t first_compute_start = 0, last_compute_end = 0, total_batch = 0;
  for (uint32_t r = 0; r < count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    const uint64_t request_start = NowNs();
    uint64_t compute_start = request_start, compute_end = request_start, batch = 1;
    TRITONBACKEND_Response* response = nullptr;
    TRITONSERVER_Error* err = TRITONBACKEND_ResponseNew(&response, request);
    if (err == nullptr) {
      try {
        err = Run(request, response, &compute_start, &compute_end, &batch);
      }
      catch (const std::exception& ex) {
        err = Err(TRITONSERVER_ERROR_INTERNAL, std::string("tlaloc backend: ") + ex.what());
      }
      catch (...) {
        err = Err(TRITONSERVER_ERROR_INTERNAL, "tlaloc backend: unknown exception");
      }
      const bool ok = err == nullptr;
      if (!ok) {
        std::string msg = "model '" + model_state_->Name() + "': " + TRITONSERVER_ErrorMessage(err);
        TRITONSERVER_Error_Code code = TRITONSERVER_ErrorCode(err);
        TRITONSERVER_ErrorDelete(err);
        err = Err(code, msg);
      }
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
          "failed to send the response");
      LOG_IF_ERROR(
          TRITONBACKEND_ModelInstanceReportStatistics(
              TritonModelInstance(), request, ok, request_start, compute_start, compute_end,
              NowNs()),
          "failed to report request statistics");
      if (err != nullptr) TRITONSERVER_ErrorDelete(err);
    } else {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, TRITONSERVER_ErrorMessage(err));
      TRITONSERVER_ErrorDelete(err);
    }
    if (first_compute_start == 0) first_compute_start = compute_start;
    last_compute_end = compute_end;
    total_batch += batch;
    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed to release the request");
  }
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          TritonModelInstance(), total_batch, exec_start, first_compute_start, last_compute_end,
          NowNs()),
      "failed to report batch statistics");
}

}  // namespace

// ---------------------------------------------------------------------------
// The Triton entry points.

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  uint32_t major = 0, minor = 0;
  RETURN_IF_ERROR(TRITONBACKEND_ApiVersion(&major, &minor));
  if (major != TRITONBACKEND_API_VERSION_MAJOR || minor < TRITONBACKEND_API_VERSION_MINOR) {
    return Err(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "tlaloc backend: built against Triton backend API " +
            std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
            std::to_string(TRITONBACKEND_API_VERSION_MINOR) + ", server provides " +
            std::to_string(major) + "." + std::to_string(minor));
  }
  auto state = std::make_unique<BackendState>();
  RETURN_IF_ERROR(ParseBackendConfig(backend, state.get()));
  std::ostringstream m;
  m << "tlaloc backend: initialized; memory_fraction=" << state->options.memory_fraction
    << ", preallocate=" << (state->options.preallocate ? "true" : "false");
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, m.str().c_str());
  if (state->options.preallocate) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN,
        "tlaloc backend: preallocate=true reserves memory_fraction of GPU memory up front "
        "for every PJRT client");
  }
  RETURN_IF_ERROR(TRITONBACKEND_BackendSetState(backend, state.release()));
  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  void* vstate = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  delete reinterpret_cast<BackendState*>(vstate);
  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  ModelState* state = nullptr;
  try {
    RETURN_IF_ERROR(ModelState::Create(model, &state));
  }
  catch (const std::exception& ex) {
    return Err(TRITONSERVER_ERROR_INTERNAL, std::string("tlaloc backend: ") + ex.what());
  }
  RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(model, state));
  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  delete reinterpret_cast<ModelState*>(vstate);
  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  TRITONBACKEND_Model* model = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));
  void* vmodel = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodel));
  ModelInstanceState* state = nullptr;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(reinterpret_cast<ModelState*>(vmodel), instance, &state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(instance, state));
  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  delete reinterpret_cast<ModelInstanceState*>(vstate);
  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  void* vstate = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  reinterpret_cast<ModelInstanceState*>(vstate)->ProcessRequests(requests, request_count);
  return nullptr;
}

}  // extern "C"

}}}  // namespace triton::backend::tlaloc
