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
using tlaloc_triton::ExecuteArg;
using tlaloc_triton::FunctionSignature;
using tlaloc_triton::HostInput;
using tlaloc_triton::PjrtBuffer;
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

// Where an argument of the entry function comes from.
enum class ArgSource {
  INPUT,   // a request tensor, by its config.pbtxt input name
  WEIGHT,  // a raw little-endian file in the version directory, uploaded at load
  STATE,   // a device buffer owned by the model instance, zero at start
};

struct ArgPlan {
  ArgSource source = ArgSource::INPUT;
  std::string name;  // input name, weight file, or state name
  size_t input = 0;  // index into the config inputs, for INPUT
};

// Where a result of the entry function goes.
enum class ResultSink {
  OUTPUT,  // a response tensor, by its config.pbtxt output name
  STATE,   // replaces the instance's state of that name
};

struct ResultPlan {
  ResultSink sink = ResultSink::OUTPUT;
  std::string name;
  size_t output = 0;  // index into the config outputs, for OUTPUT
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

// Comma-separated items, trimmed, empty items dropped.
std::vector<std::string>
SplitList(const std::string& list)
{
  std::vector<std::string> items;
  std::stringstream in(list);
  std::string item;
  while (std::getline(in, item, ',')) {
    item = Trim(item);
    if (!item.empty()) items.push_back(item);
  }
  return items;
}

class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(TRITONBACKEND_Model* model, ModelState** state);

  const std::vector<TensorSpec>& inputs() const { return inputs_; }
  const std::vector<TensorSpec>& outputs() const { return outputs_; }
  const std::vector<Bucket>& buckets() const { return buckets_; }
  const std::vector<ArgPlan>& args() const { return args_; }
  const std::vector<ResultPlan>& results() const { return results_; }
  const PjrtBuffer* weight(size_t arg) const { return weights_[arg].get(); }
  PjrtClient* client() const { return client_.get(); }

 private:
  explicit ModelState(TRITONBACKEND_Model* model) : BackendModel(model) {}
  TRITONSERVER_Error* Load(const BackendState& backend);
  TRITONSERVER_Error* ReadSpecs(const char* key, std::vector<TensorSpec>* specs);
  TRITONSERVER_Error* Parameter(const std::string& key, std::string* value);
  TRITONSERVER_Error* ParsePlan();
  TRITONSERVER_Error* CheckSignature(const std::string& path, const FunctionSignature& sig);
  TRITONSERVER_Error* LoadWeights();
  TRITONSERVER_Error* InsideVersionDir(const std::string& item, const std::string& what) const;
  std::string VersionDir() const
  {
    return JoinPath({RepositoryPath(), std::to_string(Version())});
  }
  std::string Where() const { return "model '" + Name() + "': "; }

  // Declared before the buckets and weights so that those are destroyed
  // first.
  std::shared_ptr<PjrtClient> client_;
  std::vector<TensorSpec> inputs_;
  std::vector<TensorSpec> outputs_;
  std::vector<ArgPlan> args_;
  std::vector<ResultPlan> results_;
  bool explicit_arguments_ = false;
  std::vector<Bucket> buckets_;
  std::vector<std::unique_ptr<PjrtBuffer>> weights_;  // one slot per argument
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
ModelState::InsideVersionDir(const std::string& item, const std::string& what) const
{
  if (item[0] == '/') {
    return Err(
        TRITONSERVER_ERROR_INVALID_ARG,
        Where() + what + " '" + item + "' is an absolute path; " + what +
            "s are named relative to the model version directory " + VersionDir());
  }
  std::stringstream parts(item);
  std::string part;
  while (std::getline(parts, part, '/')) {
    if (part == "..") {
      return Err(
          TRITONSERVER_ERROR_INVALID_ARG,
          Where() + what + " '" + item + "' leaves the model version directory " +
              VersionDir() + "; " + what + "s must be files inside it");
    }
  }
  return nullptr;
}

// Reads the 'arguments' and 'results' parameters, or derives the positional
// default: every config input in order, every config output in order.
TRITONSERVER_Error*
ModelState::ParsePlan()
{
  std::string arguments, results;
  RETURN_IF_ERROR(Parameter("arguments", &arguments));
  RETURN_IF_ERROR(Parameter("results", &results));

  auto split = [](const std::string& item, std::string* kind, std::string* name) {
    size_t colon = item.find(':');
    if (colon == std::string::npos) return false;
    *kind = Trim(item.substr(0, colon));
    *name = Trim(item.substr(colon + 1));
    return !name->empty();
  };

  std::vector<bool> input_used(inputs_.size(), false);
  std::set<std::string> states;
  explicit_arguments_ = !Trim(arguments).empty();
  if (!explicit_arguments_) {
    for (size_t i = 0; i < inputs_.size(); ++i) {
      args_.push_back({ArgSource::INPUT, inputs_[i].name, i});
      input_used[i] = true;
    }
  } else {
    for (const std::string& item : SplitList(arguments)) {
      std::string kind, name;
      if (!split(item, &kind, &name) || (kind != "input" && kind != "weight" && kind != "state")) {
        return Err(
            TRITONSERVER_ERROR_INVALID_ARG,
            Where() + "argument '" + item + "' in the 'arguments' parameter is not "
            "input:<name>, weight:<file> or state:<name>");
      }
      ArgPlan plan;
      plan.name = name;
      if (kind == "input") {
        plan.source = ArgSource::INPUT;
        size_t k = 0;
        while (k < inputs_.size() && inputs_[k].name != name) ++k;
        if (k == inputs_.size()) {
          return Err(
              TRITONSERVER_ERROR_INVALID_ARG,
              Where() + "the 'arguments' parameter names input '" + name +
                  "', which config.pbtxt does not declare");
        }
        if (input_used[k]) {
          return Err(
              TRITONSERVER_ERROR_INVALID_ARG,
              Where() + "the 'arguments' parameter binds input '" + name + "' twice");
        }
        input_used[k] = true;
        plan.input = k;
      } else if (kind == "weight") {
        plan.source = ArgSource::WEIGHT;
        RETURN_IF_ERROR(InsideVersionDir(name, "weight file"));
      } else {
        plan.source = ArgSource::STATE;
        if (!states.insert(name).second) {
          return Err(
              TRITONSERVER_ERROR_INVALID_ARG,
              Where() + "the 'arguments' parameter binds state '" + name + "' twice");
        }
      }
      args_.push_back(plan);
    }
    for (size_t k = 0; k < inputs_.size(); ++k) {
      if (!input_used[k]) {
        return Err(
            TRITONSERVER_ERROR_INVALID_ARG,
            Where() + "config.pbtxt input '" + inputs_[k].name +
                "' is not bound by the 'arguments' parameter");
      }
    }
  }

  std::vector<bool> output_used(outputs_.size(), false);
  std::set<std::string> written;
  if (Trim(results).empty()) {
    for (size_t j = 0; j < outputs_.size(); ++j) {
      results_.push_back({ResultSink::OUTPUT, outputs_[j].name, j});
      output_used[j] = true;
    }
  } else {
    for (const std::string& item : SplitList(results)) {
      std::string kind, name;
      if (!split(item, &kind, &name) || (kind != "output" && kind != "state")) {
        return Err(
            TRITONSERVER_ERROR_INVALID_ARG,
            Where() + "result '" + item + "' in the 'results' parameter is not "
            "output:<name> or state:<name>");
      }
      ResultPlan plan;
      plan.name = name;
      if (kind == "output") {
        plan.sink = ResultSink::OUTPUT;
        size_t k = 0;
        while (k < outputs_.size() && outputs_[k].name != name) ++k;
        if (k == outputs_.size()) {
          return Err(
              TRITONSERVER_ERROR_INVALID_ARG,
              Where() + "the 'results' parameter names output '" + name +
                  "', which config.pbtxt does not declare");
        }
        if (output_used[k]) {
          return Err(
              TRITONSERVER_ERROR_INVALID_ARG,
              Where() + "the 'results' parameter binds output '" + name + "' twice");
        }
        output_used[k] = true;
        plan.output = k;
      } else {
        plan.sink = ResultSink::STATE;
        if (states.count(name) == 0) {
          return Err(
              TRITONSERVER_ERROR_INVALID_ARG,
              Where() + "the 'results' parameter writes state '" + name +
                  "', which no state: argument in the 'arguments' parameter reads");
        }
        if (!written.insert(name).second) {
          return Err(
              TRITONSERVER_ERROR_INVALID_ARG,
              Where() + "the 'results' parameter writes state '" + name + "' twice");
        }
      }
      results_.push_back(plan);
    }
    for (size_t k = 0; k < outputs_.size(); ++k) {
      if (!output_used[k]) {
        return Err(
            TRITONSERVER_ERROR_INVALID_ARG,
            Where() + "config.pbtxt output '" + outputs_[k].name +
                "' is not bound by the 'results' parameter");
      }
    }
  }
  for (const std::string& s : states) {
    if (written.count(s) == 0) {
      return Err(
          TRITONSERVER_ERROR_INVALID_ARG,
          Where() + "state '" + s + "' is read by the 'arguments' parameter but no result "
          "in the 'results' parameter writes it");
    }
  }
  return nullptr;
}

// The entry signature must agree with config.pbtxt and with the argument and
// result plan. Weights and state keep one type across all buckets.
TRITONSERVER_Error*
ModelState::CheckSignature(const std::string& path, const FunctionSignature& sig)
{
  auto count = [&](const char* what, size_t have, size_t want) -> TRITONSERVER_Error* {
    if (have == want) return nullptr;
    const bool planned = std::string(what) == "input" ? explicit_arguments_ : false;
    return Err(
        TRITONSERVER_ERROR_INVALID_ARG,
        Where() + path + ": @" + sig.name + " has " + std::to_string(have) + " " + what +
            "s but " + (planned ? "the 'arguments' parameter binds " : "config.pbtxt declares ") +
            std::to_string(want));
  };
  RETURN_IF_ERROR(count("input", sig.args.size(), args_.size()));
  RETURN_IF_ERROR(count("output", sig.results.size(), results_.size()));

  auto basic = [&](const std::string& at, const tlaloc_triton::TensorType& t) -> TRITONSERVER_Error* {
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
    return nullptr;
  };
  auto against = [&](const std::string& at, const tlaloc_triton::TensorType& t,
                     const TensorSpec& spec) -> TRITONSERVER_Error* {
    if (t.dtype != spec.dtype) {
      return Err(
          TRITONSERVER_ERROR_INVALID_ARG,
          at + "is " + t.text + " but config.pbtxt says " + tlaloc_triton::TritonName(spec.dtype));
    }
    if (!ShapeAccepts(spec.shape, t.dims)) {
      return Err(
          TRITONSERVER_ERROR_INVALID_ARG,
          at + "is " + t.text + " but config.pbtxt allows shape " + ShapeString(spec.shape) +
              (MaxBatchSize() > 0 ? " (the leading -1 is the batch dimension)" : ""));
    }
    return nullptr;
  };
  auto fixed = [&](const std::string& at, const tlaloc_triton::TensorType& t,
                   const tlaloc_triton::TensorType& first) -> TRITONSERVER_Error* {
    if (t.dtype != first.dtype || t.dims != first.dims) {
      return Err(
          TRITONSERVER_ERROR_INVALID_ARG,
          at + "is " + t.text + " but " + buckets_[0].file + " has " + first.text +
              "; weights and state have one type across all artifacts of a model");
    }
    return nullptr;
  };

  for (size_t i = 0; i < args_.size(); ++i) {
    const auto& t = sig.args[i];
    const ArgPlan& p = args_[i];
    const char* kind = p.source == ArgSource::INPUT ? "input" :
                       p.source == ArgSource::WEIGHT ? "weight" : "state";
    std::string at = Where() + path + ": argument " + std::to_string(i) + " (" + kind + " '" +
                     p.name + "') ";
    if (!explicit_arguments_) {
      at = Where() + path + ": input " + std::to_string(i) + " ('" + p.name + "') ";
    }
    RETURN_IF_ERROR(basic(at, t));
    if (p.source == ArgSource::INPUT) {
      RETURN_IF_ERROR(against(at, t, inputs_[p.input]));
    } else if (!buckets_.empty()) {
      RETURN_IF_ERROR(fixed(at, t, buckets_[0].signature.args[i]));
    }
  }
  for (size_t j = 0; j < results_.size(); ++j) {
    const auto& t = sig.results[j];
    const ResultPlan& p = results_[j];
    std::string at = Where() + path + ": output " + std::to_string(j) + " ('" + p.name + "') ";
    RETURN_IF_ERROR(basic(at, t));
    if (p.sink == ResultSink::OUTPUT) {
      RETURN_IF_ERROR(against(at, t, outputs_[p.output]));
      continue;
    }
    for (size_t i = 0; i < args_.size(); ++i) {
      if (args_[i].source == ArgSource::STATE && args_[i].name == p.name) {
        const auto& read = sig.args[i];
        if (read.dtype != t.dtype || read.dims != t.dims) {
          return Err(
              TRITONSERVER_ERROR_INVALID_ARG,
              at + "is " + t.text + " but it replaces state '" + p.name + "', which argument " +
                  std::to_string(i) + " reads as " + read.text);
        }
      }
    }
  }
  return nullptr;
}

// Uploads every weight argument from its file. The file is the tensor's
// bytes, dense, row-major and little-endian, with no header.
TRITONSERVER_Error*
ModelState::LoadWeights()
{
  weights_.resize(args_.size());
  const uint64_t t0 = NowNs();
  uint64_t total = 0;
  size_t count = 0;
  for (size_t i = 0; i < args_.size(); ++i) {
    if (args_[i].source != ArgSource::WEIGHT) continue;
    const auto& t = buckets_[0].signature.args[i];
    const std::string path = JoinPath({VersionDir(), args_[i].name});
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
      return Err(TRITONSERVER_ERROR_NOT_FOUND, Where() + "cannot read the weight file " + path);
    }
    const uint64_t size = static_cast<uint64_t>(in.tellg());
    const uint64_t need = Elements(t.dims) * tlaloc_triton::ByteWidth(t.dtype);
    if (size != need) {
      return Err(
          TRITONSERVER_ERROR_INVALID_ARG,
          Where() + "the weight file " + path + " has " + std::to_string(size) +
              " bytes; argument " + std::to_string(i) + " is " + t.text + " and needs " +
              std::to_string(need));
    }
    std::vector<char> bytes(size);
    in.seekg(0);
    if (size > 0 && !in.read(bytes.data(), static_cast<std::streamsize>(size))) {
      return Err(TRITONSERVER_ERROR_INTERNAL, Where() + "reading the weight file " + path + " failed");
    }
    HostInput host;
    host.data = bytes.data();
    host.byte_size = size;
    host.dtype = t.dtype;
    host.dims = t.dims;
    std::string err = PjrtBuffer::Upload(client_.get(), host, &weights_[i]);
    if (!err.empty()) return Err(TRITONSERVER_ERROR_INTERNAL, Where() + path + ": " + err);
    total += size;
    ++count;
  }
  if (count > 0) {
    std::ostringstream m;
    m << "tlaloc backend: " << Where() << "uploaded " << count << " weights ("
      << total / (1024 * 1024) << " MiB) in " << (NowNs() - t0) / 1000000 << " ms";
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, m.str().c_str());
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
  RETURN_IF_ERROR(ParsePlan());

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
  const std::string version_dir = VersionDir();
  std::vector<std::string> texts;
  for (const std::string& item : SplitList(artifact_list)) {
    RETURN_IF_ERROR(InsideVersionDir(item, "artifact"));
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
    RETURN_IF_ERROR(CheckSignature(path, sig));
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
    if (buckets_[i].executable->num_outputs() != results_.size()) {
      return Err(
          TRITONSERVER_ERROR_INTERNAL,
          Where() + buckets_[i].file + " compiled to " +
              std::to_string(buckets_[i].executable->num_outputs()) + " outputs, expected " +
              std::to_string(results_.size()));
    }
    std::ostringstream m;
    m << "tlaloc backend: " << Where() << "compiled " << buckets_[i].file << " (@"
      << buckets_[i].signature.name << ") in " << (NowNs() - t0) / 1000000 << " ms";
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, m.str().c_str());
  }
  RETURN_IF_ERROR(LoadWeights());
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
  TRITONSERVER_Error* InitState();
  TRITONSERVER_Error* Run(
      TRITONBACKEND_Request* request, TRITONBACKEND_Response* response,
      uint64_t* compute_start, uint64_t* compute_end, uint64_t* batch);

  ModelState* model_state_;
  // State buffers by name, on the device, replaced after every request.
  std::map<std::string, std::unique_ptr<PjrtBuffer>> state_;
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
  TRITONSERVER_Error* err = (*state)->InitState();
  if (err != nullptr) {
    delete *state;
    *state = nullptr;
  }
  return err;
}

// Every state argument starts as zeros of its type.
TRITONSERVER_Error*
ModelInstanceState::InitState()
{
  const ModelState& model = *model_state_;
  const auto& sig = model.buckets()[0].signature;
  size_t count = 0;
  uint64_t total = 0;
  for (size_t i = 0; i < model.args().size(); ++i) {
    const ArgPlan& p = model.args()[i];
    if (p.source != ArgSource::STATE) continue;
    const auto& t = sig.args[i];
    const size_t bytes = Elements(t.dims) * tlaloc_triton::ByteWidth(t.dtype);
    std::vector<char> zeros(bytes, 0);
    HostInput host;
    host.data = zeros.data();
    host.byte_size = bytes;
    host.dtype = t.dtype;
    host.dims = t.dims;
    std::string err = PjrtBuffer::Upload(model.client(), host, &state_[p.name]);
    if (!err.empty()) {
      return Err(
          TRITONSERVER_ERROR_INTERNAL,
          "instance '" + Name() + "': state '" + p.name + "': " + err);
    }
    ++count;
    total += bytes;
  }
  if (count > 0) {
    std::ostringstream m;
    m << "tlaloc backend: instance '" << Name() << "': " << count << " state buffers ("
      << total / 1024 << " KiB) zeroed on the device";
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, m.str().c_str());
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
  const auto& plan = model.args();

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
    for (size_t a = 0; a < plan.size(); ++a) {
      if (plan[a].source != ArgSource::INPUT) continue;
      const auto& want = b.signature.args[a].dims;
      const auto& have = shapes[plan[a].input];
      match &= want == have || IsScalarAsOne(have, want);
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
      for (size_t k = 0; k < specs.size(); ++k) {
        for (size_t a = 0; a < plan.size(); ++a) {
          if (plan[a].source == ArgSource::INPUT && plan[a].input == k) {
            one += (one.empty() ? "" : ", ") + ShapeString(b.signature.args[a].dims);
          }
        }
      }
      want += (want.empty() ? "(" : "; (") + one + ")";
    }
    return Err(
        TRITONSERVER_ERROR_INVALID_ARG,
        "no compiled artifact takes input shapes (" + have + "); this model serves " + want);
  }

  std::vector<ExecuteArg> args(plan.size());
  for (size_t a = 0; a < plan.size(); ++a) {
    switch (plan[a].source) {
      case ArgSource::INPUT:
        args[a].host = host[plan[a].input];
        // PJRT gets the entry function's own shapes ([1] -> rank 0).
        args[a].host.dims = bucket->signature.args[a].dims;
        break;
      case ArgSource::WEIGHT:
        args[a].device = model.weight(a);
        break;
      case ArgSource::STATE:
        args[a].device = state_.at(plan[a].name).get();
        break;
    }
  }
  if (model.MaxBatchSize() > 0 && !shapes.empty() && !shapes[0].empty()) {
    *batch = static_cast<uint64_t>(shapes[0][0]);
  }

  *compute_start = NowNs();
  std::unique_ptr<PjrtResults> results;
  std::string err = bucket->executable->Execute(args, &results);
  *compute_end = NowNs();
  if (!err.empty()) {
    return Err(TRITONSERVER_ERROR_INTERNAL, bucket->file + ": " + err);
  }
  // The execution ran, so its state results replace the state it read, even
  // if sending the response fails below or the client asked for no output.
  const auto& sinks = model.results();
  for (size_t j = 0; j < sinks.size(); ++j) {
    if (sinks[j].sink == ResultSink::STATE) state_[sinks[j].name] = results->Release(j);
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
  for (size_t j = 0; j < sinks.size(); ++j) {
    if (sinks[j].sink != ResultSink::OUTPUT) continue;
    const TensorSpec& out = outs[sinks[j].output];
    if (requested_count > 0 && requested.count(out.name) == 0) continue;
    DType dtype;
    std::vector<int64_t> dims;
    err = results->Describe(j, &dtype, &dims);
    if (!err.empty()) return Err(TRITONSERVER_ERROR_INTERNAL, err);
    if (dtype != out.dtype) {
      return Err(
          TRITONSERVER_ERROR_INTERNAL,
          "output '" + out.name + "' came back as " + tlaloc_triton::TritonName(dtype) +
              ", expected " + tlaloc_triton::TritonName(out.dtype));
    }
    const size_t bytes = Elements(dims) * tlaloc_triton::ByteWidth(dtype);
    if (IsScalarAsOne(out.shape, dims)) dims = {1};
    TRITONBACKEND_Output* output = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(
        response, &output, out.name.c_str(),
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
            TRITONSERVER_ERROR_INTERNAL, "output '" + out.name +
                                             "': copying to GPU memory failed: " +
                                             cudaGetErrorString(e));
      }
#else
      return Err(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "output '" + out.name + "' was given GPU memory and this build has no CUDA");
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
