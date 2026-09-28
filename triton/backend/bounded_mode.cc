// Copyright 2026 Pedro N. Rodriguez
// SPDX-License-Identifier: Apache-2.0

#include "bounded_mode.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif

#include "stablehlo_text.h"

namespace triton { namespace backend { namespace tlaloc {

using tlaloc_triton::DType;
using tlaloc_triton::ExecuteArg;
using tlaloc_triton::PjrtResults;

namespace {

TRITONSERVER_Error*
Invalid(const std::string& msg)
{
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, msg.c_str());
}

TRITONSERVER_Error*
Internal(const std::string& msg)
{
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, msg.c_str());
}

uint64_t
NowNs()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int64_t
Elements(const std::vector<int64_t>& dims)
{
  int64_t n = 1;
  for (int64_t d : dims) n *= d;
  return n;
}

bool
InsideDir(const std::string& path)
{
  if (path.empty() || path[0] == '/') return false;
  std::stringstream parts(path);
  std::string part;
  while (std::getline(parts, part, '/')) {
    if (part == "..") return false;
  }
  return true;
}

// Refuses keys the reader does not know, as the Kotlin and Python readers do.
TRITONSERVER_Error*
OnlyKeys(triton::common::TritonJson::Value& obj, const std::set<std::string>& allowed, const std::string& at)
{
  std::vector<std::string> names;
  RETURN_IF_ERROR(obj.Members(&names));
  for (const std::string& n : names) {
    if (allowed.count(n) == 0) return Invalid(at + "unknown key '" + n + "'");
  }
  return nullptr;
}

TRITONSERVER_Error*
Str(triton::common::TritonJson::Value& obj, const char* key, std::string* out, const std::string& at)
{
  if (obj.MemberAsString(key, out) != nullptr) return Invalid(at + "field '" + key + "' is missing or not a string");
  return nullptr;
}

TRITONSERVER_Error*
Int(triton::common::TritonJson::Value& obj, const char* key, int64_t* out, const std::string& at)
{
  if (obj.MemberAsInt(key, out) != nullptr) return Invalid(at + "field '" + key + "' is missing or not an integer");
  return nullptr;
}

TRITONSERVER_Error*
ReadTensor(triton::common::TritonJson::Value& o, BoundedTensorSpec* t, const std::string& at)
{
  RETURN_IF_ERROR(OnlyKeys(o, {"name", "role", "dtype", "axes", "bound"}, at + "a tensor: "));
  RETURN_IF_ERROR(Str(o, "name", &t->name, at));
  RETURN_IF_ERROR(Str(o, "role", &t->role, at));
  if (t->role != "DATA" && t->role != "VALID_MASK" && t->role != "VALID_LENGTH") {
    return Invalid(at + "tensor '" + t->name + "' has unknown role '" + t->role + "'");
  }
  std::string dtype;
  RETURN_IF_ERROR(Str(o, "dtype", &dtype, at));
  t->dtype = tlaloc_triton::DTypeFromMlir(dtype);
  if (t->dtype != DType::F32 && t->dtype != DType::I32) {
    return Invalid(at + "tensor '" + t->name + "' has dtype '" + dtype + "'; a bounded program's tensors are f32 or i32");
  }
  if (o.Find("bound")) RETURN_IF_ERROR(Str(o, "bound", &t->bound, at));
  triton::common::TritonJson::Value axes;
  if (o.MemberAsArray("axes", &axes) != nullptr) return Invalid(at + "tensor '" + t->name + "' has no axes array");
  for (size_t i = 0; i < axes.ArraySize(); ++i) {
    triton::common::TritonJson::Value a;
    RETURN_IF_ERROR(axes.IndexAsObject(i, &a));
    RETURN_IF_ERROR(OnlyKeys(a, {"size", "bound"}, at + "an axis of '" + t->name + "': "));
    BoundedAxisSpec axis;
    const bool has_size = a.Find("size"), has_bound = a.Find("bound");
    if (has_size == has_bound) return Invalid(at + "an axis of '" + t->name + "' needs exactly one of size and bound");
    if (has_size) {
      RETURN_IF_ERROR(Int(a, "size", &axis.size, at));
      if (axis.size < 1) return Invalid(at + "an axis of '" + t->name + "' has size " + std::to_string(axis.size));
    } else {
      RETURN_IF_ERROR(Str(a, "bound", &axis.bound, at));
    }
    t->axes.push_back(axis);
  }
  return nullptr;
}

TRITONSERVER_Error*
InputToHost(
    TRITONBACKEND_Request* request, const std::string& name, TRITONSERVER_DataType* dtype,
    std::vector<int64_t>* shape, std::vector<char>* bytes)
{
  TRITONBACKEND_Input* input = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, name.c_str(), &input));
  const int64_t* s = nullptr;
  uint32_t rank = 0, buffers = 0;
  uint64_t size = 0;
  RETURN_IF_ERROR(TRITONBACKEND_InputProperties(input, nullptr, dtype, &s, &rank, &size, &buffers));
  shape->assign(s, s + rank);
  bytes->resize(size);
  size_t offset = 0;
  for (uint32_t b = 0; b < buffers; ++b) {
    const void* ptr = nullptr;
    uint64_t n = 0;
    TRITONSERVER_MemoryType mt = TRITONSERVER_MEMORY_CPU;
    int64_t mid = 0;
    RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(input, b, &ptr, &n, &mt, &mid));
    if (offset + n > size) return Invalid("input '" + name + "' buffers exceed its byte size");
    if (mt == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
      cudaError_t e = cudaMemcpy(bytes->data() + offset, ptr, n, cudaMemcpyDeviceToHost);
      if (e != cudaSuccess) return Internal("input '" + name + "': " + cudaGetErrorString(e));
#else
      return Invalid("input '" + name + "' is in GPU memory");
#endif
    } else {
      std::memcpy(bytes->data() + offset, ptr, n);
    }
    offset += n;
  }
  return nullptr;
}

TRITONSERVER_Error*
WriteOutput(
    TRITONBACKEND_Response* response, const std::string& name, TRITONSERVER_DataType dtype,
    const std::vector<int64_t>& dims, const void* data, size_t bytes)
{
  TRITONBACKEND_Output* output = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(
      response, &output, name.c_str(), dtype, dims.data(), static_cast<uint32_t>(dims.size())));
  void* buffer = nullptr;
  TRITONSERVER_MemoryType mt = TRITONSERVER_MEMORY_CPU;
  int64_t mid = 0;
  RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(output, &buffer, bytes, &mt, &mid));
  if (bytes == 0) return nullptr;
  if (mt == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
    cudaError_t e = cudaMemcpy(buffer, data, bytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return Internal("output '" + name + "': " + cudaGetErrorString(e));
#else
    return Invalid("output '" + name + "' was given GPU memory and this build has no CUDA");
#endif
  } else {
    std::memcpy(buffer, data, bytes);
  }
  return nullptr;
}

// Copies the leading `block` of a row-major array of `src_dims` into a
// row-major array of `dst_dims` (both at least `block` in every axis).
void
CopyBlock(
    const char* src, const std::vector<int64_t>& src_dims, char* dst, const std::vector<int64_t>& dst_dims,
    const std::vector<int64_t>& block, size_t width)
{
  const size_t rank = block.size();
  if (rank == 0) {
    std::memcpy(dst, src, width);
    return;
  }
  for (int64_t b : block) {
    if (b == 0) return;
  }
  std::vector<int64_t> ss(rank), ds(rank);
  int64_t sa = 1, da = 1;
  for (size_t i = rank; i-- > 0;) {
    ss[i] = sa;
    ds[i] = da;
    sa *= src_dims[i];
    da *= dst_dims[i];
  }
  std::vector<int64_t> idx(rank, 0);
  const size_t inner = static_cast<size_t>(block[rank - 1]) * width;
  while (true) {
    int64_t so = 0, dof = 0;
    for (size_t a = 0; a + 1 < rank; ++a) {
      so += idx[a] * ss[a];
      dof += idx[a] * ds[a];
    }
    std::memcpy(dst + dof * width, src + so * width, inner);
    size_t a = rank - 1;
    while (a-- > 0) {
      if (++idx[a] < block[a]) break;
      idx[a] = 0;
      if (a == 0) return;
    }
    if (rank == 1) return;
  }
}

}  // namespace

const BoundSpec*
BoundedModel::Bound(const std::string& name) const
{
  for (const BoundSpec& b : bounds_) {
    if (b.name == name) return &b;
  }
  return nullptr;
}

TRITONSERVER_Error*
BoundedModel::Load(
    const std::string& model_name, const std::string& version_dir, const std::string& manifest_file,
    triton::common::TritonJson::Value& config,
    const std::function<TRITONSERVER_Error*(std::shared_ptr<tlaloc_triton::PjrtClient>*)>& acquire,
    std::unique_ptr<BoundedModel>* out)
{
  std::unique_ptr<BoundedModel> m(new BoundedModel());
  m->name_ = model_name;
  m->version_dir_ = version_dir;
  if (!InsideDir(manifest_file)) {
    return Invalid(
        m->Where() + "bounded_manifest '" + manifest_file + "' must be a file inside the model version directory " +
        version_dir + " (no absolute path, no '..')");
  }
  const std::string path = JoinPath({version_dir, manifest_file});
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_NOT_FOUND, (m->Where() + "cannot read the bounded manifest " + path).c_str());
  }
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  RETURN_IF_ERROR(m->ReadManifest(text));
  RETURN_IF_ERROR(m->ReadConfig(config));
  RETURN_IF_ERROR(acquire(&m->client_));
  RETURN_IF_ERROR(m->CompileEntries());
  *out = std::move(m);
  return nullptr;
}

TRITONSERVER_Error*
BoundedModel::ReadManifest(const std::string& text)
{
  const std::string at = Where() + "bounded manifest: ";
  triton::common::TritonJson::Value doc;
  TRITONSERVER_Error* perr = doc.Parse(text);
  if (perr != nullptr) {
    std::string msg = TRITONSERVER_ErrorMessage(perr);
    TRITONSERVER_ErrorDelete(perr);
    return Invalid(at + "not valid JSON: " + msg);
  }
  std::string version;
  RETURN_IF_ERROR(Str(doc, "schemaVersion", &version, at));
  if (version != "tlaloc-bounded-v1") {
    return Invalid(at + "schemaVersion '" + version + "' is not tlaloc-bounded-v1");
  }
  RETURN_IF_ERROR(OnlyKeys(
      doc, {"schemaVersion", "name", "bounds", "padding", "inputs", "outputs", "entries", "paddingCheck"}, at));
  RETURN_IF_ERROR(Str(doc, "name", &program_, at));

  triton::common::TritonJson::Value padding;
  if (doc.MemberAsObject("padding", &padding) != nullptr) return Invalid(at + "no padding object");
  RETURN_IF_ERROR(OnlyKeys(padding, {"value"}, at + "padding: "));
  if (padding.MemberAsDouble("value", &padding_) != nullptr) return Invalid(at + "padding.value is not a number");

  triton::common::TritonJson::Value bounds;
  if (doc.MemberAsArray("bounds", &bounds) != nullptr || bounds.ArraySize() == 0) return Invalid(at + "no bounds");
  for (size_t i = 0; i < bounds.ArraySize(); ++i) {
    triton::common::TritonJson::Value b;
    RETURN_IF_ERROR(bounds.IndexAsObject(i, &b));
    RETURN_IF_ERROR(OnlyKeys(b, {"name", "max", "buckets"}, at + "a bound: "));
    BoundSpec spec;
    RETURN_IF_ERROR(Str(b, "name", &spec.name, at));
    RETURN_IF_ERROR(Int(b, "max", &spec.max, at));
    triton::common::TritonJson::Value buckets;
    if (b.MemberAsArray("buckets", &buckets) != nullptr) return Invalid(at + "bound '" + spec.name + "' has no buckets");
    for (size_t k = 0; k < buckets.ArraySize(); ++k) {
      int64_t v = 0;
      RETURN_IF_ERROR(buckets.IndexAsInt(k, &v));
      if (v < 1 || (!spec.buckets.empty() && v <= spec.buckets.back())) {
        return Invalid(at + "bound '" + spec.name + "': buckets must ascend strictly from 1");
      }
      spec.buckets.push_back(v);
    }
    if (spec.buckets.empty() || spec.buckets.back() != spec.max) {
      return Invalid(at + "bound '" + spec.name + "': the last bucket must be the max " + std::to_string(spec.max));
    }
    if (Bound(spec.name) != nullptr) return Invalid(at + "bound '" + spec.name + "' is declared twice");
    bounds_.push_back(spec);
  }

  auto check_axes = [&](const BoundedTensorSpec& t) -> TRITONSERVER_Error* {
    for (const BoundedAxisSpec& a : t.axes) {
      if (!a.bound.empty() && Bound(a.bound) == nullptr) {
        return Invalid(at + "tensor '" + t.name + "' uses undeclared bound '" + a.bound + "'");
      }
    }
    if (t.role != "DATA" && Bound(t.bound) == nullptr) {
      return Invalid(at + t.role + " input '" + t.name + "' must name a declared bound");
    }
    return nullptr;
  };
  triton::common::TritonJson::Value inputs, outputs, entries;
  if (doc.MemberAsArray("inputs", &inputs) != nullptr) return Invalid(at + "no inputs");
  for (size_t i = 0; i < inputs.ArraySize(); ++i) {
    triton::common::TritonJson::Value o;
    RETURN_IF_ERROR(inputs.IndexAsObject(i, &o));
    BoundedTensorSpec t;
    RETURN_IF_ERROR(ReadTensor(o, &t, at));
    RETURN_IF_ERROR(check_axes(t));
    inputs_.push_back(t);
  }
  if (doc.MemberAsArray("outputs", &outputs) != nullptr || outputs.ArraySize() != 1) {
    return Invalid(at + "the tlaloc backend serves a bounded program with exactly one output");
  }
  {
    triton::common::TritonJson::Value o;
    RETURN_IF_ERROR(outputs.IndexAsObject(0, &o));
    RETURN_IF_ERROR(ReadTensor(o, &output_, at));
    RETURN_IF_ERROR(check_axes(output_));
    if (output_.role != "DATA" || output_.dtype != DType::F32) return Invalid(at + "the output must be an f32 DATA tensor");
  }

  if (doc.MemberAsArray("entries", &entries) != nullptr) return Invalid(at + "no entries");
  std::set<std::map<std::string, int64_t>> seen;
  for (size_t i = 0; i < entries.ArraySize(); ++i) {
    triton::common::TritonJson::Value o;
    RETURN_IF_ERROR(entries.IndexAsObject(i, &o));
    RETURN_IF_ERROR(OnlyKeys(o, {"id", "sizes", "entryPoint", "bodyPath", "bodyHash", "programPath"}, at + "an entry: "));
    BoundedEntrySpec e;
    RETURN_IF_ERROR(Str(o, "id", &e.id, at));
    RETURN_IF_ERROR(Str(o, "entryPoint", &e.entry_point, at));
    RETURN_IF_ERROR(Str(o, "bodyPath", &e.body_path, at));
    if (!InsideDir(e.body_path)) return Invalid(at + "entry '" + e.id + "' has body path '" + e.body_path + "' outside the directory");
    triton::common::TritonJson::Value sizes;
    if (o.MemberAsObject("sizes", &sizes) != nullptr) return Invalid(at + "entry '" + e.id + "' has no sizes");
    std::vector<std::string> names;
    RETURN_IF_ERROR(sizes.Members(&names));
    for (const std::string& n : names) {
      const BoundSpec* b = Bound(n);
      int64_t v = 0;
      RETURN_IF_ERROR(Int(sizes, n.c_str(), &v, at));
      if (b == nullptr || std::find(b->buckets.begin(), b->buckets.end(), v) == b->buckets.end()) {
        return Invalid(at + "entry '" + e.id + "' is at " + n + " = " + std::to_string(v) + ", not a declared bucket");
      }
      e.sizes[n] = v;
    }
    if (e.sizes.size() != bounds_.size() || !seen.insert(e.sizes).second) {
      return Invalid(at + "entry '" + e.id + "' does not give one bucket per bound, or repeats another entry");
    }
    entries_.push_back(e);
  }
  size_t combos = 1;
  for (const BoundSpec& b : bounds_) combos *= b.buckets.size();
  if (entries_.size() != combos) {
    return Invalid(
        at + std::to_string(entries_.size()) + " entries for " + std::to_string(combos) +
        " bucket combinations; every combination needs one");
  }
  return nullptr;
}

TRITONSERVER_Error*
BoundedModel::ReadConfig(triton::common::TritonJson::Value& config)
{
  int64_t max_batch = 0;
  config.MemberAsInt("max_batch_size", &max_batch);
  if (max_batch != 0) {
    return Invalid(
        Where() + "a bounded_manifest model takes max_batch_size: 0; a batch axis, if the program has one, "
        "is one of its bounded axes");
  }
  auto check = [&](const char* key, const std::vector<const BoundedTensorSpec*>& want) -> TRITONSERVER_Error* {
    triton::common::TritonJson::Value list;
    if (config.MemberAsArray(key, &list) != nullptr || list.ArraySize() != want.size()) {
      return Invalid(
          Where() + "config.pbtxt must have " + std::to_string(want.size()) + " " + key +
          "(s), the manifest's " + (std::string(key) == "input" ? "DATA inputs" : "output"));
    }
    for (size_t i = 0; i < want.size(); ++i) {
      triton::common::TritonJson::Value io, dims;
      RETURN_IF_ERROR(list.IndexAsObject(i, &io));
      std::string name, dtype;
      RETURN_IF_ERROR(io.MemberAsString("name", &name));
      RETURN_IF_ERROR(io.MemberAsString("data_type", &dtype));
      const BoundedTensorSpec& t = *want[i];
      if (name != t.name) {
        return Invalid(Where() + key + " " + std::to_string(i) + " is '" + name + "'; the manifest's is '" + t.name + "'");
      }
      if (tlaloc_triton::DTypeFromTriton(dtype) != t.dtype) {
        return Invalid(Where() + key + " '" + name + "' is " + dtype + "; the manifest says " + tlaloc_triton::TritonName(t.dtype));
      }
      if (io.MemberAsArray("dims", &dims) != nullptr || dims.ArraySize() != t.axes.size()) {
        return Invalid(Where() + key + " '" + name + "' must have " + std::to_string(t.axes.size()) + " dims");
      }
      for (size_t k = 0; k < t.axes.size(); ++k) {
        int64_t d = 0;
        RETURN_IF_ERROR(dims.IndexAsInt(k, &d));
        const int64_t want_d = t.axes[k].bound.empty() ? t.axes[k].size : -1;
        if (d != want_d) {
          return Invalid(
              Where() + key + " '" + name + "' dim " + std::to_string(k) + " is " + std::to_string(d) + "; it must be " +
              std::to_string(want_d) + (want_d == -1 ? " (bounded by " + t.axes[k].bound + ")" : ""));
        }
      }
    }
    return nullptr;
  };
  std::vector<const BoundedTensorSpec*> data;
  for (const BoundedTensorSpec& t : inputs_) {
    if (t.role == "DATA") data.push_back(&t);
  }
  RETURN_IF_ERROR(check("input", data));
  RETURN_IF_ERROR(check("output", {&output_}));
  return nullptr;
}

TRITONSERVER_Error*
BoundedModel::CompileEntries()
{
  for (BoundedEntrySpec& e : entries_) {
    auto hit = executables_.find(e.body_path);
    const std::string path = JoinPath({version_dir_, e.body_path});
    std::ifstream in(path, std::ios::binary);
    if (!in) return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_NOT_FOUND, (Where() + "cannot read " + path).c_str());
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<tlaloc_triton::FunctionSignature> functions;
    tlaloc_triton::FunctionSignature sig;
    std::string perr;
    if (!tlaloc_triton::ListFunctions(text, &functions, &perr) ||
        !tlaloc_triton::SelectEntry(functions, e.entry_point, &sig, &perr)) {
      return Invalid(Where() + path + ": " + perr);
    }
    // The body's signature is the manifest's tensors at this entry's buckets.
    auto dims_at = [&](const BoundedTensorSpec& t) {
      std::vector<int64_t> d;
      for (const BoundedAxisSpec& a : t.axes) d.push_back(a.bound.empty() ? a.size : e.sizes.at(a.bound));
      return d;
    };
    if (sig.args.size() != inputs_.size() || sig.results.size() != 1) {
      return Invalid(Where() + path + ": @" + sig.name + " does not have the manifest's inputs and one output");
    }
    for (size_t i = 0; i < inputs_.size(); ++i) {
      if (sig.args[i].dtype != inputs_[i].dtype || sig.args[i].dims != dims_at(inputs_[i])) {
        return Invalid(
            Where() + path + ": argument " + std::to_string(i) + " ('" + inputs_[i].name + "') is " + sig.args[i].text +
            "; the manifest says " + tlaloc_triton::TritonName(inputs_[i].dtype) +
            tlaloc_triton::ShapeString(dims_at(inputs_[i])) + " at entry " + e.id);
      }
    }
    if (sig.results[0].dtype != output_.dtype || sig.results[0].dims != dims_at(output_)) {
      return Invalid(Where() + path + ": the result is " + sig.results[0].text + ", not the manifest's output at entry " + e.id);
    }
    if (hit == executables_.end()) {
      const uint64_t t0 = NowNs();
      std::unique_ptr<tlaloc_triton::PjrtExecutable> exe;
      std::string err = client_->Compile(tlaloc_triton::PrepareForXla(text, sig.name), &exe);
      if (!err.empty()) return Invalid(Where() + path + ": " + err);
      if (exe->num_outputs() != 1) return Internal(Where() + path + " compiled to the wrong number of outputs");
      std::ostringstream m;
      m << "tlaloc backend: " << Where() << "compiled bounded entry " << e.id << " (" << e.body_path << ") in "
        << (NowNs() - t0) / 1000000 << " ms";
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, m.str().c_str());
      hit = executables_.emplace(e.body_path, std::move(exe)).first;
    }
    e.executable = hit->second.get();
  }
  std::ostringstream m;
  m << "tlaloc backend: " << Where() << "bounded program '" << program_ << "': " << executables_.size()
    << " bodies for " << entries_.size() << " bucket combinations of";
  for (const BoundSpec& b : bounds_) {
    m << " " << b.name << " <= " << b.max << " (buckets";
    for (int64_t v : b.buckets) m << " " << v;
    m << ")";
  }
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, m.str().c_str());
  return nullptr;
}

TRITONSERVER_Error*
BoundedModel::Run(
    TRITONBACKEND_Request* request, TRITONBACKEND_Response* response, uint64_t* compute_start,
    uint64_t* compute_end) const
{
  struct Data {
    std::vector<int64_t> dims;
    std::vector<char> bytes;
  };
  std::map<std::string, int64_t> sizes;
  std::vector<Data> data;
  for (const BoundedTensorSpec& t : inputs_) {
    if (t.role != "DATA") continue;
    Data d;
    TRITONSERVER_DataType dt;
    RETURN_IF_ERROR(InputToHost(request, t.name, &dt, &d.dims, &d.bytes));
    if (tlaloc_triton::DTypeFromTriton(TRITONSERVER_DataTypeString(dt)) != t.dtype) {
      return Invalid(Where() + "input '" + t.name + "' is " + TRITONSERVER_DataTypeString(dt));
    }
    if (d.dims.size() != t.axes.size()) {
      return Invalid(Where() + "input '" + t.name + "' has rank " + std::to_string(d.dims.size()) + ", expected " + std::to_string(t.axes.size()));
    }
    for (size_t k = 0; k < t.axes.size(); ++k) {
      const BoundedAxisSpec& a = t.axes[k];
      const int64_t n = d.dims[k];
      if (a.bound.empty()) {
        if (n != a.size) {
          return Invalid(Where() + "input '" + t.name + "' dim " + std::to_string(k) + " is " + std::to_string(n) + ", fixed at " + std::to_string(a.size));
        }
        continue;
      }
      const BoundSpec* b = Bound(a.bound);
      if (n < 1 || n > b->max) {
        return Invalid(
            Where() + "input '" + t.name + "' dim " + std::to_string(k) + " is " + std::to_string(n) + ", outside 1.." +
            std::to_string(b->max) + " of bound " + b->name);
      }
      auto it = sizes.emplace(a.bound, n).first;
      if (it->second != n) {
        return Invalid(
            Where() + "bound " + a.bound + " is " + std::to_string(it->second) + " on one axis and " + std::to_string(n) +
            " on input '" + t.name + "' dim " + std::to_string(k) + "; every axis of one bound has one size");
      }
    }
    const size_t want = static_cast<size_t>(Elements(d.dims)) * tlaloc_triton::ByteWidth(t.dtype);
    if (d.bytes.size() != want) return Invalid(Where() + "input '" + t.name + "' has " + std::to_string(d.bytes.size()) + " bytes, expected " + std::to_string(want));
    data.push_back(std::move(d));
  }
  std::map<std::string, int64_t> buckets;
  for (const BoundSpec& b : bounds_) {
    auto it = sizes.find(b.name);
    if (it == sizes.end()) return Internal(Where() + "no input sizes bound " + b.name);
    buckets[b.name] = *std::find_if(b.buckets.begin(), b.buckets.end(), [&](int64_t v) { return v >= it->second; });
  }
  const BoundedEntrySpec* entry = nullptr;
  for (const BoundedEntrySpec& e : entries_) {
    if (e.sizes == buckets) entry = &e;
  }
  if (entry == nullptr) return Internal(Where() + "no entry for the chosen buckets");

  // Stage the arguments on the host, in the entry's parameter order.
  std::vector<std::vector<char>> staged(inputs_.size());
  std::vector<ExecuteArg> args(inputs_.size());
  size_t next_data = 0;
  for (size_t i = 0; i < inputs_.size(); ++i) {
    const BoundedTensorSpec& t = inputs_[i];
    std::vector<int64_t> dims;
    for (const BoundedAxisSpec& a : t.axes) dims.push_back(a.bound.empty() ? a.size : buckets.at(a.bound));
    const size_t width = tlaloc_triton::ByteWidth(t.dtype);
    std::vector<char>& buf = staged[i];
    buf.resize(static_cast<size_t>(Elements(dims)) * width);
    if (t.role == "DATA") {
      const Data& d = data[next_data++];
      if (padding_ == 0.0) {
        std::fill(buf.begin(), buf.end(), 0);
      } else {
        for (size_t k = 0; k < buf.size() / width; ++k) {
          if (t.dtype == DType::F32) {
            const float v = static_cast<float>(padding_);
            std::memcpy(buf.data() + k * width, &v, width);
          } else {
            const int32_t v = static_cast<int32_t>(padding_);
            std::memcpy(buf.data() + k * width, &v, width);
          }
        }
      }
      CopyBlock(d.bytes.data(), d.dims, buf.data(), dims, d.dims, width);
    } else if (t.role == "VALID_MASK") {
      const int64_t n = sizes.at(t.bound);
      for (int64_t k = 0; k < dims[0]; ++k) {
        const float v = k < n ? 1.0f : 0.0f;
        std::memcpy(buf.data() + k * width, &v, width);
      }
    } else {
      const float v = static_cast<float>(sizes.at(t.bound));
      std::memcpy(buf.data(), &v, width);
    }
    args[i].host.data = buf.data();
    args[i].host.byte_size = buf.size();
    args[i].host.dtype = t.dtype;
    args[i].host.dims = dims;
  }

  std::vector<int64_t> out_dims, real_dims;
  for (const BoundedAxisSpec& a : output_.axes) {
    out_dims.push_back(a.bound.empty() ? a.size : buckets.at(a.bound));
    real_dims.push_back(a.bound.empty() ? a.size : sizes.at(a.bound));
  }
  std::vector<char> out(static_cast<size_t>(Elements(out_dims)) * sizeof(float));
  *compute_start = NowNs();
  std::unique_ptr<PjrtResults> results;
  std::string err = const_cast<tlaloc_triton::PjrtExecutable*>(entry->executable)->Execute(args, &results);
  if (err.empty()) err = results->CopyToHost(0, out.data(), out.size());
  *compute_end = NowNs();
  if (!err.empty()) return Internal(Where() + "entry " + entry->id + ": " + err);

  std::vector<char> real(static_cast<size_t>(Elements(real_dims)) * sizeof(float));
  CopyBlock(out.data(), out_dims, real.data(), real_dims, real_dims, sizeof(float));
  return WriteOutput(response, output_.name, TRITONSERVER_TYPE_FP32, real_dims, real.data(), real.size());
}

}}}  // namespace triton::backend::tlaloc
