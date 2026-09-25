// Copyright 2026 Pedro N. Rodriguez
// SPDX-License-Identifier: Apache-2.0

#include "sequence_mode.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif

#include "stablehlo_text.h"
#include "triton/backend/backend_common.h"

namespace triton { namespace backend { namespace tlaloc {

using tlaloc_triton::DType;
using tlaloc_triton::ExecuteArg;
using tlaloc_triton::HostInput;
using tlaloc_triton::PjrtBuffer;
using tlaloc_triton::PjrtResults;

namespace {

TRITONSERVER_Error*
Err(TRITONSERVER_Error_Code code, const std::string& msg)
{
  return TRITONSERVER_ErrorNew(code, msg.c_str());
}

TRITONSERVER_Error*
Invalid(const std::string& msg)
{
  return Err(TRITONSERVER_ERROR_INVALID_ARG, msg);
}

uint64_t
NowNs()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

uint64_t
Elements(const std::vector<int64_t>& dims)
{
  uint64_t n = 1;
  for (int64_t d : dims) n *= static_cast<uint64_t>(d);
  return n;
}

std::string
Join(const std::vector<int64_t>& dims)
{
  return tlaloc_triton::ShapeString(dims);
}

// An unsigned integer member that the model configuration JSON may carry as
// a number or, for 64-bit protobuf fields, as a string.
bool
MemberAsU64(triton::common::TritonJson::Value& obj, const char* key, uint64_t* out)
{
  triton::common::TritonJson::Value v;
  if (!obj.Find(key, &v)) return false;
  std::string s;
  if (v.AsString(&s) == nullptr) {
    char* end = nullptr;
    unsigned long long x = std::strtoull(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0') return false;
    *out = x;
    return true;
  }
  uint64_t u = 0;
  if (obj.MemberAsUInt(key, &u) == nullptr) {
    *out = u;
    return true;
  }
  return false;
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

TRITONSERVER_Error*
IntMember(triton::common::TritonJson::Value& obj, const char* key, int* out, const std::string& at)
{
  int64_t v = 0;
  if (obj.MemberAsInt(key, &v) != nullptr) {
    return Invalid(at + "field '" + key + "' is missing or not an integer");
  }
  *out = static_cast<int>(v);
  return nullptr;
}

TRITONSERVER_Error*
StrMember(triton::common::TritonJson::Value& obj, const char* key, std::string* out, const std::string& at)
{
  if (obj.MemberAsString(key, out) != nullptr) {
    return Invalid(at + "field '" + key + "' is missing or not a string");
  }
  return nullptr;
}

TRITONSERVER_Error*
ReadSlot(triton::common::TritonJson::Value& o, SlotSpec* slot, const std::string& at)
{
  RETURN_IF_ERROR(StrMember(o, "name", &slot->name, at));
  RETURN_IF_ERROR(StrMember(o, "role", &slot->role, at));
  triton::common::TritonJson::Value type, dims;
  if (o.MemberAsObject("type", &type) != nullptr || type.MemberAsArray("dims", &dims) != nullptr) {
    return Invalid(at + "slot '" + slot->name + "' has no type.dims");
  }
  std::string dtype;
  RETURN_IF_ERROR(StrMember(type, "dtype", &dtype, at));
  slot->dtype = tlaloc_triton::DTypeFromMlir(dtype);
  if (slot->dtype == DType::UNSUPPORTED) {
    return Invalid(at + "slot '" + slot->name + "' has dtype " + dtype + ", which the tlaloc backend does not serve");
  }
  for (size_t i = 0; i < dims.ArraySize(); ++i) {
    int64_t d = 0;
    if (dims.IndexAsInt(i, &d) != nullptr) return Invalid(at + "slot '" + slot->name + "' has a non-integer dim");
    slot->dims.push_back(d);
  }
  return nullptr;
}

// Copies a request input into host memory, whatever memory it arrived in.
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
      if (e != cudaSuccess) {
        return Err(TRITONSERVER_ERROR_INTERNAL, "input '" + name + "': " + cudaGetErrorString(e));
      }
#else
      return Err(TRITONSERVER_ERROR_UNSUPPORTED, "input '" + name + "' is in GPU memory");
#endif
    } else {
      std::memcpy(bytes->data() + offset, ptr, n);
    }
    offset += n;
  }
  return nullptr;
}

// The first element of a control tensor as an unsigned integer.
TRITONSERVER_Error*
ControlValue(TRITONBACKEND_Request* request, const std::string& name, uint64_t* out)
{
  TRITONSERVER_DataType dt;
  std::vector<int64_t> shape;
  std::vector<char> bytes;
  RETURN_IF_ERROR(InputToHost(request, name, &dt, &shape, &bytes));
  switch (dt) {
    case TRITONSERVER_TYPE_INT32:
    case TRITONSERVER_TYPE_UINT32:
      if (bytes.size() < 4) break;
      {
        uint32_t v;
        std::memcpy(&v, bytes.data(), 4);
        *out = v;
      }
      return nullptr;
    case TRITONSERVER_TYPE_INT64:
    case TRITONSERVER_TYPE_UINT64:
      if (bytes.size() < 8) break;
      std::memcpy(out, bytes.data(), 8);
      return nullptr;
    case TRITONSERVER_TYPE_BOOL:
      if (bytes.empty()) break;
      *out = bytes[0] != 0;
      return nullptr;
    default:
      break;
  }
  return Invalid(
      "control input '" + name + "' is " + TRITONSERVER_DataTypeString(dt) + " with " +
      std::to_string(bytes.size()) + " bytes; expected one INT32, INT64, UINT64 or BOOL value");
}

const std::set<std::string> kRequestRoles = {
    "TOKEN_IDS", "POSITIONS", "BLOCK_TABLES", "SEQ_LENS", "SLOT_MAPPING"};

}  // namespace

// ---------------------------------------------------------------------------
// SequenceModel

TRITONSERVER_Error*
SequenceModel::Load(
    const std::string& model_name, const std::string& version_dir,
    const std::string& manifest_file, triton::common::TritonJson::Value& config,
    const std::function<TRITONSERVER_Error*(std::shared_ptr<tlaloc_triton::PjrtClient>*)>& acquire,
    std::unique_ptr<SequenceModel>* out)
{
  std::unique_ptr<SequenceModel> m(new SequenceModel());
  m->name_ = model_name;
  m->version_dir_ = version_dir;
  if (!InsideDir(manifest_file)) {
    return Invalid(
        m->Where() + "serving_manifest '" + manifest_file + "' must be a file inside the model "
        "version directory " + version_dir + " (no absolute path, no '..')");
  }
  m->manifest_path_ = JoinPath({version_dir, manifest_file});
  RETURN_IF_ERROR(m->ReadConfig(config));
  std::ifstream in(m->manifest_path_, std::ios::binary);
  if (!in) {
    return Err(TRITONSERVER_ERROR_NOT_FOUND, m->Where() + "cannot read the serving manifest " + m->manifest_path_);
  }
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  RETURN_IF_ERROR(m->ReadManifest(text));
  RETURN_IF_ERROR(acquire(&m->client_));
  RETURN_IF_ERROR(m->CompileEntries());
  RETURN_IF_ERROR(m->UploadWeights());
  *out = std::move(m);
  return nullptr;
}

TRITONSERVER_Error*
SequenceModel::ReadConfig(triton::common::TritonJson::Value& config)
{
  if (config.MemberAsInt("max_batch_size", &max_batch_size_) != nullptr || max_batch_size_ < 1) {
    return Invalid(
        Where() + "a serving_manifest model needs max_batch_size >= 1: each request is one "
        "sequence's step, and the sequence batcher batches the steps of different sequences");
  }
  auto one_io = [&](const char* key, std::string* name, const char* want_type,
                    const char* role) -> TRITONSERVER_Error* {
    triton::common::TritonJson::Value ios, io;
    if (!config.Find(key, &ios) || ios.ArraySize() != 1) {
      return Invalid(
          Where() + "a serving_manifest model declares exactly one " + key + " (" + role + ")");
    }
    RETURN_IF_ERROR(ios.IndexAsObject(0, &io));
    RETURN_IF_ERROR(io.MemberAsString("name", name));
    std::string dt;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &dt));
    if (dt != want_type) {
      return Invalid(Where() + key + " '" + *name + "' is " + dt + "; it must be " + want_type);
    }
    return nullptr;
  };
  RETURN_IF_ERROR(one_io("input", &tokens_input_, "TYPE_INT32", "the token ids"));
  RETURN_IF_ERROR(one_io("output", &logits_output_, "TYPE_FP32", "the last token's logits"));
  {
    triton::common::TritonJson::Value ios, io;
    config.Find("input", &ios);
    ios.IndexAsObject(0, &io);
    std::vector<int64_t> dims;
    RETURN_IF_ERROR(ParseShape(io, "dims", &dims));
    if (dims != std::vector<int64_t>{-1}) {
      return Invalid(
          Where() + "input '" + tokens_input_ + "' has dims " + Join(dims) +
          "; it must be [ -1 ]: the token ids a request appends to its sequence");
    }
  }

  triton::common::TritonJson::Value sb;
  if (!config.Find("sequence_batching", &sb)) {
    return Invalid(
        Where() + "a serving_manifest model needs sequence_batching: the backend keeps each "
        "sequence's KV pages by correlation ID, which only the sequence batcher provides");
  }
  if (sb.Find("direct")) {
    return Invalid(
        Where() + "sequence_batching uses the direct strategy; use oldest. The backend keys "
        "state by correlation ID, not by batch slot, and the oldest strategy lets the decode "
        "steps of several sequences run as one batch");
  }
  triton::common::TritonJson::Value controls;
  if (sb.Find("control_input", &controls)) {
    for (size_t i = 0; i < controls.ArraySize(); ++i) {
      triton::common::TritonJson::Value c, kinds;
      RETURN_IF_ERROR(controls.IndexAsObject(i, &c));
      std::string name;
      RETURN_IF_ERROR(c.MemberAsString("name", &name));
      if (!c.Find("control", &kinds)) continue;
      for (size_t k = 0; k < kinds.ArraySize(); ++k) {
        triton::common::TritonJson::Value ctl;
        RETURN_IF_ERROR(kinds.IndexAsObject(k, &ctl));
        std::string kind;
        RETURN_IF_ERROR(ctl.MemberAsString("kind", &kind));
        if (kind == "CONTROL_SEQUENCE_START" || kind == "CONTROL_SEQUENCE_END") {
          if (!ctl.Find("int32_false_true")) {
            return Invalid(
                Where() + "control input '" + name + "' (" + kind + ") must use "
                "int32_false_true: [ 0, 1 ]");
          }
          (kind == "CONTROL_SEQUENCE_START" ? start_input_ : end_input_) = name;
        } else if (kind == "CONTROL_SEQUENCE_CORRID") {
          std::string dt;
          ctl.MemberAsString("data_type", &dt);
          if (dt != "TYPE_UINT64" && dt != "TYPE_INT64") {
            return Invalid(
                Where() + "control input '" + name + "' (CONTROL_SEQUENCE_CORRID) is " + dt +
                "; the backend keys sequences by an integer correlation ID (TYPE_UINT64)");
          }
          corrid_input_ = name;
        }
      }
    }
  }
  if (start_input_.empty() || end_input_.empty() || corrid_input_.empty()) {
    return Invalid(
        Where() + "sequence_batching must declare control inputs for CONTROL_SEQUENCE_START, "
        "CONTROL_SEQUENCE_END and CONTROL_SEQUENCE_CORRID; missing:" +
        (start_input_.empty() ? " START" : "") + (end_input_.empty() ? " END" : "") +
        (corrid_input_.empty() ? " CORRID" : ""));
  }
  uint64_t idle_us = 1000000;  // Triton's default
  MemberAsU64(sb, "max_sequence_idle_microseconds", &idle_us);
  idle_ns_ = idle_us * 1000;
  return nullptr;
}

TRITONSERVER_Error*
SequenceModel::ReadManifest(const std::string& text)
{
  const std::string at = Where() + manifest_path_ + ": ";
  triton::common::TritonJson::Value doc;
  TRITONSERVER_Error* perr = doc.Parse(text);
  if (perr != nullptr) {
    std::string msg = TRITONSERVER_ErrorMessage(perr);
    TRITONSERVER_ErrorDelete(perr);
    return Invalid(at + "not valid JSON: " + msg);
  }
  std::string version;
  RETURN_IF_ERROR(StrMember(doc, "schemaVersion", &version, at));
  if (version != "tlaloc-serving-v1" && version != "tlaloc-serving-v2") {
    return Invalid(
        at + "schemaVersion '" + version + "' is not tlaloc-serving-v1 or tlaloc-serving-v2");
  }
  triton::common::TritonJson::Value model;
  if (doc.MemberAsObject("model", &model) != nullptr) return Invalid(at + "no 'model' object");
  RETURN_IF_ERROR(IntMember(model, "vocabSize", &vocab_, at));
  RETURN_IF_ERROR(IntMember(model, "numBlocks", &num_blocks_, at));
  RETURN_IF_ERROR(IntMember(model, "blockSize", &block_size_, at));
  triton::common::TritonJson::Value quant;
  if (model.Find("kvQuant", &quant) && !quant.IsNull()) {
    return Invalid(
        at + "the artifact's KV pools are quantized (kvQuant); a serving_manifest model "
        "serves float pools only");
  }
  if (num_blocks_ < 2) {
    return Invalid(
        at + "numBlocks " + std::to_string(num_blocks_) + " leaves no page for a sequence: "
        "page 0 is the padding page");
  }

  triton::common::TritonJson::Value entries;
  if (doc.MemberAsArray("entries", &entries) != nullptr || entries.ArraySize() == 0) {
    return Invalid(at + "no entries");
  }
  for (size_t i = 0; i < entries.ArraySize(); ++i) {
    triton::common::TritonJson::Value e;
    RETURN_IF_ERROR(entries.IndexAsObject(i, &e));
    ServingEntrySpec s;
    std::string kind;
    RETURN_IF_ERROR(StrMember(e, "kind", &kind, at));
    if (kind != "decode" && kind != "prefill") return Invalid(at + "entry kind '" + kind + "'");
    s.prefill = kind == "prefill";
    RETURN_IF_ERROR(IntMember(e, "batch", &s.batch, at));
    RETURN_IF_ERROR(IntMember(e, "context", &s.context, at));
    RETURN_IF_ERROR(IntMember(e, "tokensPerSeq", &s.tokens_per_seq, at));
    RETURN_IF_ERROR(IntMember(e, "maxBlocksPerSeq", &s.max_blocks, at));
    RETURN_IF_ERROR(StrMember(e, "bodyPath", &s.body_path, at));
    RETURN_IF_ERROR(StrMember(e, "entryPoint", &s.entry_point, at));
    s.id = kind + "_b" + std::to_string(s.batch) + "_c" + std::to_string(s.context);
    if (s.prefill && version == "tlaloc-serving-v1") {
      return Invalid(at + "a tlaloc-serving-v1 manifest has a prefill entry " + s.id);
    }
    if (!InsideDir(s.body_path)) {
      return Invalid(at + "entry " + s.id + " names body '" + s.body_path + "' outside the version directory");
    }
    const std::string eat = at + "entry " + s.id + ": ";
    triton::common::TritonJson::Value ins, outs, pairs;
    if (e.MemberAsArray("inputs", &ins) != nullptr || e.MemberAsArray("outputs", &outs) != nullptr ||
        e.MemberAsArray("donationPairs", &pairs) != nullptr) {
      return Invalid(eat + "inputs, outputs and donationPairs are required");
    }
    for (size_t k = 0; k < ins.ArraySize(); ++k) {
      triton::common::TritonJson::Value o;
      RETURN_IF_ERROR(ins.IndexAsObject(k, &o));
      SlotSpec slot;
      RETURN_IF_ERROR(ReadSlot(o, &slot, eat));
      s.inputs.push_back(slot);
    }
    for (size_t k = 0; k < outs.ArraySize(); ++k) {
      triton::common::TritonJson::Value o;
      RETURN_IF_ERROR(outs.IndexAsObject(k, &o));
      SlotSpec slot;
      RETURN_IF_ERROR(ReadSlot(o, &slot, eat));
      s.outputs.push_back(slot);
    }
    s.replaces.assign(s.outputs.size(), -1);
    for (size_t k = 0; k < pairs.ArraySize(); ++k) {
      triton::common::TritonJson::Value p;
      RETURN_IF_ERROR(pairs.IndexAsArray(k, &p));
      int64_t a = -1, b = -1;
      if (p.ArraySize() != 2 || p.IndexAsInt(0, &a) != nullptr || p.IndexAsInt(1, &b) != nullptr ||
          a < 0 || b < 0 || a >= static_cast<int64_t>(s.inputs.size()) ||
          b >= static_cast<int64_t>(s.outputs.size())) {
        return Invalid(eat + "donation pair " + std::to_string(k) + " is not [input, output]");
      }
      s.replaces[b] = static_cast<int>(a);
    }

    // The request-side slots: each role once, int32, at the entry's shapes.
    const int B = s.batch, T = s.tokens_per_seq, M = s.max_blocks;
    const std::map<std::string, std::vector<int64_t>> want = {
        {"TOKEN_IDS", {B, T}}, {"POSITIONS", {B, T}}, {"BLOCK_TABLES", {B, M}},
        {"SEQ_LENS", {B}}, {"SLOT_MAPPING", {int64_t(B) * T}}};
    std::set<std::string> seen;
    for (const SlotSpec& slot : s.inputs) {
      if (kRequestRoles.count(slot.role)) {
        if (!seen.insert(slot.role).second) return Invalid(eat + "role " + slot.role + " appears twice");
        if (slot.dtype != DType::I32 || slot.dims != want.at(slot.role)) {
          return Invalid(
              eat + "input '" + slot.name + "' (" + slot.role + ") is " +
              tlaloc_triton::TritonName(slot.dtype) + Join(slot.dims) + "; expected INT32" +
              Join(want.at(slot.role)));
        }
      } else if (slot.role != "KV_POOL_IN" && slot.role != "WEIGHT") {
        return Invalid(eat + "input '" + slot.name + "' has role " + slot.role + ", which is not an input role");
      }
    }
    if (seen.size() != kRequestRoles.size()) {
      return Invalid(eat + "lacks one of TOKEN_IDS, POSITIONS, BLOCK_TABLES, SEQ_LENS, SLOT_MAPPING");
    }
    int logits = 0;
    for (size_t j = 0; j < s.outputs.size(); ++j) {
      const SlotSpec& slot = s.outputs[j];
      if (slot.role == "LOGITS") {
        ++logits;
        if (slot.dtype != DType::F32 || slot.dims != std::vector<int64_t>{B, 1, vocab_}) {
          return Invalid(
              eat + "LOGITS '" + slot.name + "' is " + tlaloc_triton::TritonName(slot.dtype) +
              Join(slot.dims) + "; expected FP32" + Join({B, 1, vocab_}) +
              " (a tlaloc-serving-v2 artifact returns each sequence's last-token logits)");
        }
      } else if (slot.role == "KV_POOL_OUT") {
        const int i = s.replaces[j];
        if (i < 0 || s.inputs[i].role != "KV_POOL_IN" || s.inputs[i].dtype != slot.dtype ||
            s.inputs[i].dims != slot.dims) {
          return Invalid(eat + "KV_POOL_OUT '" + slot.name + "' is not paired with a KV_POOL_IN of its type");
        }
      } else {
        return Invalid(eat + "output '" + slot.name + "' has role " + slot.role);
      }
    }
    if (logits != 1) return Invalid(eat + "needs exactly one LOGITS output");
    if (s.prefill ? T != s.context : T != 1) {
      return Invalid(eat + "tokensPerSeq " + std::to_string(T) + " does not fit its kind");
    }
    if (int64_t(M) * block_size_ < s.context) {
      return Invalid(eat + "maxBlocksPerSeq does not cover the context");
    }
    entries_.push_back(std::move(s));
  }

  // Every entry binds the same pools and weights, by name and type.
  const ServingEntrySpec& first = entries_.front();
  for (const SlotSpec& slot : first.inputs) {
    if (slot.role == "KV_POOL_IN") pools_.push_back(slot);
    if (slot.role == "WEIGHT") weight_slots_.push_back(slot);
  }
  for (const ServingEntrySpec& e : entries_) {
    std::vector<SlotSpec> p, w;
    for (const SlotSpec& slot : e.inputs) {
      if (slot.role == "KV_POOL_IN") p.push_back(slot);
      if (slot.role == "WEIGHT") w.push_back(slot);
    }
    auto same = [](const std::vector<SlotSpec>& a, const std::vector<SlotSpec>& b) {
      if (a.size() != b.size()) return false;
      for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].name != b[i].name || a[i].dtype != b[i].dtype || a[i].dims != b[i].dims) return false;
      }
      return true;
    };
    if (!same(p, pools_) || !same(w, weight_slots_)) {
      return Invalid(
          at + "entries " + first.id + " and " + e.id + " bind different KV pools or weights; "
          "one model instance holds one set of each");
    }
    if (!e.prefill) {
      max_context_ = std::max(max_context_, e.context);
      max_decode_batch_ = std::max(max_decode_batch_, e.batch);
    }
  }
  if (max_decode_batch_ == 0) return Invalid(at + "has no decode entry");
  if (pools_.empty()) {
    return Invalid(at + "has no KV_POOL_IN slots; sequence mode serves a paged-KV decode artifact");
  }
  for (const SlotSpec& p : pools_) {
    if (p.dims.size() != 4 || p.dims[0] != num_blocks_ || p.dims[1] != block_size_) {
      return Invalid(at + "KV pool '" + p.name + "' is " + Join(p.dims) + ", not [numBlocks, blockSize, ...]");
    }
  }

  triton::common::TritonJson::Value weights, table;
  if (!weight_slots_.empty()) {
    if (doc.MemberAsObject("weights", &weights) != nullptr ||
        weights.MemberAsArray("table", &table) != nullptr) {
      return Invalid(at + "the entries bind staged weights but the manifest has no weights.table");
    }
    std::map<std::string, std::string> paths;
    for (size_t i = 0; i < table.ArraySize(); ++i) {
      triton::common::TritonJson::Value row;
      RETURN_IF_ERROR(table.IndexAsObject(i, &row));
      std::string name, path;
      RETURN_IF_ERROR(StrMember(row, "name", &name, at));
      RETURN_IF_ERROR(StrMember(row, "path", &path, at));
      if (!InsideDir(path)) return Invalid(at + "weight file '" + path + "' is outside the version directory");
      paths[name] = path;
    }
    for (const SlotSpec& w : weight_slots_) {
      auto it = paths.find(w.name);
      if (it == paths.end()) return Invalid(at + "WEIGHT slot '" + w.name + "' is not in weights.table");
      weight_files_.emplace_back(w.name, it->second);
    }
  }
  return nullptr;
}

TRITONSERVER_Error*
SequenceModel::CompileEntries()
{
  for (ServingEntrySpec& e : entries_) {
    auto hit = executables_.find(e.body_path);
    const std::string path = JoinPath({version_dir_, e.body_path});
    std::ifstream in(path, std::ios::binary);
    if (!in) return Err(TRITONSERVER_ERROR_NOT_FOUND, Where() + "cannot read " + path);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<tlaloc_triton::FunctionSignature> functions;
    tlaloc_triton::FunctionSignature sig;
    std::string perr;
    if (!tlaloc_triton::ListFunctions(text, &functions, &perr) ||
        !tlaloc_triton::SelectEntry(functions, e.entry_point, &sig, &perr)) {
      return Invalid(Where() + path + ": " + perr);
    }
    auto check = [&](const std::vector<tlaloc_triton::TensorType>& got,
                     const std::vector<SlotSpec>& want, const char* what) -> TRITONSERVER_Error* {
      if (got.size() != want.size()) {
        return Invalid(
            Where() + path + ": @" + sig.name + " has " + std::to_string(got.size()) + " " + what +
            "s; the manifest's entry " + e.id + " declares " + std::to_string(want.size()));
      }
      for (size_t i = 0; i < got.size(); ++i) {
        if (got[i].dtype != want[i].dtype || got[i].dims != want[i].dims) {
          return Invalid(
              Where() + path + ": " + what + " " + std::to_string(i) + " ('" + want[i].name +
              "') is " + got[i].text + " but the manifest says " +
              tlaloc_triton::TritonName(want[i].dtype) + Join(want[i].dims));
        }
      }
      return nullptr;
    };
    RETURN_IF_ERROR(check(sig.args, e.inputs, "argument"));
    RETURN_IF_ERROR(check(sig.results, e.outputs, "result"));
    if (hit == executables_.end()) {
      const uint64_t t0 = NowNs();
      std::unique_ptr<tlaloc_triton::PjrtExecutable> exe;
      std::string err = client_->Compile(tlaloc_triton::PrepareForXla(text, sig.name), &exe);
      if (!err.empty()) return Invalid(Where() + path + ": " + err);
      if (exe->num_outputs() != e.outputs.size()) {
        return Err(TRITONSERVER_ERROR_INTERNAL, Where() + path + " compiled to the wrong number of outputs");
      }
      std::ostringstream m;
      m << "tlaloc backend: " << Where() << "compiled " << e.id << " (" << e.body_path << ") in "
        << (NowNs() - t0) / 1000000 << " ms";
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, m.str().c_str());
      hit = executables_.emplace(e.body_path, std::move(exe)).first;
    }
    e.executable = hit->second.get();
  }
  return nullptr;
}

TRITONSERVER_Error*
SequenceModel::UploadWeights()
{
  const uint64_t t0 = NowNs();
  uint64_t total = 0;
  for (size_t i = 0; i < weight_slots_.size(); ++i) {
    const SlotSpec& w = weight_slots_[i];
    const std::string path = JoinPath({version_dir_, weight_files_[i].second});
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return Err(TRITONSERVER_ERROR_NOT_FOUND, Where() + "cannot read the weight file " + path);
    const uint64_t size = static_cast<uint64_t>(in.tellg());
    const uint64_t need = Elements(w.dims) * tlaloc_triton::ByteWidth(w.dtype);
    if (size != need) {
      return Invalid(
          Where() + "the weight file " + path + " has " + std::to_string(size) + " bytes; '" +
          w.name + "' is " + Join(w.dims) + " and needs " + std::to_string(need));
    }
    std::vector<char> bytes(size);
    in.seekg(0);
    if (size > 0 && !in.read(bytes.data(), static_cast<std::streamsize>(size))) {
      return Err(TRITONSERVER_ERROR_INTERNAL, Where() + "reading " + path + " failed");
    }
    HostInput host;
    host.data = bytes.data();
    host.byte_size = size;
    host.dtype = w.dtype;
    host.dims = w.dims;
    std::string err = PjrtBuffer::Upload(client_.get(), host, &weights_[w.name]);
    if (!err.empty()) return Err(TRITONSERVER_ERROR_INTERNAL, Where() + path + ": " + err);
    total += size;
  }
  std::ostringstream m;
  m << "tlaloc backend: " << Where() << "uploaded " << weight_slots_.size() << " weights ("
    << total / (1024 * 1024) << " MiB) in " << (NowNs() - t0) / 1000000 << " ms; "
    << entries_.size() << " entries, KV pool of " << num_blocks_ << " pages x " << block_size_
    << " tokens, largest context " << max_context_ << ", largest decode batch "
    << max_decode_batch_ << ", sequence idle timeout " << idle_ns_ / 1000 << " us";
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, m.str().c_str());
  return nullptr;
}

const ServingEntrySpec*
SequenceModel::Decode(int batch, int context) const
{
  const ServingEntrySpec* best = nullptr;
  for (const ServingEntrySpec& e : entries_) {
    if (e.prefill || e.batch < batch || e.context < context) continue;
    if (best == nullptr || int64_t(e.batch) * e.context < int64_t(best->batch) * best->context) best = &e;
  }
  return best;
}

const ServingEntrySpec*
SequenceModel::Prefill(int context) const
{
  const ServingEntrySpec* best = nullptr;
  for (const ServingEntrySpec& e : entries_) {
    if (!e.prefill || e.context < context) continue;
    if (best == nullptr || int64_t(e.batch) * e.context < int64_t(best->batch) * best->context) best = &e;
  }
  return best;
}

// ---------------------------------------------------------------------------
// PagePool

PagePool::PagePool(int num_blocks) : capacity_(num_blocks - 1)
{
  for (int p = 1; p < num_blocks; ++p) free_.insert(p);
}

bool
PagePool::Take(int n, std::vector<int>* pages)
{
  if (n > static_cast<int>(free_.size())) return false;
  for (int i = 0; i < n; ++i) {
    pages->push_back(*free_.begin());
    free_.erase(free_.begin());
  }
  return true;
}

void
PagePool::Give(const std::vector<int>& pages)
{
  for (int p : pages) free_.insert(p);
}

// ---------------------------------------------------------------------------
// SequenceInstance

struct SequenceInstance::Work {
  TRITONBACKEND_Request* request = nullptr;
  TRITONBACKEND_Response* response = nullptr;
  TRITONSERVER_Error* err = nullptr;
  uint64_t corrid = 0;
  bool start = false;
  bool end = false;
  std::vector<int32_t> tokens;
  SequenceState* seq = nullptr;
  int position = 0;  // position of tokens[0]
  std::vector<float> logits;
  uint64_t compute_start = 0;
  uint64_t compute_end = 0;
};

TRITONSERVER_Error*
SequenceInstance::Create(
    const SequenceModel* model, const std::string& name, TRITONBACKEND_ModelInstance* instance,
    std::unique_ptr<SequenceInstance>* out)
{
  std::unique_ptr<SequenceInstance> s(new SequenceInstance(model, name, instance));
  uint64_t total = 0;
  for (const SlotSpec& p : model->pools()) {
    const size_t bytes = Elements(p.dims) * tlaloc_triton::ByteWidth(p.dtype);
    std::vector<char> zeros(bytes, 0);
    HostInput host;
    host.data = zeros.data();
    host.byte_size = bytes;
    host.dtype = p.dtype;
    host.dims = p.dims;
    std::string err = PjrtBuffer::Upload(model->client(), host, &s->state_[p.name]);
    if (!err.empty()) {
      return Err(TRITONSERVER_ERROR_INTERNAL, "instance '" + name + "': KV pool '" + p.name + "': " + err);
    }
    total += bytes;
  }
  std::ostringstream m;
  m << "tlaloc backend: instance '" << name << "': " << model->pools().size() << " KV pools ("
    << total / (1024 * 1024) << " MiB) zeroed; " << s->pool_.capacity()
    << " pages for sequences (page 0 is the padding page)";
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, m.str().c_str());
  *out = std::move(s);
  return nullptr;
}

void
SequenceInstance::Free(uint64_t corrid, const char* why)
{
  auto it = sequences_.find(corrid);
  if (it == sequences_.end()) return;
  pool_.Give(it->second.pages);
  std::ostringstream m;
  m << "tlaloc backend: instance '" << name_ << "': sequence " << corrid << " " << why << "; freed "
    << it->second.pages.size() << " pages (" << pool_.free_count() << " of " << pool_.capacity()
    << " free)";
  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, m.str().c_str());
  sequences_.erase(it);
}

void
SequenceInstance::Reap(uint64_t now)
{
  std::vector<uint64_t> idle;
  for (const auto& kv : sequences_) {
    if (now - kv.second.last_ns > model_->idle_ns()) idle.push_back(kv.first);
  }
  for (uint64_t id : idle) {
    auto it = sequences_.find(id);
    std::ostringstream m;
    m << "tlaloc backend: instance '" << name_ << "': sequence " << id
      << " was idle longer than max_sequence_idle_microseconds (" << model_->idle_ns() / 1000
      << " us); freeing its " << it->second.pages.size() << " pages";
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, m.str().c_str());
    Free(id, "timed out");
  }
}

TRITONSERVER_Error*
SequenceInstance::Parse(Work* w)
{
  uint64_t v = 0;
  RETURN_IF_ERROR(ControlValue(w->request, model_->corrid_input(), &w->corrid));
  RETURN_IF_ERROR(ControlValue(w->request, model_->start_input(), &v));
  w->start = v != 0;
  RETURN_IF_ERROR(ControlValue(w->request, model_->end_input(), &v));
  w->end = v != 0;
  TRITONSERVER_DataType dt;
  std::vector<int64_t> shape;
  std::vector<char> bytes;
  RETURN_IF_ERROR(InputToHost(w->request, model_->tokens_input(), &dt, &shape, &bytes));
  if (dt != TRITONSERVER_TYPE_INT32 || shape.size() != 2 || shape[0] != 1) {
    return Invalid(
        "input '" + model_->tokens_input() + "' is " + TRITONSERVER_DataTypeString(dt) +
        Join(shape) + "; send INT32 [1, n]: one sequence, n token ids");
  }
  w->tokens.resize(static_cast<size_t>(shape[1]));
  if (!w->tokens.empty()) std::memcpy(w->tokens.data(), bytes.data(), w->tokens.size() * 4);
  for (int32_t t : w->tokens) {
    if (t < 0 || t >= model_->vocab()) {
      return Invalid(
          "token id " + std::to_string(t) + " is outside the vocabulary [0, " +
          std::to_string(model_->vocab()) + ")");
    }
  }
  return nullptr;
}

TRITONSERVER_Error*
SequenceInstance::Admit(Work* w)
{
  const uint64_t id = w->corrid;
  if (w->start) {
    if (sequences_.count(id)) Free(id, "was started again");
    sequences_[id] = SequenceState();
  }
  auto it = sequences_.find(id);
  if (it == sequences_.end() && w->end && w->tokens.empty()) {
    // Ending a sequence the backend holds nothing for (its START was
    // refused, or it timed out): nothing to free, and Triton still needs the
    // END to release the sequence.
    return nullptr;
  }
  if (it == sequences_.end()) {
    return Invalid(
        "sequence " + std::to_string(id) + " has no KV state in instance '" + name_ + "': its "
        "START was refused, it ended, or it was idle longer than max_sequence_idle_microseconds "
        "and its pages were freed. Start it again with sequence_start");
  }
  SequenceState& seq = it->second;
  w->seq = &seq;
  w->position = seq.length;
  const int n = static_cast<int>(w->tokens.size());
  auto refuse = [&](TRITONSERVER_Error* e) {
    if (w->start) Free(id, "was refused at START");
    w->seq = nullptr;
    return e;
  };
  if (n == 0 && !w->end) {
    return refuse(Invalid(
        "the request for sequence " + std::to_string(id) + " has no tokens; a request without "
        "tokens only ends a sequence, so send it with sequence_end"));
  }
  if (seq.length + n > model_->max_context()) {
    return refuse(Invalid(
        "sequence " + std::to_string(id) + " would reach " + std::to_string(seq.length + n) +
        " tokens; the largest context this model was compiled for is " +
        std::to_string(model_->max_context())));
  }
  const int bs = model_->block_size();
  const int need = (seq.length + n + bs - 1) / bs - static_cast<int>(seq.pages.size());
  if (need > 0 && !pool_.Take(need, &seq.pages)) {
    return refuse(Err(
        TRITONSERVER_ERROR_UNAVAILABLE,
        "KV page pool exhausted: sequence " + std::to_string(id) + " needs " +
            std::to_string(need) + " more page(s) of " + std::to_string(bs) + " tokens and " +
            std::to_string(pool_.free_count()) + " of " + std::to_string(pool_.capacity()) +
            " are free (" + std::to_string(sequences_.size()) + " sequences hold the rest). End "
            "a sequence (sequence_end) or export the artifact with more pages (numBlocks)"));
  }
  return nullptr;
}

TRITONSERVER_Error*
SequenceInstance::Run(
    const ServingEntrySpec& e, const std::vector<Work*>& rows, uint64_t* compute_start,
    uint64_t* compute_end)
{
  const int B = e.batch, T = e.tokens_per_seq, M = e.max_blocks, bs = model_->block_size();
  // The padding convention (DecodePadding): token 0, position 0, page 0,
  // sequence length 1, slot -1 (the KV write is dropped).
  std::vector<int32_t> tokens(size_t(B) * T, 0), positions(size_t(B) * T, 0);
  std::vector<int32_t> tables(size_t(B) * M, 0), lens(B, 1), slots(size_t(B) * T, -1);
  for (size_t r = 0; r < rows.size(); ++r) {
    const Work* w = rows[r];
    const int n = static_cast<int>(w->tokens.size());
    const int pad = T - n;  // right-aligned: the last token is at T - 1
    for (int j = 0; j < n; ++j) {
      const int pos = w->position + j;
      const size_t k = r * T + pad + j;
      tokens[k] = w->tokens[j];
      positions[k] = pos;
      slots[k] = w->seq->pages[pos / bs] * bs + pos % bs;
    }
    for (int j = 0; j < M && j < static_cast<int>(w->seq->pages.size()); ++j) {
      tables[r * M + j] = w->seq->pages[j];
    }
    lens[r] = w->position + n;
  }
  auto host = [](const std::vector<int32_t>& v, const std::vector<int64_t>& dims) {
    HostInput h;
    h.data = v.data();
    h.byte_size = v.size() * 4;
    h.dtype = DType::I32;
    h.dims = dims;
    return h;
  };
  std::vector<ExecuteArg> args(e.inputs.size());
  for (size_t i = 0; i < e.inputs.size(); ++i) {
    const SlotSpec& s = e.inputs[i];
    if (s.role == "TOKEN_IDS") args[i].host = host(tokens, s.dims);
    else if (s.role == "POSITIONS") args[i].host = host(positions, s.dims);
    else if (s.role == "BLOCK_TABLES") args[i].host = host(tables, s.dims);
    else if (s.role == "SEQ_LENS") args[i].host = host(lens, s.dims);
    else if (s.role == "SLOT_MAPPING") args[i].host = host(slots, s.dims);
    else if (s.role == "KV_POOL_IN") args[i].device = state_.at(s.name).get();
    else args[i].device = model_->weight(s.name);
  }
  *compute_start = NowNs();
  std::unique_ptr<PjrtResults> results;
  std::string err = e.executable->Execute(args, &results);
  if (err.empty()) {
    for (size_t j = 0; j < e.outputs.size() && err.empty(); ++j) {
      if (e.outputs[j].role != "LOGITS") continue;
      std::vector<float> all(size_t(B) * model_->vocab());
      err = results->CopyToHost(j, all.data(), all.size() * 4);
      if (!err.empty()) break;
      for (size_t r = 0; r < rows.size(); ++r) {
        rows[r]->logits.assign(all.begin() + r * model_->vocab(), all.begin() + (r + 1) * model_->vocab());
      }
    }
  }
  *compute_end = NowNs();
  if (!err.empty()) return Err(TRITONSERVER_ERROR_INTERNAL, e.id + ": " + err);
  for (size_t j = 0; j < e.outputs.size(); ++j) {
    if (e.replaces[j] >= 0) state_[e.inputs[e.replaces[j]].name] = results->Release(j);
  }
  for (Work* w : rows) {
    w->seq->length = w->position + static_cast<int>(w->tokens.size());
    w->compute_start = *compute_start;
    w->compute_end = *compute_end;
  }
  std::ostringstream m;
  m << "tlaloc backend: instance '" << name_ << "': " << e.id << " ran " << rows.size()
    << " sequence(s) in " << (*compute_end - *compute_start) / 1000 << " us";
  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, m.str().c_str());
  return nullptr;
}

TRITONSERVER_Error*
SequenceInstance::Respond(Work* w)
{
  if (w->logits.empty()) return nullptr;
  uint32_t requested = 0;
  RETURN_IF_ERROR(TRITONBACKEND_RequestOutputCount(w->request, &requested));
  bool wanted = requested == 0;
  for (uint32_t i = 0; i < requested && !wanted; ++i) {
    const char* name = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_RequestOutputName(w->request, i, &name));
    wanted = model_->logits_output() == name;
  }
  if (!wanted) return nullptr;
  const int64_t dims[2] = {1, model_->vocab()};
  const size_t bytes = w->logits.size() * 4;
  TRITONBACKEND_Output* output = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(
      w->response, &output, model_->logits_output().c_str(), TRITONSERVER_TYPE_FP32, dims, 2));
  void* buffer = nullptr;
  TRITONSERVER_MemoryType mt = TRITONSERVER_MEMORY_CPU;
  int64_t mid = 0;
  RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(output, &buffer, bytes, &mt, &mid));
  if (mt == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
    cudaError_t e = cudaMemcpy(buffer, w->logits.data(), bytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
      return Err(TRITONSERVER_ERROR_INTERNAL, std::string("copying logits to GPU memory: ") + cudaGetErrorString(e));
    }
#else
    return Err(TRITONSERVER_ERROR_UNSUPPORTED, "the output was given GPU memory and this build has no CUDA");
#endif
  } else {
    std::memcpy(buffer, w->logits.data(), bytes);
  }
  return nullptr;
}

void
SequenceInstance::ProcessRequests(TRITONBACKEND_Request** requests, uint32_t count)
{
  const uint64_t exec_start = NowNs();
  Reap(exec_start);

  std::vector<Work> works(count);
  for (uint32_t r = 0; r < count; ++r) {
    Work& w = works[r];
    w.request = requests[r];
    w.err = TRITONBACKEND_ResponseNew(&w.response, w.request);
    if (w.err == nullptr) w.err = Parse(&w);
    if (w.err == nullptr) w.err = Admit(&w);
  }

  // Prefill: a request with several tokens runs as one chunk of the smallest
  // prefill entry whose context covers the sequence afterwards. Without such
  // an entry, its tokens run as decode steps. Decode: every one-token
  // request, several sequences per call.
  std::vector<Work*> decode, prefill, stepwise;
  for (Work& w : works) {
    if (w.err != nullptr || w.tokens.empty()) continue;
    if (w.tokens.size() == 1) {
      decode.push_back(&w);
    } else if (model_->Prefill(w.position + static_cast<int>(w.tokens.size())) != nullptr) {
      prefill.push_back(&w);
    } else {
      stepwise.push_back(&w);
    }
  }
  uint64_t first_compute = 0, last_compute = 0, cs = 0, ce = 0;
  auto note = [&]() {
    if (first_compute == 0) first_compute = cs;
    last_compute = ce;
  };
  for (Work* w : prefill) {
    const ServingEntrySpec* e = model_->Prefill(w->position + static_cast<int>(w->tokens.size()));
    w->err = Run(*e, {w}, &cs, &ce);
    note();
  }
  for (Work* w : stepwise) {
    // One decode step per token, carrying the sequence forward.
    const std::vector<int32_t> all = w->tokens;
    const int start = w->position;
    for (size_t j = 0; j < all.size() && w->err == nullptr; ++j) {
      w->tokens = {all[j]};
      w->position = start + static_cast<int>(j);
      const ServingEntrySpec* e = model_->Decode(1, w->position + 1);
      if (e == nullptr) {
        w->err = Invalid("no decode entry covers batch 1 and context " + std::to_string(w->position + 1));
        break;
      }
      w->err = Run(*e, {w}, &cs, &ce);
      note();
    }
    w->tokens = all;
  }
  const size_t max_b = static_cast<size_t>(model_->max_decode_batch());
  for (size_t i = 0; i < decode.size(); i += max_b) {
    std::vector<Work*> rows(decode.begin() + i, decode.begin() + std::min(decode.size(), i + max_b));
    int context = 0;
    for (Work* w : rows) context = std::max(context, w->position + 1);
    const ServingEntrySpec* e = model_->Decode(static_cast<int>(rows.size()), context);
    TRITONSERVER_Error* err =
        e == nullptr ? Invalid(
                           "no decode entry covers batch " + std::to_string(rows.size()) +
                           " and context " + std::to_string(context))
                     : Run(*e, rows, &cs, &ce);
    note();
    for (Work* w : rows) {
      w->err = err == nullptr ? nullptr : Err(TRITONSERVER_ErrorCode(err), TRITONSERVER_ErrorMessage(err));
    }
    if (err != nullptr) TRITONSERVER_ErrorDelete(err);
  }

  for (Work& w : works) {
    if (w.err == nullptr && w.response != nullptr) w.err = Respond(&w);
    if (w.end) {
      Free(w.corrid, w.err == nullptr ? "ended" : "ended with an error");
    } else if (w.seq != nullptr && sequences_.count(w.corrid)) {
      w.seq->last_ns = NowNs();
    }
    const bool ok = w.err == nullptr;
    if (w.response != nullptr) {
      TRITONSERVER_Error* send = nullptr;
      if (!ok) {
        send = Err(
            TRITONSERVER_ErrorCode(w.err),
            "model '" + model_->name() + "': " + TRITONSERVER_ErrorMessage(w.err));
      }
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(w.response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, send),
          "failed to send the response");
      if (send != nullptr) TRITONSERVER_ErrorDelete(send);
    } else if (!ok) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, TRITONSERVER_ErrorMessage(w.err));
    }
    const uint64_t cstart = w.compute_start ? w.compute_start : exec_start;
    const uint64_t cend = w.compute_end ? w.compute_end : cstart;
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            instance_, w.request, ok, exec_start, cstart, cend, NowNs()),
        "failed to report request statistics");
    if (w.err != nullptr) TRITONSERVER_ErrorDelete(w.err);
    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(w.request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed to release the request");
  }
  if (first_compute == 0) first_compute = last_compute = NowNs();
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          instance_, count, exec_start, first_compute, last_compute, NowNs()),
      "failed to report batch statistics");
}

}}}  // namespace triton::backend::tlaloc
