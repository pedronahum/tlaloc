// Copyright 2026 Pedro N. Rodriguez
// SPDX-License-Identifier: Apache-2.0

#include "sequence_mode.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
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

// Adds output `name` to `response` and copies `bytes` of `data` into it,
// whatever memory Triton gives it.
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
  if (mt == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
    cudaError_t e = cudaMemcpy(buffer, data, bytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
      return Err(
          TRITONSERVER_ERROR_INTERNAL, "copying output '" + name + "' to GPU memory: " + cudaGetErrorString(e));
    }
#else
    return Err(TRITONSERVER_ERROR_UNSUPPORTED, "the output was given GPU memory and this build has no CUDA");
#endif
  } else {
    std::memcpy(buffer, data, bytes);
  }
  return nullptr;
}

const std::set<std::string> kRequestRoles = {
    "TOKEN_IDS", "POSITIONS", "BLOCK_TABLES", "SEQ_LENS", "SLOT_MAPPING"};
const std::set<std::string> kWindowRoles = {"WINDOW_BLOCK_TABLES", "WINDOW_SLOT_MAPPING"};
// A tlaloc-serving-v4 artifact's linear-attention layers: each row's state slot.
const std::set<std::string> kStateRoles = {"STATE_SLOTS"};
// A speculative artifact's per-token state write slots.
const std::set<std::string> kSpecRoles = {"STATE_WRITE_SLOTS"};

bool
IsPoolIn(const std::string& role)
{
  return role == "KV_POOL_IN" || role == "WINDOW_KV_POOL_IN" || role == "STATE_POOL_IN";
}

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
  {
    // One input, the token ids. A generation's length and end tokens come as
    // request parameters (max_tokens, end_tokens; see Parse).
    triton::common::TritonJson::Value ins, io;
    if (!config.Find("input", &ins) || ins.ArraySize() != 1) {
      return Invalid(Where() + "a serving_manifest model declares exactly one input (the token ids)");
    }
    RETURN_IF_ERROR(ins.IndexAsObject(0, &io));
    RETURN_IF_ERROR(io.MemberAsString("name", &tokens_input_));
    std::string dt;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &dt));
    if (dt != "TYPE_INT32") return Invalid(Where() + "input '" + tokens_input_ + "' is " + dt + "; it must be TYPE_INT32");
    triton::common::TritonJson::Value policy;
    if (config.Find("model_transaction_policy", &policy)) {
      bool d = false;
      if (policy.MemberAsBool("decoupled", &d) == nullptr) decoupled_ = d;
    }
  }
  {
    // LOGITS (FP32, the last token's logits), and optionally KV_PAGES
    // (INT32 [ 2 ], the pages the sequence holds in each pool class) and
    // NEXT_TOKEN (INT32 [ 1 ], the argmax of the logits, chosen here so that a
    // greedy client is sent four bytes instead of the logits row).
    triton::common::TritonJson::Value outs;
    if (!config.Find("output", &outs) || outs.ArraySize() < 1 || outs.ArraySize() > 3) {
      return Invalid(
          Where() + "a serving_manifest model declares the output of the last token's logits "
          "(TYPE_FP32) and, optionally, one of the pages the sequence holds (TYPE_INT32 [ 2 ]) and "
          "one of the greedy next token (TYPE_INT32 [ 1 ])");
    }
    for (size_t i = 0; i < outs.ArraySize(); ++i) {
      triton::common::TritonJson::Value io;
      RETURN_IF_ERROR(outs.IndexAsObject(i, &io));
      std::string name, dt;
      RETURN_IF_ERROR(io.MemberAsString("name", &name));
      RETURN_IF_ERROR(io.MemberAsString("data_type", &dt));
      std::vector<int64_t> int_dims;
      if (dt == "TYPE_INT32") RETURN_IF_ERROR(ParseShape(io, "dims", &int_dims));
      if (dt == "TYPE_FP32" && logits_output_.empty()) {
        logits_output_ = name;
      } else if (dt == "TYPE_INT32" && int_dims == std::vector<int64_t>{-1} && next_tokens_output_.empty()) {
        // A speculative model's NEXT_TOKENS: the tokens a request emits.
        next_tokens_output_ = name;
      } else if (dt == "TYPE_INT32" && int_dims == std::vector<int64_t>{1} && next_token_output_.empty()) {
        next_token_output_ = name;
      } else if (dt == "TYPE_INT32" && pages_output_.empty()) {
        const std::vector<int64_t>& dims = int_dims;
        if (dims != std::vector<int64_t>{2}) {
          return Invalid(
              Where() + "output '" + name + "' has dims " + Join(dims) + "; the pages output is "
              "[ 2 ]: pages held in the KV pool and in the windowed KV pool");
        }
        pages_output_ = name;
      } else {
        return Invalid(
            Where() + "output '" + name + "' is " + dt + "; a serving_manifest model has one "
            "TYPE_FP32 output (the logits) and at most one TYPE_INT32 output of each of the pages "
            "held ([ 2 ]) and the next token ([ 1 ])");
      }
    }
    if (logits_output_.empty() == next_tokens_output_.empty()) {
      return Invalid(
          Where() + "declare either a TYPE_FP32 output for the last token's logits or, for a speculative "
          "artifact, a TYPE_INT32 [ -1 ] output for the tokens a request emits; not both and not neither");
    }
  }
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
  triton::common::TritonJson::Value params;
  const bool has_params = config.Find("parameters", &params);
  auto flag = [&](const char* key, bool* out) -> TRITONSERVER_Error* {
    if (!has_params || !params.Find(key)) return nullptr;
    std::string v;
    RETURN_IF_ERROR(GetParameterValue(params, key, &v));
    if (v != "true" && v != "false") {
      return Invalid(Where() + "the '" + key + "' parameter must be true or false, got '" + v + "'");
    }
    *out = v == "true";
    return nullptr;
  };
  RETURN_IF_ERROR(flag("donate_kv_pools", &donate_pools_));
  RETURN_IF_ERROR(flag("pack_step_inputs", &pack_step_inputs_));
  RETURN_IF_ERROR(flag("overlap_logits_copy", &overlap_logits_copy_));
  RETURN_IF_ERROR(flag("backend_batching", &backend_batching_));
  if (has_params && params.Find("cohort_wait_microseconds")) {
    std::string v;
    RETURN_IF_ERROR(GetParameterValue(params, "cohort_wait_microseconds", &v));
    char* end = nullptr;
    const unsigned long long us = std::strtoull(v.c_str(), &end, 10);
    if (v.empty() || *end != '\0' || us > 10000000ULL) {
      return Invalid(
          Where() + "the 'cohort_wait_microseconds' parameter must be a whole number of microseconds "
          "from 0 to 10000000, got '" + v + "'");
    }
    cohort_wait_ns_ = us * 1000;
  }
  if (has_params && params.Find("prompt_wait_microseconds")) {
    std::string v;
    RETURN_IF_ERROR(GetParameterValue(params, "prompt_wait_microseconds", &v));
    char* end = nullptr;
    const unsigned long long us = std::strtoull(v.c_str(), &end, 10);
    if (v.empty() || *end != '\0' || us > 10000000ULL) {
      return Invalid(
          Where() + "the 'prompt_wait_microseconds' parameter must be a whole number of microseconds "
          "from 0 to 10000000, got '" + v + "'");
    }
    prompt_wait_ns_ = us * 1000;
  }
  uint64_t idle_us = 1000000;  // Triton's default
  MemberAsU64(sb, "max_sequence_idle_microseconds", &idle_us);
  idle_ns_ = idle_us * 1000;
  triton::common::TritonJson::Value oldest;
  if (sb.Find("oldest", &oldest)) MemberAsU64(oldest, "max_queue_delay_microseconds", &queue_delay_us_);
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
  if (version != "tlaloc-serving-v1" && version != "tlaloc-serving-v2" &&
      version != "tlaloc-serving-v3" && version != "tlaloc-serving-v4" && version != "tlaloc-serving-v5") {
    return Invalid(
        at + "schemaVersion '" + version + "' is not tlaloc-serving-v1, v2, v3, v4 or v5");
  }
  const bool v5 = version == "tlaloc-serving-v5";
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
  triton::common::TritonJson::Value refused;
  if (model.Find("refusedTokens", &refused) && !refused.IsNull()) {
    for (size_t i = 0; i < refused.ArraySize(); ++i) {
      triton::common::TritonJson::Value t;
      if (refused.IndexAsObject(i, &t) != nullptr) return Invalid(at + "model.refusedTokens holds a non-object");
      int id = 0;
      std::string key;
      RETURN_IF_ERROR(IntMember(t, "id", &id, at + "model.refusedTokens: "));
      RETURN_IF_ERROR(StrMember(t, "configKey", &key, at + "model.refusedTokens: "));
      if (id < 0 || id >= vocab_ || !refused_tokens_.emplace(id, key).second) {
        return Invalid(
            at + "model.refusedTokens: token " + std::to_string(id) + " (" + key +
            ") is outside the vocabulary or listed twice");
      }
    }
  }
  triton::common::TritonJson::Value lst;
  const bool has_state = model.Find("linearState", &lst) && !lst.IsNull();
  if (has_state != (version == "tlaloc-serving-v4" || v5)) {
    return Invalid(
        at + (has_state ? "a " + version + " manifest has linear-attention state pools, which only "
                          "tlaloc-serving-v4 and v5 define"
                        : "a " + version + " manifest without linear-attention state (model.linearState)"));
  }
  {
    triton::common::TritonJson::Value d;
    if (model.Find("mtpDraftTokens", &d)) RETURN_IF_ERROR(IntMember(model, "mtpDraftTokens", &mtp_drafts_, at));
  }
  if ((mtp_drafts_ > 0) != v5 || mtp_drafts_ < 0) {
    return Invalid(at + "model.mtpDraftTokens " + std::to_string(mtp_drafts_) + " in a " + version + " manifest; "
                   "speculative entries are tlaloc-serving-v5's, and a v5 manifest has them");
  }
  {
    triton::common::TritonJson::Value f;
    if (model.Find("cudaKernels", &f)) RETURN_IF_ERROR(f.AsBool(&cuda_kernels_));
  }
  if (speculative() != !next_tokens_output_.empty()) {
    return Invalid(
        at + (speculative() ? "a speculative artifact returns tokens, not logits: declare a TYPE_INT32 [ -1 ] "
                              "output (NEXT_TOKENS) in config.pbtxt instead of the logits"
                            : "config.pbtxt declares a TYPE_INT32 [ -1 ] tokens output, but the artifact is not speculative"));
  }
  if (has_state) {
    const std::string sat = at + "model.linearState: ";
    RETURN_IF_ERROR(IntMember(lst, "numSlots", &state_slots_, sat));
    triton::common::TritonJson::Value layers;
    if (lst.MemberAsArray("layers", &layers) != nullptr || layers.ArraySize() == 0) {
      return Invalid(sat + "no layers");
    }
    state_layers_ = static_cast<int>(layers.ArraySize());
    if (state_slots_ < 1) return Invalid(sat + "numSlots must be >= 1");
    if (speculative() && state_slots_ < mtp_drafts_ + 2) {
      return Invalid(sat + "numSlots " + std::to_string(state_slots_) + " holds no speculative sequence, which needs " +
                     std::to_string(mtp_drafts_ + 2));
    }
  }
  triton::common::TritonJson::Value wkv;
  const bool has_window = model.Find("windowedKv", &wkv) && !wkv.IsNull();
  if (has_window != (version == "tlaloc-serving-v3")) {
    return Invalid(
        at + (has_window ? "a " + version + " manifest has a windowed KV pool, which only "
                           "tlaloc-serving-v3 defines"
                         : "a tlaloc-serving-v3 manifest without a windowed KV pool (model.windowedKv)"));
  }
  if (has_window) {
    const std::string wat = at + "model.windowedKv: ";
    RETURN_IF_ERROR(IntMember(wkv, "window", &window_, wat));
    RETURN_IF_ERROR(IntMember(wkv, "numBlocks", &window_num_blocks_, wat));
    RETURN_IF_ERROR(IntMember(wkv, "ringPages", &ring_pages_, wat));
    triton::common::TritonJson::Value layers;
    if (wkv.MemberAsArray("layers", &layers) != nullptr || layers.ArraySize() == 0) {
      return Invalid(wat + "no layers");
    }
    for (size_t i = 0; i < layers.ArraySize(); ++i) {
      int64_t l = 0;
      if (layers.IndexAsInt(i, &l) != nullptr) return Invalid(wat + "a layer is not an integer");
      window_layers_.push_back(static_cast<int>(l));
    }
    if (window_ < 1 || ring_pages_ < 1 || window_num_blocks_ < 1 + ring_pages_) {
      return Invalid(
          wat + "window " + std::to_string(window_) + ", ringPages " + std::to_string(ring_pages_) +
          ", numBlocks " + std::to_string(window_num_blocks_) + ": the pool must hold one ring "
          "besides the padding page 0");
    }
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
    if (s.prefill && s.tokens_per_seq < s.context) s.id += "_t" + std::to_string(s.tokens_per_seq);
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
        {"SEQ_LENS", {B}}, {"SLOT_MAPPING", {int64_t(B) * T}},
        {"WINDOW_BLOCK_TABLES", {B, M}}, {"WINDOW_SLOT_MAPPING", {int64_t(B) * T}}, {"STATE_SLOTS", {B}},
        {"STATE_WRITE_SLOTS", {B, T}}};
    std::set<std::string> seen;
    for (const SlotSpec& slot : s.inputs) {
      if (kWindowRoles.count(slot.role) && !has_window) {
        return Invalid(eat + "input '" + slot.name + "' is " + slot.role + " but the model has no windowed KV pool");
      }
      if (kStateRoles.count(slot.role) && !has_state) {
        return Invalid(eat + "input '" + slot.name + "' is " + slot.role + " but the model has no linear-attention state");
      }
      if (slot.role == "STATE_POOL_IN" && !has_state) {
        return Invalid(eat + "input '" + slot.name + "' is STATE_POOL_IN but the model has no linear-attention state");
      }
      if (kSpecRoles.count(slot.role) && !speculative()) {
        return Invalid(eat + "input '" + slot.name + "' is " + slot.role + " but the artifact is not speculative");
      }
      if (kRequestRoles.count(slot.role) || kWindowRoles.count(slot.role) || kStateRoles.count(slot.role) ||
          kSpecRoles.count(slot.role)) {
        if (!seen.insert(slot.role).second) return Invalid(eat + "role " + slot.role + " appears twice");
        if (slot.dtype != DType::I32 || slot.dims != want.at(slot.role)) {
          return Invalid(
              eat + "input '" + slot.name + "' (" + slot.role + ") is " +
              tlaloc_triton::TritonName(slot.dtype) + Join(slot.dims) + "; expected INT32" +
              Join(want.at(slot.role)));
        }
      } else if (slot.role == "WINDOW_KV_POOL_IN" && !has_window) {
        return Invalid(eat + "input '" + slot.name + "' is WINDOW_KV_POOL_IN but the model has no windowed KV pool");
      } else if (!IsPoolIn(slot.role) && slot.role != "WEIGHT") {
        return Invalid(eat + "input '" + slot.name + "' has role " + slot.role + ", which is not an input role");
      }
    }
    if (seen.size() != kRequestRoles.size() + (has_window ? kWindowRoles.size() : 0) +
                           (has_state ? kStateRoles.size() : 0) + (speculative() ? kSpecRoles.size() : 0)) {
      return Invalid(
          eat + "lacks one of TOKEN_IDS, POSITIONS, BLOCK_TABLES, SEQ_LENS, SLOT_MAPPING" +
          (has_window ? ", WINDOW_BLOCK_TABLES, WINDOW_SLOT_MAPPING" : "") + (has_state ? ", STATE_SLOTS" : ""));
    }
    int logits = 0, spec_outputs = 0;
    const int K = mtp_drafts_;
    const std::map<std::string, std::vector<int64_t>> spec_want = {
        {"NEXT_TOKENS", {B, 1 + K}}, {"ACCEPTED", {B}}, {"DRAFTS", {B, K}}};
    for (size_t j = 0; j < s.outputs.size(); ++j) {
      const SlotSpec& slot = s.outputs[j];
      if (speculative() && spec_want.count(slot.role)) {
        ++spec_outputs;
        if (slot.dtype != DType::I32 || slot.dims != spec_want.at(slot.role)) {
          return Invalid(
              eat + slot.role + " '" + slot.name + "' is " + tlaloc_triton::TritonName(slot.dtype) + Join(slot.dims) +
              "; expected INT32" + Join(spec_want.at(slot.role)));
        }
      } else if (slot.role == "LOGITS") {
        ++logits;
        if (slot.dtype != DType::F32 || slot.dims != std::vector<int64_t>{B, 1, vocab_}) {
          return Invalid(
              eat + "LOGITS '" + slot.name + "' is " + tlaloc_triton::TritonName(slot.dtype) +
              Join(slot.dims) + "; expected FP32" + Join({B, 1, vocab_}) +
              " (a tlaloc-serving-v2 artifact returns each sequence's last-token logits)");
        }
      } else if (slot.role == "KV_POOL_OUT" || slot.role == "WINDOW_KV_POOL_OUT" || slot.role == "STATE_POOL_OUT") {
        const int i = s.replaces[j];
        const std::string in_role = slot.role == "KV_POOL_OUT"          ? "KV_POOL_IN"
                                    : slot.role == "WINDOW_KV_POOL_OUT" ? "WINDOW_KV_POOL_IN"
                                                                        : "STATE_POOL_IN";
        if (i < 0 || s.inputs[i].role != in_role || s.inputs[i].dtype != slot.dtype ||
            s.inputs[i].dims != slot.dims) {
          return Invalid(eat + slot.role + " '" + slot.name + "' is not paired with a " + in_role + " of its type");
        }
      } else {
        return Invalid(eat + "output '" + slot.name + "' has role " + slot.role);
      }
    }
    if (speculative() ? (logits != 0 || spec_outputs != 3) : logits != 1) {
      return Invalid(eat + (speculative() ? "needs NEXT_TOKENS, ACCEPTED and DRAFTS and no LOGITS" : "needs exactly one LOGITS output"));
    }
    // A prefill entry takes a chunk of up to its context per sequence; a decode
    // entry one token, or a speculative one the pending token and the drafts.
    if (s.prefill ? (T < 1 || T > s.context) : T != 1 + K) {
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
    if (IsPoolIn(slot.role)) pools_.push_back(slot);
    if (slot.role == "WEIGHT") weight_slots_.push_back(slot);
  }
  for (const ServingEntrySpec& e : entries_) {
    std::vector<SlotSpec> p, w;
    for (const SlotSpec& slot : e.inputs) {
      if (IsPoolIn(slot.role)) p.push_back(slot);
      if (slot.role == "WEIGHT") w.push_back(slot);
    }
    auto same = [](const std::vector<SlotSpec>& a, const std::vector<SlotSpec>& b) {
      if (a.size() != b.size()) return false;
      for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].name != b[i].name || a[i].role != b[i].role || a[i].dtype != b[i].dtype ||
            a[i].dims != b[i].dims) {
          return false;
        }
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
    } else {
      max_prefill_batch_ = std::max(max_prefill_batch_, e.batch);
      max_prefill_tokens_ = std::max(max_prefill_tokens_, e.tokens_per_seq);
    }
  }
  if (max_decode_batch_ == 0) return Invalid(at + "has no decode entry");
  if (windowed() && int64_t(ring_pages_) * block_size_ < std::min(window_, max_context_)) {
    // A ring that holds neither a window nor the largest context would write
    // a position over one a decode step still reads.
    return Invalid(
        at + "model.windowedKv: a ring of " + std::to_string(ring_pages_) + " pages of " +
        std::to_string(block_size_) + " holds " + std::to_string(ring_pages_ * block_size_) +
        " positions; the window is " + std::to_string(window_) + " and the largest context " +
        std::to_string(max_context_) + ", so a decode step would read a position already written over");
  }
  if (pools_.empty()) {
    return Invalid(at + "has no KV_POOL_IN slots; sequence mode serves a paged-KV decode artifact");
  }
  size_t window_pools = 0, state_pools = 0;
  for (const SlotSpec& p : pools_) {
    if (p.role == "STATE_POOL_IN") {
      // [numSlots, ...], f32, indexed by a sequence's state slot.
      ++state_pools;
      if (p.dims.empty() || p.dims[0] != state_slots_ || p.dtype != DType::F32) {
        return Invalid(
            at + "state pool '" + p.name + "' is " + tlaloc_triton::TritonName(p.dtype) + Join(p.dims) +
            ", not FP32 [linearState.numSlots = " + std::to_string(state_slots_) + ", ...]");
      }
      continue;
    }
    const bool w = p.role == "WINDOW_KV_POOL_IN";
    window_pools += w;
    const int blocks = w ? window_num_blocks_ : num_blocks_;
    if (p.dims.size() != 4 || p.dims[0] != blocks || p.dims[1] != block_size_) {
      return Invalid(
          at + "KV pool '" + p.name + "' is " + Join(p.dims) + ", not [" +
          (w ? "windowedKv.numBlocks" : "numBlocks") + ", blockSize, ...] = [" +
          std::to_string(blocks) + ", " + std::to_string(block_size_) + ", ...]");
    }
  }
  if (state_pools != 2 * static_cast<size_t>(state_layers_) + (speculative() ? 1 : 0)) {
    return Invalid(
        at + "the entries bind " + std::to_string(state_pools) + " STATE_POOL_IN pools; the linear "
        "state lists " + std::to_string(state_layers_) + " layers, two pools each" +
        (speculative() ? ", and the MTP head carries one more (its hidden states)" : ""));
  }
  if (window_pools != 2 * window_layers_.size()) {
    return Invalid(
        at + "the entries bind " + std::to_string(window_pools) + " WINDOW_KV_POOL_IN pools; the "
        "windowed KV pool lists " + std::to_string(window_layers_.size()) + " layers, two pools each");
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
    if (text.find("stablehlo.custom_call @tlaloc_") != std::string::npos && !client_->plugin()->kernels.empty()) {
      return Invalid(
          Where() + path + " calls a CUDA kernel of libtlaloc_kernels.so (custom_call @tlaloc_...), which is not "
                           "registered with the PJRT plugin: " + client_->plugin()->kernels);
    }
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
      tlaloc_triton::CompiledMemory mem;
      if (exe->MemoryStats(&mem).empty()) {
        m << "; XLA writes outputs over " << mem.alias / (1024 * 1024) << " MiB of its "
          << mem.argument / (1024 * 1024) << " MiB of arguments, and needs " << mem.temp / (1024 * 1024)
          << " MiB of temporary memory";
      }
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
    in.close();
    tlaloc_triton::DropFileCache(path);
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
    << max_decode_batch_ << ", largest prefill batch " << max_prefill_batch_;
  if (max_prefill_tokens_ > 0 && max_prefill_tokens_ < max_context_) {
    m << ", at most " << max_prefill_tokens_ << " tokens per sequence in a prefill call";
  }
  m << ", sequence idle timeout " << idle_ns_ / 1000 << " us";
  if (!refused_tokens_.empty()) {
    m << "; refuses token ids";
    for (const auto& kv : refused_tokens_) m << " " << kv.first << " (" << kv.second << ")";
  }
  if (windowed()) {
    m << "; windowed KV pool for " << window_layers_.size() << " sliding layers (window " << window_
      << "): " << window_num_blocks_ << " pages, a ring of at most " << ring_pages_
      << " pages per sequence";
  }
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
SequenceModel::Prefill(int batch, int context, int tokens) const
{
  const ServingEntrySpec* best = nullptr;
  for (const ServingEntrySpec& e : entries_) {
    if (!e.prefill || e.batch < batch || e.context < context || e.tokens_per_seq < tokens) continue;
    // The smallest batch x context, then the fewest tokens per call: a short
    // request takes the smallest chunk that holds it.
    const int64_t cost = int64_t(e.batch) * e.context, best_cost = best ? int64_t(best->batch) * best->context : 0;
    if (best == nullptr || cost < best_cost || (cost == best_cost && e.tokens_per_seq < best->tokens_per_seq)) best = &e;
  }
  return best;
}

int
SequenceModel::MaxPrefillBatch(int context) const
{
  int most = 0;
  for (const ServingEntrySpec& e : entries_) {
    if (e.prefill && e.context == context) most = std::max(most, e.batch);
  }
  return most;
}

int
SequenceModel::MaxTokensPerCall(int start) const
{
  const int chunk = max_prefill_tokens_ > 0 ? max_prefill_tokens_ : std::numeric_limits<int>::max();
  if (!windowed()) return chunk;
  return std::max(1, std::min(chunk, ring_pages_ * block_size_ - std::min(start, window_ - 1)));
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
  // Speculative: the tokens the request emits, and whether it is a verify step
  // (one token, the sequence's pending one, run with its drafts).
  std::vector<int32_t> emitted;
  bool verify = false;
  size_t pages_held = 0;  // pages the sequence holds after the request
  size_t ring_held = 0;   // windowed pages it holds
  uint64_t arrival = 0;  // when Triton handed the request over
  bool ok = false;       // answered without an error
  uint64_t compute_start = 0;
  uint64_t compute_end = 0;
  // Set when a request of several calls fails after an earlier call of it
  // ran: part of its tokens are in the KV and the windowed ring may have
  // written over positions the request's first call read, so the sequence
  // cannot be continued or the request sent again; it is freed.
  bool lost = false;
  // A generation (max_tokens > 0): the backend steps the sequence itself
  // until it has emitted `max_tokens` tokens or one of `end_tokens`; the
  // tokens so far; set by RunBatch when the work runs again in the next
  // batch; and, for a decoupled model, the factory of its per-step responses.
  int max_tokens = 0;
  std::vector<int32_t> end_tokens;
  std::vector<int32_t> generated;
  bool again = false;
  TRITONBACKEND_ResponseFactory* factory = nullptr;
  uint64_t first_arrival = 0;
};

SequenceInstance::SequenceInstance(
    const SequenceModel* model, const std::string& name, TRITONBACKEND_ModelInstance* instance)
    : model_(model), name_(name), instance_(instance), pool_(model->num_blocks()),
      window_pool_(model->windowed() ? model->window_num_blocks() : 1)
{
  for (int i = 0; i < model->state_slots(); ++i) free_states_.insert(i);
}

TRITONSERVER_Error*
SequenceInstance::Create(
    const SequenceModel* model, const std::string& name, TRITONBACKEND_ModelInstance* instance,
    std::unique_ptr<SequenceInstance>* out)
{
  std::unique_ptr<SequenceInstance> s(new SequenceInstance(model, name, instance));
  uint64_t total = 0;
  std::string err = s->ZeroPools(&total);
  if (!err.empty()) return Err(TRITONSERVER_ERROR_INTERNAL, "instance '" + name + "': " + err);
  err = s->AllocateStaging();
  if (!err.empty()) return Err(TRITONSERVER_ERROR_INTERNAL, "instance '" + name + "': " + err);
  s->pool_bytes_ = total;
  std::ostringstream m;
  m << "tlaloc backend: instance '" << name << "': " << model->pools().size() << " KV pools ("
    << total / (1024 * 1024) << " MiB) zeroed; " << s->pool_.capacity()
    << " pages for sequences (page 0 is the padding page); ";
  if (model->windowed()) {
    m << s->window_pool_.capacity() << " windowed pages, at most " << model->ring_pages()
      << " per sequence; ";
  }
  if (model->state_slots() > 0) {
    m << model->state_slots() << " linear-attention state slots, one per live sequence; ";
  }
  m << (model->donate_pools() ? "each execution is handed the pools to update in place"
                              : "executions are not handed the pools (donate_kv_pools is false)");
  if (s->device_staging_ != nullptr) {
    m << "; a step's integer inputs are written into " << s->staging_bytes_
      << " bytes of pinned host memory the executions read in place (no copy)";
  } else {
    m << "; a step's integer inputs are uploaded one by one"
      << (model->pack_step_inputs() ? " (the plugin cannot read device memory in place)"
                                    : " (pack_step_inputs is false)");
  }
  m << (model->overlap_logits_copy() ? "; the logits copy is queued behind each execution"
                                     : "; the logits are copied once the host sees an execution finish "
                                       "(overlap_logits_copy is false)");
  if (model->backend_batching()) {
    m << "; the backend batches steps itself: a batch waits for the sequences that decoded in the "
      << "previous one for up to " << model->cohort_wait_ns() / 1000 << " us (a lone sequence does not "
      << "wait), and a prompt for others sent with it for up to " << model->prompt_wait_ns() / 1000 << " us";
    if (model->queue_delay_us() > 0) {
      m << " (but Triton's max_queue_delay_microseconds of " << model->queue_delay_us()
        << " still holds every request of fewer than preferred_batch_size sequences before the "
        << "backend sees it; set it to 0)";
    }
  } else {
    m << "; batches are the ones Triton's sequence batcher forms (backend_batching is false)";
  }
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, m.str().c_str());
  if (model->backend_batching()) s->worker_ = std::thread([p = s.get()] { p->Loop(); });
  *out = std::move(s);
  return nullptr;
}

SequenceInstance::~SequenceInstance()
{
  if (worker_.joinable()) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stopping_ = true;
    }
    arrived_.notify_all();
    worker_.join();  // runs what is still queued first
  }
  // The views over the staging memory go before the memory.
  layouts_.clear();
#ifdef TRITON_ENABLE_GPU
  if (host_staging_ != nullptr) cudaFreeHost(host_staging_);
#endif
}

namespace {

constexpr size_t kNotStaged = std::numeric_limits<size_t>::max();
// Each input starts on its own 256 bytes, as cudaMalloc aligns its blocks.
constexpr size_t kStagingAlign = 256;

size_t
StagedBytes(const ServingEntrySpec& e, std::vector<size_t>* offsets)
{
  size_t at = 0;
  if (offsets != nullptr) offsets->assign(e.inputs.size(), kNotStaged);
  for (size_t i = 0; i < e.inputs.size(); ++i) {
    const SlotSpec& s = e.inputs[i];
    if (!kRequestRoles.count(s.role) && !kWindowRoles.count(s.role) && !kStateRoles.count(s.role) &&
        !kSpecRoles.count(s.role)) {
      continue;
    }
    if (offsets != nullptr) (*offsets)[i] = at;
    at += (Elements(s.dims) * 4 + kStagingAlign - 1) / kStagingAlign * kStagingAlign;
  }
  return at;
}

}  // namespace

std::string
SequenceInstance::AllocateStaging()
{
  if (!model_->pack_step_inputs() || !model_->client()->SupportsDeviceViews()) return "";
#ifdef TRITON_ENABLE_GPU
  size_t bytes = 0;
  for (const ServingEntrySpec& e : model_->entries()) bytes = std::max(bytes, StagedBytes(e, nullptr));
  if (bytes == 0) return "";
  // Pinned host memory mapped into the GPU's address space: the executions
  // read the inputs where the host wrote them, with no copy.
  cudaError_t c = cudaSetDevice(model_->client()->device_ordinal());
  if (c == cudaSuccess) c = cudaHostAlloc(&host_staging_, bytes, cudaHostAllocMapped);
  if (c == cudaSuccess) c = cudaHostGetDevicePointer(&device_staging_, host_staging_, 0);
  if (c != cudaSuccess) {
    return "allocating " + std::to_string(bytes) + " bytes for step inputs (pack_step_inputs): " +
           cudaGetErrorString(c);
  }
  std::memset(host_staging_, 0, bytes);
  staging_bytes_ = bytes;
#endif
  return "";
}

std::string
SequenceInstance::Layout(const ServingEntrySpec& e, StepLayout** out)
{
  auto it = layouts_.find(e.id);
  if (it != layouts_.end()) {
    *out = &it->second;
    return "";
  }
  StepLayout l;
  l.bytes = StagedBytes(e, &l.offset);
  l.views.resize(e.inputs.size());
  for (size_t i = 0; i < e.inputs.size(); ++i) {
    if (l.offset[i] == kNotStaged) continue;
    std::string err = PjrtBuffer::View(
        model_->client(), static_cast<char*>(device_staging_) + l.offset[i], DType::I32, e.inputs[i].dims,
        &l.views[i]);
    if (!err.empty()) return "input '" + e.inputs[i].name + "' in the step staging memory: " + err;
  }
  *out = &layouts_.emplace(e.id, std::move(l)).first->second;
  return "";
}

void
SequenceInstance::Clock(
    const ServingEntrySpec& e, uint64_t t0, uint64_t t1, uint64_t t2, uint64_t t3, uint64_t t4)
{
  StepClock& c = clocks_[e.id];
  c.inputs.push_back((t1 - t0) / 1000);
  c.launch.push_back((t2 - t1) / 1000);
  c.wait.push_back((t3 - t2) / 1000);
  c.after.push_back((t4 - t3) / 1000);
  if (c.inputs.size() < 100) return;
  auto median = [](std::vector<uint64_t>* v) {
    std::nth_element(v->begin(), v->begin() + v->size() / 2, v->end());
    return (*v)[v->size() / 2];
  };
  std::ostringstream m;
  m << "tlaloc backend: instance '" << name_ << "': " << e.id << " host time per run, median of "
    << c.inputs.size() << " runs: inputs " << median(&c.inputs) << " us, launch " << median(&c.launch)
    << " us, until the logits are on the host " << median(&c.wait) << " us, after " << median(&c.after)
    << " us (" << (device_staging_ != nullptr ? "packed inputs" : "inputs uploaded one by one") << ", "
    << (model_->overlap_logits_copy() ? "logits copy queued behind the execution"
                                      : "logits copied after the execution")
    << ")";
  // The first 100 runs of each entry are logged; later ones only verbosely.
  LOG_MESSAGE(
      clock_reported_.insert(e.id).second ? TRITONSERVER_LOG_INFO : TRITONSERVER_LOG_VERBOSE, m.str().c_str());
  c = StepClock();
}

std::string
SequenceInstance::ZeroPools(uint64_t* bytes)
{
  *bytes = 0;
  pool_address_.clear();
  const bool addresses = model_->client()->SupportsDeviceViews();
  for (const SlotSpec& p : model_->pools()) {
    const size_t size = Elements(p.dims) * tlaloc_triton::ByteWidth(p.dtype);
    std::vector<char> zeros(size, 0);
    HostInput host;
    host.data = zeros.data();
    host.byte_size = size;
    host.dtype = p.dtype;
    host.dims = p.dims;
    std::string err = PjrtBuffer::Upload(model_->client(), host, &state_[p.name]);
    if (!err.empty()) return "KV pool '" + p.name + "': " + err;
    if (addresses) {
      const void* at = nullptr;
      err = state_[p.name]->DeviceAddress(&at);
      if (!err.empty()) return "KV pool '" + p.name + "': " + err;
      pool_address_[p.name] = at;
    }
    *bytes += size;
  }
  return "";
}

void
SequenceInstance::CheckInPlace(const ServingEntrySpec& e)
{
  // Whether an entry writes the pools in place is fixed when it is compiled:
  // its first run tells, and the count covers the first 100 runs. Later runs
  // are not checked (each check asks PJRT for every pool's address).
  if (pool_address_.empty() || (runs_ >= 100 && reported_.count(e.id))) return;
  bool in_place = true;
  for (auto& kv : pool_address_) {
    const void* at = nullptr;
    if (!state_.at(kv.first)->DeviceAddress(&at).empty()) return;
    in_place &= at == kv.second;
    kv.second = at;
  }
  ++runs_;
  if (in_place) ++in_place_runs_;
  const std::string mib = std::to_string(pool_bytes_ / (1024 * 1024)) + " MiB";
  const std::string pools = std::to_string(pool_address_.size()) + " KV pools";
  if (reported_.insert(e.id).second) {
    std::ostringstream m;
    m << "tlaloc backend: instance '" << name_ << "': " << e.id;
    if (in_place) {
      m << " updated the " << pools << " in place (each output at the device address of its "
        << "pool; " << mib << " not copied)";
    } else {
      m << " wrote the " << pools << " to new device memory, copying " << mib << " per run ("
        << (model_->donate_pools() ? "the artifact does not alias them to its outputs"
                                   : "donate_kv_pools is false")
        << ")";
    }
    LOG_MESSAGE(in_place ? TRITONSERVER_LOG_INFO : TRITONSERVER_LOG_WARN, m.str().c_str());
  }
  if (runs_ == 100) {
    std::ostringstream m;
    m << "tlaloc backend: instance '" << name_ << "': in 100 runs the " << pools << " were updated "
      << "in place " << in_place_runs_ << " times and copied " << runs_ - in_place_runs_ << " times";
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, m.str().c_str());
  }
}

void
SequenceInstance::Free(uint64_t corrid, const char* why)
{
  auto it = sequences_.find(corrid);
  if (it == sequences_.end()) return;
  pool_.Give(it->second.pages);
  window_pool_.Give(it->second.ring);
  if (!it->second.spec_slots.empty()) {
    for (int slot : it->second.spec_slots) free_states_.insert(slot);
  } else if (it->second.state_slot >= 0) {
    free_states_.insert(it->second.state_slot);
  }
  std::ostringstream m;
  m << "tlaloc backend: instance '" << name_ << "': sequence " << corrid << " " << why << "; freed "
    << it->second.pages.size() << " pages (" << pool_.free_count() << " of " << pool_.capacity()
    << " free)";
  if (model_->windowed()) {
    m << " and " << it->second.ring.size() << " windowed pages (" << window_pool_.free_count()
      << " of " << window_pool_.capacity() << " free)";
  }
  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, m.str().c_str());
  sequences_.erase(it);
}

// Triton ends a sequence that has had no request for
// max_sequence_idle_microseconds and does not tell the backend, so the
// backend frees such sequences' pages itself, when a sequence needs pages the
// pool does not have. It must never free a sequence Triton still holds.
// Triton measures idleness from a request's arrival, and a request waiting in
// its queue keeps the sequence alive; the backend only sees `last_ns`, the end
// of the sequence's last execution, and not the requests still queued. The
// oldest strategy serves the oldest ready requests first, max_batch_size per
// execution, so a queued request waits for at most one execution per
// max_batch_size other live sequences. A sequence may therefore be freed only
// when it has no request in this batch and has been idle for longer than
//   2 * timeout + ceil(live sequences / max_batch_size) * longest execution,
// by when Triton has ended it (twice the timeout covers the reaper's own
// delay), and a later request for it must carry START. Of those, the least
// recently active are freed first, and only until the request's pages are
// free: a sequence ended with END has already given its pages back.
uint64_t
SequenceInstance::ReclaimLimit() const
{
  const uint64_t batch = static_cast<uint64_t>(std::max<int64_t>(1, model_->max_batch_size()));
  const uint64_t queue = (sequences_.size() + batch - 1) / batch * max_exec_ns_;
  return 2 * model_->idle_ns() + queue;
}

void
SequenceInstance::Reclaim(uint64_t now, int need, int need_ring, uint64_t for_id, int need_state)
{
  const uint64_t limit = ReclaimLimit();
  std::vector<std::pair<uint64_t, uint64_t>> idle;  // last_ns, id
  std::set<uint64_t> queued;  // sequences with a request waiting to run
  {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& w : inbox_) queued.insert(w->corrid);
  }
  for (const auto& kv : sequences_) {
    if (batch_.count(kv.first) || queued.count(kv.first)) continue;
    if (now > kv.second.last_ns && now - kv.second.last_ns > limit) idle.emplace_back(kv.second.last_ns, kv.first);
  }
  std::sort(idle.begin(), idle.end());
  for (const auto& e : idle) {
    if (pool_.free_count() >= need && window_pool_.free_count() >= need_ring &&
        static_cast<int>(free_states_.size()) >= need_state) {
      break;
    }
    auto it = sequences_.find(e.second);
    std::ostringstream m;
    m << "tlaloc backend: instance '" << name_ << "': reclaimed sequence " << e.second
      << " (the least recently active, idle " << (now - e.first) / 1000 << " us, past the limit of "
      << limit / 1000 << " us: twice max_sequence_idle_microseconds plus queueing) for sequence "
      << for_id << ": its " << it->second.pages.size() << " pages";
    if (model_->windowed()) m << " and " << it->second.ring.size() << " windowed pages";
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, m.str().c_str());
    Free(e.second, "timed out");
  }
}

std::string
SequenceInstance::Holders(uint64_t now, uint64_t except) const
{
  std::vector<std::pair<size_t, uint64_t>> held;  // pages, id
  for (const auto& kv : sequences_) {
    if (kv.first != except && !kv.second.pages.empty()) held.emplace_back(kv.second.pages.size(), kv.first);
  }
  std::sort(held.begin(), held.end(), [](const auto& a, const auto& b) {
    return a.first != b.first ? a.first > b.first : a.second < b.second;
  });
  std::ostringstream m;
  m << held.size() << " other sequence(s) hold pages";
  const size_t shown = std::min<size_t>(held.size(), 6);
  for (size_t i = 0; i < shown; ++i) {
    const SequenceState& s = sequences_.at(held[i].second);
    m << (i == 0 ? ": " : ", ") << held[i].second << " (" << held[i].first << " pages, " << s.length
      << " tokens, ";
    if (s.last_ns == 0) {
      m << "in this batch)";
    } else {
      m << "idle " << (now > s.last_ns ? (now - s.last_ns) / 1000000 : 0) << " ms)";
    }
  }
  if (held.size() > shown) m << ", ...";
  m << "; none of them has been idle past the reclaim limit of " << ReclaimLimit() / 1000000
    << " ms, and a sequence Triton may still hold is not preempted";
  return m.str();
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
  // A generation: the request parameters max_tokens (an integer) and end_tokens
  // (an integer, or a string of comma-separated ids).
  uint32_t count = 0;
  RETURN_IF_ERROR(TRITONBACKEND_RequestParameterCount(w->request, &count));
  for (uint32_t i = 0; i < count; ++i) {
    const char* key = nullptr;
    TRITONSERVER_ParameterType type;
    const void* value = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_RequestParameter(w->request, i, &key, &type, &value));
    const std::string k = key;
    if (k != "max_tokens" && k != "end_tokens") continue;
    std::vector<int64_t> v;
    if (type == TRITONSERVER_PARAMETER_INT) {
      v.push_back(*static_cast<const int64_t*>(value));
    } else if (type == TRITONSERVER_PARAMETER_STRING) {
      std::stringstream ss(static_cast<const char*>(value));
      std::string part;
      while (std::getline(ss, part, ',')) {
        if (part.empty()) continue;
        try {
          v.push_back(std::stoll(part));
        } catch (...) {
          return Invalid("request parameter " + k + " = '" + static_cast<const char*>(value) + "' is not a list of integers");
        }
      }
    } else {
      return Invalid("request parameter " + k + " must be an integer or a string of integers");
    }
    if (k == "max_tokens") {
      if (v.size() != 1 || v[0] < 0) return Invalid("request parameter max_tokens is one non-negative count");
      w->max_tokens = static_cast<int>(v[0]);
    } else {
      w->end_tokens.assign(v.begin(), v.end());
    }
  }
  if (w->max_tokens > 0 && model_->next_tokens_output().empty()) {
    return Invalid("max_tokens: this model returns logits, and the backend generates only with a speculative "
                   "artifact (its NEXT_TOKENS output)");
  }
  for (size_t i = 0; i < w->tokens.size(); ++i) {
    const int32_t t = w->tokens[i];
    if (t < 0 || t >= model_->vocab()) {
      return Invalid(
          "token id " + std::to_string(t) + " is outside the vocabulary [0, " +
          std::to_string(model_->vocab()) + ")");
    }
    auto refused = model_->refused_tokens().find(t);
    if (refused != model_->refused_tokens().end()) {
      return Invalid(
          "token " + std::to_string(i) + " of the request is " + std::to_string(t) + ", the model's " +
          refused->second + " placeholder; this artifact serves the text decoder only (the "
          "vision encoder is not in it, so the placeholder would be embedded as an ordinary "
          "token), and the backend refuses it. " +
          (w->start ? std::string("The sequence holds no KV state: start it again without the placeholder")
                    : std::string("The sequence's state is unchanged")));
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
  // Speculative: a one-token request carrying the sequence's pending token is a
  // verify step with the drafts. A call also writes KV past its tokens: a
  // verify step at the drafts and, like a prefill call, at the head's own
  // draft positions after the last token.
  const int K = model_->mtp_drafts();
  w->verify = K > 0 && !w->start && n == 1 && seq.pending >= 0 && w->tokens[0] == seq.pending &&
              static_cast<int>(seq.drafts.size()) == K;
  const int extra = K == 0 ? 0 : (w->verify ? 2 * K - 1 : K - 1);
  if (seq.length + n + extra > model_->max_context()) {
    return refuse(Invalid(
        "sequence " + std::to_string(id) + " would reach " + std::to_string(seq.length + n + extra) +
        " positions; the largest context this model was compiled for is " +
        std::to_string(model_->max_context())));
  }
  const int bs = model_->block_size();
  const int blocks = (seq.length + n + extra + bs - 1) / bs;
  const int need = blocks - static_cast<int>(seq.pages.size());
  const int need_ring =
      model_->windowed() ? std::min(blocks, model_->ring_pages()) - static_cast<int>(seq.ring.size()) : 0;
  // A linear-attention model's sequence holds one state slot for its whole life,
  // a speculative one 2 + drafts.
  const int need_state = model_->state_slots() > 0 && seq.state_slot < 0 ? (K > 0 ? K + 2 : 1) : 0;
  if ((need > 0 && need > pool_.free_count()) || (need_ring > 0 && need_ring > window_pool_.free_count()) ||
      need_state > static_cast<int>(free_states_.size())) {
    Reclaim(NowNs(), std::max(need, 0), std::max(need_ring, 0), id, need_state);
  }
  // What a refused request leaves behind: a new sequence nothing, a sequence
  // mid-generation its pages and KV, so the same request can be sent again.
  auto what_now = [&]() {
    const std::string sid = std::to_string(id);
    if (w->start) {
      return "Sequence " + sid + " is refused at START and holds nothing; start it again once "
             "pages are free";
    }
    return "Sequence " + sid + " keeps its " + std::to_string(seq.pages.size()) + " pages and the KV of its " +
           std::to_string(seq.length) + " tokens: send this request again once pages are free, or end "
           "sequence " + sid + " (sequence_end)";
  };
  if (need_ring > 0 && need_ring > window_pool_.free_count()) {
    return refuse(Err(
        TRITONSERVER_ERROR_UNAVAILABLE,
        "windowed KV page pool exhausted: sequence " + std::to_string(id) + " needs " +
            std::to_string(need_ring) + " more windowed page(s) and " +
            std::to_string(window_pool_.free_count()) + " of " + std::to_string(window_pool_.capacity()) +
            " are free (" + std::to_string(sequences_.size()) + " sequences hold the rest, at most " +
            std::to_string(model_->ring_pages()) + " each; " + Holders(NowNs(), id) + "). " + what_now() +
            ". Or export the artifact with more windowed pages"));
  }
  if (need > 0 && pool_.free_count() < need) {
    return refuse(Err(
        TRITONSERVER_ERROR_UNAVAILABLE,
        "KV page pool exhausted: sequence " + std::to_string(id) + " needs " +
            std::to_string(need) + " more page(s) of " + std::to_string(bs) + " tokens to grow from " +
            std::to_string(seq.length) + " to " + std::to_string(seq.length + n) + " tokens, and " +
            std::to_string(pool_.free_count()) + " of " + std::to_string(pool_.capacity()) + " are free (" +
            Holders(NowNs(), id) + "). " + what_now() + ". Or export the artifact with more pages (numBlocks)"));
  }
  if (need_state > static_cast<int>(free_states_.size())) {
    return refuse(Err(
        TRITONSERVER_ERROR_UNAVAILABLE,
        "linear-attention state slots exhausted: sequence " + std::to_string(id) + " needs " +
            std::to_string(need_state) + " and " + std::to_string(free_states_.size()) + " of " +
            std::to_string(model_->state_slots()) + " are free (" + std::to_string(sequences_.size()) +
            " sequences hold the rest; " + Holders(NowNs(), id) + "). " + what_now() +
            ". Or export the artifact with more state slots (stateSlots)"));
  }
  if (need > 0) pool_.Take(need, &seq.pages);
  if (need_ring > 0) window_pool_.Take(need_ring, &seq.ring);
  if (need_state > 0) {
    for (int i = 0; i < need_state; ++i) {
      if (K > 0) seq.spec_slots.push_back(*free_states_.begin());
      if (i == 0) seq.state_slot = *free_states_.begin();
      free_states_.erase(free_states_.begin());
    }
  }
  return nullptr;
}

TRITONSERVER_Error*
SequenceInstance::Run(
    const ServingEntrySpec& e, const std::vector<Work*>& rows, uint64_t* compute_start,
    uint64_t* compute_end)
{
  if (pools_lost_) {
    return Err(
        TRITONSERVER_ERROR_INTERNAL,
        e.id + ": an earlier execution in this batch failed and took the KV pools with it; "
        "every sequence of instance '" + name_ + "' has lost its KV state");
  }
  const int B = e.batch, T = e.tokens_per_seq, M = e.max_blocks, bs = model_->block_size();
  // The padding convention (DecodePadding): token 0, position 0, page 0,
  // sequence length 1, slot -1 (the KV write is dropped).
  std::vector<int32_t> tokens(size_t(B) * T, 0), positions(size_t(B) * T, 0);
  std::vector<int32_t> tables(size_t(B) * M, 0), lens(B, 1), slots(size_t(B) * T, -1);
  // The windowed layers' tables: logical block b is on ring page b % R.
  const bool windowed = model_->windowed();
  const int R = model_->ring_pages();
  std::vector<int32_t> wtables(windowed ? size_t(B) * M : 0, 0), wslots(windowed ? size_t(B) * T : 0, -1);
  // Each row's linear-attention state slot; a padding row's tokens are dead
  // through its slot mapping, so any slot does.
  std::vector<int32_t> sslots(B, 0);
  // Speculative: where a verify step writes the state after each token, the
  // sequence's other slots (-1 elsewhere: a prefill call writes in place).
  std::vector<int32_t> write_slots(model_->speculative() ? size_t(B) * T : 0, -1);
  std::vector<std::vector<int>> writes(rows.size());
  for (size_t r = 0; r < rows.size(); ++r) {
    const Work* w = rows[r];
    if (w->verify && !e.prefill) {
      for (int slot : w->seq->spec_slots) {
        if (slot != w->seq->state_slot && static_cast<int>(writes[r].size()) < T) writes[r].push_back(slot);
      }
      for (int j = 0; j < T; ++j) write_slots[r * T + j] = writes[r][j];
    }
    const int n = static_cast<int>(w->tokens.size());
    const int pad = T - n;  // right-aligned: the last token is at T - 1
    for (int j = 0; j < n; ++j) {
      const int pos = w->position + j;
      const size_t k = r * T + pad + j;
      tokens[k] = w->tokens[j];
      positions[k] = pos;
      slots[k] = w->seq->pages[pos / bs] * bs + pos % bs;
      if (windowed) wslots[k] = w->seq->ring[(pos / bs) % R] * bs + pos % bs;
    }
    for (int j = 0; j < M && j < static_cast<int>(w->seq->pages.size()); ++j) {
      tables[r * M + j] = w->seq->pages[j];
      if (windowed) wtables[r * M + j] = w->seq->ring[j % R];
    }
    lens[r] = w->position + n;
    if (w->seq->state_slot >= 0) sslots[r] = w->seq->state_slot;
  }
  const uint64_t t0 = NowNs();
  // The step's integer inputs: written into the mapped staging memory, which
  // the execution reads in place through views made on the entry's first run
  // (the previous execution that read it has finished: every run waits for
  // its own); or uploaded one by one.
  StepLayout* layout = nullptr;
  if (device_staging_ != nullptr) {
    std::string lerr = Layout(e, &layout);
    if (!lerr.empty()) return Err(TRITONSERVER_ERROR_INTERNAL, e.id + ": " + lerr);
  }
  std::vector<std::unique_ptr<PjrtBuffer>> uploads;
  std::vector<ExecuteArg> args(e.inputs.size());
  for (size_t i = 0; i < e.inputs.size(); ++i) {
    const SlotSpec& s = e.inputs[i];
    const std::vector<int32_t>* v = nullptr;
    if (s.role == "TOKEN_IDS") v = &tokens;
    else if (s.role == "POSITIONS") v = &positions;
    else if (s.role == "BLOCK_TABLES") v = &tables;
    else if (s.role == "SEQ_LENS") v = &lens;
    else if (s.role == "SLOT_MAPPING") v = &slots;
    else if (s.role == "WINDOW_BLOCK_TABLES") v = &wtables;
    else if (s.role == "WINDOW_SLOT_MAPPING") v = &wslots;
    else if (s.role == "STATE_SLOTS") v = &sslots;
    else if (s.role == "STATE_WRITE_SLOTS") v = &write_slots;
    if (v != nullptr) {
      if (layout != nullptr) {
        std::memcpy(static_cast<char*>(host_staging_) + layout->offset[i], v->data(), v->size() * 4);
        args[i].device = layout->views[i].get();
      } else {
        HostInput h;
        h.data = v->data();
        h.byte_size = v->size() * 4;
        h.dtype = DType::I32;
        h.dims = s.dims;
        uploads.emplace_back();
        std::string uerr = PjrtBuffer::Upload(model_->client(), h, &uploads.back());
        if (!uerr.empty()) return Err(TRITONSERVER_ERROR_INTERNAL, e.id + ": input '" + s.name + "': " + uerr);
        args[i].device = uploads.back().get();
      }
    } else if (IsPoolIn(s.role)) {
      // Donated: the artifact aliases each KV_POOL_OUT to its KV_POOL_IN, so
      // the step writes the pool in place and hands it back as that output.
      args[i].device = state_.at(s.name).get();
      args[i].donate = model_->donate_pools();
    } else {
      args[i].device = model_->weight(s.name);
    }
  }
  const uint64_t t1 = NowNs();
  *compute_start = t0;
  std::unique_ptr<PjrtResults> results;
  std::string err = e.executable->Execute(args, &results, /*wait=*/false);
  const uint64_t t2 = NowNs();
  // The logits: asked for now, the copy follows the execution on the device
  // and the host wakes once; or after the host has seen the execution end.
  const bool spec = model_->speculative();
  std::vector<float> all(spec ? 0 : size_t(B) * model_->vocab());
  size_t logits_at = e.outputs.size(), next_at = 0, accepted_at = 0, drafts_at = 0;
  for (size_t j = 0; j < e.outputs.size(); ++j) {
    if (e.outputs[j].role == "LOGITS") logits_at = j;
    if (e.outputs[j].role == "NEXT_TOKENS") next_at = j;
    if (e.outputs[j].role == "ACCEPTED") accepted_at = j;
    if (e.outputs[j].role == "DRAFTS") drafts_at = j;
  }
  const int K = model_->mtp_drafts();
  std::vector<int32_t> next(spec ? size_t(B) * (K + 1) : 0), accepted(spec ? B : 0), drafts(spec ? size_t(B) * K : 0);
  std::string copy_err;
  if (err.empty() && !spec && model_->overlap_logits_copy()) {
    copy_err = results->CopyToHost(logits_at, all.data(), all.size() * 4);
  }
  if (err.empty()) err = results->Await();
  if (!err.empty()) {
    *compute_end = NowNs();
    for (const auto& kv : state_) pools_lost_ |= kv.second->IsDeleted();
    return Err(
        TRITONSERVER_ERROR_INTERNAL,
        e.id + ": " + err +
            (pools_lost_ ? "; the execution took the KV pools with it, so every sequence of instance '" +
                               name_ + "' loses its KV state"
                         : ""));
  }
  if (spec) {
    copy_err = results->CopyToHost(next_at, next.data(), next.size() * 4);
    if (copy_err.empty()) copy_err = results->CopyToHost(accepted_at, accepted.data(), accepted.size() * 4);
    if (copy_err.empty() && K > 0) copy_err = results->CopyToHost(drafts_at, drafts.data(), drafts.size() * 4);
  } else if (!model_->overlap_logits_copy()) {
    copy_err = results->CopyToHost(logits_at, all.data(), all.size() * 4);
  }
  const uint64_t t3 = NowNs();
  // The pools first: a donated pool now lives only in the results.
  for (size_t j = 0; j < e.outputs.size(); ++j) {
    if (e.replaces[j] >= 0) state_[e.inputs[e.replaces[j]].name] = results->Release(j);
  }
  CheckInPlace(e);
  if (!copy_err.empty()) {
    *compute_end = NowNs();
    return Err(TRITONSERVER_ERROR_INTERNAL, e.id + ": " + copy_err);
  }
  for (size_t r = 0; r < rows.size(); ++r) {
    Work* w = rows[r];
    if (!spec) {
      w->logits.assign(all.begin() + r * model_->vocab(), all.begin() + (r + 1) * model_->vocab());
      w->seq->length = w->position + static_cast<int>(w->tokens.size());
      continue;
    }
    SequenceState& q = *w->seq;
    if (w->verify && !e.prefill) {
      // The accepted drafts and the token after them: the target is now valid
      // through the last accepted one, whose state is in its write slot.
      const int a = accepted[r];
      w->emitted.assign(next.begin() + r * (K + 1), next.begin() + r * (K + 1) + a + 1);
      q.length = w->position + a + 1;
      q.state_slot = writes[r][a];
    } else {
      w->emitted.assign(1, next[r * (K + 1)]);
      q.length = w->position + static_cast<int>(w->tokens.size());
    }
    q.pending = w->emitted.back();
    q.drafts.assign(drafts.begin() + r * K, drafts.begin() + (r + 1) * K);
  }
  *compute_end = NowNs();
  for (Work* w : rows) {
    w->pages_held = w->seq->pages.size();
    w->ring_held = w->seq->ring.size();
    w->compute_start = *compute_start;
    w->compute_end = *compute_end;
  }
  Clock(e, t0, t1, t2, t3, *compute_end);
  std::ostringstream m;
  m << "tlaloc backend: instance '" << name_ << "': " << e.id << " ran " << rows.size()
    << " sequence(s) in " << (*compute_end - *compute_start) / 1000 << " us";
  // The first call of each prefill entry with each number (> 1) of prompts
  // is logged, so that a log shows prompts were prefilled together.
  const bool first_batched =
      e.prefill && rows.size() > 1 && batched_reported_.insert(e.id + "/" + std::to_string(rows.size())).second;
  LOG_MESSAGE(first_batched ? TRITONSERVER_LOG_INFO : TRITONSERVER_LOG_VERBOSE, m.str().c_str());
  return nullptr;
}

void
SequenceInstance::RunPrompts(
    const std::vector<Work*>& works, uint64_t* compute_start, uint64_t* compute_end,
    const std::function<void()>& note)
{
  struct Prompt {
    Work* w;
    std::vector<int32_t> all;  // the request's tokens
    int start;                 // the position of all[0]
    size_t done = 0;           // tokens already run
    size_t n = 0;              // tokens of this round's call
  };
  std::vector<Prompt> prompts;
  for (Work* w : works) prompts.push_back({w, w->tokens, w->position});
  // Sets the error of every work of `group` from `err`, and deletes `err`.
  // `ran` says whether an earlier call of each work's request ran.
  auto fail = [](const std::vector<Work*>& group, const std::vector<bool>& ran, TRITONSERVER_Error* err) {
    if (err == nullptr) return;
    for (size_t i = 0; i < group.size(); ++i) {
      Work* w = group[i];
      if (w->err != nullptr) continue;
      std::string msg = TRITONSERVER_ErrorMessage(err);
      if (ran[i]) {
        w->lost = true;
        msg += "; an earlier call of this request had run, so sequence " + std::to_string(w->corrid) +
               " is freed: start it again";
      }
      w->err = Err(TRITONSERVER_ErrorCode(err), msg);
    }
    TRITONSERVER_ErrorDelete(err);
  };
  for (;;) {
    // This round's call of every request with tokens left: all of them, or
    // as many as the windowed ring can hold. The calls that a prefill entry
    // covers are keyed by the context of the smallest one; the rest run one
    // decode step per token.
    std::vector<std::pair<int, std::vector<Prompt*>>> by_context;  // in order of first appearance
    std::vector<Prompt*> stepwise;
    for (Prompt& p : prompts) {
      if (p.w->err != nullptr || p.done >= p.all.size()) continue;
      const int at = p.start + static_cast<int>(p.done);
      p.n = std::min(p.all.size() - p.done, static_cast<size_t>(model_->MaxTokensPerCall(at)));
      p.w->tokens.assign(p.all.begin() + p.done, p.all.begin() + p.done + p.n);
      p.w->position = at;
      // A speculative model's calls all reach the head's draft positions after their last token.
      const int reach = at + static_cast<int>(p.n) + std::max(0, model_->mtp_drafts() - 1);
      const ServingEntrySpec* e =
          p.n > 1 || model_->speculative() ? model_->Prefill(1, reach, static_cast<int>(p.n)) : nullptr;
      if (e == nullptr) {
        stepwise.push_back(&p);
        continue;
      }
      auto it = std::find_if(by_context.begin(), by_context.end(), [&](const auto& g) { return g.first == e->context; });
      if (it == by_context.end()) {
        by_context.push_back({e->context, {}});
        it = by_context.end() - 1;
      }
      it->second.push_back(&p);
    }
    if (by_context.empty() && stepwise.empty()) break;
    for (const auto& g : by_context) {
      // Rows of a context bucket are not merged into a longer one: a call
      // computes every row at its entry's context.
      const size_t most = static_cast<size_t>(std::max(1, model_->MaxPrefillBatch(g.first)));
      for (size_t i = 0; i < g.second.size(); i += most) {
        std::vector<Work*> rows;
        std::vector<bool> ran;
        int tokens = 1;
        for (size_t k = i; k < std::min(g.second.size(), i + most); ++k) {
          rows.push_back(g.second[k]->w);
          ran.push_back(g.second[k]->done > 0);
          tokens = std::max(tokens, static_cast<int>(g.second[k]->n));
        }
        const ServingEntrySpec* e = model_->Prefill(static_cast<int>(rows.size()), g.first, tokens);
        for (Work* w : rows) w->verify = false;
        TRITONSERVER_Error* err =
            e == nullptr ? Invalid(
                               "no prefill entry covers batch " + std::to_string(rows.size()) +
                               ", context " + std::to_string(g.first) + " and " + std::to_string(tokens) +
                               " tokens per sequence")
                         : Run(*e, rows, compute_start, compute_end);
        note();
        fail(rows, ran, err);
      }
    }
    for (Prompt* p : stepwise) {
      // No prefill entry covers the call: one decode step per token.
      Work* w = p->w;
      if (model_->speculative()) {
        fail({w}, {p->done > 0}, Invalid(
            "no prefill entry covers " + std::to_string(p->n) + " tokens at position " + std::to_string(w->position) +
            " and the head's drafts after them; a speculative model's decode entries take only verify steps"));
        continue;
      }
      const int at = w->position;
      for (size_t j = 0; j < p->n && w->err == nullptr; ++j) {
        w->tokens = {p->all[p->done + j]};
        w->position = at + static_cast<int>(j);
        const ServingEntrySpec* d = model_->Decode(1, w->position + 1);
        TRITONSERVER_Error* err =
            d == nullptr ? Invalid("no decode entry covers batch 1 and context " + std::to_string(w->position + 1))
                         : Run(*d, {w}, compute_start, compute_end);
        note();
        fail({w}, {p->done + j > 0}, err);
      }
    }
    for (auto& g : by_context) {
      for (Prompt* p : g.second) p->done += p->n;
    }
    for (Prompt* p : stepwise) p->done += p->n;
  }
  for (Prompt& p : prompts) {
    p.w->tokens = p.all;
    p.w->position = p.start;
  }
}

TRITONSERVER_Error*
SequenceInstance::Respond(Work* w)
{
  if (w->logits.empty() && w->emitted.empty()) return nullptr;
  uint32_t requested = 0;
  RETURN_IF_ERROR(TRITONBACKEND_RequestOutputCount(w->request, &requested));
  bool wanted = requested == 0 && !model_->logits_output().empty();
  bool pages = requested == 0 && !model_->pages_output().empty();
  bool next = requested == 0 && !model_->next_token_output().empty();
  bool tokens = requested == 0 && !model_->next_tokens_output().empty();
  for (uint32_t i = 0; i < requested; ++i) {
    const char* name = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_RequestOutputName(w->request, i, &name));
    wanted |= !model_->logits_output().empty() && model_->logits_output() == name;
    pages |= !model_->pages_output().empty() && model_->pages_output() == name;
    next |= !model_->next_token_output().empty() && model_->next_token_output() == name;
    tokens |= !model_->next_tokens_output().empty() && model_->next_tokens_output() == name;
  }
  if (next) {
    // The greedy choice, the first index of the largest logit (numpy's argmax); for a
    // speculative model the last token the request emits, the one to send back.
    int32_t best = 0;
    if (!w->emitted.empty()) {
      best = w->emitted.back();
    } else {
      for (size_t k = 1; k < w->logits.size(); ++k) {
        if (w->logits[k] > w->logits[best]) best = static_cast<int32_t>(k);
      }
    }
    RETURN_IF_ERROR(WriteOutput(w->response, model_->next_token_output(), TRITONSERVER_TYPE_INT32, {1}, &best, sizeof(best)));
  }
  if (tokens) {
    RETURN_IF_ERROR(WriteOutput(
        w->response, model_->next_tokens_output(), TRITONSERVER_TYPE_INT32, {1, static_cast<int64_t>(w->emitted.size())},
        w->emitted.data(), w->emitted.size() * sizeof(int32_t)));
  }
  if (pages) {
    // The pages the sequence holds after this request, in each pool class.
    const int32_t held[2] = {
        static_cast<int32_t>(w->pages_held), static_cast<int32_t>(w->ring_held)};
    RETURN_IF_ERROR(WriteOutput(w->response, model_->pages_output(), TRITONSERVER_TYPE_INT32, {2}, held, sizeof(held)));
  }
  if (!wanted) return nullptr;
  return WriteOutput(
      w->response, model_->logits_output(), TRITONSERVER_TYPE_FP32, {1, model_->vocab()},
      w->logits.data(), w->logits.size() * 4);
}

void
SequenceInstance::Intake(TRITONBACKEND_Request* request, uint64_t arrival, Work* w)
{
  w->request = request;
  w->arrival = arrival;
  w->first_arrival = arrival;
  // A decoupled model's responses all come from the request's factory, each
  // made when it is sent (Triton's gRPC stream fills them in that order).
  w->err = model_->decoupled() ? TRITONBACKEND_ResponseFactoryNew(&w->factory, w->request)
                               : TRITONBACKEND_ResponseNew(&w->response, w->request);
  if (w->err == nullptr) w->err = Parse(w);
}

void
SequenceInstance::ProcessRequests(TRITONBACKEND_Request** requests, uint32_t count)
{
  const uint64_t now = NowNs();
  std::vector<std::unique_ptr<Work>> works(count);
  for (uint32_t r = 0; r < count; ++r) {
    works[r].reset(new Work());
    Intake(requests[r], now, works[r].get());
  }
  if (!model_->backend_batching()) {
    std::vector<Work*> batch;
    for (auto& w : works) {
      if (w->err == nullptr && w->max_tokens > 0) {
        w->err = Invalid("max_tokens needs backend_batching: the backend steps a generation on its own thread");
      }
      batch.push_back(w.get());
    }
    RunBatch(batch, now);
    return;
  }
  {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& w : works) inbox_.push_back(std::move(w));
  }
  arrived_.notify_one();
}

void
SequenceInstance::Loop()
{
  std::vector<std::unique_ptr<Work>> pending;  // in arrival order
  for (;;) {
    bool waited = false, timed_out = false;
    std::vector<uint64_t> absent;
    {
      std::unique_lock<std::mutex> lock(mu_);
      for (;;) {
        while (!inbox_.empty()) {
          pending.push_back(std::move(inbox_.front()));
          inbox_.pop_front();
        }
        if (pending.empty()) {
          if (stopping_) return;
          arrived_.wait(lock);
          continue;
        }
        if (stopping_) break;
        const uint64_t now = NowNs();
        uint64_t until = std::numeric_limits<uint64_t>::max();
        // The sequences that decoded in the last batch, still held, without
        // a request in: waited for up to cohort_wait from when this batch
        // could first have run.
        std::set<uint64_t> in;
        for (const auto& w : pending) in.insert(w->corrid);
        absent.clear();
        for (uint64_t id : cohort_) {
          if (!in.count(id) && sequences_.count(id)) absent.push_back(id);
        }
        timed_out = false;
        if (!absent.empty()) {
          const uint64_t deadline =
              std::max(pending.front()->arrival, last_batch_end_) + model_->cohort_wait_ns();
          if (now < deadline) until = deadline;
          else timed_out = true;
        }
        // A prompt waits up to prompt_wait after its arrival for the prompts
        // sent with it, to be prefilled together.
        for (const auto& w : pending) {
          if (w->err != nullptr || w->tokens.size() < 2) continue;
          const uint64_t deadline = w->arrival + model_->prompt_wait_ns();
          if (now < deadline) until = std::min(until, deadline);
          break;  // the oldest prompt
        }
        if (until == std::numeric_limits<uint64_t>::max()) break;
        waited = true;
        arrived_.wait_for(lock, std::chrono::nanoseconds(until - now));
      }
    }
    // One request per sequence per batch (Triton hands over the next request
    // of a sequence only after the last one is released, so this only guards).
    std::vector<std::unique_ptr<Work>> later;
    std::vector<Work*> batch;
    std::set<uint64_t> ids;
    std::vector<std::unique_ptr<Work>> run;
    for (auto& w : pending) {
      if (w->corrid != 0 && !ids.insert(w->corrid).second) {
        later.push_back(std::move(w));
      } else {
        batch.push_back(w.get());
        run.push_back(std::move(w));
      }
    }
    pending = std::move(later);
    ++batches_;
    waited_batches_ += waited || timed_out;
    if (timed_out) {
      ++timed_out_batches_;
      std::ostringstream m;
      m << "tlaloc backend: instance '" << name_ << "': a batch of " << batch.size()
        << " request(s) ran without sequence(s)";
      for (size_t i = 0; i < absent.size() && i < 6; ++i) m << " " << absent[i];
      if (absent.size() > 6) m << " ...";
      m << " of the previous batch after waiting " << model_->cohort_wait_ns() / 1000
        << " us (cohort_wait_microseconds); " << timed_out_batches_ << " of " << batches_
        << " batches so far ran without all of the previous one";
      LOG_MESSAGE(timed_out_batches_ <= 10 ? TRITONSERVER_LOG_INFO : TRITONSERVER_LOG_VERBOSE, m.str().c_str());
    }
    RunBatch(batch, NowNs());
    // Generations with tokens left run again in the next batch, before
    // anything that arrived meanwhile.
    {
      std::vector<std::unique_ptr<Work>> again;
      for (auto& w : run) {
        if (w->again) again.push_back(std::move(w));
      }
      if (!again.empty()) {
        for (auto& w : pending) again.push_back(std::move(w));
        pending = std::move(again);
      }
    }
    // The next batch waits for the sequences that took a decode step in this
    // one, were not refused and are still held: a client generating tokens
    // sends its next step as soon as it has the logits. (A sequence whose
    // prompt was just prefilled is not waited for: its client may hold it.)
    cohort_.clear();
    for (Work* w : batch) {
      // A verify step's tokens are its pending token and the drafts.
      if (w->ok && (w->tokens.size() == 1 || w->verify) && sequences_.count(w->corrid)) cohort_.insert(w->corrid);
    }
    last_batch_end_ = NowNs();
    if (batches_ % 1000 == 0) {
      std::ostringstream m;
      m << "tlaloc backend: instance '" << name_ << "': of " << batches_ << " batches, " << waited_batches_
        << " waited for sequences of the previous batch and " << timed_out_batches_
        << " ran without all of them";
      LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, m.str().c_str());
    }
  }
}

void
SequenceInstance::RunBatch(const std::vector<Work*>& batch, uint64_t exec_start)
{
  const uint32_t count = static_cast<uint32_t>(batch.size());
  batch_.clear();
  for (Work* wp : batch) {
    Work& w = *wp;
    // A sequence with a request in the batch, refused or not, is not reclaimed.
    if (w.corrid != 0) batch_.insert(w.corrid);
    if (w.err != nullptr && w.start) {
      // Triton has started a new sequence under this ID all the same: the
      // KV of an earlier sequence with the ID must not be continued.
      Free(w.corrid, "was started again by a refused request");
    }
  }
  for (Work* w : batch) {
    if (w->err == nullptr) w->err = Admit(w);
  }

  // The requests with several tokens run as prefill calls, together where a
  // prefill entry of batch > 1 takes them (RunPrompts). Decode: every
  // one-token request, several sequences per call.
  std::vector<Work*> decode, several;
  for (Work* w : batch) {
    if (w->err != nullptr || w->tokens.empty()) continue;
    // Speculative: a one-token request that is not its sequence's pending token runs as a prefill call.
    (w->tokens.size() == 1 && (!model_->speculative() || w->verify) ? decode : several).push_back(w);
  }
  uint64_t first_compute = 0, last_compute = 0, cs = 0, ce = 0;
  auto note = [&]() {
    if (first_compute == 0) first_compute = cs;
    last_compute = ce;
  };
  RunPrompts(several, &cs, &ce, note);
  const size_t max_b = static_cast<size_t>(model_->max_decode_batch());
  for (size_t i = 0; i < decode.size(); i += max_b) {
    std::vector<Work*> rows(decode.begin() + i, decode.begin() + std::min(decode.size(), i + max_b));
    int context = 0;
    const int K = model_->mtp_drafts();
    for (Work* w : rows) {
      // A verify step runs the pending token and the drafts, and the head writes K - 1 positions past them.
      if (w->verify) w->tokens.insert(w->tokens.end(), w->seq->drafts.begin(), w->seq->drafts.end());
      context = std::max(context, w->position + (K > 0 ? 2 * K : 1));
    }
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

  for (Work* wp : batch) {
    Work& w = *wp;
    w.again = false;
    if (w.max_tokens > 0 && w.err == nullptr && w.seq != nullptr) {
      // A generation step: its tokens join the generation; it runs again
      // unless it has its tokens or met an end token.
      const std::vector<int32_t> step = w.emitted;
      w.generated.insert(w.generated.end(), step.begin(), step.end());
      bool stop = static_cast<int>(w.generated.size()) >= w.max_tokens || step.empty();
      for (int32_t t : step) {
        for (int32_t e : w.end_tokens) stop |= t == e;
      }
      if (w.factory != nullptr && !stop) {
        // Decoupled: this step's tokens now; the last step's go with the final response below.
        TRITONBACKEND_Response* r = nullptr;
        TRITONSERVER_Error* e = TRITONBACKEND_ResponseNewFromFactory(&r, w.factory);
        if (e == nullptr) {
          e = WriteOutput(r, model_->next_tokens_output(), TRITONSERVER_TYPE_INT32, {1, static_cast<int64_t>(step.size())},
                          step.data(), step.size() * sizeof(int32_t));
          LOG_IF_ERROR(TRITONBACKEND_ResponseSend(r, 0, e), "failed to send a generation step");
          if (e != nullptr) TRITONSERVER_ErrorDelete(e);
        } else {
          LOG_IF_ERROR(e, "failed to create a generation step's response");
        }
      }
      if (!stop) {
        if (auto it = sequences_.find(w.corrid); it != sequences_.end()) it->second.last_ns = NowNs();
        w.again = true;
        w.tokens.assign(1, w.seq->pending);
        w.start = false;
        w.logits.clear();
        w.emitted.clear();
        w.verify = false;
        w.compute_start = w.compute_end = 0;
        w.arrival = NowNs();
        continue;
      }
      // The response carries the whole generation, or (decoupled) its last step.
      w.emitted = w.factory != nullptr ? step : w.generated;
    }
    if (w.factory != nullptr && w.response == nullptr) {
      LOG_IF_ERROR(TRITONBACKEND_ResponseNewFromFactory(&w.response, w.factory), "failed to create the final response");
    }
    if (w.err == nullptr && w.response != nullptr) w.err = Respond(&w);
    if (w.end) {
      Free(w.corrid, w.err == nullptr ? "ended" : "ended with an error");
    } else if (w.lost) {
      Free(w.corrid, "lost its place: a call after the request's first failed");
    } else if (auto it = sequences_.find(w.corrid); it != sequences_.end()) {
      // Every request Triton hands over, refused or not, restarts Triton's
      // idle timer for the sequence, so it restarts the backend's too.
      it->second.last_ns = NowNs();
    }
    const bool ok = w.err == nullptr;
    w.ok = ok;
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
    // From the request's arrival: time it waited for its batch (backend
    // batching) counts as the request's input time.
    const uint64_t start = std::min(w.first_arrival != 0 ? w.first_arrival : w.arrival, exec_start);
    const uint64_t cstart = w.compute_start ? w.compute_start : exec_start;
    const uint64_t cend = w.compute_end ? w.compute_end : cstart;
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            instance_, w.request, ok, start, cstart, cend, NowNs()),
        "failed to report request statistics");
    if (w.err != nullptr) TRITONSERVER_ErrorDelete(w.err);
    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(w.request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed to release the request");
    if (w.factory != nullptr) {
      LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryDelete(w.factory), "failed to delete a response factory");
      w.factory = nullptr;
    }
  }
  if (pools_lost_) {
    // Every sequence's KV state went with the donated pools: free them all
    // (a later request for one must START again) and zero new pools.
    std::vector<uint64_t> all;
    for (const auto& kv : sequences_) all.push_back(kv.first);
    for (uint64_t id : all) Free(id, "lost its KV state with a failed execution");
    uint64_t bytes = 0;
    std::string err = ZeroPools(&bytes);
    LOG_MESSAGE(
        TRITONSERVER_LOG_ERROR,
        ("tlaloc backend: instance '" + name_ + "': a failed execution took the KV pools; " +
         std::to_string(all.size()) + " sequence(s) freed; " +
         (err.empty() ? "new pools zeroed" : "zeroing new pools failed: " + err))
            .c_str());
    pools_lost_ = !err.empty();
  }
  max_exec_ns_ = std::max(max_exec_ns_, NowNs() - exec_start);
  if (first_compute == 0) first_compute = last_compute = NowNs();
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          instance_, count, exec_start, first_compute, last_compute, NowNs()),
      "failed to report batch statistics");
}

}}}  // namespace triton::backend::tlaloc
