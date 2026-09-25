// Copyright 2026 Pedro N. Rodriguez
// SPDX-License-Identifier: Apache-2.0
//
// Sequence mode of the tlaloc backend: a model whose config.pbtxt names a
// Tlaloc serving artifact manifest (parameter "serving_manifest") and uses
// Triton's sequence batcher. The backend keeps each sequence's KV pages and
// length, keyed by correlation ID, and chooses the artifact entry (prefill or
// decode, batch and context bucket) for every request itself. A client sends
// token ids and gets the logits of the last one back.

#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "pjrt_runtime.h"
#include "triton/backend/backend_common.h"
#include "triton/core/tritonbackend.h"

namespace triton { namespace backend { namespace tlaloc {

// One tensor of an entry's signature, as the manifest describes it.
struct SlotSpec {
  std::string name;
  std::string role;  // DecodeSlotRole name: TOKEN_IDS, ..., KV_POOL_IN, WEIGHT, LOGITS, KV_POOL_OUT
  tlaloc_triton::DType dtype = tlaloc_triton::DType::UNSUPPORTED;
  std::vector<int64_t> dims;
};

// One compiled ladder point of the artifact.
struct ServingEntrySpec {
  bool prefill = false;
  int batch = 0;
  int context = 0;
  int tokens_per_seq = 0;
  int max_blocks = 0;
  std::string id;         // e.g. decode_b4_c64
  std::string body_path;  // relative to the version directory
  std::string entry_point;
  std::vector<SlotSpec> inputs;
  std::vector<SlotSpec> outputs;
  // For each output, the index of the input it replaces (KV_POOL_OUT), or -1.
  std::vector<int> replaces;
  tlaloc_triton::PjrtExecutable* executable = nullptr;
};

class SequenceModel {
 public:
  // Reads the manifest named by the model's "serving_manifest" parameter and
  // checks config.pbtxt against it; then, and only then, gets a PJRT client
  // from `acquire`, compiles every entry and uploads the staged weights.
  static TRITONSERVER_Error* Load(
      const std::string& model_name, const std::string& version_dir,
      const std::string& manifest_file, triton::common::TritonJson::Value& config,
      const std::function<TRITONSERVER_Error*(std::shared_ptr<tlaloc_triton::PjrtClient>*)>& acquire,
      std::unique_ptr<SequenceModel>* out);

  const std::string& name() const { return name_; }
  tlaloc_triton::PjrtClient* client() const { return client_.get(); }
  const std::vector<ServingEntrySpec>& entries() const { return entries_; }
  const tlaloc_triton::PjrtBuffer* weight(const std::string& name) const
  {
    return weights_.at(name).get();
  }

  int vocab() const { return vocab_; }
  int num_blocks() const { return num_blocks_; }
  int block_size() const { return block_size_; }
  int max_context() const { return max_context_; }
  int max_decode_batch() const { return max_decode_batch_; }
  uint64_t idle_ns() const { return idle_ns_; }
  const std::string& tokens_input() const { return tokens_input_; }
  const std::string& logits_output() const { return logits_output_; }
  const std::string& start_input() const { return start_input_; }
  const std::string& end_input() const { return end_input_; }
  const std::string& corrid_input() const { return corrid_input_; }
  // KV pool state names and their type, in the order entries read them.
  const std::vector<SlotSpec>& pools() const { return pools_; }

  // The cheapest decode entry with batch >= `batch` and context >= `context`,
  // or nullptr.
  const ServingEntrySpec* Decode(int batch, int context) const;
  // The cheapest prefill entry of batch >= 1 and context >= `context`, or
  // nullptr.
  const ServingEntrySpec* Prefill(int context) const;

 private:
  SequenceModel() = default;
  TRITONSERVER_Error* ReadConfig(triton::common::TritonJson::Value& config);
  TRITONSERVER_Error* ReadManifest(const std::string& text);
  TRITONSERVER_Error* CompileEntries();
  TRITONSERVER_Error* UploadWeights();
  std::string Where() const { return "model '" + name_ + "': "; }

  // Declared first, so that executables and buffers go before the client.
  std::shared_ptr<tlaloc_triton::PjrtClient> client_;
  std::string name_;
  std::string version_dir_;
  std::string manifest_path_;
  std::vector<ServingEntrySpec> entries_;
  std::map<std::string, std::unique_ptr<tlaloc_triton::PjrtExecutable>> executables_;
  std::map<std::string, std::unique_ptr<tlaloc_triton::PjrtBuffer>> weights_;
  std::vector<std::pair<std::string, std::string>> weight_files_;  // slot name, path
  std::vector<SlotSpec> weight_slots_;
  std::vector<SlotSpec> pools_;
  int vocab_ = 0;
  int num_blocks_ = 0;
  int block_size_ = 0;
  int max_context_ = 0;
  int max_decode_batch_ = 0;
  int64_t max_batch_size_ = 0;
  uint64_t idle_ns_ = 0;
  std::string tokens_input_, logits_output_, start_input_, end_input_, corrid_input_;
};

// The KV pages of one model instance. Page 0 is never handed out: it is the
// page padding rows of a batch point their block tables at.
class PagePool {
 public:
  explicit PagePool(int num_blocks);
  int free_count() const { return static_cast<int>(free_.size()); }
  int capacity() const { return capacity_; }
  // Takes `n` pages (lowest first); false, and nothing taken, if fewer are free.
  bool Take(int n, std::vector<int>* pages);
  void Give(const std::vector<int>& pages);

 private:
  int capacity_ = 0;
  std::set<int> free_;
};

struct SequenceState {
  std::vector<int> pages;
  int length = 0;  // tokens whose KV is in the pool
  uint64_t last_ns = 0;
};

class SequenceInstance {
 public:
  static TRITONSERVER_Error* Create(
      const SequenceModel* model, const std::string& name,
      TRITONBACKEND_ModelInstance* instance, std::unique_ptr<SequenceInstance>* out);

  void ProcessRequests(TRITONBACKEND_Request** requests, uint32_t count);

 private:
  struct Work;
  SequenceInstance(const SequenceModel* model, const std::string& name,
                   TRITONBACKEND_ModelInstance* instance)
      : model_(model), name_(name), instance_(instance), pool_(model->num_blocks())
  {
  }
  TRITONSERVER_Error* Parse(Work* w);
  TRITONSERVER_Error* Admit(Work* w);
  void Reap(uint64_t now);
  void Free(uint64_t corrid, const char* why);
  // Runs `entry` on the given works (one row each, `tokens` of each placed in
  // its row), stores the state results and each work's logits row.
  TRITONSERVER_Error* Run(const ServingEntrySpec& entry, const std::vector<Work*>& rows,
                          uint64_t* compute_start, uint64_t* compute_end);
  TRITONSERVER_Error* Respond(Work* w);

  const SequenceModel* model_;
  std::string name_;
  TRITONBACKEND_ModelInstance* instance_;
  PagePool pool_;
  std::unordered_map<uint64_t, SequenceState> sequences_;
  std::map<std::string, std::unique_ptr<tlaloc_triton::PjrtBuffer>> state_;
};

}}}  // namespace triton::backend::tlaloc
