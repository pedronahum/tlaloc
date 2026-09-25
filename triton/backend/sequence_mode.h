// Copyright 2026 Pedro N. Rodriguez
// SPDX-License-Identifier: Apache-2.0
//
// Sequence mode of the tlaloc backend: a model whose config.pbtxt names a
// Tlaloc serving artifact manifest (parameter "serving_manifest") and uses
// Triton's sequence batcher. The backend keeps each sequence's KV pages and
// length, keyed by correlation ID, and chooses the artifact entry (prefill or
// decode, batch and context bucket) for every request itself. A client sends
// token ids and gets the logits of the last one back.
//
// An artifact with a windowed KV pool (tlaloc-serving-v3) has a second pool
// class for its sliding-window layers. Each sequence holds a ring of at most
// ring_pages pages of it: logical block b is on the ring's page b % ring_pages,
// so position p is written over position p - ring_pages * block_size, which
// has left the window. A request is split into calls of at most
// ring_pages * block_size - min(start, window - 1) tokens, so that no call
// writes over a position one of its own rows still reads.
//
// When a sequence needs pages the pool does not have, the backend reclaims
// pages from sequences Triton has ended without telling it (idle for longer
// than the reclaim rule in sequence_mode.cc; every request Triton hands over,
// refused or not, counts as activity), least recently active first, and
// only as many as the request needs. It never takes pages from a sequence
// Triton still holds: if reclaiming is not enough the request is refused by
// name (UNAVAILABLE), the refusal lists the sequences holding pages, and a
// sequence refused mid-generation keeps its pages and KV, so the same request
// can be sent again once pages are free. There is no preemption of live
// sequences (swapping their KV out or recomputing it later).
//
// The requests of several tokens that arrive in one batch (the prompts of
// sequences started together) are prefilled together: an artifact with
// prefill entries of batch > 1 runs the prompts of a context bucket in one
// call, each right-aligned in its own row.

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
  int64_t max_batch_size() const { return max_batch_size_; }
  uint64_t idle_ns() const { return idle_ns_; }
  const std::string& tokens_input() const { return tokens_input_; }
  const std::string& logits_output() const { return logits_output_; }
  const std::string& start_input() const { return start_input_; }
  const std::string& end_input() const { return end_input_; }
  const std::string& corrid_input() const { return corrid_input_; }
  // Whether executions are handed the KV pools to write in place (parameter
  // "donate_kv_pools", true unless set to false).
  bool donate_pools() const { return donate_pools_; }
  // KV pool state names and their type, in the order entries read them
  // (full-history and windowed pools both).
  const std::vector<SlotSpec>& pools() const { return pools_; }
  // The windowed pool class (all zero without one).
  bool windowed() const { return window_ > 0; }
  int window() const { return window_; }
  int window_num_blocks() const { return window_num_blocks_; }
  int ring_pages() const { return ring_pages_; }
  // The most tokens one call may write for a sequence whose next position is
  // `start`: what the windowed ring holds (all of them without a windowed
  // pool), and no more than the largest prefill entry takes per sequence.
  int MaxTokensPerCall(int start) const;
  // The KV_PAGES output's name, or empty when config.pbtxt does not declare it.
  const std::string& pages_output() const { return pages_output_; }

  // The cheapest decode entry with batch >= `batch` and context >= `context`,
  // or nullptr.
  const ServingEntrySpec* Decode(int batch, int context) const;
  // The cheapest prefill entry with batch >= `batch`, context >= `context`
  // and at least `tokens` tokens per sequence, or nullptr.
  const ServingEntrySpec* Prefill(int batch, int context, int tokens) const;
  // The largest batch of the prefill entries of context `context` (0 if none).
  int MaxPrefillBatch(int context) const;
  // The largest batch of any prefill entry (0 without prefill entries).
  int max_prefill_batch() const { return max_prefill_batch_; }
  // Token ids the manifest refuses (model.refusedTokens), with the config key
  // naming each: a multimodal checkpoint's image and video placeholders.
  const std::map<int32_t, std::string>& refused_tokens() const { return refused_tokens_; }

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
  int window_ = 0;
  int window_num_blocks_ = 0;
  int ring_pages_ = 0;
  std::vector<int> window_layers_;
  int block_size_ = 0;
  int max_context_ = 0;
  int max_decode_batch_ = 0;
  int max_prefill_batch_ = 0;
  // The most tokens per sequence any prefill entry takes (0 without prefill).
  int max_prefill_tokens_ = 0;
  int64_t max_batch_size_ = 0;
  uint64_t idle_ns_ = 0;
  bool donate_pools_ = true;
  std::string tokens_input_, logits_output_, start_input_, end_input_, corrid_input_;
  std::string pages_output_;
  std::map<int32_t, std::string> refused_tokens_;
};

// The KV pages of one model instance, of one pool class. Page 0 is never
// handed out: it is the page padding rows of a batch point their block tables
// at.
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
  std::vector<int> ring;  // windowed pages: logical block b is on ring[b % ring_pages]
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
      : model_(model), name_(name), instance_(instance), pool_(model->num_blocks()),
        window_pool_(model->windowed() ? model->window_num_blocks() : 1)
  {
  }
  TRITONSERVER_Error* Parse(Work* w);
  TRITONSERVER_Error* Admit(Work* w);
  // Runs the requests of several tokens of one batch (each work's error is
  // set in the work). In rounds: each round takes from every request the
  // next call's worth of tokens, at most MaxTokensPerCall, and runs the
  // calls that fall in one prefill context bucket together, as many per call
  // as the largest prefill batch of that context, each group through the
  // smallest prefill entry covering it. A call no prefill entry covers runs
  // as one decode step per token.
  void RunPrompts(const std::vector<Work*>& works, uint64_t* compute_start, uint64_t* compute_end,
                  const std::function<void()>& note);
  // Frees, least recently active first, sequences Triton has ended for being
  // idle (see the definition) until `need` pages and `need_ring` windowed
  // pages are free or no such sequence is left. `for_id` is the sequence
  // that needs them (for the log).
  void Reclaim(uint64_t now, int need, int need_ring, uint64_t for_id);
  // The idle time after which the backend may free a sequence (the rule in
  // Reclaim's definition), and a description of the sequences holding pages,
  // most pages first, for a refusal.
  uint64_t ReclaimLimit() const;
  std::string Holders(uint64_t now, uint64_t except) const;
  void Free(uint64_t corrid, const char* why);
  // Runs `entry` on the given works (one row each, `tokens` of each placed in
  // its row), stores the state results and each work's logits row.
  TRITONSERVER_Error* Run(const ServingEntrySpec& entry, const std::vector<Work*>& rows,
                          uint64_t* compute_start, uint64_t* compute_end);
  TRITONSERVER_Error* Respond(Work* w);
  // Uploads zeroed KV pools (replacing any) and records their device
  // addresses; `bytes` is their total size.
  std::string ZeroPools(uint64_t* bytes);
  // After a run of `e`: whether every KV pool output is at the device address
  // its pool had before the run (written in place, not copied). Logged for
  // the first run of each entry and summed over the first 100 runs.
  void CheckInPlace(const ServingEntrySpec& e);

  const SequenceModel* model_;
  std::string name_;
  TRITONBACKEND_ModelInstance* instance_;
  PagePool pool_;
  PagePool window_pool_;
  std::unordered_map<uint64_t, SequenceState> sequences_;
  std::set<uint64_t> batch_;  // correlation IDs with a request in the batch being run
  uint64_t max_exec_ns_ = 0;  // the longest ProcessRequests call so far
  std::map<std::string, std::unique_ptr<tlaloc_triton::PjrtBuffer>> state_;
  uint64_t pool_bytes_ = 0;
  // Device address of each KV pool before the next run (empty when the
  // plugin cannot report addresses).
  std::map<std::string, const void*> pool_address_;
  uint64_t runs_ = 0;
  uint64_t in_place_runs_ = 0;
  std::set<std::string> reported_;  // entries whose first run was logged
  std::set<std::string> batched_reported_;  // "<prefill entry>/<prompts>" whose first run was logged
  // Set when a failed execution took the donated pools with it: every
  // sequence's KV state is gone. The rest of the batch is refused, then all
  // sequences are freed and the pools zeroed again.
  bool pools_lost_ = false;
};

}}}  // namespace triton::backend::tlaloc
