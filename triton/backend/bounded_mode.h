// Copyright 2026 Pedro N. Rodriguez
// SPDX-License-Identifier: Apache-2.0
//
// Bounded mode of the tlaloc backend: a model whose config.pbtxt names a
// Tlaloc bounded-program manifest (parameter "bounded_manifest",
// tlaloc-bounded.json, schema tlaloc-bounded-v1; docs/design/bounded-dims.md).
//
// The artifact holds one StableHLO body per combination of buckets of the
// program's bounded axes. For each request the backend reads each bound's size
// from the shapes of the request's inputs (every axis of one bound has one
// size), picks the smallest bucket of each bound that holds it, pads the
// inputs with the manifest's padding value along their bounded axes, fills the
// VALID_MASK inputs ([bucket], 1 for a real position) and VALID_LENGTH inputs
// (the real size, f32), runs that bucket's body, and returns the output with
// its bounded axes sliced back to the real sizes. Every body is compiled at
// load.
//
// The config's inputs are the manifest's DATA inputs, by name, with -1 for
// each bounded axis; its one output is the manifest's output, likewise.
// max_batch_size must be 0. Tensors go through the host.

#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "pjrt_runtime.h"
#include "triton/backend/backend_common.h"
#include "triton/core/tritonbackend.h"

namespace triton { namespace backend { namespace tlaloc {

struct BoundedAxisSpec {
  int64_t size = 0;   // a fixed axis, when bound is empty
  std::string bound;  // the bound's name, for a bounded axis
};

struct BoundedTensorSpec {
  std::string name;
  std::string role;   // DATA, VALID_MASK, VALID_LENGTH
  tlaloc_triton::DType dtype = tlaloc_triton::DType::UNSUPPORTED;
  std::vector<BoundedAxisSpec> axes;
  std::string bound;  // for VALID_MASK and VALID_LENGTH
};

struct BoundSpec {
  std::string name;
  int64_t max = 0;
  std::vector<int64_t> buckets;  // ascending, the last is max
};

struct BoundedEntrySpec {
  std::string id;
  std::map<std::string, int64_t> sizes;  // bucket of each bound
  std::string entry_point;
  std::string body_path;
  const tlaloc_triton::PjrtExecutable* executable = nullptr;
};

class BoundedModel {
 public:
  static TRITONSERVER_Error* Load(
      const std::string& model_name, const std::string& version_dir,
      const std::string& manifest_file, triton::common::TritonJson::Value& config,
      const std::function<TRITONSERVER_Error*(std::shared_ptr<tlaloc_triton::PjrtClient>*)>& acquire,
      std::unique_ptr<BoundedModel>* out);

  // Runs one request and adds its output to `response`.
  TRITONSERVER_Error* Run(
      TRITONBACKEND_Request* request, TRITONBACKEND_Response* response, uint64_t* compute_start,
      uint64_t* compute_end) const;

  const std::string& name() const { return name_; }

 private:
  BoundedModel() = default;
  TRITONSERVER_Error* ReadManifest(const std::string& text);
  TRITONSERVER_Error* ReadConfig(triton::common::TritonJson::Value& config);
  TRITONSERVER_Error* CompileEntries();
  std::string Where() const { return "model '" + name_ + "': "; }
  const BoundSpec* Bound(const std::string& name) const;

  std::string name_;
  std::string version_dir_;
  std::string program_;
  double padding_ = 0.0;
  std::vector<BoundSpec> bounds_;
  std::vector<BoundedTensorSpec> inputs_;
  BoundedTensorSpec output_;
  std::vector<BoundedEntrySpec> entries_;
  std::shared_ptr<tlaloc_triton::PjrtClient> client_;
  std::map<std::string, std::unique_ptr<tlaloc_triton::PjrtExecutable>> executables_;
};

}}}  // namespace triton::backend::tlaloc
