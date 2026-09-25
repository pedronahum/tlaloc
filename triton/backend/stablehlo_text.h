// Copyright 2026 Pedro N. Rodriguez
// SPDX-License-Identifier: Apache-2.0
//
// Reading the StableHLO text that Tlaloc writes: element types, the entry
// function's signature, and the rewrite XLA needs before it will compile it.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tlaloc_triton {

// The element types the backend moves between Triton and PJRT. Anything else
// is refused by name at model load.
enum class DType { F32, F64, F16, BF16, I8, I32, I64, U8, BOOL, UNSUPPORTED };

// "f32", "bf16", "i32", "ui8", "i1", ... -> DType. Unknown -> UNSUPPORTED.
DType DTypeFromMlir(const std::string& element);
// "FP32", "BF16", "INT32", ... (Triton's names without TYPE_) -> DType.
DType DTypeFromTriton(const std::string& name);
// Triton's name ("FP32", ...) for a DType.
const char* TritonName(DType t);
// Byte width of one element.
size_t ByteWidth(DType t);

struct TensorType {
  DType dtype = DType::UNSUPPORTED;
  std::string element;         // as written in the module, e.g. "bf16"
  std::vector<int64_t> dims;   // -1 for a dynamic ("?") dimension
  std::string text;            // e.g. "tensor<2x2xf32>"
};

struct FunctionSignature {
  std::string name;
  std::vector<TensorType> args;
  std::vector<TensorType> results;
};

// True when the bytes are MLIR bytecode rather than text.
bool IsMlirBytecode(const std::string& bytes);

// Parses one tensor type such as "tensor<2x2xf32>" or "tensor<f32>".
// Returns false and fills `error` if it is not a ranked tensor type.
bool ParseTensorType(const std::string& text, TensorType* out, std::string* error);

// Every `func.func` in the text, in order. Fills `error` on a signature the
// parser cannot read.
bool ListFunctions(
    const std::string& module_text, std::vector<FunctionSignature>* out,
    std::string* error);

// The function to serve: `entry` if non-empty, else `main` if present, else
// the first function in the file.
bool SelectEntry(
    const std::vector<FunctionSignature>& functions, const std::string& entry,
    FunctionSignature* out, std::string* error);

// XLA compiles a module whose entry function is public and named `main`.
// Tlaloc emits `func.func @<name>` at top level, without a module. This
// renames the entry to `main` (moving an existing, different `main` out of
// the way), drops a `private` on it, and wraps bare functions in a module.
std::string PrepareForXla(const std::string& module_text, const std::string& entry);

std::string ShapeString(const std::vector<int64_t>& dims);

}  // namespace tlaloc_triton
