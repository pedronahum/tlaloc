// Copyright 2026 Pedro N. Rodriguez
// SPDX-License-Identifier: Apache-2.0
//
// Checks for the StableHLO text reader. Built and run by build_backend.sh.

#include <cstdio>
#include <string>
#include <vector>

#include "../stablehlo_text.h"

using namespace tlaloc_triton;

static int failures = 0;

#define CHECK(cond)                                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                     \
    }                                                                 \
  } while (0)

int
main()
{
  // A Tlaloc emit: a bare function, scalar and matrix results.
  const std::string tlaloc =
      "func.func @matmul_sumsq_grad(%0: tensor<2x2xf32>) -> (tensor<f32>, tensor<2x2xf32>) {\n"
      "  // a comment mentioning func.func @nope(%x: tensor<1xf32>)\n"
      "  return %2, %9 : tensor<f32>, tensor<2x2xf32>\n"
      "}\n";
  std::vector<FunctionSignature> fns;
  std::string err;
  CHECK(ListFunctions(tlaloc, &fns, &err));
  CHECK(fns.size() == 1);
  CHECK(fns[0].name == "matmul_sumsq_grad");
  CHECK(fns[0].args.size() == 1 && fns[0].args[0].dims == std::vector<int64_t>({2, 2}));
  CHECK(fns[0].args[0].dtype == DType::F32);
  CHECK(fns[0].results.size() == 2);
  CHECK(fns[0].results[0].dims.empty() && fns[0].results[0].dtype == DType::F32);

  FunctionSignature sig;
  CHECK(SelectEntry(fns, "", &sig, &err) && sig.name == "matmul_sumsq_grad");
  CHECK(!SelectEntry(fns, "missing", &sig, &err));
  CHECK(err.find("@matmul_sumsq_grad") != std::string::npos);

  std::string prepared = PrepareForXla(tlaloc, "matmul_sumsq_grad");
  CHECK(prepared.rfind("module @tlaloc_triton {", 0) == 0);
  CHECK(prepared.find("func.func @main(") != std::string::npos);
  CHECK(prepared.find("@matmul_sumsq_grad(") == std::string::npos);

  // A module with a helper, a private entry, an existing main, attributes
  // on results, and a single unparenthesised result.
  const std::string jaxish =
      "module @m attributes {mhlo.num_partitions = 1 : i32} {\n"
      "  func.func public @main(%arg0: tensor<4xbf16>) -> tensor<4xbf16> {\n"
      "    return %arg0 : tensor<4xbf16>\n"
      "  }\n"
      "  func.func private @serve(%arg0: tensor<3x?xi32> {jax.arg_info = \"x, y\"}, "
      "%arg1: tensor<i1>) -> (tensor<3xui8> {jax.result_info = \"\"}) {\n"
      "    %0 = call @main(%arg0) : (tensor<4xbf16>) -> tensor<4xbf16>\n"
      "    return %0 : tensor<3xui8>\n"
      "  }\n"
      "}\n";
  fns.clear();
  CHECK(ListFunctions(jaxish, &fns, &err));
  CHECK(fns.size() == 2);
  CHECK(fns[0].name == "main" && fns[0].results.size() == 1);
  CHECK(fns[0].results[0].dtype == DType::BF16);
  CHECK(fns[1].name == "serve" && fns[1].args.size() == 2);
  CHECK(fns[1].args[0].dims == std::vector<int64_t>({3, -1}));
  CHECK(fns[1].args[0].dtype == DType::I32);
  CHECK(fns[1].args[1].dtype == DType::BOOL && fns[1].args[1].dims.empty());
  CHECK(fns[1].results[0].dtype == DType::U8);
  CHECK(SelectEntry(fns, "", &sig, &err) && sig.name == "main");

  prepared = PrepareForXla(jaxish, "serve");
  CHECK(prepared.rfind("module @m", 0) == 0);
  CHECK(prepared.find("func.func @main(%arg0: tensor<3x?xi32>") != std::string::npos);
  CHECK(prepared.find("func.func public @tlaloc_renamed_main(") != std::string::npos);
  CHECK(prepared.find("call @tlaloc_renamed_main(") != std::string::npos);

  // Symbol replacement is whole-word: @main2 is not @main.
  prepared = PrepareForXla("func.func @f() {\n call @main2()\n}\n", "f");
  CHECK(prepared.find("@main2") != std::string::npos);

  TensorType t;
  CHECK(!ParseTensorType("tensor<*xf32>", &t, &err));
  CHECK(ParseTensorType("tensor<2x3xf64>", &t, &err) && t.dtype == DType::F64);
  CHECK(ParseTensorType("tensor<8xcomplex<f32>>", &t, &err) && t.dtype == DType::UNSUPPORTED);
  CHECK(IsMlirBytecode(std::string("ML\xEFR\x00", 5)));
  CHECK(!IsMlirBytecode("module {}"));
  CHECK(DTypeFromTriton("TYPE_BF16") == DType::BF16 && DTypeFromTriton("INT32") == DType::I32);
  CHECK(DTypeFromTriton("TYPE_STRING") == DType::UNSUPPORTED);

  if (failures) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("stablehlo_text_test: all checks passed\n");
  return 0;
}
