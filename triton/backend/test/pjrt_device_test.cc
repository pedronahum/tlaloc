// Copyright 2026 Pedro N. Rodriguez
// SPDX-License-Identifier: Apache-2.0
//
// Device checks of pjrt_runtime against a real PJRT GPU plugin, run by
// verify.sh inside the Triton container with the GPU visible:
//
//   pjrt_device_test <plugin.so> [--perturb]
//
//   1. a client created for GPU 0 sees exactly one device, CUDA ordinal 0;
//   2. a client for a GPU the machine does not have is refused, and the
//      refusal names the GPU;
//   3. x * x + x runs on an input the runtime reads in place from cudaMalloc
//      memory (a PJRT view, no host copy), and the result is copied device
//      to device into another cudaMalloc buffer; both the view path and the
//      host path give the exact expected values.
//
// --perturb expects one wrong value, so the run must fail (negative control).
// Exit status 0 when every check passes.

#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "pjrt_runtime.h"

using namespace tlaloc_triton;

namespace {

int failures = 0;

void
Check(bool ok, const std::string& what)
{
  std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++failures;
}

const char kModule[] = R"(func.func @main(%x: tensor<1024xf32>) -> tensor<1024xf32> {
  %0 = stablehlo.multiply %x, %x : tensor<1024xf32>
  %1 = stablehlo.add %0, %x : tensor<1024xf32>
  return %1 : tensor<1024xf32>
}
)";

}  // namespace

int
main(int argc, char** argv)
{
  if (argc < 2) {
    std::fprintf(stderr, "usage: pjrt_device_test <plugin.so> [--perturb]\n");
    return 2;
  }
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const bool perturb = argc > 2 && std::strcmp(argv[2], "--perturb") == 0;
  const PjrtPlugin* plugin = nullptr;
  std::string err = LoadPjrtPlugin(argv[1], &plugin);
  if (!err.empty()) {
    std::printf("FAIL %s\n", err.c_str());
    return 1;
  }
  ClientOptions options;  // memory_fraction 0.3, preallocate false

  int cuda_devices = 0;
  cudaGetDeviceCount(&cuda_devices);
  std::printf("pjrt_device_test: CUDA sees %d GPU(s)\n", cuda_devices);

  std::unique_ptr<PjrtClient> client;
  err = PjrtClient::Create(plugin, options, 0, &client);
  Check(err.empty(), "client for GPU 0 created" + (err.empty() ? "" : ": " + err));
  if (!err.empty()) return 1;
  Check(client->device_count() == 1 && client->device_ordinal() == 0,
        "the GPU 0 client sees one device, CUDA ordinal " +
            std::to_string(client->device_ordinal()));

  {
    const int missing = cuda_devices + 6;
    std::unique_ptr<PjrtClient> none;
    std::string refused = PjrtClient::Create(plugin, options, missing, &none);
    const std::string name = "GPU " + std::to_string(missing);
    Check(!refused.empty() && refused.find(name) != std::string::npos,
          "a client for " + name + " is refused by name: " + refused.substr(0, 200));
  }

  std::unique_ptr<PjrtExecutable> exe;
  err = client->Compile(PrepareForXla(kModule, "main"), &exe);
  Check(err.empty(), "compiled x * x + x" + (err.empty() ? "" : ": " + err));
  if (!err.empty()) return 1;
  Check(client->SupportsDeviceViews(), "the plugin supports device views");

  const size_t n = 1024, bytes = n * sizeof(float);
  std::vector<float> x(n), want(n);
  for (size_t i = 0; i < n; ++i) {
    x[i] = static_cast<float>(i) * 0.5f - 256.0f;  // x*x and x*x + x are exact
    want[i] = x[i] * x[i] + x[i];
  }
  if (perturb) want[3] += 1.0f;

  void* d_in = nullptr;
  void* d_out = nullptr;
  if (cudaMalloc(&d_in, bytes) != cudaSuccess || cudaMalloc(&d_out, bytes) != cudaSuccess) {
    std::printf("FAIL cudaMalloc\n");
    return 1;
  }
  cudaMemcpy(d_in, x.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0, bytes);

  // View path: the input is read in place, the output copied device to device.
  ExecuteArg arg;
  arg.host.dtype = DType::F32;
  arg.host.dims = {static_cast<int64_t>(n)};
  arg.host.byte_size = bytes;
  arg.device_ptr = d_in;
  std::unique_ptr<PjrtResults> results;
  err = exe->Execute({arg}, &results);
  Check(err.empty(), "executed on a view of cudaMalloc memory" + (err.empty() ? "" : ": " + err));
  if (!err.empty()) return 1;
  bool dense = false;
  err = results->WithDevicePointer(0, bytes, &dense, [&](const void* src) -> std::string {
    cudaError_t e = cudaMemcpy(d_out, src, bytes, cudaMemcpyDeviceToDevice);
    return e == cudaSuccess ? "" : cudaGetErrorString(e);
  });
  Check(err.empty() && dense, "result copied device to device (dense storage)");
  std::vector<float> got(n);
  cudaMemcpy(got.data(), d_out, bytes, cudaMemcpyDeviceToHost);
  Check(std::memcmp(got.data(), want.data(), bytes) == 0,
        "view path: all 1024 values equal x * x + x bit for bit");

  // The input memory is not written (the argument is never donated).
  std::vector<float> after(n);
  cudaMemcpy(after.data(), d_in, bytes, cudaMemcpyDeviceToHost);
  Check(std::memcmp(after.data(), x.data(), bytes) == 0, "the viewed input is unchanged");

  // Host path, for comparison.
  ExecuteArg host;
  host.host.data = x.data();
  host.host.dtype = DType::F32;
  host.host.dims = {static_cast<int64_t>(n)};
  host.host.byte_size = bytes;
  err = exe->Execute({host}, &results);
  std::vector<float> via_host(n);
  if (err.empty()) err = results->CopyToHost(0, via_host.data(), bytes);
  Check(err.empty() && std::memcmp(via_host.data(), want.data(), bytes) == 0,
        "host path: all 1024 values equal x * x + x bit for bit");

  cudaFree(d_in);
  cudaFree(d_out);
  if (failures > 0) {
    std::printf("%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("all device checks passed\n");
  return 0;
}
