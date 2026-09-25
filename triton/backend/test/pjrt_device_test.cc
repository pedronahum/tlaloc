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
//      host path give the exact expected values;
//   4. a 1 MiB state whose parameter carries `tf.aliasing_output` is updated
//      in place: XLA plans 1 MiB of arguments as aliased, and over 100 steps
//      that each donate the state and keep the output as the next step's
//      state, the output stays at the device address the state was uploaded
//      to and holds every value written; the same check detects the copy in
//      the two controls, the program without the attribute and the program
//      with it whose state is not donated (new memory every step, the kept
//      input unchanged).
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

// A state of 262144 floats (1 MiB); step i writes i + 1 at index i.
const char kAliasedStep[] = R"(func.func @main(%s: tensor<262144xf32> {tf.aliasing_output = 0 : i32}, %i: tensor<i32>) -> tensor<262144xf32> {
  %one = stablehlo.constant dense<1.0> : tensor<f32>
  %f = stablehlo.convert %i : (tensor<i32>) -> tensor<f32>
  %v = stablehlo.add %f, %one : tensor<f32>
  %u = stablehlo.reshape %v : (tensor<f32>) -> tensor<1xf32>
  %0 = stablehlo.dynamic_update_slice %s, %u, %i : (tensor<262144xf32>, tensor<1xf32>, tensor<i32>) -> tensor<262144xf32>
  return %0 : tensor<262144xf32>
}
)";

struct InPlaceRun {
  std::string err;
  int64_t alias_bytes = -1;
  int same_address_steps = 0;  // steps whose output is at the address of that step's state
  bool at_home = true;         // every output at the address the state was uploaded to
  bool values_ok = false;
  bool kept_input_unchanged = false;  // only meaningful when not donating
};

// Runs `steps` steps of `module`, each keeping its output as the next step's
// state. With `donate` the state is handed to the execution.
InPlaceRun
RunInPlace(PjrtClient* client, const std::string& module, bool donate, int steps)
{
  InPlaceRun r;
  const size_t n = 262144, bytes = n * sizeof(float);
  std::unique_ptr<PjrtExecutable> exe;
  r.err = client->Compile(PrepareForXla(module, "main"), &exe);
  if (!r.err.empty()) return r;
  CompiledMemory mem;
  r.err = exe->MemoryStats(&mem);
  if (!r.err.empty()) return r;
  r.alias_bytes = mem.alias;

  std::vector<float> zeros(n, 0.0f);
  HostInput h;
  h.data = zeros.data();
  h.byte_size = bytes;
  h.dtype = DType::F32;
  h.dims = {static_cast<int64_t>(n)};
  std::unique_ptr<PjrtBuffer> state;
  r.err = PjrtBuffer::Upload(client, h, &state);
  if (!r.err.empty()) return r;
  std::unique_ptr<PjrtBuffer> first;  // kept alive when not donating, to check it
  const void* home = nullptr;
  r.err = state->DeviceAddress(&home);
  if (!r.err.empty()) return r;

  for (int i = 0; i < steps; ++i) {
    const void* in = nullptr;
    r.err = state->DeviceAddress(&in);
    if (!r.err.empty()) return r;
    int32_t index = i;
    ExecuteArg s, idx;
    s.device = state.get();
    s.donate = donate;
    idx.host.data = &index;
    idx.host.byte_size = 4;
    idx.host.dtype = DType::I32;
    std::unique_ptr<PjrtResults> results;
    r.err = exe->Execute({s, idx}, &results);
    if (!r.err.empty()) return r;
    std::unique_ptr<PjrtBuffer> next = results->Release(0);
    const void* at = nullptr;
    r.err = next->DeviceAddress(&at);
    if (!r.err.empty()) return r;
    if (at == in) ++r.same_address_steps;
    if (at != home) r.at_home = false;
    if (!donate && i == 0) first = std::move(state);
    state = std::move(next);
  }

  std::vector<float> got(n), want(n, 0.0f);
  for (int i = 0; i < steps; ++i) want[i] = static_cast<float>(i + 1);
  std::unique_ptr<PjrtResults> none;
  // Read the final state back through a one-step identity: the host copy of a
  // PjrtBuffer goes through PjrtResults, so run the step once more at an index
  // past the checked range and compare the rest.
  int32_t index = steps;
  ExecuteArg s, idx;
  s.device = state.get();
  idx.host.data = &index;
  idx.host.byte_size = 4;
  idx.host.dtype = DType::I32;
  r.err = exe->Execute({s, idx}, &none);
  if (!r.err.empty()) return r;
  r.err = none->CopyToHost(0, got.data(), bytes);
  if (!r.err.empty()) return r;
  want[steps] = static_cast<float>(steps + 1);
  r.values_ok = std::memcmp(got.data(), want.data(), bytes) == 0;
  if (first) {
    ExecuteArg f;
    f.device = first.get();
    idx.host.data = &index;
    r.err = exe->Execute({f, idx}, &none);
    if (!r.err.empty()) return r;
    r.err = none->CopyToHost(0, got.data(), bytes);
    if (!r.err.empty()) return r;
    std::vector<float> only(n, 0.0f);
    only[steps] = static_cast<float>(steps + 1);
    r.kept_input_unchanged = std::memcmp(got.data(), only.data(), bytes) == 0;
  }
  return r;
}

std::string
Replace(std::string s, const std::string& from, const std::string& to)
{
  const size_t at = s.find(from);
  if (at != std::string::npos) s.replace(at, from.size(), to);
  return s;
}

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
    if (e == cudaSuccess) e = cudaStreamSynchronize(0);  // D2D cudaMemcpy is asynchronous
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

  // In-place state: the aliased program with the state donated, then the two
  // controls the check must see copying.
  const int steps = 100;
  const int64_t mib = 262144 * 4;
  InPlaceRun in_place = RunInPlace(client.get(), kAliasedStep, true, steps);
  Check(in_place.err.empty(), "aliased state program ran" + (in_place.err.empty() ? "" : ": " + in_place.err));
  Check(in_place.alias_bytes == mib,
        "XLA plans " + std::to_string(in_place.alias_bytes) + " bytes of arguments aliased to outputs (" +
            std::to_string(mib) + " expected)");
  Check(in_place.same_address_steps == steps && in_place.at_home,
        std::to_string(in_place.same_address_steps) + " of " + std::to_string(steps) +
            " donated steps wrote the state at its own address, the one it was uploaded to "
            "(updated in place)");
  Check(in_place.values_ok, "the in-place state holds all 101 values written");

  InPlaceRun no_alias = RunInPlace(
      client.get(), Replace(kAliasedStep, " {tf.aliasing_output = 0 : i32}", ""), true, steps);
  Check(no_alias.err.empty() && no_alias.alias_bytes == 0 && no_alias.same_address_steps == 0 &&
            no_alias.values_ok,
        "control, no alias attribute: 0 bytes aliased, " + std::to_string(no_alias.same_address_steps) +
            " of " + std::to_string(steps) + " steps at their state's address (the check sees the copy)" +
            (no_alias.err.empty() ? "" : ": " + no_alias.err));
  InPlaceRun kept = RunInPlace(client.get(), kAliasedStep, false, steps);
  Check(kept.err.empty() && kept.alias_bytes == mib && kept.same_address_steps == 0 && kept.values_ok &&
            kept.kept_input_unchanged,
        "control, alias attribute but the state not donated: " + std::to_string(kept.same_address_steps) +
            " of " + std::to_string(steps) + " steps at their state's address, the kept input unchanged" +
            (kept.err.empty() ? "" : ": " + kept.err));

  if (failures > 0) {
    std::printf("%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("all device checks passed\n");
  return 0;
}
