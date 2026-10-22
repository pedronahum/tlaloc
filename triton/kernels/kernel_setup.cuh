// Shared by the kernels of libtlaloc_kernels.so.
#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <mutex>
#include <set>

namespace tlaloc_kernels {

// $TLALOC_DETERMINISTIC=1: the kernels sum in an order-independent way (integer atomics), so
// a result is the same in every run, at some cost (the decode attention's verify rows about
// 40% slower, the W4A4 experts' combine 0.7 ms a layer at 2,048 rows). Read once.
inline bool Deterministic()
{
  static const bool on = [] {
    const char* v = std::getenv("TLALOC_DETERMINISTIC");
    return v != nullptr && v[0] == '1';
  }();
  return on;
}

// Lets `kernel` take as much dynamic shared memory as the device allows (less
// its static shared memory), once per kernel. The handlers call it for every
// kernel they may launch at XLA's INITIALIZE stage, so that launches inside a
// CUDA graph being captured make no such call.
template <typename K>
inline cudaError_t AllowMaxSharedMemory(K kernel)
{
  static std::mutex mu;
  static std::set<const void*> done;
  std::lock_guard<std::mutex> lock(mu);
  const void* key = reinterpret_cast<const void*>(kernel);
  if (done.count(key)) return cudaSuccess;
  int device = 0, bytes = 0;
  cudaError_t e = cudaGetDevice(&device);
  if (e == cudaSuccess) e = cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  // The kernel's static shared memory counts against the same limit.
  cudaFuncAttributes attr{};
  if (e == cudaSuccess) e = cudaFuncGetAttributes(&attr, kernel);
  if (e == cudaSuccess) {
    e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes - static_cast<int>(attr.sharedSizeBytes));
  }
  if (e == cudaSuccess) done.insert(key);
  return e;
}

}  // namespace tlaloc_kernels
