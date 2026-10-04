// Standalone timing and check of the Gated DeltaNet kernel.
//
//   nvcc -std=c++17 -O3 -arch=sm_121 gated_delta_bench.cu -o /tmp/gdr_bench && /tmp/gdr_bench
//
// Qwen3.6-35B-A3B's layer (16 key heads, 32 value heads, 128 x 128 states):
// four sequences verifying four tokens, each token's state written to its
// own slot. Prints the largest difference from a CPU reference and the median
// time per layer.
#include "../gated_delta.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CHECK(x)                                                                      \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_ != cudaSuccess) {                                                          \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

int main()
{
  gdr::Shape sh{4, 4, 16, 32, 128, 128, 24};
  std::mt19937 gen(3);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  const size_t nq = static_cast<size_t>(sh.B) * sh.T * sh.Hk * sh.Dk, nv = static_cast<size_t>(sh.B) * sh.T * sh.Hv * sh.Dv;
  const size_t ng = static_cast<size_t>(sh.B) * sh.T * sh.Hv, slotState = static_cast<size_t>(sh.Hv) * sh.Dk * sh.Dv;
  std::vector<float> q(nq), k(nq), v(nv), g(ng), beta(ng), pool(sh.S * slotState);
  for (auto& x : q) x = u(gen) * 0.1f;
  for (auto& x : k) x = u(gen) * 0.1f;
  for (auto& x : v) x = u(gen);
  for (auto& x : g) x = -0.1f + 0.05f * u(gen);
  for (auto& x : beta) x = 0.5f + 0.4f * u(gen);
  for (auto& x : pool) x = u(gen) * 0.5f;
  // Row 0: one padding token then live; row 1 starts at position 0; rows 2, 3 continue.
  std::vector<int> slots(sh.B * sh.T), positions(sh.B * sh.T), writes(sh.B * sh.T);
  for (int b = 0; b < sh.B; ++b) {
    for (int t = 0; t < sh.T; ++t) {
      const bool pad = b == 0 && t == 0;
      slots[b * sh.T + t] = pad ? -1 : b * 6;
      positions[b * sh.T + t] = b == 1 ? t : 100 + t;
      writes[b * sh.T + t] = pad ? -1 : b * 6 + 1 + t;
    }
  }
  // CPU reference.
  std::vector<double> ref(pool.begin(), pool.end()), outRef(nv, 0.0);
  for (int b = 0; b < sh.B; ++b) {
    int first = 0;
    while (first < sh.T && slots[b * sh.T + first] < 0) ++first;
    const int slot = slots[b * sh.T + first];
    const bool reset = positions[b * sh.T + first] == 0;
    std::vector<double> st(slotState, 0.0);
    if (!reset) for (size_t x = 0; x < slotState; ++x) st[x] = pool[slot * slotState + x];
    for (int t = first; t < sh.T; ++t) {
      const size_t bt = b * sh.T + t;
      for (int h = 0; h < sh.Hv; ++h) {
        const int kh = h / (sh.Hv / sh.Hk);
        double* S = st.data() + static_cast<size_t>(h) * sh.Dk * sh.Dv;
        const double d = std::exp(static_cast<double>(g[bt * sh.Hv + h]));
        for (int x = 0; x < sh.Dk * sh.Dv; ++x) S[x] *= d;
        std::vector<double> kv(sh.Dv);
        for (int j = 0; j < sh.Dv; ++j) {
          double a = 0;
          for (int i = 0; i < sh.Dk; ++i) a += S[i * sh.Dv + j] * k[(bt * sh.Hk + kh) * sh.Dk + i];
          kv[j] = (v[(bt * sh.Hv + h) * sh.Dv + j] - a) * beta[bt * sh.Hv + h];
        }
        for (int i = 0; i < sh.Dk; ++i)
          for (int j = 0; j < sh.Dv; ++j) S[i * sh.Dv + j] += k[(bt * sh.Hk + kh) * sh.Dk + i] * kv[j];
        for (int j = 0; j < sh.Dv; ++j) {
          double a = 0;
          for (int i = 0; i < sh.Dk; ++i) a += S[i * sh.Dv + j] * q[(bt * sh.Hk + kh) * sh.Dk + i];
          outRef[(bt * sh.Hv + h) * sh.Dv + j] = a;
        }
      }
      const int w = writes[bt];
      for (size_t x = 0; x < slotState; ++x) ref[w * slotState + x] = st[x];
    }
  }
  float *dq, *dk, *dv, *dg, *db, *dp, *dout;
  int *ds, *dpos, *dw;
  CHECK(cudaMalloc(&dq, nq * 4)); CHECK(cudaMalloc(&dk, nq * 4)); CHECK(cudaMalloc(&dv, nv * 4));
  CHECK(cudaMalloc(&dg, ng * 4)); CHECK(cudaMalloc(&db, ng * 4)); CHECK(cudaMalloc(&dp, pool.size() * 4));
  CHECK(cudaMalloc(&dout, nv * 4));
  CHECK(cudaMalloc(&ds, slots.size() * 4)); CHECK(cudaMalloc(&dpos, slots.size() * 4)); CHECK(cudaMalloc(&dw, slots.size() * 4));
  CHECK(cudaMemcpy(dq, q.data(), nq * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dk, k.data(), nq * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dv, v.data(), nv * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dg, g.data(), ng * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db, beta.data(), ng * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dp, pool.data(), pool.size() * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(ds, slots.data(), slots.size() * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dpos, positions.data(), slots.size() * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dw, writes.data(), slots.size() * 4, cudaMemcpyHostToDevice));
  CHECK(gdr::Launch(dq, dk, dv, dg, db, dp, ds, dpos, dw, dout, sh, nullptr));
  std::vector<float> got(nv), gotPool(pool.size());
  CHECK(cudaMemcpy(got.data(), dout, nv * 4, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(gotPool.data(), dp, pool.size() * 4, cudaMemcpyDeviceToHost));
  double wo = 0, wp = 0;
  for (size_t x = 0; x < nv; ++x) wo = std::max(wo, std::fabs(got[x] - outRef[x]));
  for (size_t x = 0; x < pool.size(); ++x) wp = std::max(wp, std::fabs(gotPool[x] - ref[x]));
  std::printf("check: worst |d| out %.3g, pool %.3g%s\n", wo, wp, wo < 1e-4 && wp < 1e-4 ? "" : "  FAILED");
  std::vector<float> ms;
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  for (int i = 0; i < 30; ++i) {
    cudaEventRecord(a);
    CHECK(gdr::Launch(dq, dk, dv, dg, db, dp, ds, dpos, dw, dout, sh, nullptr));
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float t;
    cudaEventElapsedTime(&t, a, b);
    ms.push_back(t);
  }
  std::sort(ms.begin(), ms.end());
  const double bytes = 4.0 * slotState * 4 * (1 + 4);  // read one state, write four, per sequence
  std::printf("4 sequences x 4 tokens, per-token writes: %.3f ms per layer (%.0f GB/s of state)\n", ms[15], bytes / ms[15] / 1e6);
  return 0;
}
