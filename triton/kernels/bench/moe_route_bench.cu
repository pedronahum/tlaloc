// Standalone check and timing of the router top-k kernel.
//
//   nvcc -std=c++17 -O3 -arch=sm_121 moe_route_bench.cu -o /tmp/route_bench && /tmp/route_bench
//
// Against a CPU reference that computes the routing as the emitter's XLA form does (softmax or
// sigmoid + bias, a stable descending sort, the first k): the indices must be equal and the
// weights within 1e-6. Logits are drawn from a few values too, so ties occur. Then the median
// time for 2,048 rows of Qwen3.6-35B-A3B (256 experts, top 8) and of Flash-Next (512, top 10).
#include "../moe_route.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
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

namespace {

void Reference(const std::vector<float>& x, const std::vector<float>& bias, const route::Shape& sh, std::vector<int>& top,
               std::vector<float>& w)
{
  top.assign(static_cast<size_t>(sh.R) * sh.K, 0);
  w.assign(static_cast<size_t>(sh.R) * sh.K, 0.f);
  std::vector<float> p(sh.E), sel(sh.E);
  std::vector<int> order(sh.E);
  for (int r = 0; r < sh.R; ++r) {
    const float* row = &x[static_cast<size_t>(r) * sh.E];
    if (sh.sigmoid) {
      for (int e = 0; e < sh.E; ++e) {
        p[e] = 1.f / (1.f + std::exp(-row[e]));
        sel[e] = p[e] + bias[e];
      }
    } else {
      float mx = -INFINITY;
      for (int e = 0; e < sh.E; ++e) mx = std::max(mx, row[e]);
      double sum = 0;
      for (int e = 0; e < sh.E; ++e) sum += p[e] = std::exp(row[e] - mx);
      for (int e = 0; e < sh.E; ++e) sel[e] = p[e] = static_cast<float>(p[e] / sum);
    }
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return sel[a] > sel[b]; });
    double total = 0;
    for (int i = 0; i < sh.K; ++i) total += p[order[i]];
    const double den = sh.sigmoid ? total + 1e-20 : total;
    for (int i = 0; i < sh.K; ++i) {
      top[static_cast<size_t>(r) * sh.K + i] = order[i];
      w[static_cast<size_t>(r) * sh.K + i] = static_cast<float>(p[order[i]] / den * (sh.sigmoid ? sh.scale : 1.f));
    }
  }
}

bool Check(int R, int E, int K, bool sigmoid, bool ties, std::mt19937& g)
{
  route::Shape sh;
  sh.R = R; sh.E = E; sh.K = K; sh.sigmoid = sigmoid; sh.scale = 2.5f;
  std::normal_distribution<float> n(0.f, 2.f);
  std::vector<float> x(static_cast<size_t>(R) * E), bias(E);
  for (auto& v : x) v = ties ? static_cast<float>(static_cast<int>(g() % 5)) * 0.5f : n(g);
  for (auto& v : bias) v = ties ? 0.f : n(g) * 0.1f;
  std::vector<int> rt;
  std::vector<float> rw;
  Reference(x, bias, sh, rt, rw);
  float *dx, *db, *dw;
  int* dt;
  CHECK(cudaMalloc(&dx, x.size() * 4));
  CHECK(cudaMalloc(&db, bias.size() * 4));
  CHECK(cudaMalloc(&dt, rt.size() * 4));
  CHECK(cudaMalloc(&dw, rw.size() * 4));
  CHECK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db, bias.data(), bias.size() * 4, cudaMemcpyHostToDevice));
  CHECK(route::Launch(dx, sigmoid ? db : nullptr, dt, dw, sh, 0));
  std::vector<int> t(rt.size());
  std::vector<float> w(rw.size());
  CHECK(cudaMemcpy(t.data(), dt, t.size() * 4, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(w.data(), dw, w.size() * 4, cudaMemcpyDeviceToHost));
  size_t badIds = 0;
  float worst = 0.f;
  for (size_t i = 0; i < t.size(); ++i) {
    badIds += t[i] != rt[i];
    worst = std::max(worst, std::fabs(w[i] - rw[i]));
  }
  const bool ok = badIds == 0 && worst <= 1e-6f;
  std::printf("R %d E %d K %d %s%s: %zu ids differ, weights within %.2e  %s\n", R, E, K, sigmoid ? "sigmoid" : "softmax",
              ties ? " (ties)" : "", badIds, worst, ok ? "ok" : "FAILED");
  cudaFree(dx); cudaFree(db); cudaFree(dt); cudaFree(dw);
  return ok;
}

void Time(int R, int E, int K, bool sigmoid)
{
  route::Shape sh;
  sh.R = R; sh.E = E; sh.K = K; sh.sigmoid = sigmoid;
  float *dx, *db, *dw;
  int* dt;
  CHECK(cudaMalloc(&dx, static_cast<size_t>(R) * E * 4));
  CHECK(cudaMalloc(&db, E * 4));
  CHECK(cudaMalloc(&dt, static_cast<size_t>(R) * K * 4));
  CHECK(cudaMalloc(&dw, static_cast<size_t>(R) * K * 4));
  CHECK(cudaMemset(dx, 0, static_cast<size_t>(R) * E * 4));
  CHECK(cudaMemset(db, 0, E * 4));
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  std::vector<float> ms;
  for (int i = 0; i < 25; ++i) {
    cudaEventRecord(a);
    CHECK(route::Launch(dx, sigmoid ? db : nullptr, dt, dw, sh, 0));
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float t;
    cudaEventElapsedTime(&t, a, b);
    ms.push_back(t);
  }
  std::sort(ms.begin(), ms.end());
  std::printf("time R %d E %d K %d %s: median %.1f us\n", R, E, K, sigmoid ? "sigmoid" : "softmax", ms[ms.size() / 2] * 1000);
  cudaFree(dx); cudaFree(db); cudaFree(dt); cudaFree(dw);
}

}  // namespace

int main()
{
  std::mt19937 g(7);
  bool ok = true;
  for (int E : {8, 60, 256, 512}) {
    for (int K : {1, 2, 8}) {
      for (bool sig : {false, true}) {
        for (bool ties : {false, true}) ok &= Check(37, E, std::min(K, E), sig, ties, g);
      }
    }
  }
  ok &= Check(2048, 256, 8, false, false, g);
  ok &= Check(2048, 512, 10, true, false, g);
  Time(2048, 256, 8, false);
  Time(4, 256, 8, false);
  Time(2048, 512, 10, true);
  std::printf(ok ? "all checks passed\n" : "SOME CHECKS FAILED\n");
  return ok ? 0 : 1;
}
