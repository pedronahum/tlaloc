// Standalone check and timing of the token sampler (sample.cu).
//
//   nvcc -std=c++17 -O3 -arch=sm_121 sample_bench.cu -o /tmp/sample_bench && /tmp/sample_bench
//
// 1. The kernel's tokens equal a CPU reference (the same integer steps as
//    io.tlaloc.ir.SampleTokensAttrs) over random logits and parameters, ties included.
// 2. With many seeds, the drawn tokens follow the kept distribution (W / M over the kept
//    tokens), with a draft or without: a draft accepted with probability W(d) / M, else
//    another token drawn from the rest, gives the same marginal.
// 3. The time of 16 rows over a 248,320-token vocabulary.
#include "../sample.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

float Div(float a, float b) { volatile float r = a / b; return r; }
float Mul(float a, float b) { volatile float r = a * b; return r; }
float Sub(float a, float b) { volatile float r = a - b; return r; }

struct Kept {
  std::vector<char> in;
  std::vector<uint64_t> w;
  uint64_t mass = 0;
  int argmax = 0;
};

Kept KeptSet(const float* x, int V, float t, float p, int k)
{
  Kept r;
  for (int i = 1; i < V; ++i) if (x[i] > x[r.argmax]) r.argmax = i;
  const float m = x[r.argmax];
  r.w.resize(V);
  for (int i = 0; i < V; ++i) r.w[i] = sample::Exp2Fixed(Mul(Div(Sub(x[i], m), t), 1.4426950408889634f));
  std::vector<int> order(V);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return x[a] > x[b]; });
  const int kk = k <= 0 || k >= V ? V : k;
  uint64_t sumK = 0;
  for (int j = 0; j < kk; ++j) sumK += r.w[order[j]];
  const uint64_t p24 = p >= 1.f ? (1ull << 24) : static_cast<uint64_t>(std::max(p, 0.f) * 16777216.f);
  int kept = kk;
  if (p24 < (1ull << 24)) {
    uint64_t need = (p24 * sumK + (1ull << 24) - 1) >> 24;
    if (need == 0) need = 1;
    uint64_t cum = 0;
    for (int j = 0; j < kk; ++j) {
      cum += r.w[order[j]];
      if (cum >= need) { kept = j + 1; break; }
    }
  }
  r.in.assign(V, 0);
  for (int j = 0; j < kept; ++j) { r.in[order[j]] = 1; r.mass += r.w[order[j]]; }
  return r;
}

int Reference(const float* x, int V, int d, float t, float p, int k, int seed, int pos)
{
  int argmax = 0;
  for (int i = 1; i < V; ++i) if (x[i] > x[argmax]) argmax = i;
  if (!(t > 0.f)) return argmax;
  const Kept kp = KeptSet(x, V, t, p, k);
  const bool dKept = d >= 0 && d < V && kp.in[d];
  const uint64_t wd = dKept ? kp.w[d] : 0;
  uint32_t u0, u1;
  sample::Uniforms(seed, pos, &u0, &u1);
  if (kp.mass == 0) return argmax;
  if (dKept && static_cast<uint64_t>(u0) * kp.mass < (wd << 24)) return d;
  const uint64_t m2 = kp.mass - wd;
  if (m2 == 0) return argmax;
  const uint64_t target = (static_cast<uint64_t>(u1) * m2) >> 24;
  uint64_t cum = 0;
  for (int i = 0; i < V; ++i) {
    if (!kp.in[i] || i == d) continue;
    cum += kp.w[i];
    if (cum > target) return i;
  }
  return argmax;
}

struct Batch {
  int N, V;
  std::vector<float> x, t, p;
  std::vector<int> d, k, seed, pos;
};

std::vector<int> RunGpu(const Batch& b)
{
  float *dx, *dt, *dp;
  int *dd, *dk, *ds, *dpos, *dout;
  CHECK(cudaMalloc(&dx, b.x.size() * 4));
  CHECK(cudaMalloc(&dt, b.N * 4)); CHECK(cudaMalloc(&dp, b.N * 4));
  CHECK(cudaMalloc(&dd, b.N * 4)); CHECK(cudaMalloc(&dk, b.N * 4)); CHECK(cudaMalloc(&ds, b.N * 4));
  CHECK(cudaMalloc(&dpos, b.N * 4)); CHECK(cudaMalloc(&dout, b.N * 4));
  CHECK(cudaMemcpy(dx, b.x.data(), b.x.size() * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dt, b.t.data(), b.N * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dp, b.p.data(), b.N * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dd, b.d.data(), b.N * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dk, b.k.data(), b.N * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(ds, b.seed.data(), b.N * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dpos, b.pos.data(), b.N * 4, cudaMemcpyHostToDevice));
  CHECK(sample::Launch(dx, b.N, b.V, dd, dt, dp, dk, ds, dpos, dout, 0));
  std::vector<int> out(b.N);
  CHECK(cudaMemcpy(out.data(), dout, b.N * 4, cudaMemcpyDeviceToHost));
  cudaFree(dx); cudaFree(dt); cudaFree(dp); cudaFree(dd); cudaFree(dk); cudaFree(ds); cudaFree(dpos); cudaFree(dout);
  return out;
}

}  // namespace

int main()
{
  CHECK(sample::Prepare());
  std::mt19937 g(11);
  bool ok = true;
  // 1. Equal to the reference.
  for (int V : {7, 1000, 50000, 200000}) {
    Batch b{64, V};
    std::normal_distribution<float> n(0.f, 3.f);
    b.x.resize(static_cast<size_t>(b.N) * V);
    for (int r = 0; r < b.N; ++r) {
      const bool ties = r % 4 == 3;
      // Rows 4k + 3: few distinct values (ties); rows 8k + 5: nearly flat (many tokens of weight > 0,
      // past the fast path's kCap at large V).
      for (int i = 0; i < V; ++i) {
        b.x[static_cast<size_t>(r) * V + i] = ties ? static_cast<float>(g() % 6) : r % 8 == 5 ? 0.01f * n(g) : n(g);
      }
      b.t.push_back(r % 8 == 0 ? 0.f : 0.2f + 0.3f * (r % 5));
      b.p.push_back(r % 3 == 0 ? 1.f : 0.5f + 0.1f * (r % 4));
      b.k.push_back(r % 5 == 0 ? 0 : 1 + static_cast<int>(g() % 50));
      b.d.push_back(r % 2 == 0 ? -1 : static_cast<int>(g() % V));
      b.seed.push_back(static_cast<int>(g()));
      b.pos.push_back(static_cast<int>(g() % 100000));
    }
    const std::vector<int> got = RunGpu(b);
    int bad = 0;
    for (int r = 0; r < b.N; ++r) {
      const int want = Reference(&b.x[static_cast<size_t>(r) * V], V, b.d[r], b.t[r], b.p[r], b.k[r], b.seed[r], b.pos[r]);
      if (want != got[r]) {
        if (bad < 5) std::printf("  row %d: kernel %d, reference %d (t %g p %g k %d d %d)\n", r, got[r], want, b.t[r], b.p[r], b.k[r], b.d[r]);
        ++bad;
      }
    }
    std::printf("V %d: %d of %d rows differ from the reference%s\n", V, bad, b.N, bad ? "  FAILED" : "");
    ok &= bad == 0;
  }
  // 2. The distribution: one row's logits, many seeds; with draft -1 and with a draft.
  {
    const int V = 40, S = 200000;
    std::vector<float> row(V);
    std::normal_distribution<float> n(0.f, 1.5f);
    for (auto& v : row) v = n(g);
    for (int dmode = 0; dmode < 2; ++dmode) {
      const float t = 0.8f, p = 0.9f;
      const int k = 20;
      const Kept kp = KeptSet(row.data(), V, t, p, k);
      int dr = -1;
      if (dmode == 1) for (int i = 0; i < V; ++i) if (kp.in[i] && (dr < 0 || kp.w[i] > kp.w[dr])) dr = i;  // the likeliest kept token
      Batch b{S, V};
      b.x.resize(static_cast<size_t>(S) * V);
      for (int s = 0; s < S; ++s) {
        std::memcpy(&b.x[static_cast<size_t>(s) * V], row.data(), V * 4);
        b.t.push_back(t); b.p.push_back(p); b.k.push_back(k); b.d.push_back(dr);
        b.seed.push_back(s); b.pos.push_back(7);
      }
      const std::vector<int> got = RunGpu(b);
      std::vector<int> count(V, 0);
      for (int v : got) ++count[v];
      double chi = 0;
      int dof = -1, outside = 0;
      for (int i = 0; i < V; ++i) {
        if (!kp.in[i] || kp.w[i] == 0) { outside += count[i]; continue; }
        const double e = static_cast<double>(S) * kp.w[i] / kp.mass;
        chi += (count[i] - e) * (count[i] - e) / e;
        ++dof;
      }
      // Chi-square with dof degrees of freedom: mean dof, sd sqrt(2 dof); allow 5 sd.
      const bool good = outside == 0 && chi < dof + 5 * std::sqrt(2.0 * dof);
      std::printf("distribution%s: chi2 %.1f over %d degrees of freedom, %d draws outside the kept set%s\n",
                  dmode ? " with a draft" : "", chi, dof, outside, good ? "" : "  FAILED");
      ok &= good;
    }
  }
  // 3. Timing: logits like a language model's (a few hundred tokens near the top, the fast path)
  // and a nearly flat row (tens of thousands of tokens of weight > 0, the general path).
  for (int flat = 0; flat < 2; ++flat) {
    const int V = 248320, N = 16;
    Batch b{N, V};
    b.x.resize(static_cast<size_t>(N) * V);
    std::normal_distribution<float> n(0.f, flat ? 3.f : 2.f);
    for (auto& v : b.x) v = n(g);
    if (!flat) {
      for (int r = 0; r < N; ++r) for (int j = 0; j < 300; ++j) b.x[static_cast<size_t>(r) * V + g() % V] = 22.f + 6.f * (g() % 1000) / 1000.f;
    }
    for (int r = 0; r < N; ++r) {
      b.t.push_back(flat ? 0.7f : 1.f); b.p.push_back(0.95f); b.k.push_back(r % 2 ? 50 : 0); b.d.push_back(r % 4 ? 17 : -1);
      b.seed.push_back(r); b.pos.push_back(r);
    }
    float *dx, *dt, *dp; int *dd, *dk, *ds, *dpos, *dout;
    CHECK(cudaMalloc(&dx, b.x.size() * 4)); CHECK(cudaMalloc(&dt, N * 4)); CHECK(cudaMalloc(&dp, N * 4));
    CHECK(cudaMalloc(&dd, N * 4)); CHECK(cudaMalloc(&dk, N * 4)); CHECK(cudaMalloc(&ds, N * 4)); CHECK(cudaMalloc(&dpos, N * 4)); CHECK(cudaMalloc(&dout, N * 4));
    CHECK(cudaMemcpy(dx, b.x.data(), b.x.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dt, b.t.data(), N * 4, cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(dp, b.p.data(), N * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dd, b.d.data(), N * 4, cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(dk, b.k.data(), N * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(ds, b.seed.data(), N * 4, cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(dpos, b.pos.data(), N * 4, cudaMemcpyHostToDevice));
    cudaEvent_t a, e; cudaEventCreate(&a); cudaEventCreate(&e);
    std::vector<float> ms;
    for (int i = 0; i < 20; ++i) {
      cudaEventRecord(a);
      CHECK(sample::Launch(dx, N, V, dd, dt, dp, dk, ds, dpos, dout, 0));
      cudaEventRecord(e); cudaEventSynchronize(e);
      float t; cudaEventElapsedTime(&t, a, e); ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    // The reference for these rows too.
    std::vector<int> got(N);
    CHECK(cudaMemcpy(got.data(), dout, N * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int r = 0; r < N; ++r) bad += got[r] != Reference(&b.x[static_cast<size_t>(r) * V], V, b.d[r], b.t[r], b.p[r], b.k[r], b.seed[r], b.pos[r]);
    std::printf("16 rows of 248,320 logits, %s (top-p 0.95, half with top-k 50): median %.3f ms; %d differ from the reference%s\n",
                flat ? "nearly flat" : "a language model's shape", ms[10], bad, bad ? "  FAILED" : "");
    ok &= bad == 0;
  }
  std::printf(ok ? "all checks passed\n" : "SOME CHECKS FAILED\n");
  return ok ? 0 : 1;
}
