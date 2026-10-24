// Standalone timing and check of the Gated DeltaNet kernel.
//
//   nvcc -std=c++17 -O3 -arch=sm_121 gated_delta_bench.cu -o /tmp/gdr_bench && /tmp/gdr_bench
//
// Qwen3.6-35B-A3B's layer (16 key heads, 32 value heads, 128 x 128 states):
// four sequences verifying four tokens, each token's state written to its
// own slot. Prints the largest difference from a CPU reference and the median
// time per layer; then the committed form (a starting state and a record per
// token), checked bit for bit against the per-token states, and its time.
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

  // The committed form (records): each row's base the slot the per-token call read, its target
  // another, no pending records. The state after token t is then the target's state with the
  // first t + 1 records applied, which must equal the per-token write of the same token bit for
  // bit.
  const int W = sh.Dk + sh.Dv + 4;
  std::vector<int> commit(sh.B * 3);
  for (int b = 0; b < sh.B; ++b) {
    commit[b * 3] = b * 6;
    commit[b * 3 + 1] = b * 6 + 5;
    commit[b * 3 + 2] = 0;
  }
  float* drec;
  int* dc;
  const size_t nrec = static_cast<size_t>(sh.S) * sh.T * sh.Hv * W;
  CHECK(cudaMalloc(&drec, nrec * 4));
  CHECK(cudaMemset(drec, 0, nrec * 4));
  CHECK(cudaMalloc(&dc, commit.size() * 4));
  CHECK(cudaMemcpy(dc, commit.data(), commit.size() * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dp, pool.data(), pool.size() * 4, cudaMemcpyHostToDevice));
  gdr::Commit cm;
  cm.rows = dc;
  cm.records = drec;
  cm.R = sh.T;
  cm.mode = gdr::kRecords;
  CHECK(gdr::Launch(dq, dk, dv, dg, db, dp, ds, dpos, nullptr, dout, sh, nullptr, cm));
  std::vector<float> got2(nv), pool2(pool.size()), rec(nrec);
  CHECK(cudaMemcpy(got2.data(), dout, nv * 4, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(pool2.data(), dp, pool.size() * 4, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(rec.data(), drec, nrec * 4, cudaMemcpyDeviceToHost));
  size_t outDiff = 0, stateDiff = 0;
  for (size_t x = 0; x < nv; ++x) outDiff += got2[x] != got[x];
  // A second committed call per count c from each row's target, its starting state (written at
  // its own target, the old base) the state after token c - 1. Its rows continue (row 1 started
  // at position 0 in the first call).
  std::vector<int> later(positions);
  for (int& x : later) x += sh.T;
  int* dpos2;
  CHECK(cudaMalloc(&dpos2, later.size() * 4));
  CHECK(cudaMemcpy(dpos2, later.data(), later.size() * 4, cudaMemcpyHostToDevice));
  for (int c = 1; c <= sh.T; ++c) {
    std::vector<int> next(sh.B * 3);
    for (int b = 0; b < sh.B; ++b) {
      next[b * 3] = commit[b * 3 + 1];
      next[b * 3 + 1] = commit[b * 3];
      next[b * 3 + 2] = c;
    }
    CHECK(cudaMemcpy(dp, pool2.data(), pool.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(drec, rec.data(), nrec * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dc, next.data(), next.size() * 4, cudaMemcpyHostToDevice));
    CHECK(gdr::Launch(dq, dk, dv, dg, db, dp, ds, dpos2, nullptr, dout, sh, nullptr, cm));
    std::vector<float> pool3(pool.size());
    CHECK(cudaMemcpy(pool3.data(), dp, pool.size() * 4, cudaMemcpyDeviceToHost));
    for (int b = 0; b < sh.B; ++b) {
      // Row 0's first token is padding: its first record is the identity.
      const int w = writes[b * sh.T + c - 1];
      const int after = w >= 0 ? w : (c - 1 == 0 ? -1 : writes[b * sh.T + c - 2]);
      if (after < 0) continue;
      for (size_t x = 0; x < slotState; ++x) stateDiff += pool3[commit[b * 3] * slotState + x] != gotPool[after * slotState + x];
    }
  }
  CHECK(cudaMemcpy(dc, commit.data(), commit.size() * 4, cudaMemcpyHostToDevice));
  std::printf("committed: outputs differing %zu, replayed states differing %zu (of %zu)%s\n", outDiff, stateDiff,
              static_cast<size_t>(sh.B * sh.T - 1) * slotState, outDiff == 0 && stateDiff == 0 ? "" : "  FAILED");
  std::vector<float> mc;
  for (int i = 0; i < 30; ++i) {
    cudaEventRecord(a);
    CHECK(gdr::Launch(dq, dk, dv, dg, db, dp, ds, dpos, nullptr, dout, sh, nullptr, cm));
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float t;
    cudaEventElapsedTime(&t, a, b);
    mc.push_back(t);
  }
  std::sort(mc.begin(), mc.end());
  // Each row then applies 4 records (what the next verify step does after accepting all).
  for (int r = 0; r < sh.B; ++r) commit[r * 3 + 2] = sh.T;
  CHECK(cudaMemcpy(dc, commit.data(), commit.size() * 4, cudaMemcpyHostToDevice));
  std::vector<float> md;
  for (int i = 0; i < 30; ++i) {
    cudaEventRecord(a);
    CHECK(gdr::Launch(dq, dk, dv, dg, db, dp, ds, dpos, nullptr, dout, sh, nullptr, cm));
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float t;
    cudaEventElapsedTime(&t, a, b);
    md.push_back(t);
  }
  std::sort(md.begin(), md.end());
  std::printf("4 sequences x 4 tokens, committed: %.3f ms per layer, %.3f ms applying 4 records first\n", mc[15], md[15]);
  return 0;
}
