// Standalone timing and check of the NVFP4 mixture-of-experts kernels.
//
//   nvcc -std=c++17 -O3 -arch=sm_121 moe_fp4_bench.cu -o /tmp/moe_bench && /tmp/moe_bench [rows]
//
// Qwen3.6-35B-A3B's experts (256 of [1024, 2048] gate/up and [2048, 512] down,
// top 8): the largest relative difference from a CPU reference, then the
// median time and the rate at which the experts used are read.
#include "../moe_fp4.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
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

float E4m3(uint8_t b)
{
  __nv_fp8_e4m3 v;
  std::memcpy(&v, &b, 1);
  return static_cast<float>(v);
}

float Bf16(float f)
{
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u = (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000u;
  std::memcpy(&f, &u, 4);
  return f;
}

struct Matrix {  // one NVFP4 [N, K] in the checkpoint's layout
  int N, K;
  std::vector<uint8_t> w, sc;
  float W(int n, int k) const
  {
    const uint8_t b = w[static_cast<size_t>(n) * (K / 2) + k / 2];
    return fp4::E2m1(k % 2 ? b >> 4 : b & 15) * E4m3(sc[static_cast<size_t>(n) * (K / 16) + k / 16]);
  }
};

Matrix Random(int N, int K, std::mt19937& g)
{
  Matrix m{N, K};
  m.w.resize(static_cast<size_t>(N) * K / 2);
  for (auto& b : m.w) b = static_cast<uint8_t>(g());
  m.sc.resize(static_cast<size_t>(N) * K / 16);
  for (auto& b : m.sc) b = static_cast<uint8_t>(0x20 + g() % 0x18);
  return m;
}

}  // namespace

int main(int argc, char** argv)
{
  const int H = 2048, I = 512, E = 256, K = 8;
  std::mt19937 g(1);
  std::vector<Matrix> gu, dn;
  std::vector<uint8_t> guC, guS, dnC, dnS;
  for (int e = 0; e < E; ++e) {
    gu.push_back(Random(2 * I, H, g));
    dn.push_back(Random(H, I, g));
    auto pack = [](const Matrix& m, std::vector<uint8_t>& c, std::vector<uint8_t>& s) {
      const int tiles = (m.N + 15) / 16;
      std::vector<uint8_t> pc(static_cast<size_t>(tiles) * 512 * (m.K / 64)), ps(static_cast<size_t>(tiles) * 64 * (m.K / 64));
      fp4::PackFp4(m.w.data(), m.sc.data(), m.N, m.K, pc.data(), ps.data());
      c.insert(c.end(), pc.begin(), pc.end());
      s.insert(s.end(), ps.begin(), ps.end());
    };
    pack(gu.back(), guC, guS);
    pack(dn.back(), dnC, dnS);
  }
  std::vector<float> guS2(static_cast<size_t>(E) * 2 * I), dnS2(static_cast<size_t>(E) * H);
  for (auto& v : guS2) v = 0.01f + 0.001f * (g() % 8);
  for (auto& v : dnS2) v = 0.02f + 0.001f * (g() % 8);
  void *dGuC, *dGuS, *dDnC, *dDnS;
  float *dGuS2, *dDnS2;
  CHECK(cudaMalloc(&dGuC, guC.size()));
  CHECK(cudaMalloc(&dGuS, guS.size()));
  CHECK(cudaMalloc(&dDnC, dnC.size()));
  CHECK(cudaMalloc(&dDnS, dnS.size()));
  CHECK(cudaMalloc(&dGuS2, guS2.size() * 4));
  CHECK(cudaMalloc(&dDnS2, dnS2.size() * 4));
  CHECK(cudaMemcpy(dGuC, guC.data(), guC.size(), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dGuS, guS.data(), guS.size(), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dDnC, dnC.data(), dnC.size(), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dDnS, dnS.data(), dnS.size(), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dGuS2, guS2.data(), guS2.size() * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dDnS2, dnS2.data(), dnS2.size() * 4, cudaMemcpyHostToDevice));
  const double expertBytes = (static_cast<double>(guC.size() + guS.size() + dnC.size() + dnS.size())) / E;

  std::vector<int> sizes = {1, 4, 16, 20, 128, 512, 2048};
  if (argc > 1) sizes = {std::atoi(argv[1])};  // one size, e.g. for a profiler
  for (int R : sizes) {
    moe::Shape sh{R, H, I, E, K};
    std::vector<int> top(static_cast<size_t>(R) * K);
    std::vector<float> wts(top.size());
    std::set<int> used;
    for (int r = 0; r < R; ++r) {
      std::vector<int> ids(E);
      for (int e = 0; e < E; ++e) ids[e] = e;
      std::shuffle(ids.begin(), ids.end(), g);
      float s = 0;
      for (int j = 0; j < K; ++j) {
        top[r * K + j] = ids[j];
        used.insert(ids[j]);
        wts[r * K + j] = 0.05f + (g() % 100) / 100.f;
        s += wts[r * K + j];
      }
      for (int j = 0; j < K; ++j) wts[r * K + j] /= s;
    }
    std::vector<float> x(static_cast<size_t>(R) * H);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (auto& v : x) v = u(g);
    float *dx, *dw, *dy;
    int* dtop;
    void* scratch;
    CHECK(cudaMalloc(&dx, x.size() * 4));
    CHECK(cudaMalloc(&dw, wts.size() * 4));
    CHECK(cudaMalloc(&dtop, top.size() * 4));
    CHECK(cudaMalloc(&dy, x.size() * 4));
    CHECK(cudaMalloc(&scratch, moe::ScratchBytes(sh)));
    CHECK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dw, wts.data(), wts.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dtop, top.data(), top.size() * 4, cudaMemcpyHostToDevice));
    auto run = [&] {
      CHECK(moe::Launch(dx, dtop, dw, dGuC, dGuS, dGuS2, dDnC, dDnS, dDnS2, dy, scratch, sh, nullptr));
    };
    run();
    CHECK(cudaDeviceSynchronize());
    if (R <= 512) {
      std::vector<float> y(x.size());
      CHECK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
      double worst = 0;
      for (int r : std::set<int>{0, std::min(1, R - 1), R / 2, R - 1}) {
        std::vector<double> want(H, 0.0), mag(H, 0.0);
        for (int j = 0; j < K; ++j) {
          const int e = top[r * K + j];
          std::vector<float> hv(I);
          for (int i = 0; i < I; ++i) {
            double gs = 0, us = 0;
            for (int k = 0; k < H; ++k) {
              const double xb = Bf16(x[static_cast<size_t>(r) * H + k]);
              gs += gu[e].W(i, k) * xb;
              us += gu[e].W(I + i, k) * xb;
            }
            gs *= guS2[static_cast<size_t>(e) * 2 * I + i];
            us *= guS2[static_cast<size_t>(e) * 2 * I + I + i];
            hv[i] = Bf16(static_cast<float>(gs / (1 + std::exp(-gs)) * us));
          }
          for (int d = 0; d < H; ++d) {
            double s = 0, m = 0;
            for (int i = 0; i < I; ++i) {
              s += dn[e].W(d, i) * hv[i];
              m += std::fabs(dn[e].W(d, i) * hv[i]);
            }
            want[d] += wts[r * K + j] * s * dnS2[static_cast<size_t>(e) * H + d];
            mag[d] += wts[r * K + j] * m * dnS2[static_cast<size_t>(e) * H + d];
          }
        }
        for (int d = 0; d < H; ++d) worst = std::max(worst, std::fabs(y[static_cast<size_t>(r) * H + d] - want[d]) / (mag[d] + 1e-30));
      }
      std::printf("check R=%d: worst |d| / sum |terms| %.3g%s\n", R, worst, worst < 1e-3 ? "" : "  FAILED");
    }
    std::vector<float> ms;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int i = 0; i < 20; ++i) {
      cudaEventRecord(a);
      run();
      cudaEventRecord(b);
      cudaEventSynchronize(b);
      float t;
      cudaEventElapsedTime(&t, a, b);
      ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    const double bytes = expertBytes * used.size();
    std::printf("R=%-5d experts used %3zu  %7.3f ms  %5.0f GB/s of the experts used (one layer)\n", R, used.size(), ms[10],
                bytes / ms[10] / 1e6);
    cudaFree(dx); cudaFree(dw); cudaFree(dtop); cudaFree(dy); cudaFree(scratch);
  }
  return 0;
}
