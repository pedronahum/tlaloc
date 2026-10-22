// Standalone timing and check of the NVFP4 mixture-of-experts kernels.
//
//   nvcc -std=c++17 -O3 -gencode arch=compute_121a,code=sm_121a moe_fp4_bench.cu -o /tmp/moe_bench && /tmp/moe_bench [rows] [fp4]
//
// Qwen3.6-35B-A3B's experts (256 of [1024, 2048] gate/up and [2048, 512] down,
// top 8): the largest relative difference from a CPU reference, then the
// median time and the rate at which the experts used are read. With `fp4` the
// activations are quantized to NVFP4 (Shape::actFp4): the reference quantizes
// them the same way, and the W4A4 result's distance from the W4A16 reference
// is printed too.
#include "../moe_fp4.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
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

// A row quantized as moe::QuantizeRowsKernel does, in place: v becomes code x scale (the row's
// g left out), g the row's scale.
void Quantize(std::vector<float>& v, float& g)
{
  float amax = 0.f;
  for (float f : v) amax = std::max(amax, std::fabs(f));
  g = amax / 2688.f;
  const float g6 = g * 6.f;
  for (size_t q = 0; q < v.size() / 16; ++q) {
    float a = 0.f;
    for (int i = 0; i < 16; ++i) a = std::max(a, std::fabs(v[16 * q + i]));
    const __nv_fp8_e4m3 s8(g6 > 0.f ? a / g6 : 0.f);
    const float sv = static_cast<float>(s8), sg = sv * g;
    for (int i = 0; i < 16; ++i) {
      const uint32_t c = sg > 0.f ? moe::E2m1Code(v[16 * q + i] / sg) : 0u;
      v[16 * q + i] = fp4::E2m1(static_cast<int>(c)) * sv;
    }
  }
}

}  // namespace

int main(int argc, char** argv)
{
  auto envInt = [](const char* k, int d) { const char* v = std::getenv(k); return v != nullptr ? std::atoi(v) : d; };
  const int H = envInt("MOE_H", 2048), I = envInt("MOE_I", 512), E = envInt("MOE_E", 256), K = envInt("MOE_K", 8);
  CHECK(moe::Prepare());
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
  if (argc > 1 && std::atoi(argv[1]) > 0) sizes = {std::atoi(argv[1])};  // one size, e.g. for a profiler
  const bool fp4a = argc > 2 && std::string(argv[2]) == "fp4";
  if (fp4a) sizes.erase(std::remove_if(sizes.begin(), sizes.end(), [](int r) { return r < 16; }), sizes.end());
  for (int R : sizes) {
    moe::Shape sh{R, H, I, E, K};
    sh.actFp4 = fp4a;
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
      double worst = 0, w4a16 = 0;
      for (int r : std::set<int>{0, std::min(1, R - 1), R / 2, R - 1}) {
        for (int mode = 0; mode < (fp4a ? 2 : 1); ++mode) {
          // mode 0: the arithmetic the kernel runs; mode 1 (W4A4 only): the W4A16 reference.
          const bool quant = fp4a && mode == 0;
          std::vector<double> want(H, 0.0), mag(H, 0.0);
          std::vector<float> xr(x.begin() + static_cast<size_t>(r) * H, x.begin() + static_cast<size_t>(r + 1) * H);
          float gxr = 1.f;
          if (quant) Quantize(xr, gxr);
          for (int j = 0; j < K; ++j) {
            const int e = top[r * K + j];
            const Matrix& mgu = gu[e];
            const Matrix& mdn = dn[e];
            const float* g2 = &guS2[static_cast<size_t>(e) * 2 * I];
            const float* d2 = &dnS2[static_cast<size_t>(e) * H];
            const float wj = wts[r * K + j];
            std::vector<float> hv(I);
            for (int i = 0; i < I; ++i) {
              double gs = 0, us = 0;
              for (int k = 0; k < H; ++k) {
                const double xb = quant ? xr[k] : Bf16(xr[k]);
                gs += mgu.W(i, k) * xb;
                us += mgu.W(I + i, k) * xb;
              }
              gs *= g2[i] * (quant ? gxr : 1.f);
              us *= g2[I + i] * (quant ? gxr : 1.f);
              const float hf = static_cast<float>(gs / (1 + std::exp(-gs)) * us);
              hv[i] = fp4a ? hf : Bf16(hf);
            }
            float gh = 1.f;
            if (quant) Quantize(hv, gh);
            for (int d = 0; d < H; ++d) {
              double s = 0, m = 0;
              for (int i = 0; i < I; ++i) {
                s += mdn.W(d, i) * hv[i];
                m += std::fabs(mdn.W(d, i) * hv[i]);
              }
              want[d] += wj * s * d2[d] * gh;
              mag[d] += wj * m * d2[d] * gh;
            }
          }
          double& out = mode == 0 ? worst : w4a16;
          for (int d = 0; d < H; ++d) out = std::max(out, std::fabs(y[static_cast<size_t>(r) * H + d] - want[d]) / (mag[d] + 1e-30));
        }
      }
      const double tol = fp4a ? 5e-3 : 1e-3;
      std::printf("check R=%d: worst |d| / sum |terms| %.3g%s", R, worst, worst < tol ? "" : "  FAILED");
      if (fp4a) std::printf("; from the W4A16 reference %.3g", w4a16);
      std::printf("\n");
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
