// Standalone timing and check of the NVFP4 GEMM kernel.
//
//   nvcc -std=c++17 -O3 -arch=sm_121 fp4_gemm_bench.cu -o /tmp/fp4_bench && /tmp/fp4_bench
//
// Prints the largest relative difference from a CPU reference on a small shape,
// then, per shape, the median time and the rate at which it reads the packed
// codes and scales.
#include "../fp4_gemm.cu"

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

namespace {

struct Problem {
  int M, N, K;
  std::vector<uint8_t> w, sc, codes, scales;
  std::vector<float> x;
  float scale2 = 0.0123f;
};

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

Problem Make(int M, int N, int K, unsigned seed)
{
  Problem p{M, N, K};
  std::mt19937 g(seed);
  p.w.resize(static_cast<size_t>(N) * K / 2);
  for (auto& b : p.w) b = static_cast<uint8_t>(g());
  p.sc.resize(static_cast<size_t>(N) * K / 16);
  for (auto& b : p.sc) b = static_cast<uint8_t>(0x28 + g() % 0x20);  // e4m3 0.0625 .. 1.875
  p.x.resize(static_cast<size_t>(M) * K);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  for (auto& v : p.x) v = u(g);
  const int tiles = (N + 15) / 16;
  p.codes.resize(static_cast<size_t>(tiles) * 16 * 32 * (K / 64));
  p.scales.resize(static_cast<size_t>(tiles) * 64 * (K / 64));
  fp4::PackFp4(p.w.data(), p.sc.data(), N, K, p.codes.data(), p.scales.data());
  return p;
}

struct Dev {
  float *x, *y, *part, *scale2;
  void *codes, *scales;
};

Dev Upload(const Problem& p)
{
  Dev d;
  CHECK(cudaMalloc(&d.x, p.x.size() * 4));
  CHECK(cudaMalloc(&d.y, static_cast<size_t>(p.M) * p.N * 4));
  CHECK(cudaMalloc(&d.part, static_cast<size_t>(fp4::Splits(p.K, p.M)) * p.M * p.N * 4));
  std::vector<float> s2(p.N, p.scale2);
  CHECK(cudaMalloc(&d.scale2, 4 * p.N));
  CHECK(cudaMemcpy(d.scale2, s2.data(), 4 * p.N, cudaMemcpyHostToDevice));
  CHECK(cudaMalloc(&d.codes, p.codes.size()));
  CHECK(cudaMalloc(&d.scales, p.scales.size()));
  CHECK(cudaMemcpy(d.x, p.x.data(), p.x.size() * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d.codes, p.codes.data(), p.codes.size(), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d.scales, p.scales.data(), p.scales.size(), cudaMemcpyHostToDevice));
  return d;
}

void Free(Dev& d)
{
  for (void* q : {static_cast<void*>(d.x), static_cast<void*>(d.y), static_cast<void*>(d.part), static_cast<void*>(d.scale2), d.codes, d.scales}) cudaFree(q);
}

void Run(const Problem& p, Dev& d)
{
  CHECK(fp4::Launch(d.x, d.codes, d.scales, d.scale2, d.y, d.part, p.M, p.N, p.K, nullptr));
}

}  // namespace

int main()
{
  for (int M : {3, 13}) {
    Problem p = Make(M, 72, 320, 1);  // a partial tile and 3 x 64 + 128 of K over one chunk
    Dev d = Upload(p);
    Run(p, d);
    std::vector<float> y(static_cast<size_t>(M) * p.N);
    CHECK(cudaMemcpy(y.data(), d.y, y.size() * 4, cudaMemcpyDeviceToHost));
    double worst = 0;
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < p.N; ++n) {
        double s = 0, mag = 0;
        for (int k = 0; k < p.K; ++k) {
          const uint8_t b = p.w[static_cast<size_t>(n) * (p.K / 2) + k / 2];
          const double wv = fp4::E2m1(k % 2 ? b >> 4 : b & 15) * E4m3(p.sc[static_cast<size_t>(n) * (p.K / 16) + k / 16]);
          s += wv * Bf16(p.x[static_cast<size_t>(m) * p.K + k]);
          mag += std::fabs(wv * p.x[static_cast<size_t>(m) * p.K + k]);
        }
        worst = std::max(worst, std::fabs(y[static_cast<size_t>(m) * p.N + n] - s * p.scale2) / (mag * p.scale2 + 1e-30));
      }
    }
    std::printf("check M=%d: worst |d| / sum |w x| %.3g%s\n", M, worst, worst < 1e-5 ? "" : "  FAILED");
    Free(d);
    if (worst >= 1e-5) return 1;
  }
  struct Case {
    const char* name;
    int M, N, K;
  } cases[] = {
      {"27B gate+up, 4 rows", 4, 34816, 5120},  {"27B gate+up, 16 rows", 16, 34816, 5120},
      {"27B down, 4 rows", 4, 5120, 17408},     {"27B down, 16 rows", 16, 5120, 17408},
  };
  for (const Case& c : cases) {
    Problem p = Make(c.M, c.N, c.K, 7);
    Dev d = Upload(p);
    for (int i = 0; i < 3; ++i) Run(p, d);
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    std::vector<float> ms;
    for (int i = 0; i < 20; ++i) {
      cudaEventRecord(a);
      Run(p, d);
      cudaEventRecord(b);
      cudaEventSynchronize(b);
      float t;
      cudaEventElapsedTime(&t, a, b);
      ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    const double bytes = static_cast<double>(p.codes.size() + p.scales.size());
    std::printf("%-24s %7.3f ms  %5.0f GB/s of codes and scales (FP8 at the same rate: %.3f ms)\n", c.name, ms[10],
                bytes / ms[10] / 1e6, static_cast<double>(c.N) * c.K / (bytes / ms[10] / 1e3) / 1e3);
    Free(d);
  }
  return 0;
}
