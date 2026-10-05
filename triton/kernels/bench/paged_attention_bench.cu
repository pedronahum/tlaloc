// Standalone timing and check of the paged-attention kernel, for iterating on it
// without XLA. Includes the kernel source.
//
//   nvcc -std=c++17 -O3 -arch=sm_121 -I../../third_party/xla paged_attention_bench.cu -o /tmp/pa_bench && /tmp/pa_bench
//
// Prints, per shape, the median time of SliceKernel + CombineKernel over a 32K
// context of e4m3fn pages, and the largest difference from a CPU reference on a
// small shape.
#include "../paged_attention.cu"

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

namespace {

#define CHECK(x)                                                                           \
  do {                                                                                     \
    cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                               \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));      \
      std::exit(1);                                                                        \
    }                                                                                      \
  } while (0)

struct Run {
  Shape sh;
  std::vector<float> q;
  std::vector<uint8_t> k, v;
  std::vector<int> tables, lens;
};

Run Make(int tables, int rowsPerTable, int H, int Hkv, int D, int ctx, int numBlocks, int bs, unsigned seed, int lenMinus)
{
  Run r;
  r.sh.R = tables * rowsPerTable;
  r.sh.H = H;
  r.sh.Hkv = Hkv;
  r.sh.D = D;
  r.sh.bs = bs;
  r.sh.M = ctx / bs;
  r.sh.Tb = tables;
  r.sh.S = (ctx + kThreads - 1) / kThreads;
  r.sh.slice = kThreads;
  r.sh.scale = 1.f / std::sqrt(static_cast<float>(D));
  std::mt19937 g(seed);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  r.q.resize(size_t(r.sh.R) * H * D);
  for (auto& x : r.q) x = u(g);
  const size_t n = size_t(numBlocks) * bs * Hkv * D;
  r.k.resize(n);
  r.v.resize(n);
  for (auto& x : r.k) x = static_cast<uint8_t>((g() % 0x50) | ((g() & 1) << 7));
  for (auto& x : r.v) x = static_cast<uint8_t>((g() % 0x50) | ((g() & 1) << 7));
  r.tables.resize(size_t(tables) * r.sh.M);
  for (size_t i = 0; i < r.tables.size(); ++i) r.tables[i] = 1 + int((i * 7) % (numBlocks - 1));
  r.lens.resize(r.sh.R);
  for (int i = 0; i < r.sh.R; ++i) r.lens[i] = ctx - lenMinus + (i % rowsPerTable);
  return r;
}

float F8(uint8_t b)
{
  __nv_fp8_e4m3 x;
  std::memcpy(&x, &b, 1);
  return static_cast<float>(x);
}

std::vector<float> Reference(const Run& r)
{
  const Shape& s = r.sh;
  const int G = s.H / s.Hkv, Qr = s.R / s.Tb;
  std::vector<float> out(size_t(s.R) * s.H * s.D, 0.f);
  for (int row = 0; row < s.R; ++row) {
    const int t = row / Qr;
    for (int h = 0; h < s.H; ++h) {
      const int hk = h / G;
      std::vector<double> sc(r.lens[row]);
      double mx = -1e300;
      for (int p = 0; p < r.lens[row]; ++p) {
        const size_t base = ((size_t(r.tables[t * s.M + p / s.bs]) * s.bs + p % s.bs) * s.Hkv + hk) * s.D;
        double d = 0;
        for (int i = 0; i < s.D; ++i) d += double(r.q[(size_t(row) * s.H + h) * s.D + i]) * s.scale * F8(r.k[base + i]);
        sc[p] = d;
        mx = std::max(mx, d);
      }
      double sum = 0;
      for (auto& x : sc) sum += (x = std::exp(x - mx));
      for (int p = 0; p < r.lens[row]; ++p) {
        const size_t base = ((size_t(r.tables[t * s.M + p / s.bs]) * s.bs + p % s.bs) * s.Hkv + hk) * s.D;
        for (int i = 0; i < s.D; ++i) out[(size_t(row) * s.H + h) * s.D + i] += float(sc[p] / sum * F8(r.v[base + i]));
      }
    }
  }
  return out;
}

struct Dev {
  float *q, *out, *scratch;
  uint8_t *k, *v;
  int *tables, *lens;
};

Dev Upload(const Run& r)
{
  Dev d;
  CHECK(cudaMalloc(&d.q, r.q.size() * 4));
  CHECK(cudaMalloc(&d.k, r.k.size()));
  CHECK(cudaMalloc(&d.v, r.v.size()));
  CHECK(cudaMalloc(&d.tables, r.tables.size() * 4));
  CHECK(cudaMalloc(&d.lens, r.lens.size() * 4));
  CHECK(cudaMalloc(&d.out, size_t(r.sh.R) * r.sh.H * r.sh.D * 4));
  CHECK(cudaMalloc(&d.scratch, size_t(r.sh.S) * r.sh.R * r.sh.H * (r.sh.D + 2) * 4));
  CHECK(cudaMemcpy(d.q, r.q.data(), r.q.size() * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d.k, r.k.data(), r.k.size(), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d.v, r.v.data(), r.v.size(), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d.tables, r.tables.data(), r.tables.size() * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d.lens, r.lens.data(), r.lens.size() * 4, cudaMemcpyHostToDevice));
  return d;
}

void Free(Dev& d)
{
  for (void* p : {static_cast<void*>(d.q), static_cast<void*>(d.out), static_cast<void*>(d.scratch), static_cast<void*>(d.k),
                  static_cast<void*>(d.v), static_cast<void*>(d.tables), static_cast<void*>(d.lens)}) {
    cudaFree(p);
  }
}

float Time(const Run& r, Dev& d)
{
  auto launch = [&] {
    CHECK(Launch<__nv_fp8_e4m3>(d.q, d.k, d.v, d.tables, d.lens, d.out, d.scratch, r.sh, nullptr));
  };
  for (int i = 0; i < 3; ++i) launch();
  CHECK(cudaDeviceSynchronize());
  std::vector<float> ms;
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  for (int i = 0; i < 20; ++i) {
    cudaEventRecord(a);
    launch();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float t = 0;
    cudaEventElapsedTime(&t, a, b);
    ms.push_back(t);
  }
  std::sort(ms.begin(), ms.end());
  return ms[ms.size() / 2];
}

}  // namespace

int main()
{
  // Correctness: a small shape with GQA, verify rows and lengths across slices; then
  // Qwen3.6-35B-A3B's heads (D 256, 8 queries per KV head) with four verify rows, which
  // the tensor-core kernel takes.
  for (const Run& r : {Make(2, 4, 8, 2, 64, 1024, 140, 16, 1, 300), Make(3, 4, 16, 2, 256, 1100, 300, 16, 5, 50)}) {
    Dev d = Upload(r);
    CHECK(Launch<__nv_fp8_e4m3>(d.q, d.k, d.v, d.tables, d.lens, d.out, d.scratch, r.sh, nullptr));
    std::vector<float> got(size_t(r.sh.R) * r.sh.H * r.sh.D);
    CHECK(cudaMemcpy(got.data(), d.out, got.size() * 4, cudaMemcpyDeviceToHost));
    const std::vector<float> want = Reference(r);
    float worst = 0;
    for (size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::fabs(got[i] - want[i]));
    std::printf("check: worst |d| against the CPU reference %.3g%s\n", worst, worst < 1e-4f ? "" : "  FAILED");
    Free(d);
    if (worst >= 1e-4f) return 1;
  }
  struct Case {
    const char* name;
    int tables, rows, H, Hkv;
  } cases[] = {
      {"27B decode (4 x 1 rows)", 4, 1, 24, 4}, {"27B verify (4 x 4 rows)", 4, 4, 24, 4}, {"27B head pass (4 x 5 rows)", 4, 5, 24, 4},
      {"35B decode (4 x 1 rows)", 4, 1, 16, 2}, {"35B verify (4 x 4 rows)", 4, 4, 16, 2}, {"35B head pass (4 x 5 rows)", 4, 5, 16, 2},
  };
  const int ctx = 32768, bs = 16, nb = 8200, D = 256;
  for (const Case& c : cases) {
    Run r = Make(c.tables, c.rows, c.H, c.Hkv, D, ctx, nb, bs, 7, 30);
    Dev d = Upload(r);
    const float ms = Time(r, d);
    const double bytes = 2.0 * c.tables * (ctx - 30) * c.Hkv * D;  // live keys and values, once per table
    std::printf("%-28s %7.3f ms  %6.0f GB/s of live K/V\n", c.name, ms, bytes / ms / 1e6);
    Free(d);
  }
  return 0;
}
