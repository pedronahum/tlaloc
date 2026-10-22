// Copyright 2026 Pedro N. Rodriguez
// SPDX-License-Identifier: Apache-2.0

#include "executable_cache.h"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <set>

namespace tlaloc_triton {

namespace {

// FIPS 180-4 SHA-256.
class Sha256State {
 public:
  void Update(const unsigned char* p, size_t n)
  {
    for (size_t i = 0; i < n; ++i) {
      block_[fill_++] = p[i];
      if (fill_ == 64) {
        Compress();
        fill_ = 0;
      }
    }
    bits_ += static_cast<uint64_t>(n) * 8;
  }

  std::string Finish()
  {
    const uint64_t bits = bits_;
    const unsigned char one = 0x80, zero = 0;
    Update(&one, 1);
    while (fill_ != 56) Update(&zero, 1);
    unsigned char len[8];
    for (int i = 0; i < 8; ++i) len[i] = static_cast<unsigned char>(bits >> (56 - 8 * i));
    Update(len, 8);
    std::string out(32, '\0');
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 4; ++j) out[4 * i + j] = static_cast<char>(h_[i] >> (24 - 8 * j));
    }
    return out;
  }

 private:
  static uint32_t Rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

  void Compress()
  {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (uint32_t(block_[4 * i]) << 24) | (uint32_t(block_[4 * i + 1]) << 16) |
             (uint32_t(block_[4 * i + 2]) << 8) | uint32_t(block_[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t t1 = h + (Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
      const uint32_t t2 = (Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h_[0] += a;
    h_[1] += b;
    h_[2] += c;
    h_[3] += d;
    h_[4] += e;
    h_[5] += f;
    h_[6] += g;
    h_[7] += h;
  }

  uint32_t h_[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  unsigned char block_[64];
  size_t fill_ = 0;
  uint64_t bits_ = 0;
};

const char kMagic[] = "TLXC0001";
constexpr size_t kMagicSize = 8;
constexpr size_t kHeaderSize = kMagicSize + 8 + 32;

void
PutLe64(std::string* s, uint64_t v)
{
  for (int i = 0; i < 8; ++i) s->push_back(static_cast<char>(v >> (8 * i)));
}

std::string
Hex(const std::string& raw)
{
  static const char d[] = "0123456789abcdef";
  std::string out;
  for (unsigned char c : raw) {
    out.push_back(d[c >> 4]);
    out.push_back(d[c & 15]);
  }
  return out;
}

// Keys some thread of this process is working on.
std::mutex g_keys_mu;
std::condition_variable g_keys_cv;
std::set<std::string> g_keys_busy;

}  // namespace

std::string
Sha256(const std::string& bytes)
{
  Sha256State s;
  s.Update(reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size());
  return s.Finish();
}

std::string
CacheKey(const CacheIdentity& id, const std::string& options_bytes, const std::string& program)
{
  std::string material;
  auto field = [&material](const std::string& b) {
    PutLe64(&material, b.size());
    material += b;
  };
  field("tlaloc-pjrt-cache-v1");
  field(id.plugin_path);
  field(std::to_string(id.plugin_size));
  field(std::to_string(id.plugin_mtime_seconds));
  field(id.platform_name);
  field(id.platform_version);
  field(id.device_kind);
  field(id.xla_flags);
  field(options_bytes);
  field(program);
  return Hex(Sha256(material));
}

std::string
ReadCacheEntry(const std::string& path, std::string* payload)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) return "cannot open " + path;
  std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (all.size() < kHeaderSize) return path + " is shorter than its header";
  if (all.compare(0, kMagicSize, kMagic, kMagicSize) != 0) return path + " has no TLXC0001 magic";
  uint64_t size = 0;
  for (int i = 0; i < 8; ++i) size |= uint64_t(static_cast<unsigned char>(all[kMagicSize + i])) << (8 * i);
  if (size != all.size() - kHeaderSize) return path + " is truncated";
  std::string body = all.substr(kHeaderSize);
  if (Sha256(body) != all.substr(kMagicSize + 8, 32)) return path + " fails its checksum";
  *payload = std::move(body);
  return "";
}

std::string
WriteCacheEntry(const std::string& path, const std::string& payload)
{
  std::string header(kMagic, kMagicSize);
  PutLe64(&header, payload.size());
  header += Sha256(payload);
  std::string tmp = path + ".tmp.XXXXXX";
  int fd = mkstemp(&tmp[0]);
  if (fd < 0) return "cannot create a temporary file next to " + path;
  bool ok = write(fd, header.data(), header.size()) == static_cast<ssize_t>(header.size());
  size_t done = 0;
  while (ok && done < payload.size()) {
    ssize_t n = write(fd, payload.data() + done, payload.size() - done);
    if (n <= 0) ok = false;
    else done += static_cast<size_t>(n);
  }
  ok = close(fd) == 0 && ok;
  if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
    unlink(tmp.c_str());
    return "cannot write " + path;
  }
  return "";
}

CacheKeyLock::CacheKeyLock(const std::string& dir, const std::string& key) : key_(key)
{
  {
    std::unique_lock<std::mutex> lock(g_keys_mu);
    g_keys_cv.wait(lock, [&] { return g_keys_busy.count(key_) == 0; });
    g_keys_busy.insert(key_);
  }
  const std::string path = dir + "/" + key + ".lock";
  fd_ = open(path.c_str(), O_CREAT | O_WRONLY | O_CLOEXEC, 0644);
  if (fd_ >= 0) {
    // A POSIX record lock, the kind Java's FileChannel.lock takes, so the JVM
    // and this backend exclude each other (flock and fcntl locks do not).
    struct flock fl {};
    fl.l_type = F_WRLCK;
    fl.l_whence = SEEK_SET;
    while (fcntl(fd_, F_SETLKW, &fl) != 0 && errno == EINTR) {
    }
  }
}

CacheKeyLock::~CacheKeyLock()
{
  if (fd_ >= 0) close(fd_);  // releases the record lock
  {
    std::lock_guard<std::mutex> lock(g_keys_mu);
    g_keys_busy.erase(key_);
  }
  g_keys_cv.notify_all();
}

}  // namespace tlaloc_triton
