// Copyright 2026 Pedro N. Rodriguez
// SPDX-License-Identifier: Apache-2.0
//
// Checks for the executable cache's key and entry format. Built and run by
// build_backend.sh. The golden key is the one PjrtExecutableCacheTest pins on
// the JVM, so the two sides name an entry the same way.

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#include "../executable_cache.h"

using namespace tlaloc_triton;

static int failures = 0;

#define CHECK(cond)                                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                     \
    }                                                                 \
  } while (0)

static std::string
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

int
main()
{
  // FIPS 180-4 test vectors.
  CHECK(Hex(Sha256("")) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(Hex(Sha256("abc")) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  CHECK(Hex(Sha256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")) ==
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  CHECK(Hex(Sha256(std::string(1000000, 'a'))) ==
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");

  // RFC 4231 HMAC-SHA-256 test cases 1 and 2: the keyed tag must be a real HMAC,
  // not a bespoke construction, so the JVM's javax.crypto.Mac agrees with it.
  CHECK(Hex(HmacSha256(std::string(20, '\x0b'), "Hi There")) ==
        "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
  CHECK(Hex(HmacSha256("Jefe", "what do ya want for nothing?")) ==
        "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
  // A key longer than the 64-byte block is hashed first (RFC 2104).
  CHECK(Hex(HmacSha256(std::string(131, '\xaa'), "Test Using Larger Than Block-Size Key - Hash Key First")) ==
        "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54");

  CacheIdentity id;
  id.plugin_path = "/opt/pjrt/plugin.so";
  id.plugin_size = 1234;
  id.plugin_mtime_seconds = 1700000000;
  id.platform_name = "cuda";
  id.platform_version = "cuda 13000";
  id.device_kind = "NVIDIA GB10";
  const std::string opts("\x1A\x04\x20\x01\x28\x01", 6);
  CHECK(CacheKey(id, opts, "module {}") == "136d3eaa56d6eb4660560b0b28713d2a40864162121e469bb842471d448e3dc7");
  CacheIdentity other = id;
  other.xla_flags = "--xla_gpu_autotune_level=0";
  CHECK(CacheKey(other, opts, "module {}") != CacheKey(id, opts, "module {}"));

  char tmpl[] = "/tmp/tlaloc_cache_test_XXXXXX";
  const std::string dir = mkdtemp(tmpl);
  const std::string entry = dir + "/e.pjrt";
  const std::string payload = std::string("executable\0bytes", 16);
  CHECK(WriteCacheEntry(entry, payload).empty());
  std::string back;
  CHECK(ReadCacheEntry(entry, &back).empty() && back == payload);
  // Truncate the file: refused.
  CHECK(truncate(entry.c_str(), 40) == 0);
  CHECK(!ReadCacheEntry(entry, &back).empty());

  // Keyed entries: a round trip under a key, a wrong key refused, a tampered payload
  // refused, and each mode refusing the other's magic by name.
  const std::string kentry = dir + "/k.pjrt";
  const std::string secret = "a-cache-secret";
  CHECK(WriteCacheEntry(kentry, payload, secret).empty());
  CHECK(ReadCacheEntry(kentry, &back, secret).empty() && back == payload);
  CHECK(!ReadCacheEntry(kentry, &back, "the-wrong-secret").empty());  // wrong key
  CHECK(!ReadCacheEntry(kentry, &back).empty());                       // unkeyed reader, keyed entry
  CHECK(ReadCacheEntry(kentry, &back, secret).empty());                // still intact
  // A keyed reader refuses an unkeyed entry (it could be planted).
  const std::string uentry = dir + "/u.pjrt";
  CHECK(WriteCacheEntry(uentry, payload).empty());
  CHECK(!ReadCacheEntry(uentry, &back, secret).empty());
  CHECK(ReadCacheEntry(uentry, &back).empty() && back == payload);
  // Tamper with a keyed payload: the HMAC catches it.
  CHECK(WriteCacheEntry(kentry, payload, secret).empty());
  {
    FILE* f = std::fopen(kentry.c_str(), "r+b");
    CHECK(f != nullptr);
    if (f) {
      std::fseek(f, 48, SEEK_SET);  // first payload byte, after the 48-byte header
      std::fputc('X', f);
      std::fclose(f);
    }
  }
  CHECK(!ReadCacheEntry(kentry, &back, secret).empty());
  {
    CacheKeyLock a(dir, "k");
  }
  CacheKeyLock b(dir, "k");  // reacquired once released

  if (failures > 0) {
    std::printf("%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("executable_cache_test: all checks passed\n");
  return 0;
}
