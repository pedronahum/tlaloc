// Copyright 2026 Pedro N. Rodriguez
// SPDX-License-Identifier: Apache-2.0

// Compiled executables kept on disk, shared with the JVM's PjrtExecutableCache
// (runtime-pjrt): the same key, the same file layout, the same lock files, so a
// program compiled by either is loaded by both.
//
// Key: SHA-256, as lowercase hex, over length-prefixed fields (8-byte little
// endian length, then the bytes): "tlaloc-pjrt-cache-v1", the plugin path,
// size and modification time (seconds, decimal), the platform name and version,
// the device kind, $XLA_FLAGS, the CompileOptionsProto bytes, the program text.
//
// Entry `<dir>/<key>.pjrt`: a magic, the payload length (8 bytes, little
// endian), a 32-byte tag, the payload (PJRT_Executable_Serialize's bytes).
// `<dir>/<key>.lock` carries a POSIX record lock (fcntl, as Java's
// FileChannel.lock) while one process compiles.
//
// Two magics, selected by whether a cache key is configured (TLALOC_CACHE_KEY,
// passed to Read/Write as `key`):
//   "TLXC0001" — unkeyed: the tag is the payload's SHA-256. The default.
//   "TLXC0002" — keyed: the tag is HMAC-SHA256(key, payload). A reader with a key
//                accepts ONLY keyed entries (an unkeyed one could be planted by
//                anyone who can write the directory); a reader without a key
//                accepts ONLY unkeyed ones (it cannot verify a keyed tag). So a
//                mode switch recompiles once rather than trusting the wrong tag.
// The key (filename) is unchanged by the mode: it covers the program and the
// plugin identity, not the tag.

#pragma once

#include <cstdint>
#include <string>

namespace tlaloc_triton {

struct CacheIdentity {
  std::string plugin_path;
  int64_t plugin_size = 0;
  int64_t plugin_mtime_seconds = 0;
  std::string platform_name;
  std::string platform_version;
  std::string device_kind;
  std::string xla_flags;
};

// The SHA-256 of `bytes` as 32 raw bytes.
std::string Sha256(const std::string& bytes);

// HMAC-SHA256 of `msg` under `key`, as 32 raw bytes (RFC 2104).
std::string HmacSha256(const std::string& key, const std::string& msg);

std::string CacheKey(
    const CacheIdentity& identity, const std::string& options_bytes, const std::string& program);

// Reads and checks an entry; "" on success. `key` empty selects the unkeyed
// format (TLXC0001, SHA-256); non-empty selects the keyed format (TLXC0002,
// HMAC-SHA256) and refuses an entry of the other mode by name.
std::string ReadCacheEntry(const std::string& path, std::string* payload, const std::string& key = "");

// Writes an entry through a temporary file and a rename; "" on success. `key`
// selects the format as in ReadCacheEntry.
std::string WriteCacheEntry(const std::string& path, const std::string& payload, const std::string& key = "");

// Holds an exclusive record lock on `<dir>/<key>.lock`, and an in-process
// claim on the key (record locks belong to the process, not the thread).
class CacheKeyLock {
 public:
  CacheKeyLock(const std::string& dir, const std::string& key);
  ~CacheKeyLock();
  CacheKeyLock(const CacheKeyLock&) = delete;
  CacheKeyLock& operator=(const CacheKeyLock&) = delete;

 private:
  std::string key_;
  int fd_ = -1;
};

}  // namespace tlaloc_triton
