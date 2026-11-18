# Cache and tenant security

Status: **both implemented (2026-11-18)** — the keyed cache MAC and per-key tenant identity.
Written 2026-11-17 at `38210ed8` as designs, implemented 2026-11-18. Everything around them is done
and verified — see [SERVING_REAUDIT_2026_11_17.md](../audits/SERVING_REAUDIT_2026_11_17.md) and
[xse-lessons-plan.md](../work-log/xse-lessons-plan.md) item 2.

Evidence labels are the repository's; both sections are **Run**.

## 1. A keyed MAC for the executable cache (RT-9's full closure) — IMPLEMENTED

**Done (2026-11-18).** Entries can now be authenticated with HMAC-SHA256 under a secret, on both
the JVM and the C++ backend, closing the gap the world-writable refusal left: a group member or a
compromised same-user process who can write the directory but does not hold the key cannot plant a
loadable executable.

The decisions were taken as the design proposed:

- **Key source.** `TLALOC_CACHE_KEY` (env), read by both `PjrtExecutableCache.fromEnv` and the
  backend's `Compile`, and forwarded by `run_server.sh`. Empty (the default) keeps the unkeyed
  format and today's behaviour — an explicit mode, not a silent downgrade.
- **Wire format.** A new magic `TLXC0002` marks a keyed entry: magic + LE length + HMAC-SHA256(key,
  payload) + payload. The unkeyed `TLXC0001` (SHA-256) is unchanged. A keyed reader refuses an
  unkeyed entry by name (it could be planted); an unkeyed reader refuses a keyed entry it cannot
  verify. A mode switch recompiles once rather than trusting the wrong tag. The filename key is
  unchanged — it covers the program and plugin identity, not the tag.
- **What the MAC covers.** The payload bytes. Replay under a different plugin is already blocked
  because the filename key covers the plugin's identity, so a foreign entry is not even looked up.

The HMAC is a real RFC 2104 construction on both sides — the JVM uses `javax.crypto.Mac`, the C++
builds it on the existing hand-rolled SHA-256 — and both pin the same RFC 4231 vectors
(`hmacMatchesTheRfcVectorsTheBackendPins`, `executable_cache_test.cc`), so they agree byte for byte.
Pinned further by `aKeyedEntryIsAuthenticatedAndEachModeRefusesTheOther` (wrong key, mode mismatch,
tampered payload all refused) and the device-gated
`aKeyedCacheWritesKeyedEntriesAndAnUnkeyedCacheRefusesThem` (the write path emits `TLXC0002`, a keyed
reload hits with no compile, an unkeyed reader rejects and replaces it). The C++ test adds a keyed
round trip, wrong-key and tamper refusal, and the live keyed AOT flow warms and reloads with zero
compiles under `TLALOC_CACHE_KEY` + `TLALOC_PJRT_CACHE_ONLY`.

**What remains.** Nothing for this boundary. The key is only as secret as the environment that
holds it; a deployment that leaks `TLALOC_CACHE_KEY` loses the guarantee, as with any shared secret.

## 2. Per-key tenant identity (SV-1 / SV-9's stronger boundary) — IMPLEMENTED

**Done (2026-11-18).** Both endpoints now authenticate a tenant from the bearer key and scope KV
reuse to it. `EndpointConfig.apiKeys` (Kotlin) and `TLALOC_API_KEYS` (both, `key:tenant` pairs,
comma-separated) map a key to a tenant; the owner of a sequence is `<tenant>/<user>`, so the
`user` field still sub-scopes cooperating callers within a tenant. A caller cannot reach another
tenant's sequences without holding its key — claiming its `user` field is not enough.

The decisions were taken as the design proposed: a key→tenant map alongside the existing single
`TLALOC_API_KEY` (which maps to the `default` tenant); one key or an open endpoint means every
caller shares `default`, which is exactly today's behaviour, so nothing breaks; the tenant lives
on the owner string threaded through `acquire`, and the matching logic is unchanged. Malformed
`key:tenant` entries are dropped rather than becoming an open key.

Pinned by `aTenantCannotBeClaimedWithoutItsKey` (Kotlin: the same `user` under another key gets
`cached_tokens: 0`, the own key reuses, unknown and missing keys are 401; seen failing with tenant
resolution disabled) and `tenant_auth()` in `triton/openai_server_checks.py` (eight cases across
per-key, single-key and open). The non-loopback guard now accepts `TLALOC_API_KEYS` as sufficient
auth on both endpoints.

**What remains.** Nothing for this boundary. The only isolation still not claimed is against a
caller who holds a valid key for the *same* tenant — which is one trust domain by definition.

## The two named limitations, for completeness

These are not security decisions and are already documented where they live; they are listed so
this note is the whole remaining picture:

- **The Python `Detokenizer.add` is quadratic on purpose.** `decode` is not suffix-independent
  (146 of 175 split points disagree on a real tokenizer), and no incremental decoder is exposed in
  Python. The measured reason is in the class docstring so the obvious "fix" is not applied into a
  text-corruption bug. A correct incremental detokenizer is a real piece of work, not a one-liner.
- **An abandoned `stream:false` request runs to `max_tokens`.** `HttpExchange` has no liveness
  probe; cancelling earlier means committing to a chunked body before the answer exists. The
  endpoint KDoc states this rather than claiming cancellation it does not have.

## What this note does not claim

Both boundaries are implemented and tested: per-key tenant isolation on both endpoints (a caller
cannot reach another tenant's KV without that tenant's key), and keyed cache authentication on both
the JVM and the C++ backend (a directory writer without `TLALOC_CACHE_KEY` cannot plant a loadable
executable). What is still not claimed, by definition rather than by omission: isolation between
two callers holding keys for the *same* tenant (one trust domain), and any guarantee once
`TLALOC_CACHE_KEY` itself is leaked (a shared secret is only as secret as its environment). The two
named limitations below — the deliberate Python detokenizer quadratic and the abandoned
`stream:false` request — are unrelated to these boundaries and remain as documented.
