# Cache and tenant security

Status: **per-key tenant identity implemented (2026-11-18); the keyed cache MAC still a design
awaiting a decision.** Written 2026-11-17 at `38210ed8`, updated 2026-11-18. Everything around
these is done and verified — see
[SERVING_REAUDIT_2026_11_17.md](../audits/SERVING_REAUDIT_2026_11_17.md) and
[xse-lessons-plan.md](../work-log/xse-lessons-plan.md) item 2.

Evidence labels are the repository's. Section 2 is **Run**; section 1 is a specification.

## 1. A keyed MAC for the executable cache (RT-9's full closure)

**What is already done.** `PjrtExecutableCache` refuses a world-writable cache or shared directory
by name, on both the JVM and (since the C++ caught up) the same entry-deletion policy. An entry is
SHA-256 checksummed, so corruption is caught. That blocks the anonymous-local-user case: someone
who can write the directory because it is world-writable is refused before any entry is read.

**What remains.** The checksum is *unkeyed*. An attacker who can write the directory through some
other means — a group member on a group-writable share, or a compromised process running as the
same user — can compute a valid SHA-256 for whatever native code they plant, and the plugin will
load and run it. The world-writable refusal does not cover them.

**The fix.** Authenticate each entry with a keyed MAC (HMAC-SHA256) under a secret only the serving
processes hold, and verify it before `DeserializeAndLoad`. A planted entry without the key fails
the MAC and is refused.

**The decisions this needs:**

- **Where the key comes from.** An environment variable (`TLALOC_CACHE_KEY`) is the simplest and
  matches how `TLALOC_API_KEY` already works; a file with owner-only permissions is the alternative.
  Both the JVM (`ServingModel`) and the C++ backend must read the same key. If no key is set, the
  behaviour must be explicit: either fall back to the unkeyed checksum (today's behaviour, with the
  world-writable refusal still in force) or refuse to use a shared cache at all. Falling back
  silently would hide the downgrade.
- **The wire format.** The entry header (`TLXC0001`, magic + size + SHA-256) gains a MAC field, or a
  new magic (`TLXC0002`) marks a keyed entry. A new magic is cleaner: an old reader refuses a keyed
  entry by name rather than misparsing it, and a keyed reader can accept both during a migration.
  The key material is shared with `triton/backend/executable_cache.cc`, so both sides change
  together and the golden key test (`keyIsPinnedSoTheTritonBackendCanComputeTheSameOne`) gains a
  keyed counterpart.
- **What the MAC covers.** The payload bytes, and ideally the identity material too, so an entry
  cannot be replayed under a different plugin. The cache key already covers the plugin's identity;
  the MAC should cover at least the payload.

**Why it is not implemented here.** It is a wire-format change on two sides with a key-management
choice and a migration path. Implementing it without deciding the key source and the no-key
behaviour would ship a security feature whose actual guarantee is unclear. The world-writable
refusal already removes the common case; this closes the rest once the decision is made.

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

Per-key tenant isolation IS implemented and tested on both endpoints: a caller cannot reach
another tenant's KV without that tenant's key. What is still not claimed: isolation between two
callers holding keys for the *same* tenant (one trust domain by definition), and cache
authentication against a writer who is not blocked by the world-writable refusal — the keyed MAC
of section 1, which remains a decision-ready design (the entry header, the world-writable refusal
and the golden key test exist and are tested, so it is a confined change once the key source and
no-key behaviour are decided).
