# Cache and tenant security: the two decisions left

Status: **design, awaiting a decision**. Written 2026-11-17 at `38210ed8`. These are the two
security pieces the serving work deliberately did not implement unilaterally, because each is a
deployment decision rather than a bug fix. Everything around them is done and verified — see
[SERVING_REAUDIT_2026_11_17.md](../audits/SERVING_REAUDIT_2026_11_17.md) (eleven of thirteen
findings closed) and [xse-lessons-plan.md](../work-log/xse-lessons-plan.md) item 2.

Evidence labels are the repository's; nothing here is **Run** — it is a specification.

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

## 2. Per-key tenant identity (SV-1 / SV-9's stronger boundary)

**What is already done.** KV reuse is scoped to an `owner` — the request's OpenAI `user` field, ""
when absent — on both endpoints, and sequence ids are unpredictable. Two cooperating callers who
send different `user` values cannot read each other's context through `cached_tokens`.

**What remains.** With one shared `TLALOC_API_KEY`, a caller can simply claim another's `user`
value. The owner is a convention, not an authenticated identity. So the isolation holds against
accidental cross-caller reuse, not against a deliberate caller sharing the key.

**The fix.** Make the owner an authenticated identity derived from the credential, not a
client-supplied field: support multiple API keys, each mapped to a tenant id, and scope sequences
to the authenticated tenant. A caller cannot claim another tenant because it does not hold that
tenant's key.

**The decisions this needs:**

- **The key model.** A map of key → tenant (a file or env), versus a single key with the `user`
  field trusted (today). The former is a real boundary; the latter is what exists.
- **Backward compatibility.** A deployment with one key and no `user` fields must keep working:
  one key maps to one default tenant, and reuse within it is unchanged.
- **Where the tenant lives.** On `Live` (replacing the `owner` string's source), threaded through
  `acquire` exactly as `owner` is today. The matching logic does not change — only where the owner
  value comes from.

**Why it is not implemented here.** It changes the authentication model (one key → many), which is
a deployment decision. The plumbing (owner-scoped matching, unpredictable ids) is already in place
and tested, so the change is confined to how the owner is derived once the key model is chosen.

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

Nothing here is implemented or run. The two designs are decision-ready: the plumbing they need
(owner-scoped matching, the entry header, the world-writable refusal, the golden key test) exists
and is tested, so each is a confined change once its decision is made. No multi-tenant isolation
against a deliberate caller sharing a key is claimed today, and no cache authentication against a
writer who is not blocked by the world-writable refusal is claimed today.
