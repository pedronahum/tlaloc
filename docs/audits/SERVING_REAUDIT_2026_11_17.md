# Serving re-audit against HEAD, 2026-11-17

Branch `feat/brooklyn-eats` at `f20eee97`. Re-checks the serving findings of
[REPO_AUDIT_2026_11.md](REPO_AUDIT_2026_11.md), whose snapshot is `869950a6`. That audit is
**not wrong about its snapshot**; several findings were closed after it, so working its list
verbatim re-does shipped work. Every state below was read at HEAD, with the line cited.

Evidence: **Source** = read in the tree at HEAD; **Run** = a test or a live server confirmed it.
"Carried" means the finding was not re-verified here and stands as the November audit recorded it.

| Finding | State at HEAD | Evidence |
|---|---|---|
| SV-1 no auth, `0.0.0.0`, predictable ids, no owner check, ids in the page error, `text/plain` accepted | **Partly closed** | Closed by `1cf3bbd6`: `Main.kt:74-77` defaults to `127.0.0.1` and refuses a non-loopback host without `TLALOC_API_KEY` unless `--no-auth`; `OpenAiEndpoint.kt:51-52,166-167,208` bearer-checks every route but `/health`; `:187` refuses a non-JSON POST with 415; `run_server.sh` binds `TRITON_BIND:-127.0.0.1`. **Still open**: `Sequences.kt:39` `nextId = (currentTimeMillis/1000 % 100_000)*100 + 1` is clock-derived and sequential, and `Sequences.kt:84-99` `take()` matches ANY caller's request to ANY live sequence by token prefix with no owner or tenant check. With a key set this is one trust domain; without per-caller ownership it is still a cross-caller context leak (**Source**) |
| SV-2 a client that stops reading holds its seat forever | **Closed** | `OpenAiEndpoint.kt:58,60,62` `queueTimeoutMillis` 600 s, `writeTimeoutMillis` 60 s, `stepTimeoutMillis` 1 h; `:525` `g.next(config.stepTimeoutMillis)`; `:744-745` the write is a future with a timeout; `Sequences.kt:107-109` bounded `maxWaiting` and a queue deadline, both surfaced as 429 (`OpenAiEndpoint.kt:509`); `TritonClient.kt:171-172` polls with a deadline and throws (**Source**) |
| SV-3 chained step skips end-token truncation | **Closed for KV-only; conditional otherwise** | `sequence_mode.cc:2871-2882`: the truncation now runs when `(!issued \|\| kv_only)`, so a KV-only model truncates on a chained step too. With recurrent state a chained step may have written the end token's state slot, so the response keeps the tokens after it and the sequence must not be continued — the comment at `:2866-2870` says the OpenAI frontends end it (**Source**) |
| SV-4 Python `chat_template_kwargs` unfiltered | **Closed** | `openai_server.py:49` `TEMPLATE_KWARGS` whitelist, applied at `:270` before `apply_chat_template`, with the comment naming `chat_template` as the key that must not pass (**Source**) |
| SV-5 one long prompt blocks everyone | **Partly closed — now bounded by measurement** | The structure is unchanged (`sequence_mode.cc:3485` still passes `decode.empty() ? 0 : prefill_budget()`), but the knob is no longer a guess: `:maestro:servedSloBench` with `staggerMs` measured it. Qwen3-0.6B, 1024-token prompts, N=4: budget 0 gives p99 TPOT 145.8 ms against 85.5 ms at the default one-chunk budget, for the same throughput. So the default bounds the blocking, and **0 is the setting to avoid**. Cancellation ordering not re-verified (**Run**, [SERVING_ARCHITECTURE.md](../SERVING_ARCHITECTURE.md) §6) |
| SV-6 unbounded body, tokenize before the check, unbounded JSON depth | **Partly closed** | `OpenAiEndpoint.kt:53-54,189-190` `maxBodyBytes` 64 MiB, `readNBytes` of at most limit+1, then 413; `Json.kt:145` refuses nesting past `MAX_DEPTH`, so the recursion cannot overflow the stack. Whether tokenization still precedes the length check was not re-verified (**Source**) |
| SV-7 "must specify START" not retried after an idle timeout | **Carried** | Not re-verified. No retry marker found in `Sequences.kt` or `OpenAiEndpoint.kt`; `maxSequenceIdleMicros` is now 600 s in the brooklyn-eats lanes, which lengthens the window but does not retry |
| SV-8 `prompt + max_tokens == max_context` fails against speculative reach | **Carried** | Not re-verified |
| SV-9 prefix-probing oracle across callers | **Open** | `Sequences.kt:84-99` still selects by longest token prefix across all live sequences, with no per-tenant salt; `evictIdle` (`:120-135`) ends other callers' idle sequences to make room. Same root as SV-1's residue (**Source**) |
| SV-10 a disconnect is seen only on a failed write | **Carried** | Not re-verified. `writeTimeoutMillis` now bounds a stuck write, which limits the damage but is not disconnect detection |
| SV-11 `StopStrings.feed` cuts by UTF-16 unit | **Open** | `OpenAiEndpoint.kt:627-640`: `s.substring(0, at)` and `s.substring(0, s.length - keep)` with no surrogate-pair guard, so a held-back split emoji is emitted as a lone surrogate (**Source**) |
| SV-12 `/health` constant through a GPU fault; bad params 500; `response_format` ignored | **Open (health)** | `OpenAiEndpoint.kt:181` still replies `jsonObject("status" to "ok")` unconditionally. The parameter and `response_format` parts were not re-verified (**Source**) |
| SV-13 `Safetensors` element count overflows | **Open** | `Safetensors.kt:73` `dims.fold(1L) { a, d -> a * d }` and `:302` `e.elementCount * dt.sizeBytes`, both unchecked `Long`. `[65536,65536,65536,65536]` folds to 2^64 ≡ 0, so `want` is 0 and a zero-byte range passes the equality check at `:303` (**Source**) |

## What this changes

Three findings are closed outright (SV-2, SV-4, and SV-3 for KV-only models), two are partly
closed (SV-1, SV-6), one is now bounded by a measurement rather than a guess (SV-5), and four
are confirmed still open at the cited lines (SV-9, SV-11, SV-12's health, SV-13). Four are
carried without re-verification and are marked as such rather than assumed.

The two that matter most for a multi-tenant deployment are the same root cause: **a live
sequence is matched by token prefix across callers, with predictable ids and no owner**
(SV-1's residue and SV-9). An API key makes the endpoint one trust domain; it does not make
two callers on that key unable to read each other's context. Closing it needs an owner on
`Live` and a per-caller salt on the prefix match, not another transport guard.

The cheapest definite fixes are SV-13 (a `Math.multiplyExact`-style guard in the fold and the
product, with the audit's own four-dimension counterexample as the test) and SV-12's health
(reply from the backend's state rather than a constant). SV-11 needs a surrogate-aware cut.

## Not covered here

The audit's runtime (RT), CI, docs and distributed findings are out of scope for this pass,
except that RT-1/2/3/4 and cache eviction were verified closed and RT-9 narrowed — see
[xse-lessons-plan.md](../work-log/xse-lessons-plan.md) item 2. No finding here was closed by
this pass; it records state, and the fixes above are proposed, not done.
