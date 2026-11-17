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
| SV-7 "must specify START" not retried after an idle timeout | **Closed** | Re-verified: `TritonClient.end` already retried the START-flag answer, but `generate` did not, and the endpoint's `TritonException` catch retried only "KV page pool exhausted", so an expired reuse fell through to a 500. The catch now also retries a "START flag" answer: the sequence is already dropped, so the next acquire starts a new one from the prompt and what was generated, bounded by `maxSequences` attempts. `anExpiredSequenceIsRestartedNotRefused` drives a scripted expiry on a reused continuation and was seen failing with the retry disabled (**Run**) |
| SV-8 `prompt + max_tokens == max_context` fails against speculative reach | **Closed** | Re-verified: `sequence_mode.cc:2091-2097` refuses `length + n + extra > max_context`, where `extra` is `2K-1` for a verify step plus 2 with no recurrent state, while the endpoint allowed `prompt + max_tokens == maxModelLen`. The exporter now writes `speculative_reach = 2K+1` (conservative: it assumes no state slots, the larger reach), `Main.kt` reads it, and the endpoint reserves it from the usable context, refusing up front and naming the reserve. Pinned by `theSpeculativeReachIsReservedFromTheContextLimit` and by the tiny-Ripple artifact test asserting `speculative_reach: 7` for 3 drafts; the endpoint test was seen failing with the reserve disabled (**Run**). An artifact exported before the parameter existed has no reach and keeps the old behaviour |
| SV-9 prefix-probing oracle across callers | **Open** | `Sequences.kt:84-99` still selects by longest token prefix across all live sequences, with no per-tenant salt; `evictIdle` (`:120-135`) ends other callers' idle sequences to make room. Same root as SV-1's residue (**Source**) |
| SV-10 a disconnect is seen only on a failed write | **Closed as a false claim; the waste is named, not removed** | Re-verified: the class KDoc said "a client that goes away … cancels its generation", which is true for a streamed response (the next chunk's write fails) and false for `stream:false`, where nothing is written until the end. `HttpExchange` exposes no liveness probe, so cancelling earlier would mean committing to a chunked body before the answer exists. The KDoc now states exactly when a disconnect is noticed and that an abandoned non-streamed request runs to `max_tokens`, bounded per step by `stepTimeoutMillis`. Early cancellation is **not implemented** and is recorded as a limitation rather than claimed (**Source**) |
| SV-11 `StopStrings.feed` cuts by UTF-16 unit | **Closed** | `StopStrings` is now a top-level internal class so the cut is testable directly, and both cuts go through `safeCut`, which moves back one unit when it would separate a surrogate pair. Seven tests: an emoji straddling the hold-back cut, an emoji fed one surrogate at a time, a stop match landing inside a pair, several emoji across three-unit feeds, and the ordinary stop/hold-back/no-stop cases. Three were seen failing with `safeCut` reduced to the identity (**Run**) |
| SV-12 `/health` constant; bad params 500; `response_format` ignored | **Closed for `/health` and the parameters; the Python detokenizer carried** | `/health`: `OpenAiEndpoint.health` asks Triton's `/v2/models/<model>/ready` at `EndpointConfig.tritonHttp` (wired from `Main.kt`'s `--triton-http`), replies 503 with a detail when not ready or unreachable, and says `backend: "unchecked"` rather than "ok" with no address; bounded at 2 s. `openai_server.py` matches with a new `--triton-http`. Parameters: `intOpt`/`longOpt` turn a `JsonException` into a 400 (a seed over 2^31 was a 500), `seed` is read as a Long, `top_p` must be in (0,1], `top_k` at least 1, and `response_format` / `frequency_penalty` / `presence_penalty` are refused when they ask for something non-default and accepted as no-ops at their defaults. Pinned by `badSamplingParametersAreBadRequestsNotServerErrors` and three health tests, each seen failing with its fix disabled; a four-case Python handler check passes (**Run**). The Python `Detokenizer.add` quadratic and ignored `stop` were not re-verified |
| SV-13 `Safetensors` element count overflows | **Closed** | `Safetensors.kt`'s `elementCount` fold and the `byteLength` product now go through `mulExact`, which refuses by name. The audit's own counterexample `[65536,65536,65536,65536]` with a zero-byte range is refused (`anElementCountThatWrapsToZeroIsRefused`), as is a count that fits while the byte length does not (`[2147483647,2147483647]` F64); a legitimate zero-sized tensor still parses. The wrap test was seen failing with the unchecked fold restored (**Run**) |

## What this changes

After the two passes that followed this re-audit, **eight findings are closed** (SV-2, SV-3 for
KV-only models, SV-4, SV-7, SV-8, SV-11, SV-13, and SV-12's `/health` and parameters), SV-10 is
resolved by correcting a false claim and naming the limitation it hid, three are partly closed
(SV-1, SV-5, SV-6), and **one is open: SV-9**. The only sub-part still carried without
re-verification is the Python `Detokenizer.add` quadratic and its ignored `stop`. Every closure
above has a test that was seen failing with the fix disabled.

The remaining risk is one root cause behind two findings: **a live sequence is matched by token
prefix across callers, with predictable ids and no owner** (SV-1's residue and SV-9). An API key
makes the endpoint one trust domain; it does not make two callers on that key unable to read each
other's context. Closing it needs an owner on `Live` and a per-caller salt on the prefix match,
not another transport guard. That is the next piece of work, and it is deliberately not attempted
here: it changes who may reuse whose KV, which is a serving-semantics decision rather than a bug
fix.

## Not covered here

The audit's runtime (RT), CI, docs and distributed findings are out of scope for this pass,
except that RT-1/2/3/4 and cache eviction were verified closed and RT-9 narrowed — see
[xse-lessons-plan.md](../work-log/xse-lessons-plan.md) item 2. The re-audit itself closed
nothing; SV-13 and SV-12's `/health` were closed by the pass that followed it, each with a test
seen failing under its negative control, and the table above records that.
