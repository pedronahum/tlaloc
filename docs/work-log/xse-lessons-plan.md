# XLA Serving Engine: what transfers, and the plan that follows

Branch `feat/brooklyn-eats` at `b26f77e1` (2026-11-15); written 2026-11-17.

Goal: take the one thing worth taking from Google's XLA Serving Engine announcement, and turn
it into six scoped items with measurable bars — cold start, goodput-at-SLO, a CPU serving lane,
runtime parity, sharded serving, and the hardening that any packaging claim needs first.

Evidence labels are the repository's: **Run** (a program was compiled and executed and gave the
stated output), **Source** (traced through the code, not executed), **Doc** (quoted from a
document in this repository that recorded a measurement), **API** (read from a public endpoint).
**Nothing was re-measured for this plan.** Every performance figure below is **Doc** — quoted
from [SERVING_ARCHITECTURE.md](../SERVING_ARCHITECTURE.md),
[REPO_AUDIT_2026_11.md](../audits/REPO_AUDIT_2026_11.md), [FOLLOWUPS.md](../FOLLOWUPS.md) or a
work log — and each item's bar says what must be measured before it may be claimed.

## 1. The source

"Supercharging XLA-based inference: Announcing the launch of XLA Serving Engine", Oleg Shaldybin
and Monica Song, Google TPU Software, `opensource.googleblog.com`, 2026-11-05 (**API**, fetched
2026-11-17). Apache 2.0; `github.com/openxla/xla-serving`; docs at `xla-serving.readthedocs.io`.

What XSE is, in the post's own words: "a general-purpose model export and inference runtime for
the OpenXLA ecosystem" for a "Train in Python, Serve in C++" workflow. An export API writes an
"XLA SavedModel" holding "the StableHLO representation of their model's function", with weights
in an "Orbax checkpoint". The "XLA Serving Runtime" "wraps PJRT directly and optimizes the model
using the XLA compiler". Docker images "tested on CPU, GPU, and TPU" expose "simple gRPC APIs".

Its six named pillars: **Python-Free Serving**, **Low-latency execution**, **Orbax Checkpointing
Integration**, **Flexible Scale** ("single-host and multi-host serving by leveraging PJRT"),
**Performance Parity**, **Docker Packages**.

Its numbers: "over 6 million queries per second" in production (YouTube Shorts, Google Ads,
Search, parts of Gemini Omni); "for short requests, XSE delivers 2–5x the throughput of the best
Python server we could build, and 8–36x that of a straightforward one"; "30–70% less CPU time per
prediction"; "a quarter of the memory"; "one process is capable of using all eight cores".
Partners: Arm (Neoverse, Google Cloud Axion — "vision, embedding and recommendation workloads"),
Intel (Xeon, AMX — "Vision, Recommendation, and LLM"), Moloco (650 billion daily inferences,
10.5 million QPS peak).

Its stated future work: **Torch Support** (via TorchTPU), **Ahead-of-Time (AOT) Compilation**
("reduce cold-start time by offering users the ability to compile their models offline (on CPU)
for their target serving hardware"; internal rollout first), **C++ Data Preprocessing**.

Its stated limit: "XSE is a lightweight, one-shot inference engine." For "large autoregressive
language models with long context lengths" it recommends "popular projects such as vLLM or
SGLang, which benefit from LLM-specific memory management techniques." The post contains no
mention of a KV cache, paged attention, continuous batching, speculative decoding, quantization,
dynamic shapes, or a batching API.

## 2. Where we stand against it

The architecture is the same thesis, and on the LLM axis we are ahead of it, not behind: XSE
defers long-context autoregressive serving to vLLM and SGLang, which is the exact space our
Triton backend occupies (paged KV, windowed-pool rings for sliding layers, chunked prefill,
blockwise online-softmax attention, linear-attention state pools, speculative decoding, KV pool
donation with in-place aliasing, backend-formed batches).

We already serve through five paths, all reading one artifact:

| | Path | Process | Evidence |
|---|---|---|---|
| i | `harness/python/tlaloc_serve.py` + `tlaloc_pjrt.py` | Python, ctypes over the PJRT C API; no jax, torch or numpy | **Source**; import-guarded by tests |
| ii | `harness/python/vllm_tlaloc` | vLLM engine + platform plugin (`TlalocPlatform`) | **Source** |
| iii | `triton/backend/libtriton_tlaloc.so` | `tritonserver`, C++ backend, HTTP + gRPC | **Source**; serves v3/v4/v5 |
| iv | `ServingModel` (`:runtime-pjrt`) | in-JVM, plain-Java signatures | **Source**; `examples/java-inference`, `examples/spark-inference` |
| — | `endpoint/` (`OpenAiEndpoint`, `Http2`, `Hpack`, `Protobuf`, `TritonClient`, `Sequences`) and `triton/openai_server.py` | OpenAI-compatible frontends over (iii) | **Source** |

Pillar by pillar:

| XSE pillar | Our state | Note |
|---|---|---|
| Python-Free Serving | Have it, three ways | (iii) is C++ with no Python in the process; (iv) has no Python at all; (i) is Python but framework-free |
| Low-latency execution | Have it | (iii) keeps the KV pools on the device and donates them, so no execution copies them (**Doc**: 100 of 100 runs in place, required by `triton/verify.sh`) |
| Orbax Checkpointing | Deliberately different | Ours are headerless raw little-endian files; the file *is* the operand. No change proposed |
| Flexible Scale | **Do not have** | "Serving artifacts are single-device, so their bodies contain none. Tlaloc serving on more than one device has not been built." — [SERVING_ARCHITECTURE.md](../SERVING_ARCHITECTURE.md) §2 (**Doc**) → item 5 |
| Performance Parity | Have the numerics, not the claim | Bit-for-bit against the reference interpreter for prefill/decode, chunked prefill, windowed rings. No cross-runtime parity or perf harness → item 4 |
| Docker Packages | Have one, hardened since the audit | `triton/run_server.sh` + the Triton container. SV-1's open listeners and missing authentication were closed by `1cf3bbd6` (2026-11-11, after the audit's `869950a6` snapshot): both endpoints bind `127.0.0.1` by default and take a bearer key, and the ports publish on loopback unless `TRITON_BIND` says otherwise (**Source**) → item 6 |
| AOT compilation (their future work) | **Shipped, and fragile** | See below → item 2 |
| Throughput under a latency target (their benchmark form) | **Now have it** (2026-11-17) | `:maestro:servedSloBench`; the first result is TwIL-8B's `cohort_wait` sweep in item 1. §6's older figures are still medians at fixed concurrency → item 1 |
| CPU/GPU/TPU matrix | GPU only for serving | The CPU plugin is built and passes multi-device tests, but nothing serves on it → item 3 |

The AOT comparison, precisely. XSE names AOT as work not yet delivered externally. We shipped
it: `PjrtExecutableCache` (`TLALOC_PJRT_CACHE_DIR`, `PjrtSession(diskCache = ...)`) on the JVM
side and `triton/backend/executable_cache.cc` (`executable-cache-dir`) on the C++ side, one key
and one file format, POSIX record locks on both (**Doc**: [torch-tpu-plan.md](torch-tpu-plan.md)
item 2). The measured effect is large: Qwen3.6-35B-A3B, 36 programs, **520 s compiled → 9.8 s
loaded**; model ready **564 s → 57 s**; decode unchanged (53.7/53.2 ms compiled, 54.1/52.8 ms
loaded); 475 MB on disk. Without it, Muse Glimmer's load is "16 compiles took 2 minutes … and the
weights 90 s" (**Doc**: [SERVING_ARCHITECTURE.md](../SERVING_ARCHITECTURE.md) §3). So the gap is
not the feature; it is that the feature has three known defects and no eviction, and that it is
an environment variable rather than a product step.

## 3. Items

Ordered by value per unit of effort on the one GB10 we have. Sizes are the repository's
(Small / Medium / Large). Nothing here is blocked on hardware except item 5's measurement.

### Item 1 — goodput at a latency target (Small–Medium)

**What.** A harness that sweeps offered concurrency against a latency SLO and reports the
maximum throughput that meets it, per model: `triton/slo_bench.py`, reusing the client and
accounting already in `triton/context_bench.py` and `triton/profile.sh`. For each SLO point,
report p50/p95/p99 TTFT and TPOT, aggregate tokens/s, executions per step, and the queue time
the backend already logs.

**Why.** XSE's headline figure is "Throughput under latency target" and its linked benchmark is
`dlrm_serving`. Our §6 tables are medians of three runs at fixed concurrency, which answers
"how fast is one request" and never "how many users can we hold at p99 under X ms". Three
load-bearing constants are presently guesses:

- `cohort_wait_microseconds`, default 20000 (**Source**: `triton/backend/sequence_mode.h:53`,
  [triton/README.md](../../triton/README.md):1041) — the most a batch waits for the sequences of
  the previous one. We already know the same knob cost a lone Muse Glimmer sequence about 24 ms a
  token when it lived in Triton's queue delay (**Doc**: §6), and that 1 ms missed one of two
  prompts sent together from Python gRPC threads (**Doc**: §6, `triton/verify.sh`). It has never
  been swept.
- `-PprefillChunk` / `HfServingExport.export(prefillChunk = ...)` (**Source**:
  `maestro/src/jvmMain/kotlin/io/tlaloc/maestro/serving/HfServingExport.kt:107`) — trades a
  prompt's TTFT against the decode steps it blocks. SV-5 records that one long prompt blocks
  everyone: "with no decode in the batch the prefill budget is 0 and `RunPrompts` loops on the
  single worker" (**Doc**).
- the bucket ladder (`-PcontextLadder`, and the batch ladder) — a request runs on the smallest
  entry that holds it and pads the rest, so the ladder is a latency / compile-count / memory
  trade with no measurement behind the defaults.
- `TLALOC_PREFILL_BUDGET` (**Source**: `triton/run_server.sh`; default the largest prefill call,
  "tokens a prompt runs per batch while other sequences decode", 0 meaning a prompt runs to its
  end) — the knob SV-5's head-of-line blocking turns on, and like the others never swept.

**Bar.** One table per model (Qwen3-0.6B bf16, TinyLlama-1.1B f32, Muse Glimmer bf16, and one
speculative model) giving max goodput at p99 TPOT targets of 50/100/200/400 ms and p99 TTFT
targets of 0.5/2/8 s, at concurrency 1, 2, 4, 8. Each cell dated, with the commit, the GPU-idle
check the §6 protocol already requires, and whether it was contended. The sweep must produce a
recommended `cohort_wait_microseconds` and `prefillChunk` per model with the measurement that
chose it, and [SERVING_ARCHITECTURE.md](../SERVING_ARCHITECTURE.md) §6 gains the table. If the
sweep shows the defaults are already right, that is the result and it gets recorded as such.

**Status (2026-11-17): ported to Kotlin, and the first real goodput numbers exist.** The harness
is `:maestro:servedSloBench`, not a Python script. The repository's principle is that every stage
runs in Kotlin and Python appears only where a check compares against a reference library
([brooklyn-eats-plan.md](brooklyn-eats-plan.md)); a latency benchmark is not such a check, and
`triton/`'s 18 existing Python scripts are a convention of that directory, not of the project. So:

- `maestro/src/commonMain/.../serving/SloBench.kt` — the arithmetic: nearest-rank percentiles (the
  rank is integer, `(p * n + 99) / 100`, because a floating ceil lands one sample away for some
  pairs), the SLO judgement, the goodput table, the round merge, and the recommendation across
  settings (highest goodput; a tie goes to the smaller cohort wait, since a lone sequence should
  not wait for company that never comes).
- `maestro/src/commonTest/.../serving/SloBenchTest.kt` — 13 tests, no server and no device, so they
  run in CI.
- `maestro/src/jvmTools/.../serving/ServedSloBench.kt` — the client, over `:endpoint`'s own
  `TritonClient`, so the benchmark speaks the protocol the serving frontend speaks. The tools
  compilation is unpublished, and `:endpoint` was added as a **tools-only** dependency, so no
  published POM or ABI changes.
- `compare=a.json,b.json` reads results back and recommends a setting without touching a server.

It sweeps a concurrency ladder closed loop — each sequence sends its next request as soon as its
last answer is back, and all N start on a barrier — in `mode=step` (a request per token with
`max_tokens=1`) or `mode=stream` (one request a turn, streamed per step). It reports p50/p95/p99
TPOT and TTFT with their sample counts, aggregate tokens/s, requests against executions, peak
memory, GPU utilization, and the provenance §6 asks for. Every result is parsed back by the
repository's own strict JSON reader before it is written, so a malformed result fails by name
instead of being saved.

**The first result** (**Run**, 2026-11-17, GB10, TwIL-8B bf16, Triton 2.63.0, 128-token prompts,
32 new tokens, ladder 1/2/4, 2 rounds, GPU idle at 2–3% before each ladder; commit `b26f77e1`,
dirty tree). `cohort_wait_microseconds` — the most a batch waits for the sequences of the previous
one — had never been measured. At 0 against the backend's 20000 default:

| N | cw=0 requests/executions | cw=0 tok/s | cw=0 p50 TPOT | cw=20000 requests/executions | cw=20000 tok/s | cw=20000 p50 TPOT |
|---|---|---|---|---|---|---|
| 1 | 32 / 32 | 10.3 | 84 ms | 32 / 32 | 10.8 | 84 ms |
| 2 | 64 / 41–48 | 15.0 | 83–95 ms | 64 / 33 | 19.3 | 83 ms |
| 4 | 128 / 63 | 19.8 | 164–168 ms | 128 / 33 | 31.7 | 85 ms |

Waiting 20 ms lets the cohorts coalesce: at four sequences the same 128 requests run in 33
executions instead of 63, throughput rises 19.8 → 31.7 tok/s, and per-token latency HALVES (p50
165 → 85 ms), because a batch of four amortizes a weight read that a batch of two pays twice.
Goodput at p99 TPOT ≤ 200 ms and p99 TTFT ≤ 2 s: **N=4, 31.74 tok/s at cw=20000 against 19.79 at
cw=0**; at p99 TPOT ≤ 100 ms only N=1 qualifies either way (10.75 against 10.31). The default is
well chosen and 0 is materially worse for concurrent serving. TTFT grows with N (370 ms at N=1,
1306 ms at N=4) because prompts are prefilled a batch at a time — the head-of-line effect SV-5
names, now measured rather than asserted. Results: `maestro/build/slo/twil-cw0.json` and
`twil-cw20000.json` (a relative `json=` path resolves against the module directory); both parse
with an independent reader.

**The second model** (**Run**, same day and conditions, Qwen3-0.6B bf16, exported for this sweep
with `-PcontextLadder=2048,8192 -PnumBlocks=1024`, four `cohort_wait` values, ladder 1/2/4, 2
rounds, GPU idle 2–5% before the ladders — the cw=20000 one at 5.0%, which is AT the harness's
threshold rather than below it). This is where the knob's shape appears:

| cw (µs) | N=4 tok/s | N=4 p99 TPOT | N=4 exec/req | N=2 tok/s | N=1 tok/s |
|---|---|---|---|---|---|
| 0 | 154.2 | 22.9 ms | 0.5 | 100.6 | 79.8 |
| 2000 | **205.0** | **20.0 ms** | 0.3 | 121.0 | 78.3 |
| 10000 | 203.9 | 26.8 ms | 0.3 | 121.2 | 77.5 |
| 20000 | 205.0 | 29.4 ms | 0.3 | 120.8 | 79.2 |

Any nonzero wait buys the batching: at N=4 the same 128 requests run in 33 executions instead of
63, and throughput rises 154 → 205 tok/s. But a LONGER wait buys nothing more and costs tail
latency — p99 TPOT climbs 20.0 → 26.8 → 29.4 ms as the wait goes 2 ms → 10 ms → 20 ms. So for this
model **2000 µs is the recommendation**, and the 20000 default over-waits tenfold at ~47% higher
p99. The harness's recommendation at p99 TPOT ≤ 25 ms and TTFT ≤ 1 s is `qwen3-cw2000`, N=4,
205.02 tok/s. Results: `build/slo/qwen3-cw{0,2000,10000,20000}.json`.

Read against TwIL, the pattern is that the useful wait is a FRACTION OF A DECODE STEP, not a
constant: TwIL's step is ~85 ms, so 20 ms of waiting is a quarter of it; Qwen3-0.6B's step is
~12 ms, so 20 ms is nearly two steps and the batch waits for company that has already arrived. A
per-model value derived from the step time is the export's job (item 1's remaining work), and
20000 as a universal default is now measured to be wrong for a fast model.

One refusal worth recording: Ripple's exported config has **no LOGITS output** (a speculative
artifact with sampling omits it), so the harness cannot derive a vocabulary from the server and
refuses by name until `maxId=` is given. That is the guard working, not a defect; the vocabulary
is in the artifact's manifest (`model.vocabSize`, 100352 for Ripple).

**The prefill side** (**Run**, 2026-11-17, Qwen3-0.6B at cw=2000, 1024-token prompts — two prefill
chunks — `staggerMs=150`, ladder 1/2/4, 2 rounds, GPU idle 2–3%). Measuring this knob needed a
harness change first: `TLALOC_PREFILL_BUDGET` applies **only while other sequences decode**
(`sequence_mode.cc:3485` passes `decode.empty() ? 0 : prefill_budget()`), and a ladder whose
sequences all start on a barrier never has a prompt arriving during a decode. `staggerMs` now
delays sequence i's prompt by i×staggerMs, so later prompts arrive while earlier ones decode. A
staggered phase's wall includes the stagger, so its rate is an arrival-workload rate, not a
saturation rate.

| budget (tokens) | N=2 p99 TPOT | N=4 p99 TPOT | N=4 tok/s | N=4 p99 TTFT |
|---|---|---|---|---|
| 0 (a prompt runs to its end) | 139.3 ms | 145.8 ms | 111.5 | 158.8 ms |
| 128 | 78.0 ms | 87.3 ms | 111.4 | 173.0 ms |
| 512 (the default: one prefill call) | 78.2 ms | **85.5 ms** | 111.4 | 180.2 ms |
| 1024 | 140.2 ms | 153.8 ms | 113.2 | 158.3 ms |

Capping the budget at one chunk **halves the decode tail** (145.8 → 85.5 ms p99 at N=4) for no
throughput (111.5 → 111.4 tok/s) and ~21 ms of TTFT. Letting a prompt run to its end (0), or giving
it more than a chunk (1024), lets one prompt monopolize a batch and spikes every other sequence's
per-token latency. So the default is right and **0 is the setting to avoid** under concurrency;
128 and 512 are equivalent within noise. This is SV-5's head-of-line blocking, measured and
bounded rather than asserted. Results: `build/slo/qwen3-prefill-b{0,128,512,1024}.json`.

Defects found and fixed while porting, each caught by a check that ran: `AtomicBoolean.set()`
carried over from Python's `Event.set()`; a test helper's parameter order; an assertion naming a
percentile's rank instead of its value; `step` mode without `max_tokens`, which prefills and
answers with an EMPTY NEXT_TOKENS rather than emitting a token; two JSON objects left unclosed by
brace-counting inside nested raw strings (the reader-before-writer check caught it, and the pieces
are now built separately); and a contention label that read GPU utilization DURING a phase — which
is the model working, near 100% — and so marked every clean run unusable. Comparability now rests
only on the idle window before the ladder. And a seventh, found by Ripple's first stream-mode run:
the first streamed response's interval CONTAINS the prefill, and counting it as a per-token time
put the whole prefill into the TPOT tail — Ripple's p99 TPOT came out exactly equal to its TTFT
(1014.4 ms), which is how it was spotted. The first response now establishes TTFT only, and TPOT
samples start at the second response. `mode=step` was never affected: there the prompt request's
time goes to TTFT alone. The affected Ripple result was discarded and re-measured.

**The third model: Ripple-65B NVFP4 + DFlash2** (**Run**, same day, `mode=stream`, 3 rounds,
`maxId=100352` from the manifest because a speculative artifact exports no LOGITS output). The wait
barely matters here — 12.3 → 12.3 tok/s at N=1, 12.5 → 12.6 at N=2, 10.1 → 10.5 at N=4 — and **N=2
beats N=4**: a generation request runs 5–7 executions (draft and verify), so four sequences do not
amortize the way four plain decode steps do, and TTFT reaches 9.8 s. Goodput at p99 TPOT ≤ 400 ms and
TTFT ≤ 4 s: N=2, 12.56 tok/s. Results: `build/slo/ripple-cw{0,20000}.json`.

**Natural text, and an assumption that turned out wrong.** The harness now takes `promptFile` and
`tokenizerDir` and gives each sequence a different `promptTokens`-wide window of real text (so no
prefix cache is hit), tokenized by `:tokenizer`'s `HfTokenizer`. It was assumed that seeded random
ids are a worst case for speculative acceptance. Re-measuring the same Ripple artifact with prose
(the repo's own documentation), same ladder, 3 rounds, cw=20000, says otherwise:

| prompts | N=1 tok/s | N=2 tok/s | N=4 tok/s | N=1 p99 TPOT | exec/request at N=1 |
|---|---|---|---|---|---|
| seeded random ids | 12.3 | 12.6 | 10.5 | 384.8 ms | 5–13 |
| real prose | 5.9 | 6.8 | 9.5 | 344.0 ms | 15 |

Prose roughly halved throughput at N=1 and N=2 and needed MORE draft/verify executions per request
— lower acceptance, not higher. A degenerate continuation of random ids is easier for the DFlash2
draft to predict than varied prose. Variance is large in both (random N=1 spanned 6.1–14.8 tok/s
across rounds), so the direction is clear and the magnitude is not tight. The earlier caveat in
this plan and in §6 was an assumption stated as fact; both are corrected. Neither workload is the
Brooklyn Eats agent's traffic, and the random-id figures are, if anything, optimistic. Result:
`build/slo/ripple-text.json`.

**Done since the first result.** `exportTritonModel` takes `-PcohortWaitMicros`, and
`TritonModelRepository.SequenceOptions` carries it, writing `cohort_wait_microseconds` into
`config.pbtxt` only when a value was chosen — null leaves the backend's default in force and the
parameter unwritten, so every existing golden config is unchanged. Pinned by
`theCohortWaitIsWrittenOnlyWhenAMeasurementChoseIt`, including that 0 is a value and a negative wait
is refused. The measured tables and the per-model recommendations are now in
[SERVING_ARCHITECTURE.md](../SERVING_ARCHITECTURE.md) §6, "Goodput at a latency target". The ABI cost
is recorded rather than hidden: adding a field to a data class changes its constructor and `copy`
signatures, so `maestro/api/maestro.api` shows four removed lines and their replacements —
source-compatible for Kotlin callers using named or default arguments, binary-incompatible for a
compiled caller of the old four-argument constructor.

**Still to do for the bar.** Muse Glimmer bf16 and TinyLlama-1.1B f32 (both need an export first);
the `TLALOC_PREFILL_BUDGET` and `-PprefillChunk` knobs, unswept, so the prefill side of the trade is
still a guess; a natural-text prompt workload, without which the speculative numbers are a lower
bound; and writing the recommended values into the exports that ship — the brooklyn-eats lanes'
`.env` files are pinned by manifest hash, so changing them is a deliberate re-pin, not an edit.
Whether to sweep prefill/decode disaggregation stays out of scope: SV-5 is the reason it will come
up, and it belongs in FOLLOWUPS, not here.

**The Python prototype is gone.** `triton/slo_bench.py`, `triton/slo_compare.py` and
`triton/slo_sweep.sh` were removed once the Kotlin port was verified (the first attempt was blocked
by the session's deletion policy; Pedro approved the retry). The shell driver among them had a
defect worth remembering: its readiness poll fed Python through a heredoc after a detached
`docker run`, and that process blocked forever on a stdin pipe that never closed. The Kotlin task has
no such failure mode, and the sweeps above were driven by explicit `run_server.sh` invocations with
`printf`-edited configs and `python3 -c` readiness polls — no heredocs.

### Item 2 — cold start as a product step (Medium, **mostly landed before this plan**)

**What was intended.** Fix the cache defects, add an export-time AOT step, and add a serve-time mode
that refuses to compile. The first of those turned out to be largely done already.

**Correction (2026-11-17): most of this item had already landed.** The audit's snapshot is
`869950a6`; commits `9d3f2267` (after it) and `baa24463` fixed the defects this item was written to
fix. Verified in the tree, and the pure tests run (**Run**):

| Audit finding | State | Evidence |
|---|---|---|
| RT-1 a failed execution leaks device buffers | Fixed | `9d3f2267`: single-device, reusable and multi-device paths destroy their outputs; every read-back destroys its event even when the wait throws |
| RT-2 `OverlappingFileLockException` | Fixed | `jvmLocks` is a companion-object map — one JVM-wide lock per lock file, whichever instance asks; pinned by `twoCachesInOneJvmLoadOneProgramAtOnce` (four threads, one directory) |
| RT-3 a transient failure deletes the entry | Fixed | `isEntryBad` is false for `IOException` and for `PjrtRuntimeException.isResourceExhausted`; pinned by `aLoadThatFailedForLackOfMemoryKeepsTheEntry` |
| RT-4 a CUDA client with no allocator options | Fixed | `createClient` refuses before `PJRT_Client_Create`; pinned by `aGpuPluginIsRefusedAClientWithoutAllocatorOptions` |
| Eviction (FOLLOWUPS: "no eviction") | Implemented | `maxBytes` / `TLALOC_PJRT_CACHE_MAX_BYTES`, LRU by mtime, skipping entries another caller holds; pinned by `aBoundedCacheEvictsTheLeastRecentlyUsedEntries` |
| Shared tier | Implemented | `shared` / `publish`, `copyIn` re-checksums before moving into place; pinned by `aSharedDirectoryServesAnotherMachinesCompile` |
| RT-9 an unkeyed hash in a writable directory | **Narrowed, not closed** | below |

The cache is demonstrably working in use: Ripple's 26 programs loaded in **15.3 s with 0 compiled**,
TwIL's 10 in **3.5 s**, and Qwen3-0.6B's second load was ready in **8 s** (**Run**, 2026-11-17, each
from its own server's backend log).

**What was done here (RT-9).** `PjrtExecutableCache` now refuses a **world-writable** cache or shared
directory by name, with the remedy (`chmod o-w`), before any client exists. An entry is serialized
native code and its checksum is an unkeyed SHA-256, so a directory any local user can write is not a
trust boundary; group-writable stays allowed, because a shared group is a deliberate choice. Three
tests pin it (refused by name; group-writable allowed; the shared directory refused too), and the
refusal was **seen failing** with the check disabled. The cache directories in use on this machine
are `drwxrwxr-x`, so nothing running is broken.

**What RT-9 still needs, and this does not do.** Authenticate an entry against a secret. A keyed MAC
would have to be shared with `triton/backend/executable_cache.cc`, which reads the same format and
computes the same key — a wire-format change on both sides, with the key pinned by
`keyIsPinnedSoTheTritonBackendCanComputeTheSameOne` and its C++ counterpart. Until then the directory
must be as trusted as the plugin file itself, and the C++ side enforces no permission check at all.
Both halves are named rather than presented as closed.

**Also done here: `require-cache`, on both sides.** `PjrtExecutableCache.cacheOnly`
(`TLALOC_PJRT_CACHE_ONLY`) refuses a miss by name, naming the key, the directory and the first
line of the missing program; it also refuses when the plugin cannot serialize, and `fromEnv`
refuses the variable set without a directory. The C++ side matches: `ClientOptions::
executable_cache_only`, the `executable-cache-only` backend-config key, the same env var, a
validated value, a startup log line, and `run_server.sh` forwarding it. Verified three ways
(**Run**, 2026-11-17): `PjrtExecutableCacheTest` 19 tests, 0 skipped, including a device-gated one
that compiles a program, loads it cache-only, then refuses a different one without compiling or
writing; `pjrt_device_test` on the GB10, all checks passed, with both refusal messages; and live
serving of Qwen3-0.6B — a warm cache ready in **8 s** with `10 loaded, 0 compiled, 0 refused`, an
empty cache failing the model load with `no cached executable for key b0f53aee… refuses to
compile`, after which Triton's default `--exit-on-error=true` exits the server. Cold start is now
a contract a deployment can enforce, not a hope.

**Still open in this item.**

1. An **export-time AOT step**: compile every entry of every bucket at export and write the cache
   directory into the artifact, so it ships ready to load. The key already covers the plugin
   (**Doc**: [PERFORMANCE.md](../PERFORMANCE.md):73); verify it covers device topology and the memory
   fraction too, or widen it, before a cross-target cache can be trusted not to collide.
2. The keyed MAC above.
3. An asymmetry found while documenting this: on a rejected entry the C++ side deletes it and
   compiles again, while the JVM keeps it when the failure was transient. The C++ side has no
   error-code classification, so RT-3's fix is JVM-only. Recorded in
   [triton/README.md](../../triton/README.md); not fixed here.

**Bar for what remains.** An AOT-exported artifact loads with **zero** compiles, measured as
model-ready seconds before and after, in a new row of
[SERVING_ARCHITECTURE.md](../SERVING_ARCHITECTURE.md) §6 with date, commit and idle-GPU evidence.
`require-cache` refuses a cold artifact by name on both sides, and the refusal is tested. The TPU
lane inherits the step: `scripts/tpu/aot-check.sh` and `TpuProgramRecorder` already record the
programs a session compiles, so the AOT export should be runnable against a compile-only topology —
record whether it is, do not claim it.

**Depends on.** Nothing.

### Item 3 — a CPU serving lane for non-LLM artifacts (Medium)

**What.** Serve a `tlaloc-bounded-v1` artifact on the XLA CPU PJRT plugin, through the paths that
can reach it without a port, and measure it.

**Why.** XSE's volume story is CPU: Arm Neoverse and Google Cloud Axion, Intel Xeon with AMX,
workloads named as "vision, embedding and recommendation" — which is what our bounded-program
artifacts are. We have the plugin and no lane. There is a second, larger reason: CI-6 records
that "CI never executes PJRT or CUDA" and 383 of 3,648 tests are gated on hardware or tools. A
CPU device lane is the only one that could run on a hosted runner at all.

**What exists (Source, checked 2026-11-17).** `scripts/pjrt-cpu-plugin/build.sh` builds
`pjrt_c_api_cpu_plugin.so` at the XLA commit of the vendored headers (**Doc**: 12,313 Bazel
actions). `PjrtBinaries.cpuPluginPath` / `requireCpuPlugin` resolve it, overridable with
`TLALOC_PJRT_CPU_PLUGIN_PATH`; `PjrtTarget.LlvmCpu` exists; `CpuClientOptions` carries
`cpu_device_count` (the plugin's own default is 4); `PjrtSession.runPerDevice` and `runSharded`
work on it (**Doc**: `PjrtMultiDeviceCpuTest`, four host devices, against the interpreter;
[DISTRIBUTED.md](../DISTRIBUTED.md)). On the Python side `find_plugin` returns an explicit path
for any platform and only special-cases `tpu` in the env check (`tlaloc_pjrt.py:292-316`), and
`tlaloc_bounded.py` already takes `platform` and `plugin_path` (`:229,246,250`).

**What does not.** `plugin_candidates` has no CPU location, so a CPU run needs an explicit path or
a new candidate entry. Whether the ctypes client-creation path can express `cpu_device_count`
without also passing GPU allocator options — which `PjrtFfm.kt:1012` refuses to mix — is
unverified (**Source**, not run). And the Triton backend is not a candidate at all: it calls CUDA
directly (`sequence_mode.cc:1192,2211`, `bounded_mode.cc:152,180`,
`tlaloc_backend.cc:1103-1129`), so a CPU lane there is a port of ~9,585 untested lines (CI-8),
not a configuration change. Note also that Triton's `KIND_CPU` instance groups still run on GPU 0
(**Doc**: [triton/README.md](../../triton/README.md):1147).

**Bar.** One bounded artifact (start with the existing `bounded_softmax` example) served on the
CPU plugin through (a) `tlaloc_bounded.py` with an explicit plugin path and (b) `ServingModel`
with `PjrtTarget.LlvmCpu`, giving identical outputs to the interpreter and to the same artifact on
CUDA, within the tolerance the bounded exporter already enforces. Numbers recorded: latency per
request at `cpu_device_count` 1/4/8 on this aarch64 host (the GB10's Neoverse cores), with the
thread count and the machine's state, and the same artifact's CUDA number beside it for ratio. A
`verify`-style script that runs it, so it is repeatable. Then the honest follow-up: what it would
take to run that lane in CI, given that building the plugin is 12,313 Bazel actions — a prebuilt
artifact per runner arch, or a nightly. Record the answer; do not promise CI.

**Out of scope.** LLM artifacts on CPU. A Muse Glimmer decode step reads 53.0 GB of weights
(**Doc**: §6); at CPU memory rates that is not a serving story, and XSE does not claim one either.

### Item 4 — a runtime-parity harness (Small–Medium)

**What.** One harness that runs one artifact through every runtime that accepts it — (i), (ii),
(iii), (iv) — and emits a single matrix: schema versions accepted or refused, by name; numerical
divergence against the reference interpreter and against each other; and per-token or per-request
cost with its conditions.

**Why.** XSE's "Performance Parity" pillar is the claim "performance will be identical to its
training environment". We can make a strictly stronger claim on numerics — we already hold
bit-for-bit results against the interpreter for chunked prefill, windowed rings and batched
prefill (**Doc**: §2, §3) — but it is scattered across tests and prose, and the public claim is
currently wrong: DOC-7 records that the README's "Three runtimes load it without a JVM" is false,
because the Python runtime and the vLLM plugin refuse v3 artifacts by schema version and
`ServingModel` refuses windowed and quantized KV (`tlaloc_serve.py:540`), so "As served, Muse,
Ripple and TwIL are Triton-only" (**Doc**: audit, Docs and claims).

**Bar.** A generated table, not a hand-written one: `scripts/runtime-parity.sh` (or a `:maestro`
task) produces it from one artifact set, and the README's serving table is replaced by that output
or by a link to it, so it cannot rot the way the test count did. Every refusal it reports is a
refusal by name that the harness observed, not one remembered from a document. One row per
(runtime × artifact schema × model), each with the divergence actually measured and the date,
commit and hardware. Closing DOC-7 is part of the bar.

**Depends on.** Nothing, but it is much cheaper after item 1 exists, since the cost column can
come from the same client.

### Item 5 — sharded serving: design, then a CPU-mesh prototype (Large)

**What.** Make a serving entry runnable across a device mesh. The pieces exist separately and
nothing connects them: the emitter writes Shardy (`sdy`) mesh and sharding annotations "when a
graph declares a device mesh" but "Serving artifacts are single-device, so their bodies contain
none" (**Doc**: §2); `PjrtSession.runSharded`, `shardDims`, `shardedProgram`,
`parameterShardings` / `outputShardings` and `OpShardingLayout` (tiled, replicated, partially
replicated, iota) are built and pass against the interpreter on four CPU host devices (**Doc**:
[tpu-gap-plan.md](tpu-gap-plan.md) item 1, [DISTRIBUTED.md](../DISTRIBUTED.md)); and
[MULTIHOST_DESIGN.md](../design/MULTIHOST_DESIGN.md) covers the distributed init and the Maestro
seam. `CompileOptions` carries partitions, device assignment, SPMD and Shardy (**Doc**:
[torch-tpu-plan.md](torch-tpu-plan.md) item 1).

**Why.** It is the one pillar where XSE is genuinely ahead — "Supports both single-host and
multi-host serving by leveraging PJRT" — and the audit places us "Far behind" on distributed
("4 virtual CPU devices; multi-host refused"). It is also the answer to a real constraint on this
machine: serving Muse Glimmer under a PJRT memory fraction of 0.55 peaked the machine at 87 to
90 GiB in use, about 9 GiB of it other processes (**Doc**:
[SERVING_ARCHITECTURE.md](../SERVING_ARCHITECTURE.md) §3), and Ripple-65B peaked at 87.1 GiB on
the same box (**Doc**: the branch's baseline commit `5896d24e`). Neither leaves much room for
concurrency.

**Bar, in two stages.** *Stage A (design, no hardware needed):* a `design/SHARDED_SERVING.md`
that answers, concretely — how a bucket ladder maps to a mesh; how the KV pool and the windowed
rings are partitioned and who owns a page's shard; how `KV_POOL_IN`/`KV_POOL_OUT` donation and
`tf.aliasing_output` behave when the pool is sharded; what a new manifest schema version must
carry (mesh, per-entry sharding, per-device weight slots); which of the runtimes could ever serve
it, and which refuse it by name; and what a prefill/decode split would mean for cohort batching.
Every answer cites the code it depends on. *Stage B (prototype):* one small sharded entry — the
five-layer closed-form model the interpreter tests already use — emitted with Shardy, run through
`runSharded` on the CPU plugin's four host devices, equal to the single-device result within
tolerance, with the collective count and the per-device memory recorded. That is a real execution
and needs no second GPU. **Not claimed either way:** any multi-GPU or multi-host serving
measurement. The audit's move #10 ("Two GPUs, then two hosts: at least 1.8× on 2 GPUs") stays
blocked on hardware, and this item does not pretend otherwise.

**Depends on.** Item 3 for the CPU-plugin plumbing, and item 2's cache-key work if sharded
executables are to be cached (the key must cover the topology or a 1-device and a 4-device
executable collide — check `PjrtExecutableCache`'s key before assuming).

### Item 6 — hardening before any packaging claim (Small–Medium, **mostly landed before this plan**)

**Correction (2026-11-17).** This item was written from the November audit, whose snapshot is
`869950a6`. Commit `1cf3bbd6` (2026-11-11, in HEAD) closed most of it, so the audit's wording no
longer describes the tree. What that commit records: both OpenAI endpoints bind `127.0.0.1` by
default and take a bearer key (`TLALOC_API_KEY`; a non-loopback host without one refuses to start
unless `--no-auth`); both refuse a POST that is not JSON (415) or is too large (413), and the JSON
reader refuses nesting deeper than 256; the Kotlin endpoint bounds the queue (429 at `maxWaiting`
or `queueTimeoutMillis`) and times each backend step, each write to the client and the whole
response, so a client that stops reading frees its sequence; backend errors return a generic
message and are logged without other sequences' ids; a chained step whose tokens run past an end
token drops the sequence; and `run_server.sh` publishes Triton's ports on `127.0.0.1` unless
`TRITON_BIND` says otherwise. That is SV-1, SV-2, SV-3 and SV-6 — the audit's "Fix first" #2 and
#3. Confirmed in the tree (**Source**): `endpoint/.../Main.kt:74-77` (loopback default, key
required off loopback), `OpenAiEndpoint.kt:36,56` (`host = "127.0.0.1"`, `maxWaiting = 256`),
`Sequences.kt:108` (the bounded queue), `triton/run_server.sh` (`BIND`).

**What is left needs a re-audit before it can be listed.** The residue of SV-1 (are sequence ids
still predictable, and is there an owner check behind the key?); SV-4 (the Python endpoint's
`chat_template_kwargs` reaching `apply_chat_template`); SV-5 (one long prompt blocking everyone —
`TLALOC_PREFILL_BUDGET` is now a knob, which is a mitigation and not a fix, and it is unmeasured:
item 1 sweeps it); SV-7 to SV-13; and the audit's production gaps, "latency histograms,
OpenTelemetry, a multi-node router, queue limits and priorities, vision, structured outputs,
per-request LoRA" — of which the queue limit and the API key are now done.

**Bar.** Re-run the audit's serving section against HEAD and record, per finding, closed / open /
partly, with the commit that closed it. Then close what is open in the same order, each with a test
that fails first. `/health` reporting a GPU fault rather than a constant (SV-12) and latency
histograms are worth doing even if the rest turns out to be closed: item 1 wants the histograms,
and they are the audit's named production gap.

**Depends on.** Nothing. It no longer gates a packaging claim the way it did — the container
already binds loopback by default — but the README should not invite anyone onto a network until
the re-audit says the residue is closed.

## 4. Order

Reordered 2026-11-17: item 6's security half had already landed in `1cf3bbd6`, and item 1's
harness is written, self-tested and has run against a live server, so item 1 leads.

1. **Item 1** — started; what remains is the sweeps and the restart driver for the two knobs.
2. **Item 6's re-audit** — Small, and it says whether anything security-shaped is still open.
3. **Item 2** — already scoped by the audit as "Fix first" #4; the AOT export and `require-cache`
   are the new halves.
4. **Item 4** — Small once item 1's client exists, and it closes DOC-7.
5. **Item 3** — new axis, measurable on this aarch64 host, and the only route to a CI device lane.
6. **Item 5** — Stage A design can start any time; Stage B waits on item 3.

Items 1, 3, 4 and 6 are independent and can run in parallel. Items 2 and 5 both touch the cache
key and should not.

## 5. Deliberately not copying

- **Orbax for weights.** Our headerless raw files mean the file is the operand, the export needs
  no GPU and loads no plugin, and the format is already proven across the runtimes. Adopting a
  checkpoint library to match a blog post would add a dependency and buy nothing.
- **The one-shot positioning.** XSE declines long-context autoregressive serving and points at
  vLLM and SGLang. That is our strongest ground; narrowing to match would be a regression.
- **C++ data preprocessing.** Their future work folds preprocessing into the graph. Ours lives in
  the JVM endpoint and the tokenizer module, and the artifact deliberately carries only a
  `modelName` so a frontend finds the tokenizer. Moving tokenization into the C++ backend would
  duplicate the tokenizer under a second language for no measured gain. If the Triton-only
  deployment story ever needs a self-contained artifact, the cheaper step is recording the
  tokenizer files in the manifest — put that in FOLLOWUPS if it comes up.
- **Torch support via TorchTPU.** We are the exporter, not a consumer of one;
  [torch-tpu-capability-gap.md](torch-tpu-capability-gap.md) already covers that ground.

## 7. Second pass, 2026-11-17: what shipped after the first commit

Committed as `f20eee97` (prefill budget, `require-cache`, natural text) and the slice that
follows it. Everything below is **Run** unless marked.

**Item 1's four-model bar is met.** TinyLlama-1.1B f32 and Muse Glimmer 30B bf16 were exported
at batch 4, decoupled (TinyLlama context 512; Muse context 2048, windowed KV, 52 GB, export
161 s) and swept. The tables are in [SERVING_ARCHITECTURE.md](../SERVING_ARCHITECTURE.md) §6.
The rule holds across all four: the useful `cohort_wait` is a fraction of a decode step.
TinyLlama (~21 ms) takes 2000 or 20000 equally (141 tok/s at N=4 against 87 at cw=0); Muse
(~255 ms) takes 20000 (12.2 against 7.1 tok/s at N=4, and N=2's p99 TPOT halves 515 → 274 ms).
Muse peaked at 72 GiB in use at cw=20000 and 85 GiB at cw=0 with a 0.3 memory fraction, and the
machine never went below 37 GiB available.

**Item 3, a CPU serving lane.** `ServingModel.load(dir, target)` now takes a target
(`@JvmOverloads`, default `Cuda`). The first attempt failed and the failure was the lesson:
`PjrtSession`'s default plugin is the CUDA one *whatever the target*, so a CPU session that does
not name its plugin loads the GPU plugin and is refused for want of allocator options. The lane
now passes `PjrtBinaries.requireCpuPlugin(..)` explicitly, as `runOnPjrt`'s own refusal message
already prescribed. `aPortableArtifactServesOnTheCpuPlugin` serves the export-default portable
artifact (`cudaKernels = false`, pure StableHLO) on the XLA CPU plugin and matches the
interpreter's logits — **0.79 s, no GPU**. This is the lane a hosted CI runner could execute
(CI-6: CI never runs PJRT or CUDA). It is a correctness lane, not a throughput one.

**Item 4, a runtime-parity harness.** `:runtime-pjrt:runtimeParity` (a new unpublished `tools`
compilation, mirroring `:maestro`'s) prints, per artifact, what each runtime actually says.
Observed on three schemas: a v2 portable artifact is ACCEPTED by ServingModel/CUDA,
ServingModel/CPU and the framework-free Python runtime; a v3 windowed artifact is refused by
all three, each in its own words; a v5 artifact is refused by the JVM lanes for its windowed
pool and by Python because `tlaloc-serving-v5` is not in `('tlaloc-serving-v1',
'tlaloc-serving-v2')`. That is DOC-7's claim replaced by an observation. The Triton backend and
the vLLM plugin are reported as not exercised (they need a server and vllm), never as refusals
they did not make.

**Item 5, Stage A.** [design/SHARDED_SERVING.md](../design/SHARDED_SERVING.md) answers the six
questions the bar named, and names the two that can end the approach: whether SPMD partitioning
keeps a donated KV pool in place, and whether the paged-attention custom call survives a weight
sharding. It proposes replicating the pool and sharding the weights first, because that is the
only option that leaves the page allocator and the ring arithmetic untouched. Stage B needs no
second GPU — the CPU plugin gives four host devices — and is not done.

**Item 6, the re-audit.** [audits/SERVING_REAUDIT_2026_11_17.md](../audits/SERVING_REAUDIT_2026_11_17.md)
re-checks all thirteen SV findings at HEAD. Closed: SV-2, SV-4, and SV-3 for KV-only models.
Partly: SV-1 (auth, loopback, content-type closed; predictable ids and no owner check open) and
SV-6. Confirmed open at the cited lines: SV-9, SV-11, SV-12's constant `/health`, SV-13's
unchecked `Long` fold. Four are carried without re-verification and marked as such. The two
that matter for multi-tenancy share one root: a live sequence is matched by token prefix across
callers, with predictable ids and no owner. No finding was closed by the re-audit itself.

## 8. What this plan does not claim

Items 1, 3, 4 and 6 are done to their bars, item 2 partly, item 5 at Stage A. §7 records what
shipped and the evidence for each. `:maestro:servedSloBench` has measured **five models** — the
bar's four (Qwen3-0.6B bf16, TinyLlama-1.1B f32, Muse Glimmer 30B bf16, and Ripple-65B NVFP4 +
DFlash2 streamed) plus TwIL-8B bf16 — and the prefill budget and a natural-text workload, with the
tables in [SERVING_ARCHITECTURE.md](../SERVING_ARCHITECTURE.md) §6. `require-cache` exists on both
sides of the artifact and is verified by tests, a device test and a live cold-cache refusal.

What is NOT established: `prefillChunk` is unswept (the budget is); each model was measured at one
prompt length and one batch ladder; no recommended value has been written into a shipping export,
because the brooklyn-eats lanes are hash-pinned and changing them is a deliberate re-pin; the
export-time AOT step and the keyed MAC do not exist; the C++ backend enforces no cache-directory
permission check and deletes a rejected entry where the JVM keeps it; sharded serving is a design
with two open questions that can end it, and no sharded entry has run; the re-audit closed nothing
itself, and four of its thirteen findings are carried without re-verification. The natural-text
measurement contradicted this plan's own earlier assumption — random ids are NOT a worst case for
speculative acceptance — and both places that claimed it are corrected. No figure quoted from
another document was re-measured; every number is labelled with where it came from. Nothing here
ran on a TPU, on a second GPU, or on a second host.
