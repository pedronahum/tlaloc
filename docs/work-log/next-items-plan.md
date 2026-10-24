# Next items: plan and results

From `main` at `01be0122` (2026-10-24), worked in order on `main`. Each item ends with its
commits and its numbers here; an item that stops short says what was validated and what is
left (and FOLLOWUPS says it too).

| # | Item | Done when | Status |
|---|---|---|---|
| 1 | Long requests on Qwen3.8-27B | the four-user 100K streamed run (3 turns) completes with the new timeout; FOLLOWUPS updated | done: all 12 turns complete; first tokens 794 s, follow-ups 1.3 s at 11.7–12.1 tokens/s per user |
| 2 | Checks before a push | `scripts/prepush.sh` runs the checks CI runs that the per-module suites skip (ABI dumps, the compiler plugin, JVM targets); CONTRIBUTING names it | done: 10 gate tasks in one `--continue` run, 16 s from scratch; passes on `main`; a probe ABI change plus a plugin compile error both named |
| 3 | Long prefill on Qwen3.6-35B-A3B, measured again | one and four 100K documents to the first token, with the sort-free router, against the 53 s / 214 s and vLLM's 29–36 s / 92–113 s | done: one 100K document 49.6 s to the first token (53.1), four 149.1 / 199.4 s median / last (160.3 / 213.8); a 2,048-token chunk after 98K 1.35 s, attention 644 ms of it |
| 4 | Per-token state committed once per step | the conv state, the n-gram memory and the head's carried hidden state committed as the delta rule's state is: two state slots per sequence; interpreter, XLA, kernels and backend equal; the 35B measured | done: two slots per sequence, 140.8 MiB instead of 333.1 on the 35B; speed the same within run spread (4 streams 173.6 tokens/s aggregate, 1 stream 154.7 per reply) |
| 5 | Qwen3.8-Flash-Next served (flash-next-plan step 6) | exported with the n-gram table as NVFP4 in memory and served by Triton with chained steps; contexts past 2,048 through the CUDA kernels (indexer, masked attention) | done for contexts up to 8,192 (memory): table 26.8 GiB as NVFP4, coherent replies, a code word found 9 of 9 times in 1.5K–7K documents; 4 streams 24.1 (prose) / 36.4 (code) tokens/s per stream |
| 6 | Qwen3.8-Flash-Next measured (step 7) | the community harness at one and four streams against the published table | done, target not met: 4 streams prose 24.7 / code 36.9 tokens/s per stream (target 37 / 50; vLLM 27.7–28.4, TensorFold 37.2 / 49.6); 17.7 / 26.7 with a prompt of its own per stream; TTFT 0.78 s |
| 7 | Draft width per request | verify entries for several draft counts in one artifact and the backend choosing per batch and per request's acceptance; the 35B measured at 1 and 4 streams | done: verify entries of 5 and 3 drafts in one artifact, the count chosen per batch; 35B, 40 prompts: 1 stream 127.7 tokens/s over the run (5 drafts 121.7, 3 drafts 123.8), 4 streams 177.6 (156.3 / 172.9), 2 streams 159.5 (151.9 / 148.4); median per reply at one stream 148.0 against 160.9 with 5 drafts |
| 8 | Push and CI | `main` pushed, every workflow green | |

## 1. Long requests on Qwen3.8-27B

- `triton/multi_user_bench.py` prints each failed user's turn and error (stdout and
  stderr), counts a stream that ends without a response as a failure, and ends with a
  `FAILED: k of n users ...` line and exit status 1. Checked against a port with no server:
  both users reported, exit 1.
- The 27B export of the long-context arc (NVFP4, 3 drafts, FP8 KV, 33,600 pages,
  `-PcudaKernels=true`), its repository rewritten by `exportTritonModel` with
  `-PmaxSequenceIdleMicros=600000000 -Pdecoupled=true` (hard links): the timeout is
  4,200,000,000 µs. Served with `TLALOC_PJRT_MEMORY_FRACTION=0.74` (ready after 450 s), 4
  users × 100K, 3 streamed 256-token turns, wall 864 s, exit 0:

| Turn | Prompt | First token, median / worst | Tokens/s per user, median / worst | Aggregate | Tokens a step |
|---|---|---|---|---|---|
| 0 | 100,000 | 794.5 / 794.5 s | 9.6 / 0.4 | 32 | 2.79 |
| 1 | 13 | 1.30 / 1.56 s | 11.7 / 10.3 | 47 | 3.22 |
| 2 | 13 | 1.30 / 1.30 s | 12.1 / 11.3 | 50 | 3.09 |

  The failed run of 2026-10-17 had first tokens at 576 s (median) and 768 s (last); the
  prompts now run in 2,048-token parts in turn, so all four arrive together, the last 3%
  later. A 2,048-token part at 128K takes a median 4.08 s.
- Commits: `d5f54e21` (bench), and this one (docs).

## 2. Checks before a push

- `scripts/prepush.sh` runs `./gradlew --continue` over `checkKotlinAbi`, `classes`,
  `jvmMainClasses`, `testClasses`, `jvmTestClasses`, `verifyJvmTarget`,
  `verifyPomMetadata`, `verifyBomCoverage`, `:gradle-plugin:validatePlugins` and
  `checkKdocInternalRefs` (all part of CI's `./gradlew test` through `check`), and after a
  failure prints one line per failed task with its gate and fix (`updateKotlinAbi` for an
  ABI change). `--plugin-tests` adds `:compiler-plugin:test`.
- On `main` (`43e7b6ef`): passes. 229 tasks; 16 s with `--rerun-tasks --no-build-cache`,
  2 s up to date. `:compiler-plugin:test` alone: 603 tests, 105 s.
- Probe: a new public function in `:core` and a type error in `:compiler-plugin` main,
  together: exit 1, both reported (`abi :core:checkKotlinAbi` with the `updateKotlinAbi`
  command, `compile :compiler-plugin:compileKotlin`). Reverted.
- Not covered: the test suites, the JDK 21 lane (`-PtlalocTestJdk=21`, `jdk21-smoke.sh`),
  the vendored Maestro lane and the next-Kotlin probe.
- CONTRIBUTING names it under Building.
- Commits: `203ff512` (script, CONTRIBUTING), and this one (docs).

## 3. Long prefill on Qwen3.6-35B-A3B, measured again

- Export at `d9f2db23`: the flags of `triton/build/q36-commit` (NVFP4, 3 drafts, FP8 KV,
  committed delta rule, `-PcudaKernels=true`, prefill entries 64/128/2,048 tokens up to
  batch 4, packed 128/256, 20 state slots) with the 128K bucket of the long-context arc
  (`-PcontextLadder=2048,8192,32768,131072 -PnumBlocks=33600`): 64 entries, 20,584 MiB of
  weights, 886 s to compile at load. Routing is `tlaloc_moe_route`, attention's prefill rows
  the f16 form. Served with `TLALOC_PJRT_MEMORY_FRACTION=0.74`; `multi_user_bench.py
  --stream`, the same wikitext documents, 3 turns of 256 tokens, after a 2K warm-up request.

| | 2026-10-17 (`f191eae2`) | now | vLLM 0.29 |
|---|---|---|---|
| 1 user, 100K: first token | 53.1 s | 49.6 s | 29–36 s |
| 1 user: tokens/s, first turn / follow-ups | 86.0 / 88.2, 69.3 | 91.5 / 101.5, 86.8 | 82–97 |
| 1 user: follow-up first token | | 0.18, 0.16 s | |
| 4 users, 100K: first token, median / last | 160.3 / 213.8 s | 149.1 / 199.4 s | 92–113 s (last) |
| 4 users: tokens/s per user, follow-ups (median) | 34.9, 37.1 | 38.9, 36.8 | 28–38 |
| 4 users: follow-up first token | 0.3 s | 0.23 s | 2.3–6.1 s |

- Four documents take four times one: two prompts arrived in the first batch and prefilled
  together (`prefill_b*_t2048` with 2 sequences, 1.5–1.9 s a call) to their end; the other
  two then ran in 2,048-token parts between the first two's decode steps (prefill budget
  2,048).
- One 2,048-token chunk after 98K (`profile.sh MODE=nsys CHUNK=2048 prefill:98000:2`):
  1,356 ms at the client, 1,287 ms of kernels (8,452 launches), device span 1,345 ms.

| Part | ms | Notes |
|---|---|---|
| Attention, `PrefillKernelF16` | 611.3 | 11 layers (10 and the MTP layer's), 55.6 ms a layer |
| Attention, `CombineKernel` | 32.2 | 45 launches |
| Routed + shared experts, `ExpertGemmWideKernel` | 235.1 | 82 launches (gate-up and down for 41 layers, the MTP layer's included), 5.7 ms a layer; 4.9 more converting rows to bf16 |
| Gated DeltaNet, XLA's chunked form | about 204 | triangular solves 46.5, the 32-step chunk loop 48.6, the 64 × 64 chunk products 24.6, transposes and the rest |
| Gated DeltaNet in-projection (FP8, 2048 → 12288) | 79.2 | XLA's Triton GEMM with the e4m3 widening fused, 2.64 ms a layer, 39 TFLOP/s |
| Other projections | about 37 | out-projection 22.6, cuBLAS 11.0, the head's NVFP4 GEMM 2.3 |
| Routing, `tlaloc_moe_route` | 1.0 | was XLA's radix sort, 47.3 |

  Without attention the chunk is 642 ms of kernels; the 2026-10-17 profile (exact prefill
  form, 2,120 ms of kernels) had 674 ms besides attention and its combine. The sort's 46 ms
  went; the in-projection moved from cuBLAS on a separately widened weight (then `nvjet`
  50.5 ms for 41 launches and the widening 9.8) to XLA's fused Triton GEMM (79.2 ms).
- `--xla_gpu_enable_triton_gemm=false` (same chunk, `XLA_EXTRA_FLAGS`): the in-projection
  goes back to cuBLAS (49.7 ms plus 10.1 widening) but the Gated DeltaNet's small products
  go to MAGMA (49.8 ms, 1,980 launches): 1,273 ms of kernels, 1,329 ms at the client (2 %).
  Not adopted: within run spread and it changes every decode program too.
- Left: attention is 48 % of a chunk at 98K, then the experts (18 %), the Gated DeltaNet's
  chunked form (15 %) and its FP8 in-projection (6 %). vLLM's prefill is still 1.4–1.7 times
  faster for one document and about 1.8–2.2 times for four.
- Commits: this one (docs). Scripts, logs and profiles in the session scratchpad
  (`next/export128k.sh`, `serve128k.sh`, `bench-a.log`, `prof-pf/`, `prof-pf-notriton/`).

## 4. Per-token state committed once per step

- `CAUSAL_CONV1D` takes `commit = "records"` / `"final"` with the delta rule's commit rows
  (`[B, 3]`: base, target, count) and a records pool `[S, R, C]` of its inputs. The starting
  history is the last `K - 1` rows of the base's history followed by its first `count`
  records (zeros at position 0); a verify step writes it at the target with the step's live
  inputs as records, a prefill call writes its last history at the base. No CUDA kernel: the
  conv is XLA's (a gather, a concatenation and a window read).
- The head's carried hidden state is committed in the graph: `mtpHiddenRecords`
  (`[S, 1 + k, 1, H]`) holds the target's hidden state after each verified token, and the
  carried one is record `count - 1` (the base's own at count 0).
- A speculative export commits every state: the Gated DeltaNet conv (`convRecords<l>`), the
  n-gram memory's history and conv (`pleHistoryRecords<l>`, `pleConvRecords<l>`), the head's
  hidden state, and the delta rule as before. The entries no longer take `STATE_WRITE_SLOTS`
  (nor chain it); `NEXT_STATE_SLOTS` is the step's target. The backend gives such a sequence
  two state slots and binds four state pools per linear layer and two for the head;
  `ServingManifest` and the backend read the earlier forms (per token, delta rule only) as
  before. `TritonModelRepository` counts two slots per sequence for `max_candidate_sequences`.
- Validation:
  - `CommittedConvTest`: for every accepted count the history equals the per-token form bit
    for bit; chained steps and a prefill after them equal it too.
  - `PjrtGatedDeltaOpsTest`: the committed conv (records and final, padding and new rows) on
    the GPU equals the interpreter to 1e-6.
  - `HfQwen35MtpTest`, `HfQwen4ExpMtpTest` (two slots): transformers' ids and the oracle's
    drafts, chained steps included; a prefill after two verify steps gives the next token and
    drafts of one prefill of the whole history. `PjrtQwen35MtpTest`, `PjrtQwen4ExpMtpTest`,
    `PjrtSampledSpeculativeTest` equal them on the GPU.
  - Suites: `:ir` 1,203 tests, `:stablehlo` 331, `:maestro` 110, `:runtime-pjrt` 255, none
    failing.
  - `verify.sh` step 9 (which now exports the fixture again when its manifest lacks
    `mtpHiddenRecords`) passes on the new export, with poisoned pools unchained; and on the
    per-token export `qwen35-mtp-packed`. The 35B export of the delta rule's commit alone
    (`triton/build/q36-commit`) serves on the new backend.
- Qwen3.6-35B-A3B (the flags of `triton/build/q36-commit`, exported as `triton/build/q36-all`):
  a state slot is 70.4 MiB (66.6 before: the conv records add 3.8 MiB, the head's 32 KiB); a
  sequence holds 140.8 MiB instead of 333.1 (5 slots). The same backend build serves both:

| | delta rule committed (`q36-commit`) | every state committed (`q36-all`) | flare-plan |
|---|---|---|---|
| server compute per step, 4 streams (`profile.sh decode:256:48:4`) | 46.23, 46.08 ms | 47.01, 46.57 ms | 45.77 |
| server compute per step, 1 stream | 27.12, 27.07 ms | 26.83, 27.40 ms | 26.92 |
| kernels per step, 4 streams (nsys): all / routed experts / the rest | 39.60 / 21.29 / 18.31 ms | 40.86 / 22.92 / 17.94 ms | |
| categories, 1 stream, per reply | 152.0 tokens/s | 154.7 | 150.5 |
| categories, 4 streams, aggregate / per reply / TTFT | 171.1 / 85.9 / 0.20 s | 173.6 / 86.9 / 0.20 s (4 candidate sequences) | 169.4 / 83.7 / 0.20 |

  The two profile runs of each (3 and 5 repeats) overlap. In the nsys runs the kernels other
  than the routed experts are 0.37 ms fewer with every state committed, and the experts'
  kernel differs by 1.6 ms on the same seeded prompts. The per-token writes were not a large
  cost here: the per-token form's 30-launch scatter fusion took 0.35 ms per step, the
  committed form's two 30-launch scatter fusions 0.18 and 0.21.
- With 20 slots the new export allows 10 candidate sequences, and Triton then hands the 4
  prompts that arrive together to the backend in separate batches: TTFT 0.27 to 0.30 s and
  164 to 169 tokens/s aggregate at 4 streams (three runs), against 0.20 to 0.21 s and 169 to
  174 with the repository's `max_candidate_sequences` set to 4 (two runs); in the timeline
  each prompt is prefilled alone, 13 ms median after it arrives. An export for 4 sequences
  takes `-PstateSlots=8`; FOLLOWUPS section 13 has it.
- Commits: `50fca48c` (IR and interpreters), `12387860` (StableHLO form), `47b1774f` (export
  graph, manifest, repository), `f0312c6e` (backend, verify.sh, README), and this one (docs).
  Scripts and logs in the session scratchpad (`next/export36.sh`, `prof.sh`, `nsys.sh`,
  `serve_bench2.sh`, `slotmem.py`, `prof-q36-*`, `nsys-q36-*`, `bench-*`).

## 5. Qwen3.8-Flash-Next served

- **NVFP4 n-gram table.** `-PpleTableQuant=nvfp4` stores the table as codes, an e4m3 scale per
  16 values and a tensor scale, read by the new op `NVFP4_EMBEDDING`. Qwen3.8-Flash-Next's
  table of 320,001,536 × 160 is 26.8 GiB this way, against 47.7 GiB of FP8 in the checkpoint.
  - The FP8 shards are rounded shard by shard; 4,608 sampled rows equal an independent numpy
    rounding of the checkpoint. The NVFP4 values are 8.4% of their magnitude away from the
    FP8 ones on average.
  - On the tiny fixture the interpreter equals an f32 table holding the same values bit for
    bit, and the GB10 equals the interpreter within 3.8e-5, with the same ids.
- **Sparse attention in the CUDA kernels.** `tlaloc_indexer_topk` scores the chunks and keeps
  the top 512 by a radix select, ties in chunk order. Every form of `tlaloc_paged_attention`
  takes the keep mask. Both match the reference and the interpreter
  (`PjrtFusedSparseAttentionTest`): keep masks exactly, attention within 1.4e-4 exact and
  3.4e-2 in f16 for outputs up to 30.
- **Backend.**
  - It counts an n-gram memory's state pools.
  - Each weight of a sequence-mode model now sits in device memory of its own, outside the
    PJRT pool. With the weights in the pool at a fraction of 0.83, XLA's allocator, which
    grows by regions, took the GB10 to 3 GiB available before the last weights were in.
  - verify.sh step 9 passes; the served tiny fixture gives the interpreter's ids alone and
    together.
- **Export and serving.**
  - Export: NVFP4 weights and heads, NVFP4 table, 3 drafts, FP8 KV, `-PcudaKernels=true`,
    8 state slots; context 8,192, 2,200 pages, prefill calls of 128 and 512 tokens for up to
    2 sequences. That is 7 entries and 100,029 MiB of weights, written in 8.4 min.
  - Served with `TLALOC_PJRT_MEMORY_FRACTION=0.045` from an executable cache. The first
    compile needs about 0.6, because the autotuner allocates the 23.9 GiB table.
  - Ready in 150–175 s. 11 GiB available when idle, 6 GiB at the lowest under four streams.
  - Larger contexts or 2,048-token and 4-sequence prefill calls do not fit with headroom. A
    prefill call of 4 × 2,048 tokens needs 9.7 GiB of temporaries.
- **Checks.**
  - Greedy chat replies are coherent, the same with and without chained steps.
  - A code word at 10, 50 or 90% of documents of 1,555, 3,055 and 7,055 tokens is answered
    correctly 9 of 9 times.
  - Sent together, 3 of 4 replies part from their alone replies at a near tie after 215–458
    characters, the same in two runs.
  - An independent full-model reference is not feasible here: transformers on the CPU needs
    the dequantized checkpoint (about 350 GB in bf16), and vLLM keeps the FP8 table on the
    GPU.
- **First readings** (`streams.py fixed`, 512 tokens):

| | per stream | aggregate | TTFT | tokens per step |
|---|---|---|---|---|
| prose, 1 stream | 26.1 | 26.2 | 0.31 s | 2.13 |
| prose, 4 streams | 24.1 | 96.4 | 0.80 s | 2.13 |
| code, 1 stream | 41.9 | 42.0 | 0.31 s | 3.48 |
| code, 4 streams | 36.4 | 145.9 | 0.79 s | 3.26 |

  The published four-stream numbers are 27.7–28.4 (vLLM) and 37.2 (TensorFold) on prose,
  and 49.6 (TensorFold) on code.
- **Step time.** A verify step's server compute is 88 ms at one stream and 127 ms at four.
  - One stream (80.6 ms of kernels): the Gated DeltaNet in-projection takes 26.0 ms (FP8,
    XLA's Triton GEMM for 4 rows, about 58 GB/s) and the experts 19.2 ms.
  - Four streams: the experts take 53.0 ms and the in-projection 22.5 ms.
  - Details in flash-next-plan.md, "Step 6".
- **Commits:** `2c751f00` (NVFP4 table), `45e8ee2a` (indexer and masked attention kernels),
  `d00a1410` (backend, manifest, serving), and this one (docs). Kept for item 6:
  - the export `scratchpad/next/fn/fn8c` (artifact and repository) and the executable cache
    `scratchpad/next/fn/exe-cache`;
  - the scripts `exportfn.sh`, `servefn.sh` (with `watchdog.sh`), `endpoint.sh`, `stopfn.sh`,
    `coherent.py`, `needle.py`, `together.py` and `profile.sh` (a copy whose memory check
    allows the weights plus 10 GiB).

## 6. Qwen3.8-Flash-Next measured

- The item-5 export and server (context 8,192, 3 drafts, chained steps, fraction 0.045)
  behind `openai_server.py`; `streams.py` greedy, thinking off. Full table and analysis:
  [flash-next-plan.md](flash-next-plan.md), "Step 7"; raw results in
  [flash-next-results.jsonl](flash-next-results.jsonl).

| | 1 stream | 4 streams, per stream / aggregate | published, 4 streams |
|---|---|---|---|
| prose 512 | 26.4 | 24.7 / 99.1 | vLLM 27.7–28.4, TensorFold 37.2 |
| code 512 | 42.3 | 36.9 / 147.7 | TensorFold 49.6 |
| prose / code 512, a prompt per stream (`--distinct`) | | 17.7 / 26.7 | |
| 40 prompts: decode median / TTFT | 39.4 / 0.31 s | 23.9 / 0.78 s | vLLM 26.4 (tonyd2wild) |
| tokens per step: prose / code / 40 prompts | 2.13 / 3.48 / 2.90 | 2.13 / 3.26 / 2.90 | TensorFold 2.1 / 5 |

- Target (37 prose, 50 code per stream at four streams) not met.
- `fixed` sends one prompt to every stream; greedy, the streams then emit the same tokens in
  step and share every routed expert. A 4-stream verify step is 85 ms that way, 115 ms
  (prose) and 123 ms (code) with a prompt per stream, 80 ms at one stream. `streams.py fixed
  --distinct` gives each stream a prompt of its own.
- TTFT at four streams is two 2-sequence prefill calls of 375–378 ms in series (the export's
  largest prefill batch is 2); a 4 × 128 entry was not tried.
- Commits: `a07fea69` (`--distinct`), and this one (docs).

## 7. Draft width per request

- **Export:** `-PmtpDraftTokens=5 -PmtpVerifyDrafts=3` writes verify entries of 5 drafts and of
  3 (`decode_b*_c*_k3`, manifest field `verifyDrafts`) for every decode bucket. They share the
  pools: the records pools hold `1 + mtpDraftTokens` records per slot, and a step of 3 drafts
  writes the first 4 (`GATED_DELTA_RULE` and `CAUSAL_CONV1D` with `commit = "records"` now take
  R >= T and leave records T .. R-1 as they are; `tlaloc_gated_delta` likewise). A prefill call
  drafts 5. `DecodeGraphSpec.verifyDrafts`; the cache key adds `k3`.
- **Backend:** each batch's verify step checks one of the artifact's counts. After a step of 3
  a step of 5 takes the 3 drafts and the last again in the missing places. A chained step
  keeps its predecessor's count; a change of count is taken unchained. `TLALOC_DRAFT_WIDTH`:
  `auto` (the default), a count, or `cycle` (each count for two steps in turn; a check).
  `auto` scores each count by the rows' expected tokens per step, 1 + q1 + q1 q2 + ..., with
  qi a sequence's weighted acceptance of draft i given the drafts before it (its own counts,
  2 pseudo-counts of every sequence's recent rate, or of its rate of draft i - 1 for a depth
  no step has checked yet), over a step time of 1 + 0.1 per draft beyond 3
  (`TLALOC_DRAFT_COST`); 5 only while the step has at most 12 verified rows
  (`TLALOC_DRAFT_WIDTH_ROWS`), so 1 or 2 sequences.
- **Server step time** (`profile.sh decode:256:48`, the same artifact, `TLALOC_DRAFT_WIDTH`
  fixed): 1 stream 27.0 ms with 3 drafts, 32.4 with 5 (1.20 times); 2 streams 34.7 / 40.7
  (1.17); 4 streams 44.6 / 73.5 (1.65; 24 rows pass `tlaloc_fp4_gemm`'s 16). The 3-draft
  export `q36-all` on the same backend: 27.4 / 34.8 / 45.1 ms.
- **Validation:** interpreter (`HfQwen35MtpTest`): in an artifact of 5 drafts a step of 3
  gives bit for bit the tokens, accepted drafts and next drafts of an artifact of 3 (4 steps,
  both fixture prompts), and widths switching 5, 5, 3, 3, 5, 3 with chained steps give
  transformers' 16 ids; the same switching on the GB10 (`PjrtQwen35MtpTest`); the delta-rule
  kernel and XLA forms with a records pool of T + 2 equal the interpreter
  (`PjrtFusedGatedDeltaTest`). `verify.sh` step 9 adds a Qwen3.5-0.8B artifact of 5 and 3
  drafts: with `cycle` (unchained with poisoned pools, then chained) and with `auto`,
  `speculative_checks.py` gives transformers' ids and the 3-draft unchained server's ids over
  the 100-token turns.
- **Qwen3.6-35B-A3B** (the `q36-all` flags with `-PmtpDraftTokens=5 -PmtpVerifyDrafts=3`, 57
  entries; `max_candidate_sequences` 4), the 40-prompt harness, one artifact and backend build,
  the count fixed or chosen. Two runs at 1 and 4 streams, one at 2; tokens/s:

| | 3 drafts | 5 drafts | chosen |
|---|---|---|---|
| 1 stream: median per reply | 150.1, 147.5 | 161.3, 160.4 | 148.1, 147.8 |
| 1 stream: mean per reply | 135.6, 135.2 | 141.7, 141.2 | 143.6, 144.1 |
| 1 stream: all tokens over the run's wall time | 124.1, 123.5 | 121.9, 121.5 | 127.7, 127.7 |
| 2 streams: median / mean per reply | 109.7 / 107.3 | 118.2 / 112.3 | 117.8 / 113.6 |
| 2 streams: all tokens over wall time | 148.4 | 151.9 | 159.5 |
| 4 streams: median per reply | 85.5, 86.3 | 70.4, 71.3 | 86.4, 86.2 |
| 4 streams: all tokens over wall time | 172.9, 172.9 | 155.6, 156.9 | 178.8, 176.4 |
| tokens per step, 1 / 4 streams | 2.91 / 2.90 | 3.39 / 3.44 | 3.26 / 3.10 |

  Per category at one stream (median per reply, first run): coding 158 / 173 / 168,
  reasoning 160 / 178 / 178, json 162 / 170 / 174, html 163 / 190 / 190, prose 108 / 91 / 102,
  narrative 108 / 96 / 104, summary 111 / 105 / 102, format 96 / 80 / 95 (3 / 5 / chosen).
  With three or four sequences in a step the count is always 3 (more than 12 rows at 5);
  with one or two it is chosen. Over the five runs of the chosen count, 5,074 of the first
  8,000 verify steps checked 3 drafts and 2,926 checked 5; 122 changed the count. TTFT 0.10 s at one stream and 0.20–0.21 s at four in every mode.
- The aim (at least the better fixed count at each stream count) is met by the mean per
  reply and by the tokens over the run's wall time at 1, 2 and 4 streams, and by the median
  per reply at 2 and 4. It is not met by the median per reply at one stream (148 against
  161): the 40 replies are two groups (code, reasoning, json, html near 170; prose, narrative,
  summary, format near 100), so the median falls between them, on the slowest replies of the
  fast group (code3 146, reason3 150 chosen against 146, 161 with 5 drafts), which the choice
  starts at 3 drafts or switches to 3 for some steps.
- Against earlier exports: with 3 drafts this artifact gives 147.5–150.1 per reply at one
  stream and 85.5–86.3 at four; `q36-all` measured 150.5–154.7 and 83.7–86.9 (flare plan,
  item 4), and its 3-draft step takes the same server time (27.4 against 27.0 ms). The flare
  plan's 5-draft export (`q36-commit-d5`, conv states written per token) measured 166.9 at
  one stream; this one 160.4–161.3.
- Suites: ir 1208, stablehlo 331, maestro 111, runtime-pjrt 260, none failing;
  `scripts/prepush.sh` passes.
- Commits: `27f29c5a` (graph, ops, kernel check), `65b02005` (export), `013ce52b` (backend,
  `verify.sh`), and this one (docs).

