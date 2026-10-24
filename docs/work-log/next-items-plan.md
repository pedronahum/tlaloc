# Next items: plan and results

From `main` at `01be0122` (2026-10-24), worked in order on `main`. Each item ends with its
commits and its numbers here; an item that stops short says what was validated and what is
left (and FOLLOWUPS says it too).

| # | Item | Done when | Status |
|---|---|---|---|
| 1 | Long requests on Qwen3.8-27B | the four-user 100K streamed run (3 turns) completes with the new timeout; FOLLOWUPS updated | done: all 12 turns complete; first tokens 794 s, follow-ups 1.3 s at 11.7–12.1 tokens/s per user |
| 2 | Checks before a push | `scripts/prepush.sh` runs the checks CI runs that the per-module suites skip (ABI dumps, the compiler plugin, JVM targets); CONTRIBUTING names it | done: 10 gate tasks in one `--continue` run, 16 s from scratch; passes on `main`; a probe ABI change plus a plugin compile error both named |
| 3 | Long prefill on Qwen3.6-35B-A3B, measured again | one and four 100K documents to the first token, with the sort-free router, against the 53 s / 214 s and vLLM's 29–36 s / 92–113 s | done: one 100K document 49.6 s to the first token (53.1), four 149.1 / 199.4 s median / last (160.3 / 213.8); a 2,048-token chunk after 98K 1.35 s, attention 644 ms of it |
| 4 | Per-token state committed once per step | the conv state, the n-gram memory and the head's carried hidden state committed as the delta rule's state is: two state slots per sequence; interpreter, XLA, kernels and backend equal; the 35B measured | |
| 5 | Qwen3.8-Flash-Next served (flash-next-plan step 6) | exported with the n-gram table as NVFP4 in memory and served by Triton with chained steps; contexts past 2,048 through the CUDA kernels (indexer, masked attention) | |
| 6 | Qwen3.8-Flash-Next measured (step 7) | the community harness at one and four streams against the published table | |
| 7 | Draft width per request | verify entries for several draft counts in one artifact and the backend choosing per batch and per request's acceptance; the 35B measured at 1 and 4 streams | |
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
