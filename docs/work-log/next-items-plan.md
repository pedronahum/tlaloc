# Next items: plan and results

From `main` at `01be0122` (2026-10-24), worked in order on `main`. Each item ends with its
commits and its numbers here; an item that stops short says what was validated and what is
left (and FOLLOWUPS says it too).

| # | Item | Done when | Status |
|---|---|---|---|
| 1 | Long requests on Qwen3.8-27B | the four-user 100K streamed run (3 turns) completes with the new timeout; FOLLOWUPS updated | done: all 12 turns complete; first tokens 794 s, follow-ups 1.3 s at 11.7–12.1 tokens/s per user |
| 2 | Checks before a push | `scripts/prepush.sh` runs the checks CI runs that the per-module suites skip (ABI dumps, the compiler plugin, JVM targets); CONTRIBUTING names it | done: 10 gate tasks in one `--continue` run, 16 s from scratch; passes on `main`; a probe ABI change plus a plugin compile error both named |
| 3 | Long prefill on Qwen3.6-35B-A3B, measured again | one and four 100K documents to the first token, with the sort-free router, against the 53 s / 214 s and vLLM's 29–36 s / 92–113 s | |
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
