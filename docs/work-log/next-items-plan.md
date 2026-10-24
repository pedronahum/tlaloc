# Next items: plan and results

From `main` at `01be0122` (2026-10-24), worked in order on `main`. Each item ends with its
commits and its numbers here; an item that stops short says what was validated and what is
left (and FOLLOWUPS says it too).

| # | Item | Done when | Status |
|---|---|---|---|
| 1 | Long requests on Qwen3.8-27B | the four-user 100K streamed run (3 turns) completes with the new timeout; FOLLOWUPS updated | |
| 2 | Checks before a push | `scripts/prepush.sh` runs the checks CI runs that the per-module suites skip (ABI dumps, the compiler plugin, JVM targets); CONTRIBUTING names it | |
| 3 | Long prefill on Qwen3.6-35B-A3B, measured again | one and four 100K documents to the first token, with the sort-free router, against the 53 s / 214 s and vLLM's 29–36 s / 92–113 s | |
| 4 | Per-token state committed once per step | the conv state, the n-gram memory and the head's carried hidden state committed as the delta rule's state is: two state slots per sequence; interpreter, XLA, kernels and backend equal; the 35B measured | |
| 5 | Qwen3.8-Flash-Next served (flash-next-plan step 6) | exported with the n-gram table as NVFP4 in memory and served by Triton with chained steps; contexts past 2,048 through the CUDA kernels (indexer, masked attention) | |
| 6 | Qwen3.8-Flash-Next measured (step 7) | the community harness at one and four streams against the published table | |
| 7 | Draft width per request | verify entries for several draft counts in one artifact and the backend choosing per batch and per request's acceptance; the 35B measured at 1 and 4 streams | |
| 8 | Push and CI | `main` pushed, every workflow green | |
