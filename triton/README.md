# Triton backend for Tlaloc

`libtriton_tlaloc.so` is an [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
backend. It serves StableHLO that Tlaloc emits: at model load it compiles the
StableHLO through a PJRT plugin (the XLA CUDA plugin), and at request time it
runs the compiled executable on the GPU. Serving needs no JVM and no JAX:
the backend loads the PJRT plugin `.so` itself. Clients use Triton's standard
HTTP and gRPC (KServe v2) endpoints.

A language model exported by Tlaloc is served in sequence mode: Triton's
sequence batcher routes each sequence's requests by correlation ID, the
backend keeps the sequence's KV pages on the device, a prompt runs as one
prefill call (the prompts of several sequences that arrive together, as one
call), and the decode steps of several sequences run as one batch. A client
sends token ids and gets the next token's logits back.

A stateless model can use Triton's dynamic batcher: the backend groups the
requests by the shape of their rows, concatenates each group along dim 0 and
runs it as one call. Tensors a client places
in CUDA shared memory are read and written in place on the GPU, and each GPU
an instance group names gets its own PJRT client.

| | What | Where it ran |
|---|---|---|
| ✅ | Sequence mode, prefill and batched decode, TinyLlama-1.1B | GB10, `verify.sh` |
| ✅ | Qwen3-0.6B: 16 greedy ids equal HuggingFace's for a plain and a chat-template prompt; the tied head reads the embedding table (no second copy) | GB10, `verify.sh` |
| ✅ | Qwen3-0.6B with bf16 weights (`-PweightDType=bf16`): half the device memory, all 32 ids equal HuggingFace's, logits within 3.0e-3 of the largest | GB10, `verify.sh` |
| ✅ | A text prompt tokenized by the checkpoint's tokenizer.json (`tokenizers` package), byte-level BPE included | GB10, `verify.sh` (Qwen3) |
| ✅ | Image and video placeholder ids refused by name (the manifest lists them) | GB10, `verify.sh` (the window models; Muse Glimmer with `MUSE_GLIMMER=1`) |
| ✅ | A live sequence that needs a page when the pool is full is refused by name and keeps its KV; idle sequences' pages are reclaimed least recently active first | GB10, `verify.sh` (TinyLlama) |
| 📐 | Preempting a live sequence (swapping its KV out or recomputing it) | Designed, not built |
| ✅ | Muse Glimmer 30B, text decoder, bf16 weights: 32 greedy ids equal HuggingFace's run with the same arithmetic | GB10, `verify.sh` with `MUSE_GLIMMER=1` |
| ✅ | Muse Glimmer at contexts 512 to 32,768 with up to four sequences, prefill in 512-token calls; a 2,305-token prompt whose fact is outside the sliding window gives HuggingFace's 16 ids | GB10, `verify.sh` with `MUSE_GLIMMER=1`, `context_bench.py` |
| ✅ | Speculative generation in the backend with chained verify steps: Qwen3.5-0.8B (3 MTP drafts) gives transformers' 16 ids per prompt alone, batched and arriving mid-generation, and the ids of an unchained server | GB10, `verify.sh` step 9, `speculative_checks.py` |
| ✅ | Prompts of several sequences prefilled in one call: 2 and 4 TinyLlama and Qwen3-0.6B prompts give their solo argmax and the same 8 greedy ids as alone | GB10, `verify.sh` |
| ✅ | Dynamic batching (`max_batch_size > 0`, `dynamic_batching`), ragged batches grouped by the shape of their rows | GB10, `verify.sh` |
| ✅ | Bounded programs (`bounded_manifest`): a request of any length up to the bound runs on the smallest compiled bucket that holds it, padded and masked by the backend, and gets the interpreter's result for its own length | GB10, `verify.sh` (HTTP; gRPC when `tritonclient` is installed) |
| ✅ | CUDA shared memory inputs read in place, outputs written device to device | GB10, `verify.sh` |
| ✅ | KV pools updated in place: donated to each execution, never copied (100 of 100 runs checked by device address; a control without donation is seen copying) | GB10, `verify.sh` |
| ✅ | FP32, FP64, FP16, BF16, INT8, INT32, INT64, UINT8, BOOL over HTTP and gRPC | GB10, `verify.sh` |
| ✅ | GPU 0 selected by ordinal; a GPU the machine does not have is refused by name | GB10, `verify.sh` |
| 🧪 | Instances on GPUs other than 0, one PJRT client per GPU | Not run: the GB10 has one GPU |

How the backend fits with the Kotlin export and the other two ways to serve
an artifact is in [docs/SERVING_ARCHITECTURE.md](../docs/SERVING_ARCHITECTURE.md).
[examples/triton-llm](../examples/triton-llm/) exports Qwen3-0.6B (or TinyLlama,
or Muse Glimmer), serves it with this backend and streams a chat answer, in
one script.

This directory is not a Gradle module and is not published to Maven.

```
triton/
  backend/              the backend source (C++17); sequence_mode.{h,cc} is sequence mode
  third_party/          vendored upstream headers and sources (see docs/vendoring.md)
  examples/
    model_repository/   the example models, emitted by Tlaloc (one written by hand, and marked so)
    reference/          the DXIR interpreter's results for them
  build_backend.sh      builds the backend inside the Triton container
  fetch_pjrt_plugin.sh  downloads the PJRT CUDA plugin
  run_server.sh         starts tritonserver with the backend and a model repository
  verify.sh             end-to-end check: start, infer over HTTP and gRPC, compare, stop
  verify_client.py      the client half of verify.sh
  perf_client.py        the measurements verify.sh prints (batching, zero copy)
  sequence_client.py    a client for a sequence-mode model (Python, tritonclient)
  multi_user_bench.py   concurrent users with long prompts and follow-ups: first-token time and tokens/s
  generate_client.py    greedy decoding from text against a sequence-mode model
  sequence_checks.py    the TinyLlama sequence-mode checks verify.sh runs
  fixture_checks.py     greedy decoding against a HuggingFace fixture (the Qwen3 checks)
  context_bench.py      prefill and decode timings of a sequence-mode model at several contexts
  profile.sh            times decode steps and prefill calls, optionally under Nsight Systems
  profile_client.py     the client half of profile.sh
  profile_report.py     summarises a capture: executions, kernels by model component
```

`backends/` (the built `.so`), `pjrt/` (the downloaded plugin) and `build/`
(the device test and the TinyLlama and Qwen3 models verify.sh writes) are
build outputs and are not committed.

## Prerequisites

- Docker with the NVIDIA Container Toolkit (`docker run --gpus all` works).
- The Triton container `nvcr.io/nvidia/tritonserver:25.11-py3` (Triton 2.63.0,
  CUDA 13.0, Ubuntu 24.04). The backend is built inside this image so that
  it matches the server's compiler, glibc and `libtritonserver.so`.
- A PJRT CUDA plugin whose PJRT C API major version is 0.
  `fetch_pjrt_plugin.sh` downloads `jax-cuda13-pjrt` 0.10.0 from PyPI (PJRT C
  API 0.104, CUDA 13), checks its sha256 and unpacks the one `.so` needed. Its
  CUDA major version matches the container, so it uses the container's CUDA
  libraries and nothing else has to be mounted.
- For `verify.sh`: `curl`, and a Python with `tritonclient[all]` whose
  `cuda-python` matches the CUDA major version of the host driver's
  libraries that the client can load. On the GB10 (driver 580, CUDA 13)
  tritonclient 2.72 needs `cuda-python` 12.x: with `cuda-bindings` 13 its
  CUDA shared memory helper fails (`cudaIpcMemHandle_t` has no `reserved`).
  For example `pip install "tritonclient[all]==2.72.0" "cuda-python>=12.8,<13"`
  in a virtual environment; point `TRITON_CLIENT_PYTHON` at its `python`.

Tested on an NVIDIA GB10 (aarch64, unified memory) with host driver
580.126.09.

## Build, verify, run

```bash
triton/fetch_pjrt_plugin.sh     # -> triton/pjrt/xla_cuda13/xla_cuda_plugin.so
triton/build_backend.sh         # -> triton/backends/tlaloc/libtriton_tlaloc.so
TRITON_CLIENT_PYTHON=/path/to/venv/bin/python triton/verify.sh
triton/run_server.sh            # serve the example repository on ports 8000-8002
```

`build_backend.sh` also builds and runs the unit test for the StableHLO text
reader (`backend/test/stablehlo_text_test.cc`) before it links the backend,
and builds `build/tests/pjrt_device_test` (`backend/test/pjrt_device_test.cc`),
which needs a GPU and is run by `verify.sh`.

`verify.sh` first runs `pjrt_device_test` in the Triton container against the
real plugin: a PJRT client created for GPU 0 sees exactly one device, CUDA
ordinal 0; a client for a GPU the machine does not have (GPU 7 here) is
refused with a message naming it; and `x * x + x` over 1024 values read in
place from `cudaMalloc` memory, with the result copied device to device, gives
the same bits as the host path and leaves the input unchanged. Run again with
one wrong expected value, it must fail.

It then starts the server, waits for it to be ready, and runs
`verify_client.py`, which checks:

- `matmul_sumsq` over KServe v2 JSON, tritonclient HTTP and tritonclient gRPC,
  bit for bit against the DXIR interpreter's result
  (`examples/reference/matmul_sumsq.json`) and against the closed form
  (value 54, gradient `[[7, 11], [9, 13]]`);
- `dtypes` (FP64, BF16, INT32), `int64_bool` (INT64, BOOL) and
  `dtypes_small` (FP16, INT8, UINT8) over HTTP and gRPC, exactly, including a
  request that asks for only one output. The INT64 values go past the INT32
  range (`3e9 * 3e9 + 3e9`);
- `grad_batched` and `grad_unbatched` over gRPC: 8 rows in one request and 3
  rows (run on the batch-4 artifact with a row of padding) against the DXIR
  interpreter (`examples/reference/grad_batched.json`) and the closed form
  `rowsum(A)[q] + colsum(A)[p]`, bit for bit; then 512 one-row requests from
  32 threads to each model. Every request gets exactly its own row back, the
  two models' answers are identical, Triton's statistics show that
  `grad_batched` ran them in fewer executions than requests (162 and 181
  executions for the 512 in two runs) and `grad_unbatched` in one execution each;
- `ragged_batched` and `ragged_consecutive` over gRPC: three rounds of 72
  requests sent at once, each one or two rows of width 4 or 8 chosen at
  random, so that Triton's batches mix the two widths. Every request gets
  exactly its own rows of `x · x + x` back from both models, and the two
  models' answers are identical. Grouped by the shape of their rows
  (`ragged_batched`) the 216 requests must run in fewer executions than
  grouped consecutively (`ragged_consecutive`), and in at most half as many
  executions as requests (measured: 74 to 76 executions against 119 to
  134; with one-row requests only, 72 requests ran in 23 executions against
  43). The log must show each model's grouping at load;
- CUDA shared memory over gRPC: `matmul_sumsq` with `A`, `GRAD` and `VALUE` in
  CUDA shared memory against the interpreter, and `large_io` and
  `large_io_host` (16 MiB in, 16 MiB out) against the closed form, bit for
  bit, with the input region unchanged afterwards; `large_io` without shared
  memory gives the same bits. The server log must then say that `large_io`
  read `X` in place and copied `Y` device to device, that `large_io_host`
  took both through the host, and not the other way round;
- `buckets` at both compiled lengths, and a third length that the backend
  refuses;
- `reference_decode` over HTTP and gRPC: five decode steps (two batch/context
  entries) with the KV pools held by the backend between requests, against
  the DXIR interpreter's logits for the same steps
  (`examples/reference/reference_decode.json`), within 1e-3 of each step's
  largest logit and with the same argmax. XLA runs the GPU's f32 dot as
  TF32, which is why this one is not bit for bit. A step sent on a page that
  no earlier request wrote must then disagree with the reference, which shows
  that the agreement depends on the carried state;
- `reference_sequence` over HTTP and gRPC: the same artifact in sequence mode
  (the log must show its KV pools updated in place).
  The reference steps' first sequence sent one token per request, then all
  four tokens in one request, and two more one-token sequences, each against
  the interpreter within 1e-3; then requests the backend or Triton must
  refuse by name (a prompt longer than the compiled context, a token outside
  the vocabulary, a step without START, a step after END, an empty request
  without END) and an empty END request that returns no logits;
- thirteen model configurations the backend or Triton must refuse at load
  (including `zero_copy: "yes"` and `gpus: [ 7 ]`), two batching
  configurations (`max_batch_size` above the largest artifact's batch, and
  state with `max_batch_size > 0`), and nine sequence-mode configurations (no `sequence_batching`, the direct strategy,
  no CORRID control, `max_batch_size: 0`, `serving_manifest` combined with
  `artifact`, a manifest outside the version directory, two inputs, a
  `KV_PAGES` output of three, and `window_sequence` pointed at
  `tlaloc-serving-short-ring.json`, its manifest with a ring of one page,
  which holds 4 positions of a window of 8), each by
  the expected message, each followed by a reload of the good configuration;
- that the server is still live at the end.

Then `window_checks.py` runs `window_sequence` against `window_sequence_full`
(the same decoder with full-history pages):

- one sequence grown to 60 positions (a 13-token prompt, single tokens, a
  9-token request once the window is full, single tokens): after every
  request `KV_PAGES` reports `ceil(length / 4)` full pages for both models,
  and `min(3, ceil(length / 4))` windowed pages for `window_sequence` (so 3
  while the full pages reach 15) and none for the full-history model;
- the logits of every request equal those of the full-history model sent the
  same calls (the backend splits the 13 tokens into 12 and 1 and the 9 into 5
  and 4 for the ring; the client sends the full-history model those calls),
  within 1e-4 of the largest logit; the run measures 1.2e-6 (1 of 40 bit
  for bit: the windowed model's sliding layers score their 12-position ring
  and the full-history model's score 64 positions, so the GPU sums in
  another order), and a ring error moves them by about 1. The full-history model sent the requests whole
  (one prefill call instead of two, a different entry under TF32) agrees
  within 5e-3 (7.6e-4 measured), with the same argmax. Over HTTP and gRPC;
- three sequences stepped concurrently get the logits they get alone, within
  5e-3;
- with every ring of the windowed pool taken, one more START is refused by
  name (`windowed KV page pool exhausted`), and runs after an END;
- ids 62 and 63, which the manifest lists as the model's image and video
  placeholders (stand-ins: the window models are text only), are refused by
  name in a START and in a later request, over HTTP and gRPC; the sequence
  a later request was refused for then gives the logits of a sequence that
  never sent it, bit for bit; id 61 is accepted. The load log must list the
  refused ids;
- the log must show the windowed pool at load and 100 of 100 runs with its 6
  pools updated in place.

`window_sequence_chunked` is the same decoder with its prefill entries at
6 tokens per sequence and rings of 4 pages (`ceil((8 - 1 + 6) / 4)`). Grown
the same way, the 13- and 9-token requests run as calls of at most 6: it must
hold at most 4 windowed pages, and its logits must equal the full-history
model's sent the same calls within 1e-4 of the largest (measured: 5 of 40
bit for bit, the rest within 9.9e-7) and sent the requests whole within 5e-3
with the same argmax. The load log must state the 6-token chunk and the
4-page ring.

Then `prefill_checks.py` runs on `window_sequence`, `window_sequence_full`
and `window_sequence_chunked`: one prompt, then two prompts sent together from two
threads (13 and 5 random ids; the 13 are split 12 and 1 for the ring). The
two must run in one execution, the log must show `prefill_b2_c16 ran 2
sequence(s)`, and each prompt's last-token logits must be within 5e-3 of its
solo run's largest logit (measured: 0 to 3.8e-4), with the solo argmax where
the solo top two are further apart than that.

It then runs the three clients again with deliberately wrong expected values
(the window checks compare each request with the next one, the prefill checks
each prompt with the next one), and all three runs must fail. It prints the measurements of `perf_client.py`
(`SKIP_PERF=1` skips them) and stops the container.

Last, if the TinyLlama-1.1B checkpoint is in
`~/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0`
(`TINYLLAMA_CHECKPOINT` overrides), `verify.sh` exports its serving artifact
(decode entries for batches 1, 2 and 4 and a prefill entry, all at context 64)
and writes it as the sequence-mode Triton model `tinyllama` under
`triton/build/tinyllama/`, with a 5 s sequence idle timeout. Later runs reuse
that model (`TINYLLAMA_REEXPORT=1` exports again). Gradle runs only while no
server is up. It then starts a server on that repository and:

- greedy-decodes "The capital of France is" with `generate_client.py`. The six
  generated ids must equal HuggingFace transformers' ids for the same
  checkpoint, `[3681, 29889, 13, 13, 29906, 29889]` (" Paris.\n\n2.");
- runs `sequence_checks.py`:
  - prefill: the prompt in one request and the prompt one token per request
    give the same argmax, and logits within 5e-3 of the largest (measured:
    9e-3 to 1.3e-2 on logits of about 14; two different executables, TF32).
    Greedy decoding reproduces HuggingFace's ids over HTTP and gRPC;
  - concurrent: four sequences with different prompts, decoded from four
    threads at once, each produce exactly the 12 ids they produce alone, and
    the server's statistics show fewer executions than requests (the decode
    steps were batched);
  - END frees pages: 43 sequences of three pages each, one after the other,
    on a pool that holds 21 at once, all run;
  - exhaustion: with 21 sequences holding the whole pool, one more START is
    refused with `KV page pool exhausted`, the server stays live, the refused
    sequence has no state, and after the others end a new sequence decodes
    the expected ids;
  - pressure: 21 live sequences hold the whole pool, and the first one (a
    47-token prompt of ordinary words) needs a fourth page for its second
    generated token. That request is refused with `KV page pool exhausted`,
    the message lists the 20 other sequences holding pages and says the
    sequence keeps its 3 pages and the KV of its 48 tokens, and the log shows
    no sequence reclaimed. After another sequence ends, the same request is
    sent again and succeeds, and the sequence's 6 ids equal its solo run's;
  - idle: 21 sequences abandoned without END lose their pages after twice the
    idle timeout (plus the queueing allowance described under "Idle
    timeout"); a new sequence then decodes correctly, the abandoned
    sequence's next step is refused, and 21 new three-page sequences fit
    again. From the server log (`--log`): the new one-page sequence
    reclaimed only the least recently active abandoned sequence, and the 21
    new sequences reclaimed the other 20 in the order they went idle;
  - queued (a second server, the same model with a 200 ms idle timeout):
    12, 24 and 32 concurrent sequences, so steps wait in Triton's queue past
    the timeout; no step Triton accepts is refused for having no KV state.
    Then one sequence is sent a refused request (a token outside the
    vocabulary) every 50 ms for 2.5 s while 22 new sequences run the pool
    out: it is not reclaimed (read from the log) and continues with its solo
    ids. A backend that counted only the requests that ran reclaimed it, and
    its next step was refused for having no KV state;
  - restart-refused: a second START for a live sequence ID, refused for a
    token outside the vocabulary, still ends the earlier sequence in Triton,
    so the next step is refused for having no KV state instead of continuing
    the earlier sequence's KV (which a backend without this rule did);
- runs `sequence_checks.py --perturb` (wrong expected ids, for the prefill
  and pressure checks), which must fail;
- runs `prefill_checks.py` with `sequence_checks.py`'s four prompts: one, two
  and four prompts sent together, over HTTP and gRPC, each run in one
  execution (the log must show `prefill_b4_c64 ran 4 sequence(s)`), and each
  prompt's argmax and 8 greedy ids equal its solo run's, with logits within
  5e-3 of the largest (measured: at most 6.2e-4; a batch-2 or batch-4 entry
  is a different executable under TF32). Then with `--perturb`, which must
  fail. It prints the prefill throughput (below). It runs again with four
  prompts of 2, 55, 9 and 30 tokens, so that one call holds a 2-token row
  behind 62 padding positions next to a 55-token row (measured: logits
  within 5.6e-4, the same ids), and with `--perturb`;

Without the checkpoint this step prints `SKIP tinyllama` and the run can
still pass; `SKIP_TINYLLAMA=1` skips it on purpose.

Then, if Qwen/Qwen3-0.6B (Apache-2.0) is in the HuggingFace cache at the
revision the fixture names (`hf download Qwen/Qwen3-0.6B`; `QWEN3_CHECKPOINT`
overrides the directory), `verify.sh` exports it the same way into
`triton/build/qwen3/` as the sequence-mode model `qwen3` (`QWEN3_REEXPORT=1`
exports again), starts a server on it and runs `fixture_checks.py` against
`ir/src/jvmTest/resources/io/tlaloc/ir/inference/qwen3_0_6b_greedy.json`. That
fixture is HuggingFace transformers greedy-decoding 16 tokens in float32 on
the CPU with eager attention (`harness/python/hf_greedy_fixture.py`), for
" The capital of France is" and for "What is the capital of France? Answer in
one word." rendered with Qwen3's chat template (thinking off). For each prompt,
over HTTP and gRPC, the prompt is one prefill request and the 16 generated ids
must equal the fixture's; the first position's top-20 logits and the chosen
logit at every position must be within 2e-3 of the largest logit (measured:
2.6e-4 and 3.3e-4; TF32). It prints the prefill and decode timings, then runs
`fixture_checks.py --perturb` (wrong expected ids), which must fail, and
`prefill_checks.py` as for TinyLlama, on four prompts from the fixture (its two
prompts, and each followed by the first tokens of its continuation: 5, 24, 11
and 27 tokens; logits measured within 9.0e-4 of the largest), and with
`--perturb`, which must fail. Qwen3-0.6B ties its head to the embedding
table, and the model must upload 310 weights: the head contracts against the
table's hidden axis and no transposed copy is staged (2273 MiB, against 2867
MiB with the copy; measured logits within 5.6e-4, decode step 15.4 ms against
16.4 ms). When the client's Python has the `tokenizers` package,
`generate_client.py` then sends the fixture's text prompt as text: the prompt
must tokenize to the fixture's 5 ids and the 16 generated ids must be the
fixture's; with one wrong expected id it must fail. Then the same checkpoint
is exported with bf16 weights (`-PweightDType=bf16`) into
`triton/build/qwen3-bf16/` and served: the upload must be half the f32
model's MiB (1136 against 2273), all 32 ids must equal the fixture's, the
logits must be within 6e-3 of the largest (measured 3.0e-3: every projection
rounds its input to bf16), and `--perturb` must fail. As a control on the
logit check, the bf16 model must fail the f32 model's tolerance of 2e-3. A decode step takes
11.3 ms against 15.4 ms with f32 weights. Last, the checkpoint is exported
with bf16 weights and int8 projections (`-PweightQuant=int8`) into
`triton/build/qwen3-bf16-int8/`: the upload must be 506 weights and under
70% of the bf16 model's MiB (718 against 1136), all 32 ids must equal the
fixture's (the logits are not compared: quantization changes them), and
`--perturb` must fail. Without the checkpoint this step prints
`SKIP qwen3`; `SKIP_QWEN3=1` skips it.

With `MUSE_GLIMMER=1`, and meta-models/Muse-Glimmer-30B (Apache-2.0, 59 GB,
not gated) in the HuggingFace cache at the revision the fixtures name
(`hf download meta-models/Muse-Glimmer-30B`), `verify.sh` also serves its text
decoder. The step is opt-in because it puts 56 GB of bf16 weights on the
device; before the export and before the load it refuses by name if
MemAvailable is below what the step needs plus 16 GiB, because the GB10's GPU
shares system memory and a runaway allocation there can take the machine
down. It exports into `triton/build/muse-glimmer/` for contexts 512, 2,048,
8,192 and 32,768, decode batches 1, 2 and 4, batch-1 prefill in calls of at
most 512 tokens and a pool for four sequences of 32,768 positions (8,193
pages), with a sequence idle timeout of 600 s
(`MUSE_GLIMMER_REEXPORT=1` exports again; so does a model exported for
another ladder). It drops the page cache of the checkpoint and the artifact
and starts the server with a PJRT memory fraction of the weights plus 15 GiB
over MemTotal (0.55 on the GB10: 4 GiB of KV pools, the temporary memory
of the largest entry, under 0.5 GiB, which the compile log states, and room
to spare), printed. Three fixtures, all from `harness/python/muse_glimmer_fixture.py`
(transformers 5.17 on the CPU, eager attention, 16 greedy tokens for
" The capital of France is", a chat-template prompt with a fixed
`current_date`, and a long text prompt):

- `muse_glimmer_30b_mixed_greedy.json`: bf16 weights, f32 activations, each
  projection's input rounded to bf16, RoPE tables in float64: the graph's own
  arithmetic. All 32 ids must be equal, and the logits within twice the
  fixture's `referenceNoise`, which is how far that same run moves when its
  activations are float64 instead of float32 (up to 9.1e-3 of the largest
  logit).
- `muse_glimmer_30b_needle_mixed_greedy.json`: the same arithmetic, for a
  2,305-token field report with one fact near its start ("the vault code for
  the Lindqvist archive is 4719") and a question about it at the end, 2,190
  tokens later: outside the 2,048-token window of the 39 sliding layers, so
  only the 13 full-attention layers can read it. transformers answers
  " 4719." and goes on reasoning. The prompt runs as five prefill calls.
  All 16 ids must be equal and the logits within 2e-2 of the largest (the
  fixture has no `referenceNoise`; twice the other prompts' is 1.2e-2 and
  1.8e-2); measured 7.7e-3. `--perturb` on this fixture must fail. The
  fixture holds the prompt's text (`prompts[0].input`); `muse_glimmer_fixture.py
  real --precision mixed --f64-rope --text "<that text>"` writes it again (263 s
  on the GB10's CPU, about 60 GB of memory; run it with no server up).
- `muse_glimmer_30b_bf16_greedy.json`: transformers in bfloat16 as it runs by
  default. Only the ids are compared (`--ids-only`), and only as far as the
  two fixtures agree (`--common-prefix-with`): they choose differently at the
  chat prompt's 16th token (margins 0.31 and 0.12), which is reported.

The log must list the refused placeholder ids (`refuses token ids 200091
(video_token_id) 200092 (image_token_id)`), and a START holding either is
refused by name. Then `--perturb`, which must fail, and the peak memory in
use. An artifact exported before its manifest listed the placeholder ids is
exported again. With `MUSE_CONTEXT_BENCH=512,2048,8192` it also runs
`context_bench.py` at those contexts (below; adding 32768 takes about 20
minutes more).

A request with curl:

```bash
curl -s localhost:8000/v2/models/matmul_sumsq/infer \
  -d '{"inputs":[{"name":"A","shape":[2,2],"datatype":"FP32","data":[1,2,3,4]}]}'
# {"model_name":"matmul_sumsq","model_version":"1","outputs":[
#   {"name":"VALUE","datatype":"FP32","shape":[1],"data":[54.0]},
#   {"name":"GRAD","datatype":"FP32","shape":[2,2],"data":[7.0,11.0,9.0,13.0]}]}
```

## The example models

| Model | Function | Inputs | Outputs |
|---|---|---|---|
| `matmul_sumsq` | `f(A) = sum(A · A)` and `df/dA = 1 · Aᵀ + Aᵀ · 1`, one function returning both | `A` FP32 [2,2] | `VALUE` FP32 [1], `GRAD` FP32 [2,2] |
| `dtypes` | `x · x + x` for three dtypes | `X_F64` FP64 [4], `X_BF16` BF16 [4], `X_I32` INT32 [4] | `Y_F64`, `Y_BF16`, `Y_I32` |
| `buckets` | `x · x + x`, compiled for length 4 and length 8 | `X` FP32 [-1] | `Y` FP32 [-1] |
| `reference_decode` | one decode step of Tlaloc's reference decode graph (embedding, paged attention over a KV cache, LM head), six batch/context entries | `tokenIds`, `positions` INT32 [-1,1], `blockTables` INT32 [-1,-1], `seqLens`, `slotMapping` INT32 [-1] | `logits` FP32 [-1,1,11] |
| `reference_sequence` | the `reference_decode` artifact in sequence mode | `TOKENS` INT32 [-1] (batch dim added, max batch 4), START/END/CORRID controls | `LOGITS` FP32 [11], `KV_PAGES` INT32 [2] |
| `window_sequence` | a three-layer decoder with seeded random weights, sequence mode: layers 0 and 1 attend over a sliding window of 8 positions and keep their KV in a windowed pool (rings of 3 pages of 4 tokens), layer 2 over the full history; contexts 16, 32 and 64, decode and prefill entries for batches 1 and 2 | as `reference_sequence` (max batch 2) | `LOGITS` FP32 [64], `KV_PAGES` INT32 [2] |
| `window_sequence_full` | the same decoder and weights with full-history pages for every layer | as `window_sequence` | as `window_sequence` |
| `int64_bool` | `x · x + x` over i64, `(x · x > x) and b`, `not b` | `X` INT64 [4], `B` BOOL [4] | `Y` INT64, `ABOVE_AND_B` BOOL, `NOT_B` BOOL |
| `dtypes_small` | `x · x + x` over f16, i8 and u8. **Written by hand**: Tlaloc's `DType` has no f16, i8 or u8, so this model shows only that the backend moves those types unchanged | `X_F16` FP16 [4], `X_I8` INT8 [4], `X_U8` UINT8 [4] | `Y_F16`, `Y_I8`, `Y_U8` |
| `grad_batched` | `df/dA` of `sum(A · A)` per 2x2 row, compiled for batch 1, 2, 4 and 8, with `dynamic_batching` | `A` FP32 [2,2] (batch dim added, max batch 8) | `GRAD` FP32 [2,2] |
| `grad_unbatched` | the same files without `dynamic_batching` | as `grad_batched` | as `grad_batched` |
| `ragged_batched` | `x · x + x` over rows of 4 or of 8 values, compiled for batch 1, 2, 4 and 8 at each width, with `dynamic_batching` and `allow_ragged_batch`, so one batch can hold both widths | `X` FP32 [-1] (batch dim added, max batch 8) | `Y` FP32 [-1] |
| `ragged_consecutive` | the same files and batching with `group_by_shape: "false"` (the control) | as `ragged_batched` | as `ragged_batched` |
| `large_io` | `x · x + x` over 4Mi f32 values (16 MiB each way) | `X` FP32 [4194304] | `Y` FP32 [4194304] |
| `large_io_host` | `large_io` with `zero_copy: "false"` | as `large_io` | as `large_io` |
| `bounded_mean` | the mean of the rows of `x0` over a sequence axis of at most 16 positions, bounded mode, buckets 4, 8 and 16 | `x0` FP32 [-1,4] | `y0` FP32 [4] |
| `bounded_softmax` | a softmax over at most 16 positions, bounded mode, buckets 4, 8 and 16; the output keeps the bounded axis | `x0` FP32 [-1] | `y0` FP32 [-1] |

The `.mlir` files (except `dtypes_small`'s), the `reference_decode`,
`reference_sequence`, `window_sequence`, `window_sequence_full`, `bounded_mean`
and `bounded_softmax` model directories and the files in `examples/reference/`
are generated by Tlaloc. The gradients are Tlaloc's reverse-mode transform of
the DXIR graph, not hand-written StableHLO. To regenerate them (JDK 25, from the
repository root):

```bash
./gradlew :maestro:exportTritonExamples -PoutDir=$PWD/triton/examples
```

The source is `maestro/src/jvmTools/kotlin/io/tlaloc/maestro/serving/ExportTritonExamples.kt`.
The `config.pbtxt` files of the other models are written by hand;
`reference_decode`, `reference_sequence`, `window_sequence` and
`window_sequence_full` are written whole from their serving artifact, as
described in the next section.

## Exporting a Triton model repository from Kotlin

A Tlaloc serving artifact (the directory `:maestro:exportServingArtifact` and
`:maestro:exportLlamaServingArtifact` write: a manifest, one StableHLO body
per entry, and staged weight files) becomes a Triton model with
`TritonModelRepository` in `:maestro`:

```kotlin
import io.tlaloc.maestro.serving.TritonModelRepository

TritonModelRepository.write(artifactDir, repositoryDir, "tinyllama")
// -> <repositoryDir>/tinyllama/config.pbtxt
//    <repositoryDir>/tinyllama/1/tlaloc-serving.json, bodies/, programs/, weights/
```

`TritonModelRepository.config(manifest, name)` returns the `config.pbtxt`
text alone. The same from Gradle, for an artifact already on disk:

```bash
./gradlew :maestro:exportHfServingArtifact \
    -PckptDir=$HOME/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0 \
    -PoutDir=/abs/tinyllama-artifact
./gradlew :maestro:exportTritonModel -PartifactDir=/abs/tinyllama-artifact \
    -PoutDir=/abs/model_repository -PmodelName=tinyllama
MODEL_REPOSITORY=/abs/model_repository triton/run_server.sh
python triton/generate_client.py --model tinyllama \
    --tokenizer $HOME/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0/tokenizer.json
```

`exportHfServingArtifact` (also registered under its earlier name,
`exportLlamaServingArtifact`) reads Llama, Qwen3 and Muse Glimmer checkpoints
(the last text only: the vision encoder's tensors are listed and not read). `-PckptDir`
may be a HuggingFace cache snapshot, for example
`~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<revision>`; the
manifest's model name is then the repo id, `Qwen/Qwen3-0.6B`.
`generate_client.py` tokenizes `--text` with the checkpoint's `tokenizer.json`
through the `tokenizers` package when it is installed, so a Qwen3 or Muse
Glimmer (byte-level BPE) prompt can be text; without the package it reads a
SentencePiece vocabulary word by word (TinyLlama) and a byte-level one needs
ids (`--prompt`). `-PweightDType=bf16` stages the weights as bf16 (the
default is f32 for Llama and Qwen3, bf16 for Muse Glimmer): half the device
memory, and each projection rounds its input to bf16 and sums in f32.

`-PweightQuant=int8` (opt-in; the default is `none`) stores each layer's
projection weights as int8 with one f32 scale per output channel, next to
their scales in the weight table (`qProj0`, `qProj0Scale`); the embedding
table, the norms and the head keep the weight dtype. It halves the bytes a
decode step reads for a bf16 model. It changes the numerics, so an int8
model is compared with the unquantized one rather than expected to equal
it: `quant_checks.py` reports, for the greedy fixtures, how many ids it
reproduces, and, for the first tokens of the wikitext-103 test split, the
perplexity and the share of argmaxes equal to a saved run of the other
model:

```bash
python triton/quant_checks.py --grpc localhost:8021 --model muse \
    --fixture ir/src/jvmTest/resources/io/tlaloc/ir/inference/muse_glimmer_30b_bf16_greedy.json \
    --tokenizer <checkpoint>/tokenizer.json --text-tokens 480 --bos 200000 \
    --label int8 --save int8.json --compare bf16.json
```

Measured on the GB10 (section 6 of
[SERVING_ARCHITECTURE.md](../docs/SERVING_ARCHITECTURE.md#int8-weights-opt-in)
has the tables): Muse Glimmer with int8 weights keeps all 48 fixture ids of
the bf16 model (the needle answer is still " 4719."), its perplexity moves
by −0.14% and 99.0% of the argmaxes agree; with the GPU idle a decode step
takes 138 ms instead of 244 ms (7.2 tokens/s instead of 4.1), a 512-token
prefill call about 160 ms longer, and the weights take 29,143 MiB instead of
53,128 MiB. Qwen3-0.6B
keeps its 32 ids, +1.1% perplexity, 96% argmax agreement, and a decode
step takes 8.8 ms instead of 10.4 (bf16 weights). `verify.sh`
serves Qwen3-0.6B with int8 weights and requires its 32 fixture ids, and
serves it again with the codes quantized along the input axis
(`int8_wrong_axis.py`, the scales left per output channel), where the ids
must fail.

`-PweightQuant=fp8` stores the same projections as e4m3fn codes with an f32
scale per output channel (the row's largest absolute weight over 448), the
routed experts of a MoE model per expert and channel. A checkpoint already
quantized (ModelOpt FP8 or NVFP4, block FP8) is dequantized on load and
quantized again. `-PkvDtype=fp8` stores the KV pools as e4m3fn: a write
clamps the keys and values to ±448 and rounds them, and attention widens the
pages it gathers. Both change the numerics and are opt-in. Measurements on
the GB10 with Qwen3.5-family models are in
[qwen35-progress.md](../docs/work-log/qwen35-progress.md).

A checkpoint that ties its head to the embedding table (Qwen3) has no head
weight in the artifact: the head is one `dot_general` that contracts the
final hidden state against the table's hidden axis. A multimodal checkpoint's
image and video placeholder ids (`image_token_id`, `video_token_id` in its
config) are listed in the manifest (`model.refusedTokens`), and the backend
refuses a request holding one by name: the artifact is the text decoder only,
so the placeholder would be embedded as an ordinary token.

The artifact's files are hard-linked into the version directory when both are
on one file system, and copied otherwise. They are real files either way, so
the repository can be mounted into the container on its own.

For an artifact with KV pools the model is written in sequence mode (next
section). `TritonModelRepository.write(..., kv = KvMode.CLIENT)`, or
`-PkvMode=client`, writes the client-managed form instead, in which the client
names pages and slots in every request; `reference_decode` is written that
way. `-PmaxSequenceIdleMicros` and `-PmaxQueueDelayMicros` set the sequence
batcher's idle timeout (default 60 s) and queue delay (default 0: the
backend forms the batches, see "Batching" under "Sequence mode").

The TinyLlama commands above export decode entries for batch 1 only; add
`-PmaxBatch=4` to `exportHfServingArtifact` to let the backend batch up to
four sequences' decode steps and prefill up to four prompts in one call
(three decode entries and three prefill entries: six XLA compiles at load,
about 25 s for TinyLlama). `-PprefillMaxBatch=1` keeps the batch-1 prefill
entry alone.

In client mode the generated `config.pbtxt` maps each slot of the manifest
by its role:

| Manifest slot | In Triton |
|---|---|
| `tokenIds`, `positions`, `blockTables`, `seqLens`, `slotMapping` | inputs, INT32, under the slot's own name. A dimension that differs between entries (batch size, block-table width) is `-1`; each request runs on the entry compiled for exactly its shapes. |
| `logits` | output |
| staged weights | `weight:` arguments. The backend reads each file when the model loads and keeps it on the device; no request carries weights. |
| KV pools (`keyCacheN`, `valueCacheN`) and their `…Out` results | `state:` arguments and results. Each model instance holds its pools on the device, zero at start; every request reads them and replaces them with the results the manifest's donation pairs name. The backend donates every `state:` buffer to the execution, so a body that aliases it to its result (as a Tlaloc serving artifact's does) updates it in place. |

The pools belong to a model instance, so the generated model has one
instance, and requests to it run one at a time. The client allocates pages,
as for any paged KV cache: it chooses the pages each sequence uses in
`blockTables` and the slot each new token is written to in `slotMapping`. The
backend never clears a page; attention reads only the first `seqLens`
positions of a sequence, so a page that is reused is overwritten before it is
read. The `kv_block_size` and `kv_num_blocks` parameters of the generated
configuration give a client the page geometry; the backend ignores them.

## Sequence mode

A model whose `config.pbtxt` has the parameter `serving_manifest` (the
manifest file in the version directory) runs in sequence mode. The manifest
names every entry (body file, kind, batch and context bucket, signature with
slot roles), the staged weights and the KV pool geometry; the backend reads it
at load, checks each body's signature against it, compiles every entry and
uploads the weights. `TritonModelRepository` writes this configuration:

```
max_batch_size: 4                      # the artifact's largest decode batch
input  [ { name: "TOKENS" data_type: TYPE_INT32 dims: [ -1 ] allow_ragged_batch: true } ]
output [ { name: "LOGITS" data_type: TYPE_FP32 dims: [ 32000 ] },
         { name: "KV_PAGES" data_type: TYPE_INT32 dims: [ 2 ] } ]
sequence_batching {
  max_sequence_idle_microseconds: 60000000
  control_input [
    { name: "START"  control [ { kind: CONTROL_SEQUENCE_START  int32_false_true: [ 0, 1 ] } ] },
    { name: "END"    control [ { kind: CONTROL_SEQUENCE_END    int32_false_true: [ 0, 1 ] } ] },
    { name: "CORRID" control [ { kind: CONTROL_SEQUENCE_CORRID data_type: TYPE_UINT64 } ] }
  ]
  oldest { max_candidate_sequences: 63 preferred_batch_size: [ 4 ] max_queue_delay_microseconds: 0 }
}
parameters: { key: "serving_manifest" value: { string_value: "tlaloc-serving.json" } }
```

### Protocol

Each request is one step of one sequence: `TOKENS` `[1, n]` with the
sequence's correlation ID (`sequence_id` in tritonclient), and the
`sequence_start` / `sequence_end` flags. The response is `LOGITS` `[1, vocab]`,
the logits of the last token sent, and, when the client asks for it,
`KV_PAGES` `[1, 2]`: the pages the sequence holds after the request in the KV
pool and in the windowed KV pool (0 without one). The output is optional in
`config.pbtxt`; a model written before it existed serves without it. A client
may ask for `NEXT_TOKEN` `[1, 1]` instead of `LOGITS`: the greedy next token
(the first index of the largest logit), chosen by the backend, so a greedy
client is not sent the logits row (1 MB a token for a 248K vocabulary).
`sequence_client.py`'s `step_token` does that. A
client

1. sends the prompt with START. The backend runs it as one prefill call;
2. sends each chosen token alone. Each is a decode step;
3. sets END on the request that produces the last token it wants, or sends
   an empty `TOKENS` `[1, 0]` with END, which runs nothing and returns no
   `LOGITS`.

A request with several tokens appends them to the sequence, so a later
request may also carry several tokens (it runs as another prefill chunk,
starting at the sequence's current length). `sequence_client.py` implements
this; `generate_client.py` adds a tokenizer.

### What the backend does per request

- **START** creates the sequence's state (its page list and length). A START
  for a correlation ID that already has state restarts it and frees its old
  pages.
- **Pages** are allocated as the sequence grows, `ceil(length / blockSize)`
  of them, lowest page first, from the instance's pool. Page 0 is never
  allocated: padding rows of a batch point at it. The block table and slot of
  every token are derived from the page list; the client never sends them.
- **Windowed pages.** An artifact with a windowed KV pool (`tlaloc-serving-v3`;
  a model with sliding-window layers, such as Muse Glimmer) has a second page
  pool for those layers. Each sequence holds a ring of at most `ringPages`
  of its pages, taken as it grows: logical block `b` is on the ring's page
  `b % ringPages`, so a position is written over the one `ringPages *
  blockSize` positions before it, which has left the window. The window
  block table and slots are derived from the ring. The load log states the
  pool (`windowed KV pool for 39 sliding layers (window 2048): 64 pages, a
  ring of at most 8 pages per sequence`). See
  [SERVING_ARCHITECTURE.md](../docs/SERVING_ARCHITECTURE.md#sliding-window-layers-the-windowed-kv-pool).
- **Linear-attention state.** An artifact with Gated DeltaNet layers
  (`tlaloc-serving-v4`, the Qwen3.5 family) keeps, for each of them, a conv and a
  recurrent state per sequence instead of KV pages. A sequence takes one state
  slot at START and holds it to its end; every call is handed each row's slot.
  A sequence that starts at position 0 starts from zero state, so a slot is
  reused without clearing. Slots are reclaimed from idle sequences as pages are,
  a sequence that finds none free is refused (UNAVAILABLE), and
  `max_candidate_sequences` is no more than the slots.
- **Speculative decoding.** An artifact exported with `-PmtpDraftTokens=k`
  (`tlaloc-serving-v5`, the Qwen3.5 family's MTP head) has no `LOGITS` output.
  Its `NEXT_TOKENS` output (INT32 `[1, n]`) holds the tokens a request emits;
  `NEXT_TOKEN` is the last of them.
  - **A prompt** (a request of several tokens, or of one token that is not the
    pending one) is prefilled. It emits one token, the greedy next token, which
    becomes the sequence's pending token, and the head drafts `k` more.
  - **A decode step** is a request carrying the pending token alone. The step
    verifies it with the drafts in one call and emits the drafts the target
    agrees with and the target's own token after them: 1 to `k + 1` tokens.
    The client sends the last of them back. `sequence_client.py`'s
    `step_tokens` does this.
  - **Slots and pages:** a sequence holds `k + 2` state slots, since a step
    writes the state after each of its tokens to a slot of its own and the next
    step reads the accepted one. Pages are reserved for the positions the head
    writes past a call.
- **Entry selection.** A request of `n > 1` tokens runs on the smallest
  prefill entry whose context covers the sequence's length after it, one call,
  with the tokens right-aligned in the chunk. An artifact exported with
  `-PprefillChunk=N` has prefill entries of at most `N` tokens per sequence
  (`tokensPerSeq` below the context); a longer request runs as calls of at
  most `N` tokens, each on the smallest entry whose context holds its last
  position, and the load log says so (`at most 512 tokens per sequence in a
  prefill call`). `-PprefillChunk=128,2048` exports entries of both sizes (the
  smaller ones' ids end in `_t128`): a call runs on the entry of the fewest
  tokens that holds it, so a short follow-up does not run a whole chunk. With a windowed KV pool, a
  request is first split into calls of at most `ringPages * blockSize -
  min(start, window - 1)` tokens, the most the ring holds while the call's
  first row still reads its window; each call runs on its own prefill entry
  (or as a decode step when it is one token). A one-token request runs on a
  decode entry. All one-token requests in one batch (different
  sequences, which the oldest strategy guarantees) run as one call on the
  smallest decode entry whose batch and context cover them, with padding rows
  for the rest of the batch. Without a prefill entry (a v1 artifact, or
  `-Pprefill=false`), the tokens of a longer request run as decode steps.
- **Batching.** Triton hands each request over as soon as it arrives (the
  export sets `max_queue_delay_microseconds` to 0), and the backend forms
  the batches on a thread of its own. A batch runs once every sequence that
  took a decode step in the previous batch, and is still held, has its next
  request in; a lone sequence therefore never waits, and sequences decoding
  together stay together however far apart their clients' replies arrive.
  If one of them does not come back within `cohort_wait_microseconds`
  (default 20000, counted from when the batch could first have run), the
  batch runs without it and the log says so (`a batch of 1 request(s) ran
  without sequence(s) 90001 90002 90003 of the previous batch after waiting
  20000 us`: three clients that stopped stepping without END).
  A prompt waits up to `prompt_wait_microseconds` (default 2000) after it
  arrived for prompts sent with it. The time a request waits in the backend
  is part of Triton's compute input time for it. With the model parameter
  `backend_batching` set to `false` the batches are the ones Triton's
  sequence batcher forms, after its queue delay; the load log states which.
  `sequence_checks.py --batching` checks the policy: one sequence alone
  waits under 0.8 ms a step before its execution starts, and four sequences
  whose clients pause 0 to 5 ms before each step run 60 steps in at most 17
  executions; batched by Triton with a 1 ms queue delay, the same checks
  fail (1.2 ms of waiting, 35 executions).
- **Per step on the host.** The integer inputs of a step (token ids,
  positions, block tables, lengths, slots) are written into pinned host
  memory mapped into the GPU, which the executions read in place through
  PJRT views made on each entry's first run, with no copy
  (`pack_step_inputs`, default `true`; `false` uploads each input as its own
  buffer). The logits copy is asked for as soon as the execution is queued,
  so it follows the execution on the device and the host wakes once
  (`overlap_logits_copy`, default `true`). The first 100 runs of each entry
  log the host time of a run (`decode_b1_c64 host time per run, median of
  100 runs: inputs 94 us, launch 963 us, until the logits are on the host
  8365 us, after 91 us`, Qwen3-0.6B bf16 on a GPU shared with the desktop).
- **Prompts together.** The requests of several tokens in one batch
  (several sequences started at once) run in rounds: each round takes every
  request's next call (all its tokens, or what the windowed ring holds), and
  the calls whose smallest covering prefill entry has the same context run
  as one call on the smallest prefill entry whose batch holds them, each
  right-aligned in its own row. Calls of different context buckets are not
  merged: a call computes every row at its entry's context. The first call
  of each prefill entry with several prompts is logged
  (`prefill_b4_c64 ran 4 sequence(s) in 39170 us`). An artifact exported with
  `-PprefillMaxBatch=1` has batch-1 prefill entries only and runs one call
  per prompt.
- **KV pools** are the instance's, on the device, zeroed at load. Every
  execution is handed them (donated) and returns the updated pools; the
  artifact's bodies alias each `KV_POOL_OUT` to its `KV_POOL_IN`
  (`tf.aliasing_output` on the parameter), so XLA writes each pool where it
  is and nothing is copied. After every run the backend compares each pool's
  device address with the one it had before the run, logs the result for
  the first run of each entry (`decode_b1_c64 updated the 44 KV pools in
  place ...`, or a warning that they were written to new device memory) and
  after 100 runs (`in 100 runs the 44 KV pools were updated in place 100
  times and copied 0 times`); later runs are not checked. The model parameter `donate_kv_pools` set to
  `false` hands the executions the pools without donating them, so XLA
  copies them first; it exists as the control for that check. An artifact
  exported before its bodies carried the alias is served with the pools
  copied, and the log says so. If a failed execution takes the donated pools
  with it, the rest of that batch is refused, every sequence is freed (a
  later request must START again) and new pools are zeroed. A request of
  several calls (a prompt split by the chunk or the windowed ring) that
  fails after one of its calls ran frees its sequence and says so: part of
  its tokens are in the KV, and the ring may have written over positions its
  first call read, so neither continuing nor sending it again would be
  right.
- **END** frees the sequence's pages (and its ring) after its step.
- **Pages run short.** When a request needs more pages than are free, the
  backend reclaims pages from sequences Triton has already ended without
  telling it (the idle rule below), the least recently active first, and
  only until the request's pages are free (`reclaimed sequence n (the least
  recently active, idle ... us, past the limit of ... us) for sequence m`).
  Sequences Triton ended with END have already given their pages back. It
  never takes pages from a sequence Triton still holds: if reclaiming is not
  enough, the request is refused by name (`KV page pool exhausted`,
  UNAVAILABLE), and the refusal lists the sequences holding pages (the six
  largest, with their pages, tokens and idle time). A sequence refused
  mid-generation keeps its pages and KV (`Sequence n keeps its 3 pages and
  the KV of its 48 tokens: send this request again once pages are free, or
  end sequence n`), so the client can send the same request again later; a
  START that is refused leaves nothing. Live sequences are not preempted:
  there is no swapping of a sequence's KV to host memory and no
  recomputation of it later.
- **Idle timeout.** In the oldest strategy Triton ends a sequence that has
  been idle longer than `max_sequence_idle_microseconds` without telling the
  backend (its log says `Reaper: CORRID n: max sequence idle exceeded`). The
  backend therefore frees such sequences' pages itself, when a sequence needs
  more pages than are free. Triton's clock starts at a request's arrival and
  a request waiting in Triton's queue keeps its sequence alive, while the
  backend only sees when each sequence last ran. So the backend frees a
  sequence only when it has no request in the current batch and has not run
  for twice the timeout plus the longest wait the oldest strategy allows a
  queued request: one execution (the longest seen so far) per
  `max_batch_size` live sequences. Every request Triton hands over counts as
  activity, including one the backend refuses (Triton restarts its own timer
  for it too). With the timeout alone the backend freed
  sequences Triton still held once steps queued behind other sequences;
  `sequence_checks.py --queued` (a 200 ms timeout, 12 to 32 concurrent
  sequences) catches that, and fails against a backend without the rule. The
  next request of a sequence Triton has ended is refused by Triton unless it
  carries START.

Refused by name, with the server staying up: a START (or growth) that needs
more pages than are free after reclaiming (`KV page pool exhausted: sequence n
needs k more page(s) of 16 tokens to grow from 48 to 49 tokens, and 0 of 63
are free (...). ... Or export the artifact with more pages (numBlocks)`, or
`windowed KV page pool exhausted: ...`); a sequence that would pass the
largest compiled context; a step for a sequence the backend holds nothing
for; a token id outside the vocabulary; a token id the manifest refuses
(`token 3 of the request is 200092, the model's image_token_id placeholder;
...`), which leaves the sequence as it was; a request with no tokens and no
END. A refused request that carries START still starts a new sequence in
Triton, which ends any earlier sequence with that ID, so the backend frees
the earlier sequence's KV. A refused START leaves no
state behind; Triton still counts the sequence as live until END or the idle
timeout, so a client should send END for it.

### Why one model with an entry chosen per request

Triton's model is the unit that owns instances, and an instance is the unit
that owns device state. Prefill and decode must write and read the same KV
pools, so they have to be one model: two Triton models (a prefill model and a
decode model) would be two sets of instances with no shared state, and the
sequence batcher routes a correlation ID within one model only. Within the
model, the number of tokens in the request already says which entry runs, so
there is no entry-name input for a client to get wrong.

### Why the oldest strategy

The backend keys state by correlation ID, not by batch slot, so the direct
strategy's slot pinning buys nothing; and the oldest strategy forms batches
from the next request of several different sequences, which is exactly a
batched decode step. The direct strategy is refused at load by name.

### Measured on the GB10

From `verify.sh` (TinyLlama-1.1B, f32 weights, context 64, over HTTP): the
model loads in about 17 s (four XLA compiles of 3.0 to 5.7 s, 2.0 s to upload
4196 MiB of weights); the 44 KV pools (44 MiB, 63 usable pages of 16 tokens)
are zeroed per instance. A 6-token prompt's prefill takes about 26 ms (the
64-token prefill entry, padded) where the same prompt sent as six decode steps
takes about 6 x 24 ms; a decode step of one sequence takes about 24 ms (41
tokens/s). Four sequences decoding concurrently take about 42 ms per step
each, and together produce about 82 to 87 tokens/s, about twice one sequence's
rate; 48 requests ran in 22 or 23 executions. The Python driver in
`docs/SERVING_RUNBOOK.md` takes 1.35 s per step, because it copies every KV
pool to the host and back. These are single measurements, not a benchmark.

Prompts sent together (`prefill_checks.py`, HTTP, the median of 5 rounds, each
round one execution; the sequences are ended after the timed prefill):

| | 1 prompt | 2 together | 2 one after the other | 4 together | 4 one after the other |
|---|---|---|---|---|---|
| TinyLlama-1.1B (prompts of 6, 9, 5, 9 tokens) | 25 ms | 29 ms | 54 ms | 35 ms (113 prompts/s) | 112 ms (36 prompts/s) |
| Qwen3-0.6B (5, 24, 11, 27 tokens) | 23 ms | 35 ms | 52 ms | 61 ms (66 prompts/s) | 101 ms (40 prompts/s) |

Every entry is padded to 64 tokens a row, so a batch-4 call computes 256
token rows; it still reads the weights once and dispatches once. A prompt
whose request also carries END was not batched with the others: four such
prompts ran in two executions, two in two, in every round measured.

KV pools in place against copied (`donate_kv_pools: false`), measured with
another process keeping the GPU 95% busy, the two servers alternated three
times, server-side time of a batch-1 decode step over 235 steps each:
Qwen3-0.6B (56 pools, 224 MiB) takes 11.5 ms at its fastest in place against
16.2 ms copied (medians 13.1 to 23.8 ms against 25.8 to 32.5 ms); TinyLlama
(44 pools, 44 MiB) 18.3 ms against 19.3 ms at its fastest, with medians that
the other process's load makes too noisy to separate. The timings above and
below were taken before the pools were updated in place, on an idle GPU.

Qwen3-0.6B (f32 weights, context 64, gRPC, medians of 5 to 10 runs of 16
tokens): the model loads in about 20 s (four XLA compiles of 3.5 to 6.6 s,
1.2 s to upload 2867 MiB of weights, the tied head then staged as a copy of the embedding table; 2273 MiB without it). Prefill takes about 25 ms for the 5-token
and for the 24-token prompt alike, since both run the padded 64-token prefill
entry. A decode step of one sequence takes about 19 ms (about 52 tokens/s).

Muse Glimmer 30B (bf16 weights, the long-context export `verify.sh` serves:
contexts 512 to 32,768, decode batches 1, 2 and 4, 512-token prefill calls):
the model loads in about 3.5 minutes (16 XLA compiles in 2 minutes, from 3.7
s to 13.7 s each, and 82 s to upload 53128 MiB of weights); the largest
entry, prefill at 32,768, needs 453 MiB of temporary memory (4125 MiB before
prefill attended block by block), and the KV pools take 4109 MiB. With
`context_bench.py` (gRPC, four sequences per context, prompts of seeded
random ids, medians over 16 steps per sequence), measured with the GPU
shared and before sliding layers read their ring:

| Context bucket | Prompt | Prefill | 1 sequence | 2 sequences | 4 sequences |
|---|---|---|---|---|---|
| 512 | 432 tokens | 0.56 s | 265 ms a token | 267 ms a step, 7.5 tokens/s | 238 ms a step, 16.7 tokens/s |
| 2,048 | 1,968 | 2.6 s | 268 ms | 272 ms, 7.3 tokens/s | 248 ms, 16.0 tokens/s |
| 8,192 | 8,112 | 17.0 s | 275 ms | 286 ms, 7.0 tokens/s | 279 ms, 14.3 tokens/s |
| 32,768 | 32,688 | 230 s | 317 ms | 352 ms, 5.7 tokens/s | 416 ms, 9.6 tokens/s |

A decode step reads the 52 GiB of weights once whatever the batch, so four
sequences cost about what one does until attention over 32,768 positions
shows. These runs were batched by Triton after a 20 ms queue delay: with
1 ms, the steps of two and four sequences (each client receiving an 800 KB
logits vector) arrived more than 1 ms apart and ran 32 steps in 31
executions and 64 in 35; with 20 ms every round ran as one execution, and a
lone sequence paid about 24 ms a token for the wait (265 ms against 241 ms).
The backend now forms the batches itself and the export sets no delay (see
"Batching" above, and `docs/SERVING_ARCHITECTURE.md`, section 6). A 512-token prefill
call took about 0.55 s at context 512 and about 4.5 s at 32,768, because
every call scored its tokens against every position of its entry's context,
the sliding layers' included. Now a sliding layer scores its ring and a call
attends block by block up to its last position: on an idle GPU a call at
positions 8,704 takes 0.88 s instead of 4.49 s, one at 31,744 takes 1.39 s
instead of 4.50 s, and a 31,744-token prompt 62.6 s instead of 222.8 s
(`docs/SERVING_ARCHITECTURE.md`, section 6). Under a PJRT memory fraction of 0.55 the
machine peaked at 87 to 90 GiB in use during these runs, and at 96 GiB in
the `verify.sh` run. Those runs shared the GPU with another process; the
numbers below did not.

### Profiling

`profile.sh` serves one model on ports 8020 to 8022 and times workloads on
it: `decode:PREFIX:STEPS` (single-token steps after a PREFIX-token prompt)
and `prefill:PREFIX:CHUNKS` (512-token requests, each one prefill call at
the bucket that holds it). It first checks that the GPU is idle (nvidia-smi
sampled for 10 s, before loading and again before timing). If the GPU is
busy it waits up to 45 minutes, then labels every number contended. It
reports the client's time per request (median of three runs) with Triton's
compute and queue time for it:

```bash
TRITON_CLIENT_PYTHON=/path/to/venv/bin/python MAX_ID=200000 REFUSED=200091,200092 \
  triton/profile.sh triton/build/muse-glimmer muse decode:400:16 prefill:8704:3
```

With `MODE=nsys XLA_DUMP=1` the server runs under the Triton container's own
`nsys launch`, each workload's first run is captured, and XLA writes its
optimized HLO. `profile_report.py` then ties every kernel to the HLO
instruction that launched it and sums kernel time by model component (the
weight each matmul reads, KV writes and gathers, attention dots, softmax),
with the bytes and FLOPs behind each. It also splits a step into host time,
input copies and launch, kernels, and the wait for the logits copy, and
lists each dot's operand types and the largest intermediate buffers.

Measured with it on an idle GPU (gRPC, one sequence, medians of three runs):

| | Client | Triton compute | Queue |
|---|---|---|---|
| TinyLlama-1.1B decode step (f32) | 22.9 ms | 20.5 ms | 1.2 ms |
| Qwen3-0.6B decode step (f32) | 17.1 ms | 13.9 ms | 1.2 ms |
| Qwen3-0.6B decode step (bf16) | 11.3 ms | 8.6 ms | 1.2 ms |
| Muse Glimmer decode step, context 512 | 268 ms | 246 ms | 20.5 ms |
| Muse Glimmer decode step, context 8,192 | 277 ms | 255 ms | 20.6 ms |
| Muse Glimmer 512-token prefill call, context 2,048 | 722 ms | 699 ms | 20.5 ms |
| same, context 8,192 | 1,301 ms | 1,277 ms | 20.6 ms |
| same, context 32,768 | 4,598 ms | 4,574 ms | 20.4 ms |

Where that time goes, kernel by kernel, is in
[SERVING_ARCHITECTURE.md](../docs/SERVING_ARCHITECTURE.md#6-where-the-time-goes).
Those runs were batched by Triton after a queue delay (1 ms, and 20 ms for
Muse Glimmer), which is the "Queue" column. With the backend batching and no
delay the queue is about 0.3 ms. On an idle GPU, the old and new ways served
in turn for three rounds: Qwen3-0.6B bf16 10.70 ms a token against 11.33 ms,
f32 15.39 against 16.79, TinyLlama 21.66 against 22.86, Muse Glimmer 244.3
against 265.9 (tables in the same section, with the first measurements, taken
on a GPU the desktop kept 15 to 22% busy).

## Bounded mode

A bounded program is a Tlaloc program whose inputs have axes typed
`Bounded<B>` in Kotlin: a size known only at run time, at most `B.max`
([docs/design/bounded-dims.md](../docs/design/bounded-dims.md)).
`BoundedProgramExport` writes one StableHLO body per bucket of each bound and a
manifest, `tlaloc-bounded.json`; `TritonModelRepository.write` turns that
directory into a model whose `config.pbtxt` names it (`bounded_manifest`):

```
name: "bounded_mean"
backend: "tlaloc"
max_batch_size: 0
input [
  { name: "x0" data_type: TYPE_FP32 dims: [ -1, 4 ] }
]
output [
  { name: "y0" data_type: TYPE_FP32 dims: [ 4 ] }
]
instance_group [ { kind: KIND_GPU count: 1 gpus: [ 0 ] } ]
parameters: { key: "bounded_manifest" value: { string_value: "tlaloc-bounded.json" } }
```

- The config's inputs are the manifest's `DATA` inputs, by name, with `-1` for
  each bounded axis; its one output is the manifest's output. Anything else is
  refused at load by name, and so is `max_batch_size` other than 0 (a batch
  axis, if the program has one, is one of its bounded axes).
- At load the backend checks every body's signature against the manifest at
  that body's buckets and compiles every body once.
- Per request it reads each bound's size from the inputs' shapes (every axis
  of one bound must have one size), takes the smallest bucket of each bound
  that holds it, pads the inputs with the manifest's padding value, fills the
  program's `VALID_MASK` inputs (1 for a real position) and `VALID_LENGTH`
  inputs (the real size), runs, and returns the output sliced to the real
  sizes. A size outside `1..max` is refused by name.
- Tensors go through the host, and each request runs alone.
- The exporter refuses a program whose padded result differs from its
  exact-size result in the interpreter, so a program that reads padding
  without masking it never reaches the backend.

## Model configuration

A model uses the backend with `backend: "tlaloc"`. Without the `arguments`
and `results` parameters, inputs and outputs in `config.pbtxt` map to the
entry function's arguments and results by position: the first `input` is the
first argument, the first `output` the first result.

| Parameter | Required | Meaning |
|---|---|---|
| `artifact` | yes, except in bounded and sequence mode | StableHLO text file(s) inside the model version directory (`<model>/<version>/`), named relative to it; an absolute path or a `..` component is refused. A comma-separated list declares shape buckets: each file is compiled at load, and a request runs on the file whose input shapes it matches exactly. |
| `entry` | no | The function to serve. Default: `main` if the file has one, otherwise the first `func.func` in the file. |
| `pjrt_plugin_path` | no | PJRT plugin `.so` for this model. Overrides the backend setting below. |
| `bounded_manifest` | no | A bounded-program manifest (`tlaloc-bounded.json`) in the version directory: the model runs in bounded mode (see "Bounded mode"). Cannot be combined with `artifact`, `entry`, `arguments`, `results` or `serving_manifest`. |
| `arguments` | no | One item per argument of the entry function, in order, comma-separated: `input:<name>` (a config input), `weight:<file>` (a raw little-endian file in the version directory, dense and row-major with no header, uploaded once at load; its size must match the argument's type), or `state:<name>` (a buffer the model instance keeps on the device, zero at start). Every config input must appear exactly once. |
| `results` | no | One item per result, in order: `output:<name>` (a config output) or `state:<name>` (replaces that state after the request runs). Every config output must appear exactly once, and every state read by `arguments` must be written by exactly one result. |
| `zero_copy` | no | `true` (default) or `false`. With `false`, tensors in GPU memory go through the host; see "GPU memory in and out". |
| `donate_kv_pools` | no | Sequence mode only. `true` (default) or `false`. With `false` the executions are not handed the KV pools to write in place, so XLA copies them every run; see "Sequence mode". |
| `backend_batching` | no | Sequence mode only. `true` (default): the backend forms the batches, waiting only for the sequences that decoded in its previous batch; `false`: the batches Triton's sequence batcher forms. See "Batching" under "Sequence mode". |
| `cohort_wait_microseconds` | no | Sequence mode with backend batching. The most a batch waits for the sequences of the previous one (default 20000). |
| `prompt_wait_microseconds` | no | Sequence mode with backend batching. The most a prompt waits for prompts sent with it (default 2000). |
| `pack_step_inputs` | no | Sequence mode only. `true` (default): a step's integer inputs are written into mapped pinned host memory the executions read in place; `false`: one upload per input. |
| `overlap_logits_copy` | no | Sequence mode only. `true` (default): the logits copy is queued behind the execution; `false`: it is asked for after the host sees the execution finish. |

At load the backend reads each artifact's entry signature and checks it
against `config.pbtxt`: the number of inputs and outputs, each data type, and
each shape (`-1` in `config.pbtxt` matches any size). Weight and state
arguments must have the same type in every artifact of the model, and a
state result the same type as the state argument it replaces. A mismatch
refuses the model with a message naming the file, the tensor and both types.

Triton has no rank-0 tensors in a model with `max_batch_size: 0`. Declare a
rank-0 argument or result as `dims: [ 1 ]`; the backend maps one to the
other.

Data types: FP32, FP64, FP16, BF16, INT8, INT32, INT64, UINT8 and BOOL, all
run by `verify.sh` over HTTP and gRPC. Any other type is refused by name at
load. Tlaloc emits FP32, FP64, BF16, INT32, INT64 and BOOL; FP16, INT8 and
UINT8 are run from a hand-written module (`dtypes_small`).

### Batching (`max_batch_size > 0`)

With `max_batch_size > 0` the leading dimension of every tensor is the batch.
Every artifact of the model takes one batch size `B` as dim 0 of each input
and each output (the backend checks this at load), and the largest `B` must
be at least `max_batch_size`. Rows must be independent: row `i` of every
output depends only on row `i` of the inputs. That is Triton's contract for
batching, and it is why `state:` arguments are refused with
`max_batch_size > 0`.

When Triton hands the backend several requests in one call (the dynamic
batcher, `dynamic_batching { ... }`), the backend keys each request by the
shape of its rows (every input's dimensions after the batch), groups the
requests of one key in arrival order, as many rows per group as the largest
artifact of that shape takes, concatenates each group along dim 0 on the
host, runs the artifact with the smallest `B` that holds the group's rows,
with the remaining rows zero, and gives each request its own rows of every
output. Requests of different shapes reach one batch only with
`allow_ragged_batch: true` on the input; a batch of two widths then runs as
two executions. The model parameter `group_by_shape: "false"` groups only
consecutive requests of one shape (the example `ragged_consecutive`, which
`verify.sh` uses as the control: 216 requests of two widths ran in 74 to 76
executions grouped by shape and 119 to 134 grouped consecutively, with the
same rows). A single request whose rows fill a bucket exactly runs on its own
tensors (and can read GPU memory in place, below). Without
`dynamic_batching` each request runs alone, padded to the smallest bucket
that holds it. Triton's statistics count one execution per group.

Export one artifact per batch size you want to run at; more sizes mean more
compiles at load and less padding per call. The example `grad_batched`
compiles batch 1, 2, 4 and 8.

### GPU memory in and out

A client can put input and output tensors in CUDA shared memory
(`tritonclient.utils.cuda_shared_memory`, registered with
`register_cuda_shared_memory`). Triton then hands the backend GPU pointers.

- An input in one buffer on the instance's GPU, 16-byte aligned, is wrapped
  in a PJRT buffer with `PJRT_Client_CreateViewOfDeviceBuffer` and read in
  place: no copy through the host. The argument is marked non-donatable, so
  XLA never writes into the client's memory.
- An output Triton places in GPU memory on the instance's GPU is copied
  device to device (`cudaMemcpy` from `PJRT_Buffer_OpaqueDeviceMemoryDataPointer`,
  holding an external reference for the copy) when PJRT stores it densely in
  exactly the tensor's byte size, which is the case for the XLA GPU plugin's
  default layout.
- Anything else (host memory, several buffers, another GPU, unaligned memory,
  `zero_copy: "false"`, a plugin without these functions, a batched group)
  goes through the host as before.

The server log names the path the first time a model uses it, per tensor:
`input 'X' is read in place from GPU memory (no host copy)`, `output 'Y' is
copied device to device into GPU memory`, or `... goes through the host:
<why>`. `run_server.sh` starts the container with `--ipc=host`, which CUDA
IPC between the client's process and the server needs.

### Measured on the GB10 (batching, GPU memory)

`perf_client.py`, run by `verify.sh` over gRPC; the ranges are two runs. These
are not a benchmark.

| | Requests/s | Executions for 4000 requests |
|---|---|---|
| `grad_unbatched`, one-row requests, 32 in flight | 1,440 to 1,590 | 4000 |
| `grad_batched`, the same requests | 7,700 to 7,810 | 760 to 830 (4.8 to 5.2 requests each) |

| `large_io`, 16 MiB in and 16 MiB out | Client latency (median) | Server time per request |
|---|---|---|
| CUDA shared memory, read and written in place | 1.3 to 1.4 ms | 0.5 to 0.6 ms |
| CUDA shared memory, through the host (`large_io_host`) | 5.2 to 5.7 ms | 4.5 to 4.8 ms (0.7 input, 1.5 to 1.7 compute including the upload, 2.2 to 2.3 output) |
| gRPC bytes, no shared memory | 61 to 62 ms | 3.9 to 4.1 ms |

On this machine the host round trip that reading and writing in place removes
is about 4 ms of the server's 4.5 to 4.8 ms for 32 MiB of traffic. The GB10's GPU
memory is the system RAM; the saving on a discrete GPU, where these copies
cross PCIe, has not been measured. Most of the 62 ms without
shared memory is moving 32 MiB through gRPC.

### GPUs

The backend creates one PJRT client per plugin and GPU, with the plugin's
`visible_devices` option set to that GPU's CUDA ordinal, and checks that the
client's one device has that ordinal (`PJRT_Device_LocalHardwareId`). At load
a model compiles its artifacts and uploads its weights once for every GPU its
`instance_group` names (`KIND_CPU` and `KIND_MODEL` instances run on GPU 0),
and each instance runs on its own GPU's executables. In sequence mode each GPU
gets its own copy of the model, and each instance its own pools, as before.

A `gpus` entry the machine does not have is refused by Triton itself before
the backend is called (`instance group ... specifies invalid or unsupported
gpu id 7`); `verify.sh` checks that message. If the plugin were asked for such
a GPU anyway, the backend refuses the client by name: the XLA CUDA plugin
0.10.0 does not fail on `visible_devices: [7]` on a one-GPU machine, it gives
back GPU 0, and the ordinal check catches that (`the PJRT client created for
GPU 7 runs on GPU 0 instead`); `pjrt_device_test` certifies it.

🧪 Instances on a GPU other than 0 have not run: the GB10 has one GPU. What
ran is GPU 0 selected through `visible_devices` and the refusal above.

## Backend configuration and GPU memory

| `--backend-config=tlaloc,<key>=<value>` | Environment variable | Default |
|---|---|---|
| `pjrt-plugin-path` | `TLALOC_PJRT_PLUGIN_PATH` | none; a model without a plugin path is refused by name |
| `memory-fraction` | `TLALOC_PJRT_MEMORY_FRACTION` | `0.3` |
| `preallocate` | `TLALOC_PJRT_PREALLOCATE` | `false` |

The backend-config value wins over the environment variable. For the plugin
path, a model's `pjrt_plugin_path` parameter wins over both.

Every PJRT client is created with `memory_fraction` and `preallocate` as
client options. Without them the XLA CUDA plugin reserves 75% of GPU memory
when the client is created. On a GPU whose memory is the system RAM (GB10)
that is 75% of the machine's memory, and a few such clients can take the
machine down. The default here is 0.3 rather than Tlaloc's usual 0.5 because
Triton itself also allocates GPU memory in the same process.

The backend creates one PJRT client per plugin path and GPU and shares it
between all models that use that plugin on that GPU. It is created when the
first such model loads and destroyed when the last one unloads. Each model's
executables are destroyed when the model unloads. `memory_fraction` applies
to each client, that is to each GPU.

`run_server.sh` starts the container with `--ipc=host` (for CUDA shared
memory clients) and mounts the model repository at `/models`, the backend at
`/opt/tritonserver/backends/tlaloc`, the plugin at
`/opt/pjrt/xla_cuda_plugin.so` (all read-only) and the host's `/dev/shm` (for
system shared memory clients), sets `TLALOC_PJRT_PLUGIN_PATH`, and passes any
extra arguments to `tritonserver`. `MODEL_REPOSITORY`, `PJRT_PLUGIN`,
`HTTP_PORT`, `GRPC_PORT`, `METRICS_PORT` and `CONTAINER_NAME` override its
defaults.

## Limits

- StableHLO text only. MLIR bytecode is refused at load, because the backend
  reads the text to find and rename the entry function and to check its
  signature.
- Static shapes only. A `?` dimension in the entry signature is refused;
  export one artifact per shape and list them in `artifact`, or export a
  bounded program and serve it in bounded mode.
- More than one GPU has not run (see "GPUs").
- Host-memory inputs are copied to the device and outputs back to host memory
  for each request. Requests batched together are concatenated on the host,
  including those whose tensors are in GPU memory. Sequence mode reads its
  token ids from host memory (they are a few bytes) and writes logits through
  the host.
- A model that fails to load stays in the repository index as
  `UNAVAILABLE`. With Triton's default `--strict-readiness=true` the server
  then answers `/v2/health/ready` with 400 until that model loads, while
  `/v2/health/live` and the other models keep serving. Pass
  `--strict-readiness=false` if readiness should track only the server.
- No optional inputs, no string tensors, no decoupled (streaming) responses.
- Sequence mode: one instance per model (the pools are its state); pages are
  allocated as a sequence grows and live sequences are not preempted, so a
  sequence that needs a page when none is free (and none can be reclaimed
  from a sequence Triton has ended) is refused mid-generation, keeping its
  KV, rather than paused; the client sends the request again. Only the oldest strategy. Prompts in one batch share a
  prefill call only when they fall in one context bucket, and
  `prompt_wait_microseconds` (2 ms by default) is all the time the backend
  waits for them. Prompts whose request also carries END were not batched together
  by Triton in the measurements above. Sampling is the client's (the backend
  returns logits).
- Client mode: state is per model instance and is not tied to a Triton
  sequence ID. The client that allocates pages must be the only client of
  that model, or the clients must agree on the pages.
- The weights and KV pools are on the device in the artifact's dtypes: f32
  by default for Llama and Qwen3 (TinyLlama: 4.1 GiB of weights; Qwen3-0.6B:
  2.2 GiB), bf16 with `-PweightDType=bf16` (Qwen3-0.6B: 1.1 GiB) and for
  Muse Glimmer. The KV pools are f32.

## Generation in the backend

A speculative model (an artifact exported with `-PmtpDraftTokens`) can generate without a request
per step. A request with the parameter `max_tokens` (and optionally `end_tokens`, comma-separated
ids) appends its tokens, and the backend then steps the sequence itself, in the same batches as the
other sequences, until it has emitted `max_tokens` tokens (a step may emit up to the number of drafts
more) or one of `end_tokens`. Without streaming the response carries every generated token in
`NEXT_TOKENS`. A model exported with `-Pdecoupled=true` (`model_transaction_policy { decoupled: true }`)
answers once per step over a gRPC stream; the last response is marked final.
`sequence_client.py`'s `generate_tokens` and `stream_tokens` send them.

Generation steps are chained. A verify step's program also returns the next verify step's inputs:
- the token after the accepted drafts and the new drafts;
- their positions and the sequence lengths;
- the slot mapping, made through the step's own block tables;
- the state slot of the last accepted token and the next write slots.

The backend issues the next step with those buffers before it has read the current one, then reads
the current step's tokens and streams them. The host's time per step (about 3.3 ms of
`PJRT_LoadedExecutable_Execute` for a Qwen3.6-35B-A3B entry) then overlaps the GPU's work.

A step is chained only if all of these hold:
- the batch is the same generations;
- each continues past the step whatever it emits;
- no request is waiting;
- the step's block tables reach the next step's tokens.

A generation's verify step takes pages two steps ahead for that.

A chained step that the next batch does not take is dropped. Nothing of a sequence's own state is
lost: it wrote only past each sequence's length and into slots other than its state slot.
`TLALOC_CHAIN_STEPS=0` (`run_server.sh` passes it) turns chaining off. An artifact exported
before the chained inputs existed runs unchained.

Qwen3.6-35B-A3B NVFP4 with 3 drafts, streamed, tokens/s per user, unchained and chained on the same
build:

| | unchained | chained |
|---|---|---|
| 1 user, 2K, first turn | 95.4 | 111.3 |
| 1 user, 30K, first turn | 92.5 | 110.5 |
| 4 users, 2K, follow-up turn | 46.6 | 51.5 |
| 4 users, 30K, follow-up turn | 37.5 | 41.0 |
