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
prefill call, and the decode steps of several sequences run as one batch. A
client sends token ids and gets the next token's logits back.

A stateless model can use Triton's dynamic batcher: the backend concatenates
the requests along dim 0 and runs them as one call. Tensors a client places
in CUDA shared memory are read and written in place on the GPU, and each GPU
an instance group names gets its own PJRT client.

| | What | Where it ran |
|---|---|---|
| ✅ | Sequence mode, prefill and batched decode, TinyLlama-1.1B | GB10, `verify.sh` |
| ✅ | Qwen3-0.6B: 16 greedy ids equal HuggingFace's for a plain and a chat-template prompt | GB10, `verify.sh` |
| ✅ | Muse Glimmer 30B, text decoder, bf16 weights: 32 greedy ids equal HuggingFace's run with the same arithmetic | GB10, `verify.sh` with `MUSE_GLIMMER=1` |
| ✅ | Dynamic batching (`max_batch_size > 0`, `dynamic_batching`) | GB10, `verify.sh` |
| ✅ | CUDA shared memory inputs read in place, outputs written device to device | GB10, `verify.sh` |
| ✅ | FP32, FP64, FP16, BF16, INT8, INT32, INT64, UINT8, BOOL over HTTP and gRPC | GB10, `verify.sh` |
| ✅ | GPU 0 selected by ordinal; a GPU the machine does not have is refused by name | GB10, `verify.sh` |
| 🧪 | Instances on GPUs other than 0, one PJRT client per GPU | Not run: the GB10 has one GPU |

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
  generate_client.py    greedy decoding from text against a sequence-mode model
  sequence_checks.py    the TinyLlama sequence-mode checks verify.sh runs
  fixture_checks.py     greedy decoding against a HuggingFace fixture (the Qwen3 checks)
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
- `reference_sequence` over HTTP and gRPC: the same artifact in sequence mode.
  The reference steps' first sequence sent one token per request, then all
  four tokens in one request, and two more one-token sequences, each against
  the interpreter within 1e-3; then requests the backend or Triton must
  refuse by name (a prompt longer than the compiled context, a token outside
  the vocabulary, a step without START, a step after END, an empty request
  without END) and an empty END request that returns no logits;
- thirteen model configurations the backend or Triton must refuse at load
  (including `zero_copy: "yes"` and `gpus: [ 7 ]`), two batching
  configurations (`max_batch_size` above the largest artifact's batch, and
  state with `max_batch_size > 0`), and seven sequence-mode configurations (no `sequence_batching`, the direct strategy,
  no CORRID control, `max_batch_size: 0`, `serving_manifest` combined with
  `artifact`, a manifest outside the version directory, two inputs), each by
  the expected message, each followed by a reload of the good configuration;
- that the server is still live at the end.

It then runs the client again with deliberately wrong expected values, and
that run must fail. It prints the measurements of `perf_client.py`
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
  - idle: 21 sequences abandoned without END lose their pages after the idle
    timeout; a new sequence then decodes correctly, the abandoned sequence's
    next step is refused, and 21 new three-page sequences fit again;
- runs `sequence_checks.py --perturb` (wrong expected ids), which must fail.

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
`fixture_checks.py --perturb` (wrong expected ids), which must fail. Without
the checkpoint this step prints `SKIP qwen3`; `SKIP_QWEN3=1` skips it.

With `MUSE_GLIMMER=1`, and meta-models/Muse-Glimmer-30B (Apache-2.0, 59 GB,
not gated) in the HuggingFace cache at the revision the fixtures name
(`hf download meta-models/Muse-Glimmer-30B`), `verify.sh` also serves its text
decoder. The step is opt-in because it puts 56 GB of bf16 weights on the
device; before the export and before the load it refuses by name if
MemAvailable is below what the step needs plus 16 GiB, because the GB10's GPU
shares system memory and a runaway allocation there can take the machine
down. It exports into `triton/build/muse-glimmer/` (batch 1, context 128,
since the chat prompt is 68 tokens; `MUSE_GLIMMER_REEXPORT=1` exports again),
drops the page cache of the checkpoint and the artifact, and starts the server
with a PJRT memory fraction of the weights plus 8 GiB over MemTotal (0.49 on
the GB10), printed. Two fixtures, both from `harness/python/muse_glimmer_fixture.py`
(transformers 5.17 on the CPU, eager attention, 16 greedy tokens for
" The capital of France is" and a chat-template prompt with a fixed
`current_date`):

- `muse_glimmer_30b_mixed_greedy.json`: bf16 weights, f32 activations, each
  projection's input rounded to bf16, RoPE tables in float64: the graph's own
  arithmetic. All 32 ids must be equal, and the logits within twice the
  fixture's `referenceNoise`, which is how far that same run moves when its
  activations are float64 instead of float32 (up to 9.1e-3 of the largest
  logit).
- `muse_glimmer_30b_bf16_greedy.json`: transformers in bfloat16 as it runs by
  default. Only the ids are compared (`--ids-only`), and only as far as the
  two fixtures agree (`--common-prefix-with`): they choose differently at the
  chat prompt's 16th token (margins 0.31 and 0.12), which is reported.

Then `--perturb`, which must fail, and the peak memory in use.

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
| `reference_sequence` | the `reference_decode` artifact in sequence mode | `TOKENS` INT32 [-1] (batch dim added, max batch 4), START/END/CORRID controls | `LOGITS` FP32 [11] |
| `int64_bool` | `x · x + x` over i64, `(x · x > x) and b`, `not b` | `X` INT64 [4], `B` BOOL [4] | `Y` INT64, `ABOVE_AND_B` BOOL, `NOT_B` BOOL |
| `dtypes_small` | `x · x + x` over f16, i8 and u8. **Written by hand**: Tlaloc's `DType` has no f16, i8 or u8, so this model shows only that the backend moves those types unchanged | `X_F16` FP16 [4], `X_I8` INT8 [4], `X_U8` UINT8 [4] | `Y_F16`, `Y_I8`, `Y_U8` |
| `grad_batched` | `df/dA` of `sum(A · A)` per 2x2 row, compiled for batch 1, 2, 4 and 8, with `dynamic_batching` | `A` FP32 [2,2] (batch dim added, max batch 8) | `GRAD` FP32 [2,2] |
| `grad_unbatched` | the same files without `dynamic_batching` | as `grad_batched` | as `grad_batched` |
| `large_io` | `x · x + x` over 4Mi f32 values (16 MiB each way) | `X` FP32 [4194304] | `Y` FP32 [4194304] |
| `large_io_host` | `large_io` with `zero_copy: "false"` | as `large_io` | as `large_io` |

The `.mlir` files (except `dtypes_small`'s), the `reference_decode` and
`reference_sequence` model directories and the files in `examples/reference/`
are generated by Tlaloc. The gradients are Tlaloc's reverse-mode transform of
the DXIR graph, not hand-written StableHLO. To regenerate them (JDK 25, from the
repository root):

```bash
./gradlew :maestro:exportTritonExamples -PoutDir=$PWD/triton/examples
```

The source is `maestro/src/jvmTools/kotlin/io/tlaloc/maestro/serving/ExportTritonExamples.kt`.
The `config.pbtxt` files of the models other than these two are written by hand;
`reference_decode` and `reference_sequence` are written whole from their
serving artifact, as described in the next section.

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
manifest's model name is then the repo id, `Qwen/Qwen3-0.6B`. Qwen3's
tokenizer is byte-level BPE, which `generate_client.py`'s word lookup does not
implement, so give it ids with `--prompt` (the fixture's `promptTokens`, for
example).

The artifact's files are hard-linked into the version directory when both are
on one file system, and copied otherwise. They are real files either way, so
the repository can be mounted into the container on its own.

For an artifact with KV pools the model is written in sequence mode (next
section). `TritonModelRepository.write(..., kv = KvMode.CLIENT)`, or
`-PkvMode=client`, writes the client-managed form instead, in which the client
names pages and slots in every request; `reference_decode` is written that
way. `-PmaxSequenceIdleMicros` and `-PmaxQueueDelayMicros` set the sequence
batcher's idle timeout (default 60 s) and batching delay (default 1 ms).

The TinyLlama commands above export decode entries for batch 1 only; add
`-PmaxBatch=4` to `exportHfServingArtifact` to let the backend batch up to
four sequences' decode steps (three decode entries plus the prefill entry:
four XLA compiles at load, about 15 s for TinyLlama).

In client mode the generated `config.pbtxt` maps each slot of the manifest
by its role:

| Manifest slot | In Triton |
|---|---|
| `tokenIds`, `positions`, `blockTables`, `seqLens`, `slotMapping` | inputs, INT32, under the slot's own name. A dimension that differs between entries (batch size, block-table width) is `-1`; each request runs on the entry compiled for exactly its shapes. |
| `logits` | output |
| staged weights | `weight:` arguments. The backend reads each file when the model loads and keeps it on the device; no request carries weights. |
| KV pools (`keyCacheN`, `valueCacheN`) and their `…Out` results | `state:` arguments and results. Each model instance holds its pools on the device, zero at start; every request reads them and replaces them with the results the manifest's donation pairs name. |

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
output [ { name: "LOGITS" data_type: TYPE_FP32 dims: [ 32000 ] } ]
sequence_batching {
  max_sequence_idle_microseconds: 60000000
  control_input [
    { name: "START"  control [ { kind: CONTROL_SEQUENCE_START  int32_false_true: [ 0, 1 ] } ] },
    { name: "END"    control [ { kind: CONTROL_SEQUENCE_END    int32_false_true: [ 0, 1 ] } ] },
    { name: "CORRID" control [ { kind: CONTROL_SEQUENCE_CORRID data_type: TYPE_UINT64 } ] }
  ]
  oldest { max_candidate_sequences: 63 preferred_batch_size: [ 4 ] max_queue_delay_microseconds: 1000 }
}
parameters: { key: "serving_manifest" value: { string_value: "tlaloc-serving.json" } }
```

### Protocol

Each request is one step of one sequence: `TOKENS` `[1, n]` with the
sequence's correlation ID (`sequence_id` in tritonclient), and the
`sequence_start` / `sequence_end` flags. The response is `LOGITS` `[1, vocab]`,
the logits of the last token sent. A client

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
- **Entry selection.** A request of `n > 1` tokens runs on the smallest
  prefill entry whose context covers the sequence's length after it, one call,
  with the tokens right-aligned in the chunk. A one-token request runs on a
  decode entry. All one-token requests in one Triton batch (different
  sequences, which the oldest strategy guarantees) run as one call on the
  smallest decode entry whose batch and context cover them, with padding rows
  for the rest of the batch. Without a prefill entry (a v1 artifact, or
  `-Pprefill=false`), the tokens of a longer request run as decode steps.
- **END** frees the sequence's pages after its step.
- **Idle timeout.** In the oldest strategy Triton ends a sequence that has
  been idle longer than `max_sequence_idle_microseconds` without telling the
  backend (its log says `Reaper: CORRID n: max sequence idle exceeded`). The
  backend therefore applies the same timeout itself: at the start of every
  execution it frees the pages of every sequence it has not served for longer
  than the timeout. The next request of such a sequence is refused by Triton
  unless it carries START.

Refused by name, with the server staying up: a START (or growth) that needs
more pages than are free (`KV page pool exhausted: sequence n needs k more
page(s) ... End a sequence (sequence_end) or export the artifact with more
pages (numBlocks)`); a sequence that would pass the largest compiled context;
a step for a sequence the backend holds nothing for; a token id outside the
vocabulary; a request with no tokens and no END. A refused START leaves no
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

Qwen3-0.6B (f32 weights, context 64, gRPC, medians of 5 to 10 runs of 16
tokens): the model loads in about 20 s (four XLA compiles of 3.5 to 6.6 s,
1.2 s to upload 2867 MiB of weights). Prefill takes about 25 ms for the 5-token
and for the 24-token prompt alike, since both run the padded 64-token prefill
entry. A decode step of one sequence takes about 19 ms (about 52 tokens/s).

Muse Glimmer 30B (bf16 weights, context 128, gRPC, medians of 3 runs of 16
tokens): the model loads in about 105 s (two XLA compiles of 8 and 10 s, 86 s
to upload 53128 MiB of weights). Prefill takes about 310 ms (the padded
128-token entry), a decode step about 245 ms (4.1 tokens/s), which is about
what reading 56 GB of weights per token at the GB10's memory bandwidth allows.
The server process held 60 GiB of device memory under a memory fraction of
0.49; the machine peaked at 85 to 90 GiB in use over two runs.

## Model configuration

A model uses the backend with `backend: "tlaloc"`. Without the `arguments`
and `results` parameters, inputs and outputs in `config.pbtxt` map to the
entry function's arguments and results by position: the first `input` is the
first argument, the first `output` the first result.

| Parameter | Required | Meaning |
|---|---|---|
| `artifact` | yes | StableHLO text file(s) inside the model version directory (`<model>/<version>/`), named relative to it; an absolute path or a `..` component is refused. A comma-separated list declares shape buckets: each file is compiled at load, and a request runs on the file whose input shapes it matches exactly. |
| `entry` | no | The function to serve. Default: `main` if the file has one, otherwise the first `func.func` in the file. |
| `pjrt_plugin_path` | no | PJRT plugin `.so` for this model. Overrides the backend setting below. |
| `arguments` | no | One item per argument of the entry function, in order, comma-separated: `input:<name>` (a config input), `weight:<file>` (a raw little-endian file in the version directory, dense and row-major with no header, uploaded once at load; its size must match the argument's type), or `state:<name>` (a buffer the model instance keeps on the device, zero at start). Every config input must appear exactly once. |
| `results` | no | One item per result, in order: `output:<name>` (a config output) or `state:<name>` (replaces that state after the request runs). Every config output must appear exactly once, and every state read by `arguments` must be written by exactly one result. |
| `zero_copy` | no | `true` (default) or `false`. With `false`, tensors in GPU memory go through the host; see "GPU memory in and out". |

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
batcher, `dynamic_batching { ... }`), the backend groups consecutive requests
whose non-batch dimensions agree, concatenates each group along dim 0 on the
host, runs the artifact with the smallest `B` that holds the group's rows,
with the remaining rows zero, and gives each request its own rows of every
output. A single request whose rows fill a bucket exactly runs on its own
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
  export one artifact per shape and list them in `artifact`.
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
  allocated as a sequence grows and there is no preemption, so a sequence
  that needs a page when none is free is refused mid-generation rather than
  paused. Only the oldest strategy. Prefill entries are batch 1, so two
  prompts in one Triton batch run as two calls. Sampling is the client's
  (the backend returns logits). The pools are copied by each execution
  rather than donated (the PJRT execute call does not donate buffers yet).
- Client mode: state is per model instance and is not tied to a Triton
  sequence ID. The client that allocates pages must be the only client of
  that model, or the clients must agree on the pages.
- The weights and KV pools are f32 on the device, as the artifact stores them
  (TinyLlama: 4.1 GiB of weights; Qwen3-0.6B: 2.8 GiB, of which 594 MiB is the
  tied head, staged as a second, transposed copy of the embedding table).
