# Triton backend for Tlaloc

`libtriton_tlaloc.so` is an [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
backend. It serves StableHLO that Tlaloc emits: at model load it compiles the
StableHLO through a PJRT plugin (the XLA CUDA plugin), and at request time it
runs the compiled executable on the GPU. Serving needs no JVM and no JAX:
the backend loads the PJRT plugin `.so` itself. Clients use Triton's standard
HTTP and gRPC (KServe v2) endpoints.

This directory is not a Gradle module and is not published to Maven.

```
triton/
  backend/              the backend source (C++17)
  third_party/          vendored upstream headers and sources (see docs/vendoring.md)
  examples/
    model_repository/   four example models, emitted by Tlaloc
    reference/          the DXIR interpreter's results for them
  build_backend.sh      builds the backend inside the Triton container
  fetch_pjrt_plugin.sh  downloads the PJRT CUDA plugin
  run_server.sh         starts tritonserver with the backend and a model repository
  verify.sh             end-to-end check: start, infer over HTTP and gRPC, compare, stop
  verify_client.py      the client half of verify.sh
  generate_client.py    greedy decoding against a decode model (TinyLlama)
```

`backends/` (the built `.so`), `pjrt/` (the downloaded plugin) and `build/`
(the TinyLlama model verify.sh writes) are build outputs and are not
committed.

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
- For `verify.sh`: `curl`, and a Python with `tritonclient[http,grpc]`
  (`pip install "tritonclient[http,grpc]"` in any virtual environment; point
  `TRITON_CLIENT_PYTHON` at its `python`).

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
reader (`backend/test/stablehlo_text_test.cc`) before it links the backend.

`verify.sh` starts the server, waits for it to be ready, and runs
`verify_client.py`, which checks:

- `matmul_sumsq` over KServe v2 JSON, tritonclient HTTP and tritonclient gRPC,
  bit for bit against the DXIR interpreter's result
  (`examples/reference/matmul_sumsq.json`) and against the closed form
  (value 54, gradient `[[7, 11], [9, 13]]`);
- `dtypes` (FP64, BF16, INT32) over HTTP and gRPC, bit for bit, including a
  request that asks for only one output;
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
- eleven model configurations the backend must refuse at load, each by the
  expected message, followed by a reload of the good configuration;
- that the server is still live at the end.

It then runs the client again with deliberately wrong expected values, and
that run must fail. It stops the container.

Last, if the TinyLlama-1.1B checkpoint is in
`~/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0`
(`TINYLLAMA_CHECKPOINT` overrides), `verify.sh` exports its decode artifact and
writes it as the Triton model `tinyllama` under `triton/build/tinyllama/`,
reusing that model on later runs (`TINYLLAMA_REEXPORT=1` exports again).
Gradle runs only while no server is up. It then starts a server on that
repository and greedy-decodes "The capital of France is" with
`generate_client.py`. The six generated ids must equal HuggingFace
transformers' ids for the same checkpoint, `[3681, 29889, 13, 13, 29906,
29889]` (" Paris.\n\n2."). Without the checkpoint this step prints
`SKIP tinyllama` and the run can still pass; `SKIP_TINYLLAMA=1` skips it on
purpose.

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

The `.mlir` files, the `reference_decode` model directory and the files in
`examples/reference/` are generated by Tlaloc. The gradient is Tlaloc's reverse-mode transform of the DXIR graph for
`f`, not hand-written StableHLO. To regenerate them (JDK 25, from the
repository root):

```bash
./gradlew :maestro:exportTritonExamples -PoutDir=$PWD/triton/examples
```

The source is `maestro/src/jvmTools/kotlin/io/tlaloc/maestro/serving/ExportTritonExamples.kt`.
The `config.pbtxt` files of the first three models are written by hand;
`reference_decode` is written whole from its serving artifact, as described in
the next section.

## Exporting a Triton model repository from Kotlin

A Tlaloc serving artifact (the directory `:maestro:exportServingArtifact` and
`:maestro:exportLlamaServingArtifact` write: a manifest, one StableHLO body
per batch/context entry, and staged weight files) becomes a Triton model with
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
./gradlew :maestro:exportLlamaServingArtifact \
    -PckptDir=$HOME/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0 \
    -PoutDir=/abs/tinyllama-artifact
./gradlew :maestro:exportTritonModel -PartifactDir=/abs/tinyllama-artifact \
    -PoutDir=/abs/model_repository -PmodelName=tinyllama
MODEL_REPOSITORY=/abs/model_repository triton/run_server.sh
python triton/generate_client.py --model tinyllama \
    --tokenizer $HOME/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0/tokenizer.json
```

The artifact's files are hard-linked into the version directory when both are
on one file system, and copied otherwise. They are real files either way, so
the repository can be mounted into the container on its own.

The generated `config.pbtxt` maps each slot of the manifest by its role:

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
`generate_client.py` gives one sequence pages 1 to N (page 0 is the page
padded rows use) and runs the prompt as single-token decode steps, as
`examples/gpu-inference/serve.py` does.

On the GB10, the 22-layer TinyLlama loads in 5.1 s: the XLA compile takes
3.0 s, and uploading 201 weights (4196 MiB as f32) takes 2.0 s. The 44 KV
pools (44 MiB) are then zeroed on the device. A decode
step, sent over HTTP and answered with 32000 logits, then takes about 23 ms
(median of 11 steps); the first step took 62 ms. For comparison, the Python
driver in `docs/SERVING_RUNBOOK.md` takes 1.35 s per step, because it copies
every KV pool to the host and back on every step. These are single
measurements from `verify.sh`, not a benchmark.

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

At load the backend reads each artifact's entry signature and checks it
against `config.pbtxt`: the number of inputs and outputs, each data type, and
each shape (`-1` in `config.pbtxt` matches any size). Weight and state
arguments must have the same type in every artifact of the model, and a
state result the same type as the state argument it replaces. A mismatch
refuses the model with a message naming the file, the tensor and both types.

Triton has no rank-0 tensors in a model with `max_batch_size: 0`. Declare a
rank-0 argument or result as `dims: [ 1 ]`; the backend maps one to the
other.

Data types: FP32, FP64, FP16, BF16, INT8, INT32, INT64, UINT8 and BOOL. Any
other type is refused by name at load. `verify.sh` exercises FP32, FP64, BF16
and INT32; the other five use the same code path with a different element
width and have not been run.

With `max_batch_size > 0` the leading `-1` of each tensor is the batch
dimension and the artifact must have been emitted for the exact batch sizes
that will arrive (one bucket per batch size). The backend does not
concatenate requests: each request in a Triton batch runs on its own. None
of the example models batches, so this path has not been run.

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

The backend creates one PJRT client per plugin path and shares it between all
models that use that plugin. It is created when the first such model loads
and destroyed when the last one unloads. Each model's executables are
destroyed when the model unloads.

`run_server.sh` mounts the model repository at `/models`, the backend at
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
- One device. Every instance runs on PJRT device 0; a `KIND_GPU` instance on
  any other GPU is refused. On a multi-GPU host, set `gpus: [ 0 ]` or
  restrict the container to one GPU.
- Inputs are copied to the device and outputs back to host memory for each
  request. Input tensors in GPU memory (CUDA shared memory) are first copied
  to the host, and outputs Triton places in GPU memory are copied there from
  the host; `verify.sh` does not exercise these two paths. There is no
  zero-copy path yet.
- A model that fails to load stays in the repository index as
  `UNAVAILABLE`. With Triton's default `--strict-readiness=true` the server
  then answers `/v2/health/ready` with 400 until that model loads, while
  `/v2/health/live` and the other models keep serving. Pass
  `--strict-readiness=false` if readiness should track only the server.
- No sequence batching, no optional inputs, no string tensors, no decoupled
  (streaming) responses.
- State is per model instance and is not tied to a Triton sequence ID. The
  client that allocates pages must be the only client of that model, or the
  clients must agree on the pages.
- A decode artifact has no prefill entry yet, so a prompt runs as one decode
  step per token. `generate_client.py` handles one sequence at a time, on a
  model with one block-table width.
- The weights and KV pools are f32 on the device, as the artifact stores them
  (TinyLlama: 4.1 GiB of weights).
