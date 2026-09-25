# triton-llm: a chat model served by NVIDIA Triton, compiled by Kotlin

```
$ examples/triton-llm/run.sh
question  Why is the sky blue? Answer in two sentences.
answer    The sky appears blue because of the way light interacts with the Earth's atmosphere, ...
decode    median 20.5 ms a token over 45 requests (48.8 tokens/s)
```

Qwen3-0.6B, all 28 layers, answering through
[NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server).
Kotlin read the HuggingFace checkpoint and wrote the programs Triton runs.
Triton runs them through `libtriton_tlaloc.so`, a C++ backend that compiles
Tlaloc's StableHLO with the XLA CUDA plugin. The server holds the weights, the
KV cache and each conversation's pages on the GPU; the client sends token ids
and prints the answer as the tokens arrive.

```
  Kotlin (this Gradle build)         Triton container                     chat.py
  ──────────────────────────         ────────────────                     ───────
  read config.json + safetensors     tritonserver                         chat template (jinja2)
  build prefill + decode graphs      └ libtriton_tlaloc.so                tokenizer.json (tokenizers)
  emit StableHLO                       └ xla_cuda_plugin.so (PJRT)        prompt ──► one prefill call
  write weights, manifest,           weights + KV pages on the GPU        token  ──► one decode step
    config.pbtxt                     picks the entry per request          ◄── logits; argmax; print
  EXIT
```

How each stage works is in
[docs/SERVING_ARCHITECTURE.md](../../docs/SERVING_ARCHITECTURE.md).

## Running it

```bash
# from the repo root, once: publish Tlaloc to mavenLocal (this build resolves it there)
./gradlew publishToMavenLocal -x test

# once: the PJRT CUDA plugin and the Triton image
triton/fetch_pjrt_plugin.sh
docker pull nvcr.io/nvidia/tritonserver:25.11-py3

# once: the checkpoint, into the HuggingFace cache (never into this repo)
hf download Qwen/Qwen3-0.6B

# once: the client's packages (no torch, no transformers)
python3 -m venv /tmp/chat-venv
/tmp/chat-venv/bin/pip install -r examples/triton-llm/requirements.txt

CHAT_PYTHON=/tmp/chat-venv/bin/python examples/triton-llm/run.sh
CHAT_PYTHON=/tmp/chat-venv/bin/python examples/triton-llm/run.sh --question "What is the capital of France? Answer in one word."
CHAT_PYTHON=/tmp/chat-venv/bin/python examples/triton-llm/run.sh --model tinyllama
CHAT_PYTHON=/tmp/chat-venv/bin/python examples/triton-llm/run.sh --model muse-glimmer   # read the warning below first
```

`run.sh` does six things:

1. checks what it needs and skips by name, with exit status 0, when something
   is missing (below);
2. runs this directory's Gradle build, which exports the model into
   `build/<model>/`: the serving artifact and the Triton model repository.
   Later runs reuse it; `REEXPORT=1` exports again;
3. builds `triton/backends/tlaloc/libtriton_tlaloc.so` with
   `triton/build_backend.sh` if it is not there yet (the build runs inside the
   Triton image);
4. starts Triton in its container with `triton/run_server.sh`, on ports
   8000 to 8002 (`HTTP_PORT`, `GRPC_PORT`, `METRICS_PORT` change them), and
   waits until the model is loaded;
5. runs `chat.py`, which asks the question over gRPC and prints the answer
   token by token;
6. removes the container, also when a step fails or you press Ctrl-C.

The export is a standalone Gradle project like every other example: its own
`settings.gradle.kts` and `build.gradle.kts`, resolving
`io.github.pedronahum:tlaloc-*` from mavenLocal. The serving side uses the
repository's `triton/` directory for the backend and the server script.

### The models

| `--model` | Checkpoint (HuggingFace) | Licence | Weights on the GPU | Status |
|---|---|---|---|---|
| `qwen3` (default) | Qwen/Qwen3-0.6B | Apache-2.0 | 2.8 GiB, f32 | ✅ ran on the GB10 |
| `tinyllama` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | Apache-2.0 | 4.1 GiB, f32 | ✅ ran on the GB10 |
| `muse-glimmer` | meta-models/Muse-Glimmer-30B, text decoder only | Apache-2.0 | 52 GiB, bf16 | ✅ ran on the GB10 |

`run.sh` finds the checkpoint in the HuggingFace cache
(`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`);
`CHECKPOINT=<dir>` points it anywhere else. For TinyLlama it also looks in
`~/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0`, where the
repository's other examples keep it.

Each model is compiled for batch 1 and contexts 64, 128 and 256 tokens: a
decode entry and a prefill entry for each, six XLA compiles when the model
loads. The server runs each request on the smallest entry that holds the
conversation, so a short one does not pay for 256 positions. The prompt and
the answer together must fit in 256 tokens; `chat.py` stops there.

**Muse Glimmer needs about 60 GB of GPU memory.** Its 52 GiB of bf16 weights
are exported to `build/muse-glimmer/` (the export writes 52 GiB to disk and
took 165 s here) and uploaded at load (85 s here). On a GB10 the GPU's memory
is the system RAM, so the weights come out of the memory everything else on
the machine uses. `run.sh` prints a warning, skips by name unless 80 GiB are
available before the export, checks again before the load, drops the
checkpoint and the artifact from the page cache, and gives the PJRT client a
memory fraction of the weights plus 8 GiB (0.492 on the GB10) instead of the
default 0.3. Do not run it next to another large GPU process.

## What it printed

Run on 2026-09-25 on the GB10 (aarch64, driver 580.126.09, JDK 25.0.3,
Triton 25.11, XLA CUDA plugin from jax-cuda13-pjrt 0.10.0, tritonclient 2.72.0,
tokenizers 0.23.2). Timings are single runs, not a benchmark.

### Qwen3-0.6B (the default)

`REEXPORT=1 examples/triton-llm/run.sh`:

```
== export (Kotlin)
checkpoint  /home/pedro/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
family      qwen3, 28 layers, hidden 1024, vocab 151936, weights as f32
entries     decode at batch 1..1 and prefill at batch 1, for contexts 64, 128, 256; KV pages of 16 tokens
artifact    6 entries, 311 weight files (2867 MiB) in 7.9 s
              decode_b1_c64
              decode_b1_c128
              decode_b1_c256
              prefill_b1_c64
              prefill_b1_c128
              prefill_b1_c256
triton      /home/pedro/programming/tlaloc/examples/triton-llm/build/qwen3/repository/qwen3/config.pbtxt
== starting Triton (tlaloc-triton-llm, PJRT memory fraction 0.3, log: /home/pedro/programming/tlaloc/examples/triton-llm/build/qwen3/server.log)
ready in 30 s: uploaded 311 weights (2867 MiB) in 1679 ms; 6 entries, KV pool of 64 pages x 16 tokens, largest context 256, largest decode batch 1, sequence idle timeout 60000000 us
== chat (gRPC, localhost:8001)
question  Why is the sky blue? Answer in two sentences.
prompt    23 tokens after the chat template
answer    The sky appears blue because of the way light interacts with the Earth's atmosphere, scattering shorter wavelengths of light (blue) while longer wavelengths (red and orange) are scattered more. This scattering causes the sky to appear blue.

generated 45 tokens, stopped by end-of-turn token 151645
prefill   23 tokens in one request: 82 ms
decode    median 20.5 ms a token over 45 requests (48.8 tokens/s)
```

`run.sh --question "What is the capital of France? Answer in one word."`
(the export reused):

```
== export: reusing /home/pedro/programming/tlaloc/examples/triton-llm/build/qwen3/repository (REEXPORT=1 writes it again)
== starting Triton (tlaloc-triton-llm, PJRT memory fraction 0.3, log: /home/pedro/programming/tlaloc/examples/triton-llm/build/qwen3/server.log)
ready in 30 s: uploaded 311 weights (2867 MiB) in 4471 ms; 6 entries, KV pool of 64 pages x 16 tokens, largest context 256, largest decode batch 1, sequence idle timeout 60000000 us
== chat (gRPC, localhost:8001)
question  What is the capital of France? Answer in one word.
prompt    24 tokens after the chat template
answer    Paris

generated 1 token, stopped by end-of-turn token 151645
prefill   24 tokens in one request: 80 ms
decode    median 27.5 ms a token over 1 request (36.4 tokens/s)
```

That prompt is the chat prompt of the Qwen3 fixture that `triton/verify.sh`
checks against HuggingFace transformers
(`ir/src/jvmTest/resources/io/tlaloc/ir/inference/qwen3_0_6b_greedy.json`):
`chat.py`'s 24 prompt ids are the fixture's, and transformers' first two
greedy ids are `59604` ("Paris") and `151645`, the end-of-turn token this run
stopped on.

### TinyLlama-1.1B-Chat

`run.sh --model tinyllama` (export from an earlier run, 6.8 s for 4196 MiB):

```
== export: reusing /home/pedro/programming/tlaloc/examples/triton-llm/build/tinyllama/repository (REEXPORT=1 writes it again)
== starting Triton (tlaloc-triton-llm, PJRT memory fraction 0.3, log: /home/pedro/programming/tlaloc/examples/triton-llm/build/tinyllama/server.log)
ready in 32 s: uploaded 201 weights (4196 MiB) in 6781 ms; 6 entries, KV pool of 64 pages x 16 tokens, largest context 256, largest decode batch 1, sequence idle timeout 60000000 us
== chat (gRPC, localhost:8001)
question  Why is the sky blue? Answer in two sentences.
prompt    27 tokens after the chat template
answer    The sky is blue because of the presence of blue light, which is emitted by the Earth's atmosphere. The blue color is caused by the interaction of sunlight with the molecules in the atmosphere, which absorb and scatter the blue light. The blue light then reflects off the Earth's surface and reaches the observer's eye, creating the blue color.

generated 76 tokens, stopped by end-of-turn token 2
prefill   27 tokens in one request: 73 ms
decode    median 26.5 ms a token over 76 requests (37.7 tokens/s)
```

A 1.1B model's physics; the serving is what is being shown.

### Muse Glimmer 30B (text decoder)

The export, from the first run:

```
WARNING: Muse Glimmer puts 56 GB of bf16 weights on the GPU; the export
         writes 56 GB to /home/pedro/programming/tlaloc/examples/triton-llm/build/muse-glimmer and the load takes minutes.
== export (Kotlin)
checkpoint  /home/pedro/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B/snapshots/a4e59da52a7bc87ae7251dd5545c0dd437c44b68
family      muse_glimmer, 52 layers, hidden 6656, vocab 202048, weights as bf16
            809 checkpoint tensors are not part of the text decoder and are not read
entries     decode at batch 1..1 and prefill at batch 1, for contexts 64, 128, 256; KV pages of 16 tokens
artifact    6 entries, 627 weight files (53128 MiB) in 165.3 s
              decode_b1_c64
              decode_b1_c128
              decode_b1_c256
              prefill_b1_c64
              prefill_b1_c128
              prefill_b1_c256
triton      /home/pedro/programming/tlaloc/examples/triton-llm/build/muse-glimmer/repository/muse-glimmer/config.pbtxt
```

`run.sh --model muse-glimmer`, the second run:

```
WARNING: Muse Glimmer puts 56 GB of bf16 weights on the GPU; the export
         writes 56 GB to /home/pedro/programming/tlaloc/examples/triton-llm/build/muse-glimmer and the load takes minutes.
== export: reusing /home/pedro/programming/tlaloc/examples/triton-llm/build/muse-glimmer/repository (REEXPORT=1 writes it again)
== starting Triton (tlaloc-triton-llm, PJRT memory fraction 0.492, log: /home/pedro/programming/tlaloc/examples/triton-llm/build/muse-glimmer/server.log)
ready in 139 s: uploaded 627 weights (53128 MiB) in 82917 ms; 6 entries, KV pool of 64 pages x 16 tokens, largest context 256, largest decode batch 1, sequence idle timeout 60000000 us
== chat (gRPC, localhost:8001)
question  Why is the sky blue? Answer in two sentences.
prompt    67 tokens after the chat template
answer     to=self<|message|>Why is the sky blue? Answer in two sentences.
          
          Answer in two sentences. Provide explanation. Probably Rayleigh scattering.
          
          Two sentences. Ensure exactly two sentences.
          
          Let's output two sentences.<|eom|><|start|>assistant to=user<|message|>The sky appears blue because sunlight is scattered by molecules in Earth's atmosphere, and shorter blue wavelengths scatter more than longer red wavelengths. This Rayleigh scattering sends blue light toward our eyes from all directions, while the sun itself looks whiter or redder near the horizon.

generated 97 tokens, stopped by end-of-turn token 200008
prefill   67 tokens in one request: 368 ms
decode    median 252.1 ms a token over 97 requests (4.0 tokens/s)
```

Muse Glimmer's chat template has the model reason in a channel addressed to
itself (`to=self`) before it writes to the user (`to=user`). `chat.py` prints
special tokens, so the channel markers show. `run.sh` passes the template's
`reasoning_strength=low` (`REASONING_STRENGTH` changes it; the template's own
default is `high`, which reasoned for the whole 96-token budget of an earlier
run without reaching the answer) and today's date as `current_date`. At 4
tokens/s the decode speed is about what reading 52 GiB of weights per token
allows at the GB10's memory bandwidth.

### Timings

| | Qwen3-0.6B | TinyLlama-1.1B | Muse Glimmer 30B |
|---|---|---|---|
| Export (Kotlin) | 7.9 s | 6.8 s | 165 s |
| Server ready (six XLA compiles, weight upload) | 30 s | 28 to 32 s | 133 to 139 s |
| Prefill of the chat prompt, one request | 80 to 82 ms (23 to 24 tokens) | 73 ms (27 tokens) | 368 ms (67 tokens) |
| Decode, median per token | 20.5 ms | 26.5 ms | 252 ms |

The prefill is the first request the server runs after loading, and it runs on
the 64-token entry (Muse Glimmer's 67-token prompt on the 128-token entry).
The client measures each request from gRPC call to response, so the numbers
include the round trip.

One run of the Qwen3 example shared the GPU with another process that had it at
96% utilization: the decode step took 39 to 43 ms instead of about 20. Check
`nvidia-smi` before reading anything into a timing.

## When something is missing

`run.sh` checks, in this order, and prints one line and exits 0 at the first
thing missing:

```
SKIP: no docker on PATH. Triton runs in a container.
SKIP: no NVIDIA GPU: nvidia-smi is missing or lists no GPU.
SKIP: no Triton image nvcr.io/nvidia/tritonserver:25.11-py3 on this machine (docker pull nvcr.io/nvidia/tritonserver:25.11-py3, about 13 GB).
SKIP: no PJRT CUDA plugin at <repo>/triton/pjrt/xla_cuda13/xla_cuda_plugin.so (triton/fetch_pjrt_plugin.sh downloads it).
SKIP: no Qwen/Qwen3-0.6B checkpoint in /home/you/.cache/huggingface/hub (hf download Qwen/Qwen3-0.6B, or set CHECKPOINT).
SKIP: /usr/bin/python3 lacks tritonclient[grpc] tokenizers numpy (pip install -r <repo>/examples/triton-llm/requirements.txt, then set CHAT_PYTHON).
```

Each line was produced on the GB10 by taking that one thing away (a `PATH`
without `docker`, a `PATH` without `nvidia-smi`, an image tag that does not
exist, a plugin path that does not exist, an empty cache, the system Python);
the paths and the tag above are the defaults rather than the ones the test
used. A port already in use or a leftover `tlaloc-triton-llm` container is
an error (exit 1) naming the port or the container, not a skip.

## What to look at in the source

- `src/main/kotlin/Export.kt`: the two library calls that do the export,
  `HfServingExport.export` and `TritonModelRepository.write`, and the
  compiled shapes.
- `chat.py`: the whole client protocol in `Sequence.send`: `TOKENS` in,
  `LOGITS` out, the sequence id and the START and END flags. The chat
  template is rendered in the same jinja2 environment transformers'
  `apply_chat_template` uses; for Qwen3 and Muse Glimmer it gives the same
  prompt ids as transformers (checked against both fixtures).
- `build/<model>/repository/<model>/config.pbtxt`: what Kotlin generated for
  Triton (sequence batcher, controls, `serving_manifest`).
- `build/<model>/server.log`: the server's log, including each entry's
  compile time and the weight upload.

## Limits

- Greedy decoding only: the server returns logits and `chat.py` takes the
  argmax. Sampling would be a change to `chat.py` alone.
- One question, no conversation history, no system message beyond the
  template's own.
- Batch 1 is compiled, so concurrent sequences decode one after another. The
  backend batches decode steps when the export compiles larger batches
  (`maxBatch` in `Export.kt`; `triton/verify.sh` runs four TinyLlama sequences
  at once).
- Weights are f32 on the device for Qwen3 and TinyLlama, as the artifact
  stores them; Muse Glimmer's are bf16.
- The model weights are never committed: they stay in the HuggingFace cache,
  and `build/` is ignored by git.
