# gpu-inference — compile a model into an artifact and serve it with no framework

```
$ ./gradlew -p examples/gpu-inference run        # Kotlin writes a directory, exits
$ python3 serve.py                               # no jax, no torch, no numpy, no JVM

    prompt      ' The capital of France is'
    completion  ' Paris.\n\n2.'
```

That is a real TinyLlama-1.1B — all 22 layers, 4.1 GiB of its own weights — run
by `/usr/bin/python3` on a machine where jax, torch, numpy and transformers are
**not installed at all**. Those six generated token ids are the same six
HuggingFace `transformers` produces from the same checkpoint.

**What it shows.** Two processes, and the gap between them is the whole point.

```
   Kotlin                                    Python
   ──────                                    ──────
   read a model                              dlopen one PJRT plugin .so
   build a decode graph        ───────►      load the directory
   emit StableHLO           a DIRECTORY      compile it with XLA
   stage the weights                         run it on the GPU
   write the manifest                        decode tokens
   EXIT
```

The first process never loads a PJRT plugin, never opens a CUDA context and
never runs the graph it just built — it does not even depend on
`io.github.pedronahum:tlaloc-runtime-pjrt`. The second has **no JVM in it, and no jax, no torch,
no numpy and no transformers either**; `serve.py` asks the import system and
prints the answer, so you do not have to take that on faith.

What the serving process depends on is exactly two things:

1. **a PJRT plugin `.so`** — `xla_cuda_plugin.so` on an NVIDIA box,
   `/lib/libtpu.so` on a Cloud TPU VM. It is found as a *file*, never imported.
2. **a driver** for whatever that plugin drives.

That is the sentence this example exists to make runnable: **the artifact is
the deployment.**

## Which model you get, without asking for one

The export looks for a checkpoint in the cache this repo already uses
(`~/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0`):

- **it is there** → the real 22-layer TinyLlama is exported, and the default
  prompt is `"The capital of France is"`, encoded with the checkpoint's own
  `tokenizer.json`. You get the run at the top of this page.
- **it is not there** → the **reference decode graph** is exported instead: one
  attention layer, an 11-word vocabulary, weights from a fixed LCG. It needs no
  download, no GPU to export, and it exercises the identical code path — which
  is the point of it. `--reference` forces this lane even when the checkpoint
  exists.

Nothing writes a token id by hand in either lane. `serve.py` reads
`tokenizer.json` with the standard library's `json` — a vocabulary lookup, not
a tokenizer library — so the ids come from the model's own vocabulary or the
script refuses. `--text "..."` encodes your own prompt the same way (exact
vocabulary matches only; it will not guess a subword split that might differ
from the oracle's).

## Running it

Standalone Gradle project; it resolves Tlaloc from **mavenLocal** exactly as
your own project would, so publish first.

```bash
# from the repo root
./gradlew publishToMavenLocal

# half one — Kotlin compiles and exports. No GPU needed, no checkpoint needed.
./gradlew -p examples/gpu-inference run

# half two — Python serves. Point it at a plugin .so; any interpreter will do.
export TLALOC_PJRT_PLUGIN_PATH=/path/to/xla_cuda_plugin.so
/usr/bin/python3 examples/gpu-inference/serve.py

# your own prompt, encoded with the model's own vocabulary
/usr/bin/python3 examples/gpu-inference/serve.py --text "The capital of Japan is"

# the toy graph on purpose, even if a checkpoint is cached
./gradlew -p examples/gpu-inference run --args="--reference"
```

Want the real model and do not have it cached? It is one download into the
cache the export already looks in:

```bash
# tooling, not runtime — any venv with huggingface_hub; the serving process
# never sees it.
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --local-dir ~/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0
```

`serve.py` needs `tlaloc_serve` on its path. A repo checkout has it — the
script adds `harness/python` itself. A deployment gets it with
`pip install -e <tlaloc>/harness/python`, which **pulls in nothing**: that
distribution's dependency list is empty, and a test pins it empty.

**No GPU — or no `TLALOC_PJRT_PLUGIN_PATH`?** Half one still runs and still
writes a complete artifact — it is complete whether or not the box that made it
can execute it. Half two then prints a `SKIP:` line naming what is missing and
exits `0`:

```
SKIP: no PJRT plugin on this machine, so there is nothing to run on.
      no PJRT plugin found. Set TLALOC_PJRT_PLUGIN_PATH to a plugin .so (a jax
      install's jax_plugins/xla_cuda12/xla_cuda_plugin.so, a TPU VM's
      /lib/libtpu.so, or a standalone plugin a deployment ships). The serving
      runtime needs that file and a driver — nothing else.
```

That message is about this process's *inputs*, not about the machine. A box with
a working plugin prints exactly the same thing if the `export` above was skipped:
the JVM side (`PjrtBinaries`, §0.4.503) searches seven roots for a plugin, and the
serving runtime deliberately searches none — it takes the path it is given, because
a deployment ships its own `.so` and guessing is not a serving-time behaviour.
Measured at §0.4.506, on a GB10 whose JVM lane was running on CUDA at the time.

---

## Half one: the reference decode graph

The default export is a **reference decode graph** — one attention layer, an
11-word vocabulary, weights from a fixed LCG. It is deliberately not a Llama:
this half of the example is about the *seam*, and a seam is certified by a
graph small enough that a disagreement is attributable. It needs no download,
no checkpoint and no GPU.

Real output of `./gradlew -p examples/gpu-inference run`:

```
model     tlaloc-reference-decode (the reference decode graph)
           vocab 11  hidden 8  layers 1
           4 heads / 2 kv-heads x headDim 2
           KV pool 6 pages x blockSize 2
weights   in-body constants from a fixed LCG (so two exports are byte-identical)
out       /home/pedro/programming/tlaloc/examples/gpu-inference/build/artifact

wrote the artifact in 0.1s

  /home/pedro/programming/tlaloc/examples/gpu-inference/build/artifact/
    bodies/          6 files  56.3 KiB
    programs/        6 files  3.8 KiB
    tlaloc-serving.json        8.6 KiB

  model tlaloc-reference-decode
  hash  reference-decode-lcg-v1
  6 compiled entries:
    decode_b1_c2  ->  bodies/917676809d69dc0f565c94bec3043e03a76a3319ca40031040e8bc6f67978da6.mlir
    decode_b1_c4  ->  bodies/b92b107ce14c2294bbc405eae1af38d561237792aad36beb1b27ddce403a3a65.mlir
    decode_b2_c2  ->  bodies/d0df315053ba7d7e4fdae8b1edadae90124771436ec33ffc5ec28bd51213ff51.mlir
    decode_b2_c4  ->  bodies/4f7a236ffdf59327aae686cafcef8a52bb922efcb5f49bd7bbb5f755139785a3.mlir
    decode_b4_c2  ->  bodies/ab02c7c995492bf2f539ae0f93f4ab43e606ce861fa17cbc3ed10d1fb059556b.mlir
    decode_b4_c4  ->  bodies/57eee34ace6f0bb42c9ba8578df71b3cdd67c810cb7451eda3cdd4e81acc7cd2.mlir

Half one is done. Nothing below this line is Kotlin's business:
    python3 serve.py --artifact .../examples/gpu-inference/build/artifact
```

Six entries because the ladder is batch {1,2,4} × context {2,4}. `maxBatch = 4`
is chosen so that a batch of *three* has to be padded — a ladder every request
lands on exactly would exercise the export path without ever exercising the
padding convention the whole bucketing story rests on.

**The body filename is a SHA-256 of the body, not a timestamp.** Two exports of
one model are byte-identical directories, and `grep` is a legitimate debugging
tool on a deployment, which is why the bodies are textual MLIR rather than a
serialised blob. Try it:

```bash
grep -c stablehlo examples/gpu-inference/build/artifact/bodies/*.mlir
```

## Half two: serving it

Real output of `python3 serve.py`, GB10 / NVIDIA Blackwell, CUDA driver
580.126.09:

```
the entire runtime of this process:
  PJRT plugin   .../jax_plugins/xla_cuda12/xla_cuda_plugin.so
  platform      cuda (requested; the client's own answer is below)
  Python        /usr/bin/python3 (3.12.3)
  frameworks    none installed in this interpreter (jax, torch, numpy: not found)

artifact      .../examples/gpu-inference/build/artifact
  model       tlaloc-reference-decode
  hash        reference-decode-lcg-v1
  layers      1   vocab 11
  bodies      6 compiled entries, all re-hashed OK
  weights     in-body constants (nothing staged)
  engine      CtypesEngine on cuda

greedy decode  prompt [1, 2] + 2 new (compiled context 4 = 2 pages x 2)
  step  0  prompt  token      1  -> argmax      2      245.7 ms
  step  1  prompt  token      2  -> argmax      9        1.6 ms
  step  2  gen     token      9  -> argmax      7      133.2 ms
  step  3  gen     token      7  -> argmax      9        1.0 ms

prompt     [1, 2]
generated  [9, 7]
all        [1, 2, 9, 7]

  XLA compiles     2 (one per distinct ladder point actually touched)
  first step       246 ms  (weight upload + compile)
  median step      2 ms
```

The plugin path there sits inside a jax install, and that is the *only* thing
jax is doing in this story: it is a place a `.so` happens to sit on this
machine. The loader walks `site-packages` for the file; it never runs
`import jax_plugins`. Copy that one `.so` onto a machine with nothing else on
it and the run above is unchanged.

Two entries compiled, not six: a ladder point is compiled the first time a
request lands on it. Steps 0 and 2 cross a page boundary into a wider bucket
and pay a compile; steps 1 and 3 reuse it, which is why they are two orders of
magnitude faster.

---

## The real thing: a 1.1B-parameter Llama, same two halves

Everything above runs on a laptop. The same code path, with a
`--checkpoint` flag, exports **TinyLlama-1.1B — all 22 layers, 201 tensors read
by role out of its own HuggingFace `model.safetensors`** — and serves it. The
argument parsing in `Export.kt` is the only difference between the two lanes.

Fetch the checkpoint once (2.2 GB; it goes under `$HOME`, inside no venv):

```bash
# in any venv that has huggingface_hub — this is TOOLING, not runtime, and it
# never touches the serving process. On this machine that was ~/.local/venvs/vllm.
python -c "from huggingface_hub import snapshot_download; snapshot_download( \
  'TinyLlama/TinyLlama-1.1B-Chat-v1.0', \
  local_dir='$HOME/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0')"
```

Export it (~9 s, and it writes ~4.2 GiB, so put it somewhere with room):

```bash
./gradlew -p examples/gpu-inference run --args="\
  --checkpoint $HOME/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0 \
  --out /tmp/tl-llama-example"
```

Real output:

```
model     TinyLlama__TinyLlama-1.1B-Chat-v1.0
           vocab 32000  hidden 2048  layers 22 of 22
           32 heads / 4 kv-heads
weights   read by ROLE from the checkpoint's own safetensors, widened to f32
           and TRANSPOSED host-side (HF stores nn.Linear as [out, in])
ladder    batch 1 x context 64, blockSize 16
out       /tmp/tl-llama-example

wrote the artifact in 8.9s

  /tmp/tl-llama-example/
    bodies/          1 files  3.2 MiB
    programs/        1 files  14.4 KiB
    tlaloc-serving.json       65.1 KiB
    weights/       201 files  4.1 GiB

  model TinyLlama__TinyLlama-1.1B-Chat-v1.0
  hash  hf-llama:TinyLlama__TinyLlama-1.1B-Chat-v1.0:L22:h2048:v32000
  1 compiled entry:
    decode_b1_c64  ->  bodies/8cbf1976...1333.mlir
```

`weights/NNNN_<slot>.bin` are raw little-endian, dense, row-major, **with no
header: the file IS the operand**, already transposed into math layout by a
host-side pass. The loader's whole job is `open`, `readinto`, upload — it never
creates a Python number out of a weight, because 1.1e9 Python floats is not a
slow path, it is an impossible one.

Now the ids. **Nothing in this repo writes a token id by hand** — the tokenizer
is the frontend's, so the frontend owns the ids. Get them from HuggingFace, in
a venv that has `transformers`:

```bash
# again: tooling. A venv with torch + transformers, never the serving interpreter.
~/.local/venvs/vllm/bin/python harness/python/hf_llama_greedy_oracle.py \
  --checkpoint $HOME/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0 \
  --prompt "The capital of France is" --max-new 6 --output /tmp/oracle.json
```

which returned, really:

```json
{"promptTokens": [1, 450, 7483, 310, 3444, 338],
 "generatedTokens": [3681, 29889, 13, 13, 29906, 29889],
 "generatedText": "Paris.\n\n2.", "torch": "2.13.0+cu130"}
```

And then serve it — back in the interpreter with nothing installed in it:

```bash
export TLALOC_PJRT_PLUGIN_PATH=/path/to/xla_cuda_plugin.so
/usr/bin/python3 examples/gpu-inference/serve.py \
  --artifact /tmp/tl-llama-example \
  --prompt 1,450,7483,310,3444,338 --max-new 6 --verify-weights
```

Real output, on the same GB10:

```
the entire runtime of this process:
  PJRT plugin   .../jax_plugins/xla_cuda12/xla_cuda_plugin.so
  platform      cuda (requested; the client's own answer is below)
  Python        /usr/bin/python3 (3.12.3)
  frameworks    none installed in this interpreter (jax, torch, numpy: not found)

artifact      /tmp/tl-llama-example
  model       TinyLlama__TinyLlama-1.1B-Chat-v1.0
  hash        hf-llama:TinyLlama__TinyLlama-1.1B-Chat-v1.0:L22:h2048:v32000
  layers      22   vocab 32000
  bodies      1 compiled entry, all re-hashed OK
  weights     201 staged operand files
  engine      CtypesEngine on cuda

greedy decode  prompt [1, 450, 7483, 310, 3444, 338] + 6 new (compiled context 64 = 4 pages x 16)
  step  0  prompt  token      1  -> argmax    529     6626.1 ms
  step  1  prompt  token    450  -> argmax  29871     1355.9 ms
  step  2  prompt  token   7483  -> argmax    310     1321.6 ms
  step  3  prompt  token    310  -> argmax    278     1359.7 ms
  step  4  prompt  token   3444  -> argmax  29892     1287.2 ms
  step  5  prompt  token    338  -> argmax   3681     1362.2 ms
  step  6  gen     token   3681  -> argmax  29889     1285.8 ms
  step  7  gen     token  29889  -> argmax     13     1352.7 ms
  step  8  gen     token     13  -> argmax     13     1261.1 ms
  step  9  gen     token     13  -> argmax  29906     1354.4 ms
  step 10  gen     token  29906  -> argmax  29889     1257.4 ms
  step 11  gen     token  29889  -> argmax    350     1347.9 ms

prompt     [1, 450, 7483, 310, 3444, 338]
generated  [3681, 29889, 13, 13, 29906, 29889]
  XLA compiles     1
  first step       6626 ms  (weight upload + compile)
  median step      1348 ms
```

`[3681, 29889, 13, 13, 29906, 29889]` is what HuggingFace transformers
produced, token for token — **`Paris.\n\n2.`**, six for six. Run the two
commands yourself and compare the two lists; that comparison is the claim.

**Why token ids and not a logit tolerance.** XLA-GPU lowers a default-precision
f32 `dot_general` through TF32 while the oracle is fp32 on CPU, so a logit
tolerance here would be a number chosen to pass. An argmax is not. Greedy
decoding agrees *exactly* until the two arithmetics disagree about a top-1, so
the honest claim is the **length of the prefix that agrees** — asserted here at
the full requested budget, with no divergence to report at 6 tokens.

**The 1.35 s median is not a throughput claim, and the reason is worth knowing.**
The KV pools still round-trip to the host every step as flat Python lists — 44
pools per token, built and unpacked in pure Python. The fix is buffer donation,
which has ridden `donationPairs` in the manifest since the pools were designed
and is now, measurably, the dominant cost of a real decode. The number above is
what it costs today, printed by the example rather than hidden by it.

---

## What to look at in the source

| file | the interesting part |
|---|---|
| `src/main/kotlin/Export.kt` | `exportReference` and `exportLlama` — two ~20-line functions with one `export` call each. Note the dependency list in `build.gradle.kts`: no `runtime-pjrt`. |
| `serve.py` | the imports (there are five, all stdlib), `frameworks()`, and the page arithmetic in the decode loop. |

## What does NOT work here, by name

* **`vllm serve` / `LLM.generate()`.** vLLM's platform discovery, config hook
  and whole v1 worker API are certified live against vLLM 0.29.0, and the same
  artifact called through vLLM's worker and called directly agrees bit for bit
  — but one classmethod the v1 engine core calls unconditionally is still
  refused by name, so `vllm serve` is not a thing this example can show you.
* **TPU.** `serve.py --platform tpu` is the whole change, `/lib/libtpu.so` is
  already on a Cloud TPU VM image, and the artifact is hardware-neutral by
  construction — but **nothing in this repo has ever run on a TPU**, so that
  path is untested and this example does not claim it. See `docs/TPU_BRINGUP.md`.
  Note that `--platform` is a *request*, not a device: a CUDA plugin handed
  `--platform tpu` opens CUDA anyway. §0.4.490 found this example printing the
  flag back as if it were the hardware, and now it prints what the live client
  answers instead, plus a `MISMATCH` line when the two disagree:

  ```
    platform      tpu (requested; the client's own answer is below)
    engine      CtypesEngine on cuda
    MISMATCH    you asked for 'tpu' and the plugin opened 'cuda'. The plugin .so, not the flag, chose the device.
  ```
* **Prefill as one call.** The prompt runs as N decode steps: the ragged
  chunked-prefill form of `PAGED_ATTENTION` is an open performance deferral.
* **Sampling.** Greedy argmax only, host-side, in the driver.
* **bf16 weight tables.** The writer refuses a non-f32 staged slot by name;
  halving the artifact needs the graph itself to be bf16.
* **A tokenizer anywhere in the runtime.** Ids in, ids out, deliberately.

The full runbook for all of this, with what is certified and what is not, is
[`docs/SERVING_RUNBOOK.md`](../../docs/SERVING_RUNBOOK.md).
