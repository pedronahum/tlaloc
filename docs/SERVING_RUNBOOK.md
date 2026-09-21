# Serving runbook — export an artifact, run it, plug it into vLLM

**Status (§0.4.476, H6b).** This is the reproduction script for the whole
serving path: it takes a fresh machine to a Tlaloc serving artifact, runs
that artifact from Python with **no JVM, no JAX, no torch and no numpy in
the process — a PJRT plugin `.so` and a driver are the entire runtime** —
and hands it to vLLM through the `vllm-tlaloc` platform plugin. Every step is marked **CERTIFIED** (a test in `./gradlew test`
proves it) or **UNCERTIFIED** (written, never executed here, with the
reason and the command that would settle it).

The design and the decisions behind all of this live in
[INFERENCE_SERVING_AUDIT.md](INFERENCE_SERVING_AUDIT.md); its §5 ARC
STATE block is the ledger. This file is the *commands*.

The one sentence that motivates the whole shape: **the artifact is the
deployment.** After step 2 returns, a directory of JSON and textual
StableHLO is the entire input to serving; nothing downstream calls back
into Kotlin, and the same directory is what will serve on TPU the day
G2b lands ([TPU_BRINGUP.md](TPU_BRINGUP.md)).

---

## 0. The machine

Certified on: **GB10 / DGX Spark, aarch64, DGX OS**, CUDA 13 driver with
the CUDA-12 jaxlib plugin, JDK 25 LTS at `~/.local/jdks/jdk-25.0.3+9`.

```bash
export JAVA_HOME=~/.local/jdks/jdk-25.0.3+9      # every gradle command
export TLALOC_VENV=~/.local/venvs/iree            # the project venv
```

What the venv must already have, and why. **Read the middle column
carefully: since §0.4.476 the serving path needs none of it.**

| package | needed to SERVE? | why it is here |
|---|---|---|
| `jax` + `jaxlib` (+ `jax-cuda12-plugin`) | **no** | it is where a `xla_cuda_plugin.so` happens to sit on this box, and it is the ORACLE for the 1e-5 XLA-CPU semantics lane (jaxlib ships no CPU PJRT plugin `.so`) |
| `numpy` | **no** | the jax oracle engine's array type. The ctypes loader's wire format is flat Python lists |
| `torch`, `safetensors` | **no** | oracles only (H2's bf16 parity reads bytes torch wrote) |
| a PJRT plugin `.so` | **YES — and it is the only one** | `tlaloc_pjrt` dlopens it and calls `GetPjrtApi` |

### What a SERVING machine actually needs (§0.4.476)

```bash
python -m venv /srv/tlaloc-venv && . /srv/tlaloc-venv/bin/activate
pip install -e /home/pedro/programming/tlaloc/harness/python   # zero dependencies
export TLALOC_PJRT_PLUGIN_PATH=/path/to/xla_cuda_plugin.so
```

`pip install -e harness/python` pulls **nothing**: the distribution's
`dependencies` list is empty and pinned empty by a test
(`vllm_tlaloc_test.LoaderIsStandardLibraryOnly` / the registration lane).
The plugin `.so` comes from one of exactly three places, and the loader
looks for them in this order — all three are FILE lookups, never imports:

1. `TLALOC_PJRT_PLUGIN_PATH` — what a deployment that ships its own plugin
   sets. Same variable `PjrtBinaries` reads JVM-side, so one export
   configures both halves of the box.
2. `/lib/libtpu.so` — a Cloud TPU VM's plugin, already on the image (G2b).
3. `<site-packages>/jax_plugins/*/xla_cuda_plugin.so` — a jax install used
   as a *place a file sits*. This is the convenience path on THIS machine
   and it is a directory walk, not an `import jax_plugins`.

### The development venv (the oracles)

```bash
python -m venv "$TLALOC_VENV" && . "$TLALOC_VENV/bin/activate"
pip install "jax[cuda12]" numpy            # CPU-only lane: pip install jax numpy
python -c "import jax; print(jax.devices())"
```

**This is the only sanctioned write to that venv.** Once it exists it is
frozen — read §0.1 before typing `pip` at it again.

`jax.devices()` printing a `CudaDevice` is the gate for the CUDA lane;
the CPU lane needs nothing further and is the one the tolerances are
tightest on (see §3).

---

## 0.1 `~/.local/venvs/iree` is FROZEN — the oracle venv (§0.4.474)

**Never `pip install`, `pip install -U` or `pip uninstall` anything into
`~/.local/venvs/iree`.** Creating it from nothing on a fresh machine (the
recipe above) is the one sanctioned write. After that it is frozen
infrastructure, and this is the most consequential rule in the runbook.

### Why

That venv is not a convenience, it is **the measurement apparatus**. It
is where every cross-language claim in this repository gets its *other
side*, and none of those claims fail loudly when it moves:

| what runs there | what it certifies |
|---|---|
| `run_pytorch_llama.py`, `run_pytorch_llama_grad.py`, `run_pytorch*.py` | the PyTorch agreement harness — Tlaloc's forward **and its gradients** |
| `write_llama_safetensors.py` | H2's bit-exact safetensors parity (torch writes the bytes we then claim to read) |
| `tlaloc_serve.py`, `run_tlaloc_serve_check.py` | §4's "a Python process with no JVM runs the artifact", at 1e-5 CPU / 1e-3 CUDA |
| the `vllm_tlaloc` package, `run_vllm_tlaloc_check.py` | the plugin contract and the two-step decode runner |
| `jax_plugins/xla_cuda12/xla_cuda_plugin.so` | **every PJRT lane in the repo, JVM-side included** — `PjrtBinaries` resolves the plugin out of this venv's site-packages |
| `iree-base-compiler`, `iree-base-runtime` | the IREE compile-and-run lanes |

A `pip install` that swapped `torch 2.11.0+cpu` for a CUDA build, or
bumped `jax` one minor version, or set CUDA-13 wheels down beside jax's
CUDA-12 plugin, would leave `./gradlew test` **green** while silently
moving every number this repo compares itself against. That is not a
build break; it is a quiet redefinition of "correct".

`torch` is pinned **including its `+cpu` local tag**, and the tag is the
point: this torch is a *numerical reference*, not an accelerator. A CUDA
torch here is a silent swap of the instrument, not an upgrade.

### The precedent (§0.4.470)

The H3b agent needed vLLM and dry-ran `pip install vllm` into this venv.
The plan came back with **186 packages**, `torch 2.13` replacing the
oracle torch, **33 CUDA-13 wheels** beside jax's CUDA-12 plugin, and a
numpy downgrade. It refused — which is why the live-vLLM certification
is still recorded in
[INFERENCE_SERVING_AUDIT.md](INFERENCE_SERVING_AUDIT.md) §5 as a named
deferral rather than as a number nobody could reproduce.

### Instead: give the slice its own venv

```bash
python -m venv ~/.local/venvs/<slice> && . ~/.local/venvs/<slice>/bin/activate
pip install <whatever that slice needs>
# point the harness at it explicitly; do NOT re-point TLALOC_VENV
```

Venvs are cheap and disk is not the constraint. A second copy of torch
costs a few GB; a moved oracle costs the arc's credibility.

### The canary

`OracleVenvIntegrityTest` (in `:maestro:jvmTest`, beside the other
Python-subprocess certifications) runs
`harness/python/check_oracle_venv.py` in that interpreter and asserts the
stack is intact: `jax`/`jaxlib` `0.10.0`, the `jax-cuda12-plugin` /
`jax-cuda12-pjrt` family **version-matched to jax**, `torch 2.11.0+cpu`
with `torch.version.cuda is None`, `numpy` / `safetensors` / the IREE
pair present, and **no `jax-cuda13` or `*-cu13` wheel anywhere**. The
failure message says why it matters and hands over the separate-venv
recipe.

Two design points worth keeping:

- **No `TLALOC_TORCH_PYTHON` override**, unlike every other subprocess
  certification here. Those ask "is there an interpreter that can run my
  reference?" and any answer will do; this one asks "is *the* oracle
  intact?", and an override would let the thing under test point the
  test elsewhere.
- **The policy is a pure function, and it is tested against a venv that
  is wrong** — the §0.4.470 inventory (torch 2.13, a `jax-cuda13`
  plugin, `nvidia-cublas-cu13`) must produce a problem naming each. A
  canary that has only ever seen a healthy venv is an untested canary.

**Bumping a pin in that file is a deliberate act**: it means re-running
the oracles that depend on it — at minimum the PyTorch agreement
harness, the safetensors parity test and both serving lanes — and saying
so in the commit that moves it. It is not maintenance.

Self-skips: no venv at the path, or a venv there carrying neither `jax`
nor `torch` (some other venv, not the oracle). A fresh clone stays green.

---

## 1. Build and certify (CERTIFIED)

```bash
cd /home/pedro/programming/tlaloc
JAVA_HOME=~/.local/jdks/jdk-25.0.3+9 ./gradlew test --rerun-tasks 2>&1 | tee /tmp/cert1.log | tail -5
JAVA_HOME=~/.local/jdks/jdk-25.0.3+9 ./gradlew \
  :vendored-maestro:maestro-tlaloc:test :vendored-maestro:maestro-common:test \
  --tests "*KubernetesCommandTlalocFieldsTest*"
./scripts/count-tests.sh
```

**Use `--rerun-tasks`, not `--rerun`.** `--rerun` only forces the task
you named; in a multi-module KMP build the per-module `:<m>:jvmTest`
lanes stay `UP-TO-DATE` and the "clean-room recount" recounts yesterday's
XML. This was found at this close-out (a 31-second "full suite"), and it
is the strongest form of the §0.4.470 landmine below.

**Landmine (§0.4.470).** `:maestro:jvmTest` does **not** re-run when only
`harness/python/**` changes — the Python files are not declared task
inputs. Any edit to `tlaloc_serve.py` or `vllm_tlaloc/**` needs
`--rerun-tasks` or it certifies the *old* Python.

---

## 2. Export a serving artifact (CERTIFIED)

The exporter is `io.tlaloc.maestro.serving.ServingArtifactWriter.export`,
and the reference model's entry point is
`io.tlaloc.maestro.serving.ReferenceDecodeGraphKt.main`:

```bash
JAVA_HOME=~/.local/jdks/jdk-25.0.3+9 ./gradlew :maestro:jvmJar
CP=$(find . -path '*/build/libs/*-jvm-*.jar' -o -path '*/build/libs/*.jar' | tr '\n' ':')
CP="$CP$(find ~/.gradle/caches/modules-2 -name 'kotlin-stdlib-2*.jar' | head -1)"
"$JAVA_HOME/bin/java" -cp "$CP" io.tlaloc.maestro.serving.ReferenceDecodeGraphKt \
  /tmp/tlaloc-serving-artifact
```

**Named deferral: there is no `./gradlew exportServingArtifact` task.**
The `main` exists and is the sanctioned entry point; a JavaExec task that
assembles this classpath is a one-file build change and was left out of
the docs-only close-out on purpose. Until it lands, the *certified* way
to produce an artifact is the export test itself, which writes one into a
temp directory on every run:

```bash
JAVA_HOME=~/.local/jdks/jdk-25.0.3+9 ./gradlew :maestro:jvmTest --rerun-tasks \
  --tests "*ServingArtifactExportRunTest*"
```

### What lands on disk

```
  <artifact>/
    tlaloc-serving.json          the ServingManifest — model shape, KV-pool
                                 axis order, the bucket ladder, the weights
                                 pointer, one entry per compiled ladder point
    bodies/<sha256>.mlir         StableHLO+SDY, content-addressed and
                                 de-duplicated; @main is the entry point
    programs/<entryId>.json      one :maestro ProgramManifest per entry,
                                 unmodified, cross-checked against the entry
    weights/…                    pointed at, not necessarily present
                                 (v1: `embedded: true`, weights are body consts)
```

Two exports of one model are **byte-identical** — the body name is a
content address, not a timestamp. `grep` is a legitimate debugging tool
on this directory, which is why the bodies are textual MLIR.

### Exporting *your* model instead of the reference

`export(dir, modelName, modelHash, model, ladder, specs, weights, build)`
takes a `build: (DecodeGraphSpec) -> DxirFunction`. The contract the graph
must satisfy is `DecodeGraphSpec.verifySignature`, which the exporter runs
on **every** graph at export time — a signature mismatch found in Python
is an XLA shape error with the diagnosis removed.

```
  in   tokenIds [B,T] I32 · positions [B,T] I32 · blockTables [B,maxBlocksPerSeq] I32
       seqLens [B] I32 · slotMapping [B*T] I32 · (key,value) pools × numLayers
  out  logits [B,T,V] · the same pools, updated
```

**Weights.** H2's reader
(`io.tlaloc.core.io.SafetensorsFile.openCheckpoint(dir)`) loads an HF
safetensors checkpoint — single file or sharded `model.safetensors.index.json`
— and brands each tensor with `asF32<S>()` / `asBf16` / `asI32` / `asF64`.
F16, I64 and fp8 are **refused by name**. In v1 the weights end up as graph
constants; **staged weights and the HF name-mapping are H3c**, the arc's
largest open item.

---

## 3. Run the artifact from Python (CERTIFIED, both lanes)

No JVM, no gradle, no Kotlin in this process — and since §0.4.476 no
framework either. The directory plus a plugin `.so` is the input.

```bash
. "$TLALOC_VENV/bin/activate"
export PYTHONPATH=/home/pedro/programming/tlaloc/harness/python
python - <<'PY'
import tlaloc_serve                      # stdlib + ctypes; no jax, no numpy

print(tlaloc_serve.find_pjrt_plugin())   # the entire runtime dependency
art = tlaloc_serve.ServingArtifact.load("/tmp/tlaloc-serving-artifact", platform="cuda")
art.verify_bodies()                       # re-hash every body against its filename
print(art.manifest["model"])
b, c = art.select_bucket(batch=3, context=4)   # rounds UP; refuses over-cap BY NAME
entry = art.entry_for("DECODE", b, c)
print(entry.entry_id, entry.cache_key)
PY
```

`select_bucket` **refuses** an over-cap request rather than clamping it: a
clamped context silently truncates a user's history, which is a wrong
answer dressed as a slow one, and only a scheduler can split the request.

The padding convention is a **wire contract between two processes** and is
duplicated in `tlaloc_serve.py` deliberately and loudly:

| field | padded value | why |
|---|---|---|
| `tokenIds` | `0` | |
| `positions` | `0` | there is no correct position to derive for a dead row |
| `blockTables` | `0` | page 0 is the reserved scratch page — **never allocated to a live sequence** |
| `slotMapping` | `-1` | `KV_CACHE_WRITE` drops an out-of-bounds scatter update |
| `seqLens` | **`1`, never `0`** | `0` masks every lane to −Inf ⇒ `0/0 = NaN` in the emission |

`check_padding_constants()` exists so the certification pins the two
copies together instead of trusting a comment.

Host arrays are **flat row-major Python lists**, not ndarrays (§0.4.476):
`empty_pools()` hands them out in the form `run_decode` takes, and
`run_decode` returns `logits` nested to the manifest's declared trailing
dims — `[tokensPerSeq, vocab]` per real row, token axis KEPT even at 1 so
`last_token_logits` still works the day a prefill entry exists. Shapes come
from the manifest, which is where they were authoritative anyway.
`tlaloc_serve.unflatten` / `.flatten` convert.

### The certified end-to-end check

```bash
python harness/python/run_tlaloc_serve_check.py \
  --artifact /tmp/tlaloc-serving-artifact \
  --request /tmp/req.json --output /tmp/result.json --platform cuda
```

Exit codes: `0` ran · `1` the run failed (the interesting failure) · `2`
the environment cannot run it at all (no plugin, no device, no jax for the
oracle lane) — the Kotlin side self-skips on 2 and fails on 1.
`ServingArtifactExportRunTest` drives exactly this, builds the request, and
compares against the host interpreter.

**Which engine ran, and why there are two (§0.4.476).**
`--engine ctypes` is the default and the deployment path: `tlaloc_serve`
compiles and executes through `tlaloc_pjrt`, and the whole run happens under
a `sys.meta_path` guard that raises on `jax`, `jaxlib`, `torch` and `numpy`
— in the venv where jax *is* installed. `--engine jax` is an **oracle** and
nothing else: jaxlib ships no CPU PJRT plugin `.so` (its CPU client is a C++
class inside the jaxlib extension), so the tight 1e-5 XLA-CPU *semantics*
lane has nothing for ctypes to `dlopen`. `tlaloc_serve.default_engine_for`
is the one place that rule is written down — `cpu` gets jax, every
accelerator platform gets ctypes.

**The two floors, and why there are two.** The identical artifact runs on
the PJRT **CPU** client at **1e-5** relative and the PJRT **CUDA** client
at **1e-3** relative. The looseness is *measured*, not hedged: XLA-GPU
lowers a default-precision f32 `dot_general` through **TF32**, putting the
step ~5e-4 from the host while the CPU lane sits at ~1e-7 on the same
bytes. Two claims escape the tolerance entirely and are pinned with exact
`==` on **both** lanes — the poisoned pools (a misplaced write is three
orders of magnitude out) and the padded row's scratch page, because an
untouched slot is copied, not computed.

---

## 3.1 PJRT without a framework — `tlaloc_pjrt.py` (CERTIFIED, §0.4.475)

`tlaloc_serve.py` above reaches PJRT through **jaxlib**. `tlaloc_pjrt.py`
does the same job with **`ctypes` and the standard library only** — no jax,
no jaxlib, no torch, no numpy. It is a mirror of the JVM's FFM binding
(`runtime-pjrt/.../ffm/PjrtFfm.kt`), § numbers cited in place.

The only thing it needs is a PJRT plugin `.so`, and that file can live
anywhere — inside someone else's jax install, at `/lib/libtpu.so` on a TPU
VM, or in a directory a deployment ships:

```bash
export TLALOC_PJRT_PLUGIN_PATH=$HOME/.local/venvs/iree/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
export PYTHONPATH=/home/pedro/programming/tlaloc/harness/python
python3 - <<'PY'                     # any python3 — nothing is installed
import tlaloc_pjrt as P
api = P.PjrtApi.load()               # dlopen + GetPjrtApi
with api.create_client(platform="cuda") as client:   # create_options ALWAYS (§0.4.333)
    dev = client.addressable_devices()[0]
    print(client.platform_name())
    with client.compile(open("body.mlir").read()) as exe:
        x = client.buffer_from_host_f32(dev, [1.0, 2.0], [2])
        outs = exe.execute([x], dev)
        print(outs[0].to_f32(2))
        for b in outs: b.close()
        x.close()
PY
```

**Never create a CUDA client without create_options.** The binding refuses
it by name: with none, the plugin defaults to `preallocate=true`,
`memory_fraction=0.75`, which on this unified-memory box pins ~98 GB per
client and hangs the machine (§0.4.333). A TPU client, by contrast, gets
**no** options at all — those knobs are the XLA GPU plugin's (§0.4.459).

The two certification modes, both run by `PjrtCtypesBindingTest`:

```bash
# ABI mirror — no plugin, no GPU, runs anywhere
python3 harness/python/run_pjrt_ctypes_check.py --layouts /tmp/layouts.json

# real XLA: compile + execute four graphs, jax/jaxlib/torch/numpy BLOCKED
python3 harness/python/run_pjrt_ctypes_check.py --run /tmp/job.json /tmp/out.json
```

Both report a `guard` block: the driver installs a `sys.meta_path` finder
that raises on those four packages *before* importing anything, runs the
whole PJRT path underneath it, and then deliberately imports jax to prove
the guard fires. The JVM asserts both.

**Still true today**: `tlaloc_serve.py` has not been re-pointed at this
binding (that is H6b). The framework-free claim is about the binding, not
yet about the loader.

## 4. Plug it into vLLM (plugin CERTIFIED below vLLM's API, live path UNCERTIFIED)

```bash
pip install -e harness/python      # registers the vllm.platform_plugins entry point
export TLALOC_SERVING_ARTIFACT=/tmp/tlaloc-serving-artifact
python -c "from vllm.platforms import current_platform; print(current_platform)"
vllm serve <tokenizer/config> --max-num-seqs 4 --max-model-len 4 --block-size 2
```

The first command is the **discovery** check (vLLM's platform resolution
must land on `TlalocPlatform`); the second is a single-request
`generate()`. **`--block-size` must equal the artifact's compiled
`blockSize` and `--max-model-len` must lie on the ladder** —
`check_and_update_config` refuses rather than adjusts, because a different
block size is a different KV layout, not a preference.

### vLLM is NOT installed here, and that is a decision

`pip install vllm` **resolves** on this box (aarch64, CPython 3.12,
vllm 0.29.0). A `--dry-run --report` stated the closure exactly: **186
packages**, including **torch 2.13.0** (replacing this venv's
`torch 2.11.0+cpu`), **33 CUDA-13 wheels** landing beside the venv's
`jax-cuda12-plugin 0.10.0`, and a numpy downgrade 2.4.4 → 2.3.5. That venv
is the measurement apparatus for every oracle in this repo. Install vLLM
**in a venv of its own**, with jaxlib present, and settle the
CUDA-13-beside-CUDA-12 question there:

```bash
python -m venv ~/.local/venvs/vllm && . ~/.local/venvs/vllm/bin/activate
pip install vllm                     # jax NO LONGER REQUIRED — see below
pip install -e /home/pedro/programming/tlaloc/harness/python   # zero deps
export TLALOC_PJRT_PLUGIN_PATH=/path/to/xla_cuda_plugin.so
```

#### What the vLLM venv will and will not need (§0.4.476)

**Will need:** vLLM itself and its own closure (torch included — vLLM's
scheduler, tokenizer and API server are torch-native and that is the host's
business, not ours); `vllm-tlaloc` installed for its entry point; and a PJRT
plugin `.so` reachable through `TLALOC_PJRT_PLUGIN_PATH`.

**Will NOT need:** `jax`, `jaxlib`, `jax-cuda12-plugin`, or numpy on
Tlaloc's account. Before H6b the recipe above said `pip install vllm
"jax[cuda12]"`, and that second half was the thing that made the venv
question hard — jax's CUDA-12 wheel family landing beside vLLM's CUDA-13
one. `tlaloc_serve` no longer imports any of it. What remains is a single
`.so`, and the honest way to get one into that venv without installing jax
is to copy it (or point at the oracle venv's copy, which is a read of a file
and changes nothing in it — **that is not a write to the frozen venv, and it
is the only interaction with it this recipe has**):

```bash
export TLALOC_PJRT_PLUGIN_PATH=$HOME/.local/venvs/iree/lib/python3.12/\
site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
```

**Still open, and H6b does not close it:** whether vLLM's CUDA-13 torch and
the CUDA-12 XLA plugin coexist in one process. H6b removes jax from the
question; it does not answer it. That is the live-serving certification's
first finding, whichever way it goes.

### What is certified without vLLM present

The plugin is split along exactly one line — whether a file imports vLLM:

| file | imports vLLM? | certified? |
|---|---|---|
| `__init__.py` — `register()`, `PLATFORM_CLASS_PATH` | no | **yes** |
| `paging.py` — KV page pool, block tables, slot arithmetic | no | **yes** |
| `batching.py` — requests → one padded bucket-selected call | no | **yes** |
| `runner.py` — `TlalocModelRunner`: artifact + pools + sampling | no | **yes** |
| `platform.py` — `TlalocPlatform` | **yes** | **no** |
| `worker.py` — `TlalocWorker` | **yes** | **no** |

`VllmPluginContractTest` runs three lanes inside `./gradlew test`:
registration metadata (entry point ↔ `register()` ↔ `PLATFORM_CLASS_PATH`
↔ the class in the file the path names), 31 stdlib-only `unittest` cases
over the page pool and the marshalling, and **two decode steps of three
sequences** driven by `TlalocModelRunner` against a real exported artifact
on both PJRT clients. Two steps and not one: a single step cannot tell a
KV cache threaded across steps from one recomputed.

You can drive the vLLM-free unit lane yourself:

```bash
python harness/python/vllm_tlaloc_test.py
python harness/python/run_vllm_tlaloc_check.py --help
```

### Driving the runner directly (no vLLM at all)

```python
from vllm_tlaloc.runner import TlalocModelRunner
r = TlalocModelRunner("/tmp/tlaloc-serving-artifact", platform="cuda")
r.add_sequence(0, [7])
print(r.generate(0, [7], max_new_tokens=4))     # greedy, host-side sampling
r.free_sequence(0)
```

A prompt longer than one token is **refused by name**: there is no
PREFILL entry in the artifact yet, and a runner that quietly decoded a
prompt one token at a time would be "working" while doing what no serving
system accepts.

---

## 5. The KPTX paged-attention kernel (CERTIFIED, and deliberately opt-in)

Tlaloc's own PTX can replace the gather-composed `PAGED_ATTENTION`
lowering inside the XLA executable. The claiming pass keys on
**`OpKind`**, and its decline path is *the op itself* — so a machine with
no KPTX tier runs the same program with the same numbers.

`defaultInferenceKernelTemplates` is **EMPTY on purpose**, and should stay
that way in any deployment today. The correctness-tier kernel measured
**465 µs vs 310 µs** for the lowering it replaces — 1.5× slower, the
expected §0.4.358 outcome for three f32 scalar loops against XLA's tiled
tensor cores. It is the *claiming* milestone, not a speedup. The
performance tier (warp specialization, shared-memory staging of the page
window) is the named follow-up, and 465 µs is the floor it must beat.

To opt in, a pipeline passes `kptxInferenceKernelTemplates` and the
serving process registers the chain
(`io.tlaloc.runtime.pjrt.kptx.KptxPagedAttention.register`).

**Landmine (§0.4.471).** `ptxas` rejects a **non-ASCII byte anywhere in
the PTX file, comments included**, and the failure surfaces as
`CUDA_ERROR_INVALID_PTX` out of `cuModuleLoadData` deep inside an XLA
execution, naming nothing. Pinned in `PagedAttentionModuleTest`.

---

## 6. KV-quant (contract CERTIFIED, bytes DEFERRED)

`KvQuantPool` is the symmetric-absmax codec; `OpKind.DEQUANTIZE_KV` is its
in-graph half; `ServingModelShape.kvQuant` carries it in the manifest. The
error bound is **derived, not tuned**: `|x − x̂| ≤ scale/2 = absmax/254`
for int8, and `maxRoundTripError` recomputes it so the tests assert the
measurement against the derivation.

Read the manifest's two fields together, because today they disagree and
that is the point:

- `dtype: "int8"` — what the codes **mean** (the accuracy story),
- `codeDtype: "i32"` — what the codes **ride** (the byte budget).

There is no `I8` DType yet, so **v1 buys the contract, not the bytes**. A
deployment sizing a KV pool reads `codeDtype`; one reasoning about answer
quality reads `dtype`. **fp8 is refused by name** — it is a float format
whose code carries its own exponent, so `value = code * scale` is simply
not its dequantization.

---

## 7. SGLang (UNCERTIFIED — design only)

The precedent is **SGL-JAX**, and the integration surface is a *model
runner*. The prediction manifest-as-artifact makes: the loader, the
manifest reader, the bucket selection and the PJRT execution path are
**frontend-agnostic**; what a second frontend rewrites is the adapter
class and its `ForwardBatch → slot` translation. That prediction is
**untested** until an environment carries SGLang, for the same venv reason
as vLLM (`pip install sglang` pulls a CUDA torch stack and a compiled
kernel library). The one IR-level item SGLang's radix path would actually
need from this side is the **ragged/chunked-prefill form of
`PAGED_ATTENTION`**, H1a's named deferral since the first slice.

---

## 8. On TPU (GATED — the artifact is ready, the hardware is not)

Nothing in the artifact mentions CUDA. The bodies are StableHLO+SDY, the
manifest is JSON, and `tlaloc_serve.ServingArtifact.load(..., platform=…)`
takes the platform as a parameter. On a Cloud TPU VM the whole of §3 is
the same commands with `platform="tpu"`, once G2b turns
`PjrtTpuSmokeTest`'s self-skips into passes — the step-by-step is
[TPU_BRINGUP.md](TPU_BRINGUP.md), and the ordered queue is
[TPU_READINESS_AUDIT.md](TPU_READINESS_AUDIT.md) §5.

What a TPU session owes this document, specifically: re-run §3 on the
`tpu` platform and record **its** floor, because §3's two floors are
CPU's and XLA-GPU-TF32's and a TPU's default `dot_general` precision is
neither.

---

## 9. Troubleshooting, by symptom

| symptom | cause |
|---|---|
| `BUILD SUCCESSFUL in 31s` on a "full" suite | `--rerun` instead of `--rerun-tasks`; the module `jvmTest` lanes were `UP-TO-DATE` |
| a Python edit changes nothing | `:maestro:jvmTest` does not declare `harness/python/**` as inputs (§0.4.470) |
| `CUDA_ERROR_INVALID_PTX` naming nothing | a non-ASCII byte in a PTX comment (§0.4.471) |
| NaN logits on a padded batch | `seqLens = 0` on a padding row; the convention is **1** |
| a padded row reads live KV | page 0 was allocated to a sequence; it is the reserved scratch page |
| vLLM says "no platform found" | the distribution is not installed (`pip install -e harness/python`) |
| `--block-size` refused | it disagrees with the artifact's compiled `blockSize`; that is a KV layout, not a preference |
| the machine reboots under XLA | never create a PJRT client without `create_options` (§0.4.333) |
