# Serving runbook — export an artifact, run it, plug it into vLLM

**Status (§0.4.483, the H3c + H4b close-out).** This is the reproduction script for the whole
serving path: it takes a fresh machine to a Tlaloc serving artifact, runs
that artifact from Python with **no JVM, no JAX, no torch and no numpy in
the process — a PJRT plugin `.so` and a driver are the entire runtime** —
and hands it to vLLM through the `vllm-tlaloc` platform plugin, which since
§0.4.477 is **run against a real vLLM 0.29.0 in a venv of its own** rather
than merely written. Every step is marked **CERTIFIED** (a test in
`./gradlew test` proves it) or **UNCERTIFIED** (written, never executed
here, with the reason and the command that would settle it).

**Start at §10 if you want the demo.** As of §0.4.480 a real
TinyLlama-1.1B — all 22 layers, from its own HuggingFace checkpoint —
greedy-decodes through an exported artifact on PJRT-CUDA and produces the
**same token ids HuggingFace transformers produces**: `Paris.\n\n2.`, six
for six. §§1–4 are the machinery under it, in the order it was built.

Exactly one serving step is still UNCERTIFIED and it is §4's last block:
`vllm serve` / `LLM.generate()`. After §0.4.480 attempted it, the reason is
no longer a missing model — it is **one classmethod**,
`get_attn_backend_cls`, which vLLM 0.29.0's v1 engine core calls
unconditionally and which the plugin refuses by name. That is H3c-4.

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

§0.4.503 made the JVM half as general as the Python half. `PjrtBinaries`
used to fall back to one literal string —
`~/.local/venvs/iree/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so`,
one venv name, one Python minor version, one plugin package — and it now
globs all three across `$VIRTUAL_ENV`, `~/.local/venvs/*`, `~/.venv`,
`~/venv`, `~/.local`, `/usr/local` and `/usr`. `PjrtBinaries.pluginSearchReport`
prints every location it tried and what was at each, which is what the
examples' "GPU lane unavailable" reason now carries.

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

```bash
JAVA_HOME=~/.local/jdks/jdk-25.0.3+9 ./gradlew :maestro:exportServingArtifact \
  -PoutDir=/tmp/tlaloc-serving-artifact
```

Six entries and their bodies land in that directory (§0.4.477). The task is
a `JavaExec` over the **jvmMain** runtime classpath plus the unpublished
`tools` compilation that holds `main` — not jvmTest, because
if the exporter needed a test fixture the artifact would be a test fixture,
and the claim that the directory is the deployment would be a claim about
the test source set.

The export test also writes one into a temp directory on every run, and is
still the thing that certifies the bytes:

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

## 4. Plug it into vLLM (CERTIFIED through the worker API, §0.4.477)

### The venv recipe that worked — copy this one

Verified on this box on 2026-09-21. **~25 minutes**, ~7.7 GB, 197
distributions, and **not one byte written to `~/.local/venvs/iree`.**

```bash
python3 -m venv ~/.local/venvs/vllm                  # FRESH. Its own venv. Always.
~/.local/venvs/vllm/bin/python -m pip install --upgrade pip
~/.local/venvs/vllm/bin/python -m pip install --only-binary=:all: vllm
~/.local/venvs/vllm/bin/python -m pip install -e /home/pedro/programming/tlaloc/harness/python

export TLALOC_SERVING_ARTIFACT=/tmp/tlaloc-serving-artifact
export TLALOC_PJRT_PLUGIN_PATH=$HOME/.local/venvs/iree/lib/python3.12/\
site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
~/.local/venvs/vllm/bin/python -c \
  "from vllm.platforms import current_platform; print(current_platform)"
```

Three things in that recipe are load-bearing:

* **`--only-binary=:all:`.** A source build of any one of 197 packages on
  aarch64 is a stall with no upper bound. Wheels exist for all of them; the
  flag turns "this will finish or tell you why" into a property of the
  command instead of a hope. Drop it and the slice's time box is gone.
* **No `jax`.** The older version of this recipe said `pip install vllm
  "jax[cuda12]"`, and that second half was the hard part — jax's CUDA-12
  wheels beside vLLM's CUDA-13 ones. Since H6 the serving path is ctypes,
  so **do not install jax here.** `pip install -e harness/python` pulls
  nothing at all.
* **`TLALOC_PJRT_PLUGIN_PATH` pointing into the oracle venv.** That is a
  `dlopen` of a file, not an import of that venv's Python, and nothing in
  that directory is written. It is the *only* interaction this recipe has
  with the frozen venv. (A deployment ships its own `.so`; this is the
  convenience path on this machine.)

The `print(current_platform)` must say `TlalocPlatform`, and vLLM logs
`Platform plugin tlaloc is activated`. **`--block-size` must equal the
artifact's compiled `blockSize` and `--max-model-len` must lie on the
ladder** — `check_and_update_config` refuses rather than adjusts, because a
different block size is a different KV layout, not a preference.

### What §0.4.477 certified, and the one thing it did not

`VllmLivePluginTest` (in `./gradlew test`, self-skipping when
`~/.local/venvs/vllm` is absent) runs, live against **vLLM 0.29.0**:
platform discovery; `check_and_update_config` against real
`vllm.config.CacheConfig`/`ParallelConfig` with all four refusals firing by
name; and `TlalocWorker` through the whole v1 worker API —
`load_model`, `determine_available_memory`, `get_kv_cache_spec`,
`initialize_from_config` (accept *and* refuse), two `execute_model` steps
returning real `vllm.v1.outputs.ModelRunnerOutput`, and the chunked-prefill
refusal. It compares those logits against the oracle venv's runner lane on
the same artifact: **bit-for-bit, `==`.**

**The coexistence question this file used to leave open is ANSWERED: yes.**
vLLM's CUDA-13 torch and the CUDA-12 XLA PJRT plugin live in one process
without complaint — `import vllm` loads torch 2.13.0+cu130, and the same
process then compiles and executes through `xla_cuda_plugin.so` on the
GB10 and gets the right numbers.

**NOT certified: `vllm serve` / `LLM.generate()` end to end — and since
§0.4.480 the reason is ONE CLASSMETHOD, not a missing model.** §10 of the
runbook below exports a real TinyLlama-1.1B artifact and serves it. Pointing
vLLM at that artifact gets through platform discovery and dies inside
`EngineCore` startup:

```
NotImplementedError: tlaloc: attention is compiled into the serving artifact's
programs (OpKind.PAGED_ATTENTION); there is no runtime-selectable attention
backend to name
```

That is `vllm_tlaloc/platform.py`'s `get_attn_backend_cls`, refusing by name
as designed. vLLM 0.29.0's v1 engine core calls it **unconditionally**, so it
is not an optional hook. The fix is **H3c-4**
([INFERENCE_SERVING_AUDIT.md](INFERENCE_SERVING_AUDIT.md) §5, §0.4.480's
entry): hand vLLM a backend class whose `get_kv_cache_shape` agrees with the
manifest and whose forward is never reached. The command that will then work:

```bash
export TLALOC_SERVING_ARTIFACT=/tmp/tl-llama
vllm serve ~/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0 \
    --max-num-seqs 1 --max-model-len 64 --block-size 16
```

Until it does, **§10 below is the demo path.**

### Why it is a second venv and not this one (§0.4.470, still the rule)

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

**ANSWERED by §0.4.477:** vLLM's CUDA-13 torch and the CUDA-12 XLA plugin
DO coexist in one process. It was the live certification's first finding,
and it went the good way.

### What is certified without vLLM present

The plugin is split along exactly one line — whether a file imports vLLM:

| file | imports vLLM? | certified? |
|---|---|---|
| `__init__.py` — `register()`, `PLATFORM_CLASS_PATH` | no | **yes** |
| `paging.py` — KV page pool, block tables, slot arithmetic | no | **yes** |
| `batching.py` — requests → one padded bucket-selected call | no | **yes** |
| `runner.py` — `TlalocModelRunner`: artifact + pools + sampling | no | **yes** |
| `platform.py` — `TlalocPlatform` | **yes** | **yes**, in the vLLM venv (§0.4.477) |
| `worker.py` — `TlalocWorker` | **yes** | **yes**, in the vLLM venv (§0.4.477) |

The split still earns its keep: the four vLLM-free files are certified by a
stdlib `unittest` run that anyone can execute in eight milliseconds, and
the two vLLM-facing ones by a lane that costs a 7.7 GB venv. Keeping the
first four out of the second lane is why the arithmetic was ever exercised
at all. What §0.4.477 changed is that the bottom two rows are no longer
**no** — and the bug it found was in the TOP half, reached only by driving
the bottom half for real.

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

## 4.1 Fetch a real Llama checkpoint (CERTIFIED as far as ingestion, §0.4.478)

`vllm serve` needs a real model. Fetch one **into a cache under `$HOME`, not
into either venv** — 2.2 GB of weights are not a Python package:

```bash
~/.local/venvs/vllm/bin/python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0', \
    local_dir='$HOME/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0', \
    allow_patterns=['*.json','*.safetensors','tokenizer*'])"
```

That directory is what `HfLlamaCheckpoint.open(dir)` reads: `config.json`
through `:core`'s strict parser into an `HfLlamaConfig`, and any of the
checkpoint's 201 tensors by `LlamaWeightRole` — shape-verified against the
config on every load. Sharded checkpoints work unchanged (§0.4.468's
`SafetensorsIndex`); this one is single-file.

**The layout fact you will need downstream: HuggingFace stores `nn.Linear`
weights TRANSPOSED as `[out_features, in_features]`.** So `k_proj.weight` is
`[numKvHeads*headDim, hiddenSize]` — on TinyLlama, `[256, 2048]`. A matmul
against it needs `x @ W^T` or an explicit transpose. This is checked against
the real file for all 201 tensors, not assumed.

Verify the ingestion lane, oracle included:

```bash
JAVA_HOME=~/.local/jdks/jdk-25.0.3+9 ./gradlew :ir:jvmTest --tests "*HfLlama*"
# 24 tests; the two real-checkpoint ones self-skip if the cache dir is absent.
# Override the location with TLALOC_HF_LLAMA_CHECKPOINT=<dir>.
```

**Since §0.4.480 this is no longer the end of the road.** H3c-2 built the
decode graph from those weights and certified it against transformers at
1e-5; H3c-3 gave the artifact a staged weight table and served it. **§10
is the demo**, and it starts by downloading this same checkpoint. What
remains uncertified is `vllm serve` alone — §4's last block, H3c-4.

## 5. The KPTX paged-attention kernel (CERTIFIED, and deliberately opt-in)

Tlaloc's own PTX can replace the gather-composed `PAGED_ATTENTION`
lowering inside the XLA executable. The claiming pass keys on
**`OpKind`**, and its decline path is *the op itself* — so a machine with
no KPTX tier runs the same program with the same numbers.

`defaultInferenceKernelTemplates` is **EMPTY on purpose** and stays that
way — see [KPTX_PAGED_PERF.md §8](KPTX_PAGED_PERF.md) for the decision and
the alternative (shape-conditional registration) that was rejected with
its reasons.

**Read the verdict that replaced the old one.** This file used to say the
kernel was "465 µs vs 310 µs — 1.5× slower". **That number is RETIRED**
(§0.4.481): it was a host round trip over a ~1 MB fixture, ≈ 95% staging
traffic with an attention somewhere inside it. Measured on the device,
from one session, interleaved, against pre-staged buffers:

| decode point | KPTX / XLA-lowering device floor |
|---|---|
| TinyLlama-shaped, batch 1, ctx 256 | **1.9× slower** |
| TinyLlama-shaped, batch 8, ctx 512 | **1.9× slower** |
| Llama-3-8B-shaped, batch 8, ctx 1024 | **0.73× — 1.4× FASTER** |
| Llama-3-8B-shaped, batch 16, ctx 1024 | **0.62× — 1.6× FASTER** |

The sign of that comparison reproduced across five sessions; the
absolutes drift (the §0.4.337 rule). So the honest recommendation has two
halves:

- **Do NOT enable it by default.** The registry stays empty until the
  claimed lane wins at *every* point above, small ones included.
- **Enabling it explicitly is a measured 1.4–1.9× win at 8B-shaped
  decode on a GB10** — reproduced in 6/6 sessions by §0.4.495
  ([KPTX_PAGED_PERF.md §11](KPTX_PAGED_PERF.md)), with no session's ratio
  inside 28% of parity — and a ~1.6–1.8× loss at toy shapes. Measure your
  own shapes with `KptxPagedAttentionBenchTest` before you do, and note
  that at toy shapes the test's own dispatch floor is 12–76% of the
  measurement, so what it tells you there is not yet trustworthy.

### The registration step (opt-in, two lines)

```kotlin
// compile side: claim OpKind.PAGED_ATTENTION when lowering
import io.tlaloc.ir.recognizer.kernel.kptxInferenceKernelTemplates
lowerKernels(graph, KernelTarget.NVIDIA_GB10,
             inferenceRegistry = kptxInferenceKernelTemplates)

// serving process: register the PTX chain with the plugin before execute
io.tlaloc.runtime.pjrt.kptx.KptxPagedAttention.register()
```

Both halves are required and neither is a default. The claiming pass's
**decline path is the op itself**, so a process that does neither runs the
same program with the same numbers — which is exactly why an empty
registry is safe and a shape-conditional one would not be.

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


---

## 10. Serve a REAL Llama (CERTIFIED, §0.4.480 — this is the demo)

A real TinyLlama-1.1B, all 22 layers, greedy-decoding on PJRT-CUDA from a
Tlaloc serving artifact, in a process with **no JVM and no framework** — and
agreeing with HuggingFace transformers token for token.

### 10.0 The whole demo, in the order a colleague runs it

Four steps, ~15 minutes on a cold machine, most of it the 2.2 GB download.
Prerequisites: the two venvs of §0 and §4, a CUDA GPU, JDK 25, and ~7 GB
free (2.2 GB checkpoint + 4.2 GiB artifact).

| step | what it costs | where |
|---|---|---|
| 1. download the checkpoint (`huggingface_hub`, vLLM venv) | 2.2 GB, once | 10.1 |
| 2. export the artifact (`./gradlew :maestro:exportLlamaServingArtifact`) | ~9 s, 4.2 GiB on disk | 10.2 |
| 3. run the HF oracle to get the token ids (vLLM venv, transformers) | ~1 min CPU fp32 | 10.3 |
| 4. generate on PJRT-CUDA with **no framework in the process** | ~7 s first step, ~1.35 s/token | 10.3 |

Step 4 is the one that matters: the process it runs in imports `ctypes`,
`json`, `hashlib`, `os`, `struct` and `pathlib`, loads one PJRT plugin
`.so`, and that is the entire runtime. Steps 1 and 3 are *tooling* — they
may use torch and transformers, and they run in the serving venv, never in
the frozen oracle venv (§0.1).

**Where vLLM sits in this, honestly (§0.4.492):** it runs. `LLM.generate()`
against this artifact produces the same six token ids the driver below
produces — see **§11**, which is one command. The demo below is still the
DRIVER path, and it is still the one to run first, because it is the one
that shows the runtime with no framework in the process; §11 then shows the
same artifact answering through vLLM's scheduler, block manager and
tokenizer. `vllm serve` — the HTTP server — has not been started here.

### 10.1 The checkpoint (once)

```bash
~/.local/venvs/vllm/bin/python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0', \
  local_dir='$HOME/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0')"
```

2.2 GB, single-file `model.safetensors`, 201 tensors, every one bf16. The
cache directory is under `$HOME` and **inside neither venv**.

### 10.2 Export the artifact (~9 s, 4.2 GiB)

```bash
JAVA_HOME=~/.local/jdks/jdk-25.0.3+9 ./gradlew :maestro:exportLlamaServingArtifact \
  -PckptDir=$HOME/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0 \
  -PoutDir=/tmp/tl-llama \
  -PmaxBatch=1 -PmaxContext=64 -PblockSize=16 -PnumBlocks=64
```

`-PnumLayers=2` gives the cheap lane — the exact reduced model §0.4.479
certified against transformers, the same code path with one integer changed.

What lands beside `bodies/` and `programs/` is new in §0.4.480:

```
  <artifact>/weights/NNNN_<slot>.bin     raw little-endian, dense row-major,
                                         NO HEADER — the file IS the operand
```

201 of them, in the spec's own slot order, **already transposed** into math
layout by §0.4.479's host-side pass. The manifest's `weights.table` names
each one's dtype, dims, byte length and SHA-256. The loader's whole job is
`open`, `readinto`, upload: it never interprets a format and never creates a
Python number for a weight (1.1e9 Python floats is not a slow path, it is an
impossible one).

### 10.3 Tokenize and generate

The oracle owns the tokenizer, so it owns the ids — nothing in this repo
writes a token id by hand.

```bash
~/.local/venvs/vllm/bin/python harness/python/hf_llama_greedy_oracle.py \
  --checkpoint $HOME/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0 \
  --prompt "The capital of France is" --max-new 6 --output /tmp/oracle.json

export TLALOC_PJRT_PLUGIN_PATH=$HOME/.local/venvs/iree/lib/python3.12/\
site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
export PYTHONPATH=/home/pedro/programming/tlaloc/harness/python

cat > /tmp/req.json <<'J'
{"promptTokens": [1, 450, 7483, 310, 3444, 338], "maxNewTokens": 6}
J
python3 harness/python/run_llama_generate.py \
  --artifact /tmp/tl-llama --request /tmp/req.json --output /tmp/gen.json \
  --platform cuda --verify-weights
```

(`promptTokens` above is what the oracle's `promptTokens` field returned for
that prompt; copy it from `/tmp/oracle.json` rather than trusting this file.)

Both sides produce `[3681, 29889, 13, 13, 29906, 29889]` — **`Paris.\n\n2.`**

### 10.4 What is certified, and what the numbers are

`HfLlamaServingArtifactTest` (in `./gradlew test`, self-skipping without the
checkpoint / the vLLM venv / a plugin `.so`) runs exactly the two commands
above and asserts the generated token ids are **equal**.

**Why token ids and not a logit tolerance.** XLA-GPU's default f32
`dot_general` policy is TF32 (§3's 1e-3 CUDA floor is the same fact) and the
oracle is fp32 on CPU. A logit tolerance here would be a number chosen to
pass; an argmax is not. Greedy decoding agrees EXACTLY until the two
arithmetics disagree about a top-1, so the claim is the **length of the
prefix that agrees**, asserted at the full requested budget. At 6 tokens on
this prompt there is no divergence to report.

| | |
|---|---|
| export (read bf16, widen, transpose, hash, write 4196 MiB) | **9.0 s** |
| first decode step (weight upload + XLA compile, 22 layers) | **7.0 s** |
| median decode step | **1.35 s** |

**The median is not a throughput claim.** The KV pools still round-trip to
the host every step as flat Python lists — 44 pools × 32768 floats per token,
built and unpacked in pure Python. That is buffer DONATION, which has ridden
`donationPairs` in the manifest since H3a and has been the named "next
measurable win" three times; it is now the dominant cost of a real decode,
and measurable for the first time. The weights, by contrast, are uploaded
once and held.

### 10.5 What does NOT work, by name

* **`vllm serve`** — the HTTP server has never been started here. The
  ENGINE under it runs: `LLM.generate()` is certified in §11 (§0.4.492).
* **The `jax` engine** refuses a staged weight table BY NAME. It exists only
  because jaxlib ships no CPU PJRT plugin `.so`; a staged-weight artifact
  therefore has **no CPU lane in this loader**, and its oracle is
  transformers rather than the tight 1e-5 XLA-CPU semantics lane.
* **bf16 weight tables** — the writer refuses a non-f32 staged slot by name.
  Halving the artifact needs the GRAPH to be bf16 (the G1 path).
* **More than one ladder point, batch > 1, context > 64** — the demo exports
  one point because each is a full XLA compile of a 22-layer model. H1c
  already certified that bucketing does not change the answer.
* **Prefill as one call** — the prompt runs as N decode steps, because
  `PAGED_ATTENTION`'s ragged chunked-prefill form is H1a's open deferral.
  A performance deferral, not a correctness one.
* **A tokenizer inside the runtime** — ids in, ids out, deliberately.
* **Sampling** — greedy/argmax only, host-side, in the driver.

---

## 11. Serve it through vLLM (CERTIFIED, §0.4.492 — H3c-4b)

The same artifact §10 built, answering through **vLLM 0.29.0's own
`LLM.generate()`**: vLLM's scheduler, its paged-KV block manager, its
tokenizer and detokenizer, and underneath them the compiled Tlaloc
StableHLO programs on PJRT-CUDA, reached through the `vllm-tlaloc` platform
plugin. No torch model is ever built; vLLM's weight loading never runs.

### 11.1 The command

Prerequisite: §10.2 has run and `/tmp/tl-llama` exists.

```bash
export TLALOC_PJRT_PLUGIN_PATH=$HOME/.local/venvs/iree/lib/python3.12/\
site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
export PYTHONPATH=/home/pedro/programming/tlaloc/harness/python

~/.local/venvs/vllm/bin/python harness/python/run_vllm_generate_check.py \
    --artifact /tmp/tl-llama \
    --checkpoint $HOME/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0 \
    --prompt "The capital of France is" --max-new 6 \
    --max-context 64 --block-size 16 \
    --output /tmp/vllm-gen.json
```

```json
{ "vllmVersion": "0.29.0",
  "promptTokens":    [1, 450, 7483, 310, 3444, 338],
  "generatedTokens": [3681, 29889, 13, 13, 29906, 29889],
  "text": " Paris.\n\n2.", "finishReason": "length" }
```

Identical to §10.3's `generatedTokens` and to the transformers oracle's.
`--checkpoint` names the **tokenizer and HF config**, never the weights —
those come from the artifact's `weights/` directory, and
`$TLALOC_SERVING_ARTIFACT` (which the script sets from `--artifact`) is
what says which.

### 11.2 What the flags are, and why none of them is a workaround

| flag | why |
|---|---|
| `--max-context 64` / `--block-size 16` | the artifact's COMPILED ladder and page size. `TlalocPlatform.check_and_update_config` refuses a disagreement by name rather than overriding it; they are passed because vLLM validates them against the HF config before the platform hook runs |
| `max_num_seqs=1` (in the script) | the artifact's top batch bucket |
| `enable_prefix_caching=False` (in the script) | this runner's KV pool holds no cross-request prefix, so a cache hit would start a sequence in the middle of a context nothing wrote. `batching.py` refuses it by name; the flag means it never arises |
| **no `load_format`, no `gpu_memory_utilization`** | deliberately absent. `load_format="dummy"` would start the engine and would be a lie about which weights answered, and the KV pool is a compiled shape rather than a memory-profiling result |

### 11.3 What is certified, and what is refused BY NAME

**Certified**: one sequence, greedy sampling, a prompt that fits the
compiled context, token ids equal to the direct driver AND to HuggingFace,
6/6. The lane is `HfLlamaServingArtifactTest` inside `./gradlew test`,
self-skipping without the checkpoint / the vLLM venv / a plugin `.so`.

**Refused by name** — each of these raises with an explanation rather than
degrading: `vllm serve`'s HTTP layer (never started here, not refused —
simply unmeasured), chunked prefill, a prefix-cache hit, a resumed chunk or
speculative draft, a grammar bitmask (structured outputs: the mask must be
applied before the argmax and this worker's argmax has already run), a
`--block-size` or `--max-model-len` or `--max-num-seqs` past the artifact's,
`--attention-backend`, MLA, sparse attention, and `world_size > 1`.

**The performance fact**: a prompt is served as N single-token decode steps,
because `PAGED_ATTENTION`'s ragged chunked-prefill form is H1a's open
deferral. Causal attention makes them compute exactly what a fused prefill
would; at ~1.35 s/step a six-token prompt spends ~7 s before its first
generated token. That is what the ragged form is worth.
