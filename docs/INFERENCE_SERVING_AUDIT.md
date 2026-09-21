# Inference serving — vLLM / SGLang integration audit (2026-09-21)

**Status: PHASE H CLOSED (§0.4.473) — H1–H5 all landed, suite
2119 → 2290.** Ratified by Pedro 2026-09-21; GO on the H1–H5 slicing in
§4, and every slice of it is in. **Read §5's ARC STATE block first**: it
carries the per-slice ledger, what is CERTIFIED (with each claim's oracle
and floor), what is WRITTEN BUT UNCERTIFIED (the live vLLM and SGLang
paths, with the commands that would settle them), and what awaits the
Cloud TPU VM. The commands to reproduce the whole path are
[SERVING_RUNBOOK.md](SERVING_RUNBOOK.md). §2's gap list is swept at HEAD.

Researched against the vLLM TPU unified-backend material
(`tpu-inference`), the SGLang × Google TPU announcement, and the
TorchTPU serving integrations.

## 1. The landscape, and which precedent fits Tlaloc

TorchTPU integrates with vLLM and SGLang, but there are **two distinct
plugin shapes** in flight and they differ in exactly the way that
matters to us:

- **`torchtpu-vllm`** — torch-native: built on `torch_tpu`, subclassing
  vLLM's `GPUModelRunner` / `WorkerBase` / `FusedMoE`, one process per
  worker. This shape requires *being a PyTorch device*. **Not Tlaloc's
  shape**, and pursuing it would mean writing a PyTorch backend.
- **`tpu-inference`** (vLLM's current TPU backend) — **the precedent
  that fits**: vLLM's Python side keeps the scheduler, continuous
  batching, paged-KV bookkeeping and the API surface, while the plugin
  replaces *execution* with a compiled-graph lowering path
  (JAX→XLA; PyTorch model definitions ride in through Torchax).
  vLLM supports this as an **out-of-tree platform plugin**.
  SGLang's equivalent precedent is **SGL-JAX** (with SGL-torchtpu
  arriving as the torch-native sibling).

**Tlaloc's natural shape is the second**: a `vllm-tlaloc` platform
plugin where the serving loop stays vLLM's Python and the prefill /
decode graphs are **Tlaloc-emitted StableHLO executed through PJRT**.

The property that makes this attractive rather than merely possible:
**no JVM in the serving path**. Tlaloc AOT-compiles the graphs; the
`:maestro` `ProgramManifest` (content-addressed StableHLO+SDY body,
backend matrix, `kvQuantDtype` — a schema that anticipated serving) *is*
the deployment artifact; the Python plugin loads and runs it via
jaxlib/PJRT — the §0.4.299 spike already proved jaxlib consumes Tlaloc
MLIR verbatim. The same artifact serves on GPU today and TPU the day
G2b lands.

## 2. The honest gap list — swept at the arc close (§0.4.473)

The list below is the **original 2026-09-21 audit text**, kept verbatim so
the arc's starting position stays legible, each item followed by **AT THE
CLOSE**: what is true at HEAD.

1. **Paged attention** — vLLM's core primitive: attention over a
   block-table-indexed KV page pool. We have FlashAttention/GQA
   recognition and a KPTX attention kernel, but **no paged-KV form**.
   Inference-only ⇒ **no VJP required** (the single biggest scope
   relief): a coarse op + gather-composed reference emission, with
   vendor/KPTX kernel claiming as the follow-on. Biggest single item.

   **AT THE CLOSE — CLOSED for decode (H1a §0.4.465), one form deferred.**
   `OpKind.PAGED_ATTENTION` with `scale` as its only attribute (every other
   quantity derived from operand shapes and refused as an attr by name),
   interpreter arm, gather-composed StableHLO emission, cost-model arm,
   inference-only refusals in both AD transforms and the renderer. Certified
   against a dense contiguous walk, a hand-exact uniform-softmax case, and
   real XLA on the GB10 at worst |Δ| = 6e-8. **DEFERRED: the ragged /
   chunked-prefill `[numTokens, …]` form** — prefill rides the dense
   FlashAttention path meanwhile, as vLLM's own TPU backend does. It is the
   one IR-level item both frontends eventually want.
2. **KV-cache ops** — cache-write scatter + page-table gather.
   `SCATTER`/`GATHER` exist in narrow forms; the already-recorded
   "gather/scatter axis+list forms" tail is exactly what is missing.

   **AT THE CLOSE — CLOSED (H1b §0.4.466), and the second half turned out
   not to be needed.** `OpKind.KV_CACHE_WRITE` (no attributes at all; flat
   slots, `-1` = padding, functional-returning-an-updated-pool) lowers to ONE
   `stablehlo.scatter`, agrees with the interpreter EXACTLY on real XLA, and
   composes with item 1 in a round-trip oracle. The **page-table gather** was
   NOT built as a separate kind: it already exists inside `PAGED_ATTENTION`'s
   emission with its dimension numbers pinned, and a standalone kind buys
   nothing until something other than paged attention needs it. The audit's
   "gather/scatter axis+list forms" tail therefore remains open as a
   **training-side AD-surface item**, which is not what this arc was about.
3. **Decode-graph shapes + bucketing** — single-token decode with
   cache-in/cache-out signatures, compiled per (batch, seq-bucket) and
   cached in `PjrtSession`. Our static-shape story **suffices**: vLLM's
   TPU backend buckets identically (bounded dynamism is their roadmap
   item too).

   **AT THE CLOSE — CLOSED (H1c §0.4.467).** `DecodeGraphSpec` (the
   signature, `verifySignature`, `executableCacheKey`, `donationPairs`) and
   `DecodeBucketPolicy` (two ladders multiplied, context aligned up to whole
   pages, over-cap **refused by name and never clamped**) in
   `ir/…/inference/`. `PjrtSession.runOn`/`prepare` take the cache key, so a
   hit does not re-emit. The padding convention's one real decision —
   `seqLens = 1`, never 0 — is pinned by the batch- and context-bucket
   padding-invariant oracle (bit-identical rows in the reference
   interpreter). **PREFILL is covered by the contract, not deferred**; only
   item 1's prefill *attention form* is.
4. **Weight ingestion** — HF safetensors → `DTensor` loader, then
   decode-graph parity vs the HF reference (`LlamaDecoderPrimal` and the
   PyTorch-agreement harness are the substrate).

   **AT THE CLOSE — the READER is closed (H2 §0.4.468); the REAL Llama is
   H3c and open.** `io.tlaloc.core.io` carries a strict JSON parser (written
   because a checkpoint is untrusted input: duplicate keys refused, raw
   number text kept, escapes handled) and a safetensors reader with the
   header validated against the file before a byte is decoded, four dtypes
   mapped and the rest refused **by name** (F16, I64, fp8), per-tensor reads,
   and the sharded `model.safetensors.index.json` implemented with
   `weight_map` values required to be bare filenames. Parity is certified
   against a checkpoint **torch wrote**, bit-for-bit on bf16 patterns and
   1e-3 relative on the decode loss through the IREE-CPU lane. **OPEN: the HF
   NAME-MAPPING** (`model.layers.N.self_attn.q_proj.weight` → graph
   parameter) and a real Llama end to end — H3c, the arc's largest open item.
5. **Dtypes** — bf16 landed in Phase G (§0.4.455–458, native PJRT BF16
   buffers included); int8/fp8 KV-quant stays a named deferral with a
   manifest slot already reserved.

   **AT THE CLOSE — int8/int4 got the CONTRACT (H5 §0.4.472); the BYTES and
   fp8 did not.** `KvQuantPool` (symmetric absmax, per-head scales derived
   from the scale operand's extent, round-half-away-from-zero, error bound
   `scale/2 = absmax/254` **derived and asserted against the measurement**),
   `OpKind.DEQUANTIZE_KV` with interpreter arm + emission + cost arm +
   refusals, and `kvQuant` fed end to end into the manifest. **There is no
   `I8` DType**, so the codes ride I32 and the manifest says so in two
   separate fields (`dtype` = what they mean, `codeDtype` = what they ride) —
   v1 buys the accuracy contract, not the byte budget. **fp8 is REFUSED BY
   NAME**, not missing: it is a float format whose code carries its own
   exponent, so `value = code * scale` is not its dequantization.
6. **The plugin itself** — conform to vLLM's platform-plugin API
   (worker + model-runner classes), sampling host-side in v1.

   **AT THE CLOSE — the artifact and the plugin are WRITTEN and certified
   below vLLM's API surface (H3a §0.4.469, H3b §0.4.470); the LIVE vLLM path
   is UNCERTIFIED.** `ServingArtifactWriter` writes the deployment directory
   (manifest + content-addressed StableHLO bodies + per-entry
   `ProgramManifest`s, with `ProgramManifest` extended by ZERO fields);
   `harness/python/tlaloc_serve.py` loads and runs it with **no JVM in the
   process**, certified on the PJRT CPU client at 1e-5 and the CUDA client at
   1e-3 (the gap is XLA-GPU's TF32 dot policy, measured). `vllm_tlaloc/`'s
   four vLLM-free files (page pool, batching, runner, registration) are
   certified inside `./gradlew test`; `platform.py` and `worker.py` import
   vLLM and are not, because **vLLM was deliberately not installed** — see
   H3b for the 186-package closure and why that venv is the measurement
   apparatus. Sampling is host-side greedy in v1, as planned.

**Explicitly NOT built**: schedulers, continuous batching, prefix
caching / RadixAttention. Integrating *into* vLLM/SGLang is precisely
what buys those.

## 3. The strategic point

This arc **does not wait for TPU hardware**: vLLM runs on the GB10, so
`vllm-tlaloc` certifies end to end on CUDA locally, and the identical
artifacts serve on TPU when G2b lands — which is the entire StableHLO +
PJRT bet paying out on the inference side.

## 4. Phase H slicing (ratified)

| Slice | Content |
|---|---|
| H1 | Inference-graph surface: paged-attention op (no VJP — inference-only), KV-cache scatter/gather forms, decode-graph shape contract + bucketing |
| H2 | Safetensors → `DTensor` weight ingestion + Llama decode-graph parity vs the HF reference |
| H3 | The `vllm-tlaloc` out-of-tree platform plugin, certified on CUDA (manifest-as-artifact, PJRT execution, no JVM at serve time) |
| H4 | KPTX paged-attention kernel + recognizer claiming |
| H5 | SGLang variant (the SGL-JAX precedent) + KV-quant (int8/fp8) |

## 5. Running record (Phase H)

### H1a — `PAGED_ATTENTION` (§0.4.465)

Gap-list item 1, the arc's biggest single piece, landed as a coarse op that
is **complete where it runs and refuses where it does not**.

**The op.** `PAGED_ATTENTION(query, keyCache, valueCache, blockTables,
seqLens) → out`, with
`query [numSeqs, numHeads, headDim]`,
`key/valueCache [numBlocks, blockSize, numKvHeads, headDim]`,
`blockTables [numSeqs, maxBlocksPerSeq] I32`,
`seqLens [numSeqs] I32`, and **`scale` as the only attribute**.
`blockSize`, `numKvHeads`, the GQA `group` and `maxBlocksPerSeq` are DERIVED
from operand shapes and REFUSED as attrs by name — the house sentinel-dims
rule, plus the stronger property that attr-vs-operand disagreement becomes
unrepresentable. One parser (`ir/.../PagedAttentionAttrs.kt`, the
`AllReduceAttrs` precedent) so no two layers can disagree about what a legal
paged attention is.

**Shape decision.** The DECODE shape (one query token per sequence), matching
vLLM's `paged_attention_v1`. REJECTED: the unified `[numTokens, …]` ragged
form — it needs a `queryStartLoc` operand *and* intra-chunk causal masking,
which is a different mask algebra, not a different shape; folding both in
would make the mask a mode flag. **Named deferral**: the prefill/chunked
form; prefill rides the existing dense FlashAttention/GQA path today, as
vLLM's own TPU backend does.

**Inference-only by design.** A new sibling of the demoted set —
`INFERENCE_ONLY_OP_KINDS` in `ir/.../passes/DemotedOpKinds.kt` — with the
distinction spelled out there: a demoted kind is partial in the execution
layers too, an inference-only kind is COMPLETE where it runs (interpreter arm
+ StableHLO emission are mandatory; serving executes) and absent only in the
two AD transforms. Both transforms and `KotlinSourceRenderer` refuse BY NAME,
each message carrying the rationale (the KV pool is state mutated across
decode steps, addressed by integer allocator bookkeeping — not a
differentiable intermediate) and the TRAINING spelling (FlashAttention /
the GQA coarsener). No gradient math was written.

**Emission**: the gather-composed reference form — gather the pages the block
table names into a dense `[numSeqs, maxBlocksPerSeq*blockSize, numKvHeads,
headDim]` window, mask past `seqLens` to −Inf, batched `dot_general`s over
(sequence, kv head), GQA by reshape. REJECTED: per-sequence dynamic slicing
(data-dependent shapes, which XLA forbids and which is why vLLM's TPU backend
masks-and-buckets too), and per-position flat-index gathering (a second
indexing convention to keep honest, for no gain at decode sizes). A fused
vendor/KPTX paged kernel with recognizer claiming is **H4**, and the coarse
kind exists precisely so that swap is local.

**Oracles.** (1) Paged-vs-DENSE equivalence: the same attention built two
ways — the paged walk, and a dense contiguous walk over a window materialised
by separately gathering the named pages — pinned elementwise under an
identity table, a permuted table with ragged lengths, and a full sweep of
`seqLens` from 1 to the window width. (2) A hand-exact case: tied scores make
the softmax exactly uniform, so the answer is the mean of the live V rows,
with the slots at and past `seqLen` POISONED at 1000.0 so an over-read is
unmissable. (3) GPU: `PjrtPagedAttentionSmokeTest` compiles the emission
under real XLA on the GB10 and agrees with the interpreter to **worst
|Δ| = 6e-8** (permuted table, partially-filled last page) — the emission and
the walk are genuinely different programs (one skips dead lanes, the other
computes and cancels them), so that agreement pins the mask algebra, the
gather dimension numbers and the GQA convention at once. (4) Offline
structure pins in `PagedAttentionEmitTest` so certification never depends on
a GPU being present.

**Named deferrals from this slice**: the prefill/chunked `[numTokens, …]`
form; bf16 paged attention (the dtype exists; the smoke runs f32);
mixed-dtype PJRT input lanes (`runOn` is all-F32, so the GPU smoke rides
`blockTables`/`seqLens` as I32 consts — a harness limit, not a shape limit,
and H3 feeds these as device buffers through the manifest); the cost model
prices the WORST-CASE window because `seqLens` is a runtime value and pricing
may never read tensor contents.

**Still open in H1** (after H1a): gap-list item 2 (KV-cache write scatter +
page-table gather forms) and item 3 (decode-graph shape contract + bucketing).

### H1b — `KV_CACHE_WRITE` (§0.4.466)

Gap-list item 2's **write** half: the step that fills the pool H1a reads.

**The op.** `KV_CACHE_WRITE(cache, newKv, slotMapping) → updatedCache`, with
`cache [numBlocks, blockSize, numKvHeads, headDim]`,
`newKv [numTokens, numKvHeads, headDim]`,
`slotMapping [numTokens]` integer, and **no attributes at all** — every
quantity is derived from operand shapes, and `blockSize`/`numKvHeads`/
`headDim`/`numTokens`/`numBlocks` are REFUSED as attrs by name (the sentinel-
dims rule). One parser: `ir/.../KvCacheWriteAttrs.kt`, the
`PagedAttentionAttrs` precedent.

**Did an existing kind already express this?** Checked first, because adding a
duplicate kind is the failure mode here. **No.** `SCATTER`/`SCATTER_ADD` take a
**scalar** I32 index and replace **one** row; a KV write needs a *vectorized*
index, since `slotMapping` is a runtime rank-1 tensor. Expressing it with the
narrow kinds costs `numTokens` GATHERs (to pull each slot out as a scalar) plus
`numTokens` SCATTERs — an O(numTokens) op explosion for what one StableHLO
scatter does, and it would dissolve the single recognizable kind that H4's
fused kernel and H3's buffer donation both need to claim. The general
vectorized-scatter kind that *would* subsume this is a **differentiable**
surface needing a `VjpRule`; this one is inference-only and needs none. The
audit's "gather/scatter axis+list forms" tail therefore stays open as a
*training-side* item, and is **not** what H1b landed.

**The name.** vLLM calls it `reshape_and_cache`, after a reshape its CUDA
kernel happens to do on the way in. The house names ops for what they mean, and
H4's kernel may not reshape at all — hence `KV_CACHE_WRITE`.

**Decisions, and what they rejected.**
- **Flat slots** (`slot = blockIdx*blockSize + offset`), vLLM's own
  convention. REJECTED: a `[numTokens, 2]` (block, offset) pair tensor — every
  consumer would re-derive the flat index, "is this slot live" becomes two
  comparisons, and the plugin's first act would be flattening back what the
  scheduler already had flat. The payoff is visible in the lowering: **no
  block/offset arithmetic appears anywhere** (pinned as a test).
- **One pool per op**, called twice (K then V). REJECTED: fusing both pools
  into one two-result op, `reshape_and_cache`'s literal shape — that shape
  exists to save a kernel launch, which is the kernel's concern, and here it
  would buy the §0.4.448 multi-result-index landmine and forbid a graph that
  writes only K. A fused recognizer pattern over the adjacent pair is available
  in H4 if a kernel wants it.
- **Functional, returning an UPDATED pool.** REJECTED: in-place mutation via an
  inout operand — it would make every pass that reorders or CSEs ops
  responsible for a memory model none of them has. The notional full-pool copy
  is answered by **XLA buffer donation**: the serving loop donates the cache
  buffer, XLA writes in place, no copy and no new IR concept. **Named
  follow-on**: wiring donation through the manifest + `PjrtSession` (H3).
- **Negative slot = padding**, vLLM's `-1`. Load-bearing for the static-shape
  story: bucketed decode graphs always carry slack lanes.

**Emission**: ONE `stablehlo.scatter` bracketed by two reshapes (pool ↔
`[numBlocks*blockSize, numKvHeads, headDim]`), `update_window_dims = [1, 2]`,
`inserted_window_dims = [0]`, `scatter_dims_to_operand_dims = [0]`,
`index_vector_dim` = the indices' rank, replace body. Padding rides StableHLO's
rule that an **out-of-bounds scatter update is dropped**. REJECTED: masking
padding explicitly with a `slot >= 0` predicate and a `select` — three more ops
plus a gather of the current pool rows to select back, to defend against a rule
the spec states. `unique_indices = false` deliberately: not a statement about
the live slots (those are required distinct) but about the padding lanes, which
repeat `-1`; promising uniqueness where indices are literally equal is a lie
with an optimizer behind it.

**Inference-only by design.** Joins `INFERENCE_ONLY_OP_KINDS`. Both AD
transforms and `KotlinSourceRenderer` refuse BY NAME, with the rationale (the
pool is serving-runtime state threaded across decode steps; `slotMapping` is
integer allocator bookkeeping) and the differentiable spelling (the
`SCATTER`/`SCATTER_ADD` family, which carries certified rules). No gradient math
was written.

**Oracles.** (1) Hand-exact slot-mapping writes against a pool
*position-encoded* so every element equals its own flat index — a write one
slot, head or lane off yields a value that names where it actually went — with
the expectation rebuilt by an independent pool-major walk, and with two tokens
landing in the SAME BLOCK and a write into a partially-filled block whose other
slot must survive. (2) The **round trip**: a decode graph that writes K/V then
runs `PAGED_ATTENTION` with `seqLens = 1`, where a one-element softmax is
exactly 1.0, so attention must read back the written V **bit for bit** — with
the pool poisoned at 1000.0 so a write that did not land is off by three orders
of magnitude. This composes H1a and pins that both ops agree what a flat slot
is. (3) Conventions: padding writes nothing, the input array is not mutated,
and duplicate/out-of-range slots refuse loudly (the sign is what separates
padding from an allocator bug). (4) GPU: `PjrtKvCacheWriteSmokeTest` compiles
the emission under real XLA on the GB10 and agrees with the interpreter
**exactly** (a write does no arithmetic) — padding lane included, which is the
only honest way to confirm that XLA *drops* rather than *clamps* an
out-of-bounds scatter index. It does. (5) Offline structure pins in
`KvCacheWriteEmitTest`.

**Named deferrals from this slice**: buffer donation (H3); a standalone
page-gather kind — the page-table gather already exists *inside*
`PAGED_ATTENTION`'s emission with its dimension numbers pinned, and a separate
kind buys nothing until something other than paged attention needs it; int8/fp8
KV-quant, which would make the write a converting one (H5); bf16 pools (the
dtype exists and the parser accepts it; the smoke runs f32); mixed-dtype PJRT
input lanes, so the smoke rides `slotMapping` as an I32 const.

**Still open in H1**: gap-list item 3 (decode-graph shape contract +
bucketing). Item 2's write half is closed; its training-side vectorized
gather/scatter tail remains an AD-surface item, not an inference one.

### H1c — the decode-graph shape contract and bucketing (§0.4.467)

Gap-list item 3, and the piece that makes the plugin's **compile story** real.
H1a gave the paged read and H1b the paged write; what was missing was the
*signature* a serving loop calls and the rule that turns a dynamic batch into a
static one. **No new op kind** — hence no new AD refusal, no new renderer arm,
no gradient math. This slice is a contract, a policy, and the oracles that hold
them to it.

**Placement.** `ir/.../inference/` (`DecodeGraphSpec.kt`,
`DecodeBucketPolicy.kt`), not `:runtime-pjrt`. The contract is expressed in
`DxirType`, and its consumers are graph construction, the emitter's shape
checks, the `:maestro` manifest and the PJRT executable cache — all of which
already depend on `:ir`, and only the last of which is JVM-only. REJECTED: a
new `:inference` module (two files and a circular pull on `:ir`, for no
isolation) and `:core` (no `DxirType` there).

**The signature**, static per bucket, `T = 1` for decode and the bucket's
context width for prefill:

```
  in   tokenIds [B,T] I32 · positions [B,T] I32 · blockTables [B,maxBlocksPerSeq] I32
       seqLens [B] I32 · slotMapping [B*T] I32 · (key,value) pools × numLayers
  out  logits [B,T,V] · the same pools, updated
```

**Prefill is COVERED, not deferred**: it is the same contract with a longer
token axis — `DecodeGraphKind.PREFILL`, pinned by a test asserting that every
non-token-axis type is byte-identical to the decode spec's. What stays deferred
by name is H1a's prefill *attention form* (the ragged `[numTokens, …]` shape);
prefill rides the dense FlashAttention path meanwhile, as vLLM's own TPU
backend does.

**Decisions, and what they rejected.**
- **Token ids in, not embeddings.** REJECTED: a `[B,T,hiddenSize]` entry point
  — it ships `B·T·hidden` floats across PCIe every step instead of `B·T` ints,
  for a table that is already a device weight. Speculative decoding and
  multimodal prefixes do want one; that is a **named deferral**, and a sibling
  spec rather than a mode flag on this one.
- **Pools as ordinary in/out operands**, `(key, value)` per layer ascending, so
  the graph is functional end to end (H1b's decision) and XLA's buffer donation
  then makes it free — `donationPairs` names the aliases for H3 to wire.
  REJECTED: a stateful side-channel resource, which costs every pass a memory
  model and buys nothing here.
- **`positions` is an operand, not derived from `seqLens`.** For a padded row
  there is no correct position to derive, so the convention states one instead
  of computing a lie.
- **Sampling stays host-side in v1**: each sampling config would otherwise fork
  the executable cache on an axis that has nothing to do with shape.
- **Weights are graph CONSTANTS**, which is why the cache key carries a model
  hash — pinned by building the end-to-end test graph that way.

**The bucketing policy.** Two independent ladders, multiplied: batch = powers
of two to `maxBatch` with the cap appended, context = powers of two from 16,
each **aligned up to a whole number of pages**, with the aligned cap appended.
Page alignment is load-bearing: a context bucket names `maxBlocksPerSeq =
context / blockSize`, and a bucket that is not a whole number of pages would
make that width a rounding decision taken twice by two layers. REJECTED: a
single fused "total token" ladder (decode is not token-budgeted — a 1×4096 and
a 64×64 request are not interchangeable); a 1.25× geometric ladder (halves the
padding waste, triples the executable count, and every executable is a startup
compile and a slot of device memory); and rounding DOWN with a second pass,
which recreates the partial-batch shape the ladder exists to avoid. Over-cap
requests are **REFUSED BY NAME, never clamped** — clamping a context request
silently truncates a sequence's history, which is a wrong answer dressed as a
slow one, and the scheduler is the only layer that can split it.

**The padding convention, and its one real decision.** Padding rows carry
`slotMapping = -1` (H1b drops the write), `tokenIds = 0`, `positions = 0`,
`blockTables = 0` (page 0, the conventional scratch page — it must be an
*allocated* page so the gather stays in bounds), and **`seqLens = 1`, never 0**.
That last one is the whole reason this section exists: `seqLens = 0` masks
every context lane to −Inf and the softmax becomes `0/0 = NaN` in the
gather-composed emission, while the interpreter takes a `len == 0` early-out
and leaves zeros — H1a's emitter already names this as the single shape where
the two disagree. A padding convention routing through it would make every
padded batch differ between oracle and device, and would seed NaN into a buffer
any later fused cross-row reduction would spread into the *real* rows. REJECTED:
teaching the emission the interpreter's zero arm, which puts a branch in every
sequence's hot path to serve a row that is thrown away.

**The invariant, and exactly what is claimed.** Every op in the contract is
row-independent along the batch axis (per-position embedding, per-sequence
paged attention, disjoint cache writes, a logits projection that contracts
hidden and never batch), so row *r* of the output is a function of row *r* of
the input and the weights — and not of `B`. Therefore, **in the reference
interpreter, a batch of 3 run in a bucket-4 graph is BIT-IDENTICAL on rows 0–2,
and produces a bit-identical pool, against the same batch in a bucket-3 graph**
— pinned elementwise with `==`, and again with the *context* bucket widened
(bucket (4,8) vs (3,4)), where the extra block-table entries are all scratch
page and the mask must make them contribute exactly nothing.

What is **NOT** claimed: bit-identity on device. Two buckets are two compiled
executables and XLA may tile a `[4,H]×[H,V]` matmul differently from `[3,H]`;
a different tiling is a different accumulation order and float addition is not
associative. The honest device claim is the house GPU-vs-host floor, **~4e-5,
never pinned tighter than 1e-4**, and the measurement belongs on the real Llama
decode step in **H3**, where "the same logits" has a top-1 consequence worth
reporting — not on a toy. Stating the reason rather than pinning a floor on a
5-token vocabulary is the honest version of this line.

**The executable-cache key.** `tlaloc-decode-v1 / modelHash / kind / bB / cC /
tT / dtype / kvDtype`, and it is now a real front door: `PjrtSession.runOn` and
`prepare` take an optional `cacheKey`, and on a hit the StableHLO is **not
re-emitted**. REJECTED: keying on the emitted MLIR text alone, which is what
the session did — correct (structurally identical programs hash together) but
it pays a full re-emission per lookup, which at decode rates is the one thing a
cache exists to avoid; it stays as the back-stop *behind* the key, so two
different keys over an identical program still share one executable. REJECTED:
`DxirFunction` identity — a plugin that rebuilds its graph per request, which
is the obvious thing to write, would miss every time.

**Oracles.** (1) The padding-invariant pin, elementwise `==` on logits and both
pools, across batch buckets and across context buckets. (2) An **end-to-end
mini decode graph** — H1a and H1b composed into a real step (`embed → q/k/v
projections → KV_CACHE_WRITE ×2 → PAGED_ATTENTION → lm head → logits`),
`verifySignature`-checked against the contract on every build so the contract
cannot drift from the code, and agreeing with an independent hand walk that
touches none of the kinds under test: with `seqLens = 1` the softmax is exactly
`exp(0)/exp(0) = 1.0`, so attention returns the just-written V and the step
collapses to a chain of small matmuls. The pool is poisoned at 1000+ so a write
that never landed is three orders of magnitude off, not epsilon. (3) The
padding row is **inert**: exactly the three real slots changed, and scratch page
0 — which the padded row's block table names and which it therefore *read* — is
byte-for-byte untouched. (4) A bucket-selection suite: exactly-on-boundary takes
that bucket and not the next, one-past takes the next, selection is covering /
monotone / minimal over the whole legal domain, over-cap and empty-step refuse
by name, and a hand-written ladder gets no discount on the invariants. (5) GPU:
`PjrtSessionSmokeTest` pins the keyed cache path on real XLA on the GB10 — a
repeated key lowers once, still agrees with the interpreter, and two keys over
one program share one executable.

**Sensitivity check.** The invariant pins were deliberately mutated before
being trusted (a changed padded token id, and a padding slot of 0 instead of
−1): both oracles failed, then passed again on revert.

**Named deferrals from this slice**: the embeddings entry point (speculative
decoding / multimodal); the device-side padding-invariant measurement (H3, on
the real decode step); RoPE, which the mini graph declares as a `positions`
input and does not consume, because it needs H2's rotary tables; warm-up
scheduling across `allBuckets` (the API is there, the policy of *which* buckets
a deployment actually compiles at startup is a tuning question with no answer
yet); and a manifest field carrying the spec, so the artifact states its own
bucket ladder — H3, where the manifest is written.

**H1 is now CLOSED**: gap-list items 1 (paged attention), 2's write half (cache
write) and 3 (contract + bucketing) are all landed. Next is **H2** — safetensors
→ `DTensor` weight ingestion and Llama decode-graph parity vs the HF reference.

### H2 — safetensors weight ingestion + decode parity (§0.4.468)

Gap-list item 4. **No new op kind, no new attrs, no gradient math** — and
therefore no new AD refusal and no new `KotlinSourceRenderer` arm: this slice
is I/O and an oracle. The north star is untouched because nothing new entered
the IR.

**The reader.** `io.tlaloc.core.io` — `Json.kt` (a strict minimal JSON reader)
and `Safetensors.kt` (the format) in `commonMain`, `SafetensorsFile.kt` (a
`FileChannel` and the sharded index) in `jvmMain`. The format is: `[0,8)` a u64
little-endian header length, `[8, 8+N)` a UTF-8 JSON header mapping name →
`{dtype, shape, data_offsets}` where the offsets are relative to the DATA
BUFFER at `8+N`, then tensors packed back to back, row-major, little-endian.

**Placement.** `:core`. The reader's output type IS `DTensor`, and
`HostF32Storage` / `HostBf16Storage` / `HostI32Storage` / `DType` all live
there. REJECTED: a new `:io` module — it would name a `:core` type in every
signature it has and be depended on by everything that loads weights: a module
boundary with no API of its own. REJECTED: `:maestro` — that module is about
the deployment MANIFEST, and weights are not a program. The commonMain/jvmMain
split is the honest one: decoding the format is arithmetic over bytes and is
tested without a filesystem; opening a file is the only jvm-specific part.

**A JSON parser in `:core`, which has no dependencies at all.** REJECTED:
kotlinx-serialization-json — a first dependency on the module everything else
depends on, to read a flat map. REJECTED, more importantly: scanning the header
with `indexOf`, which is what "quick" loaders do and which is wrong the first
time a tensor name contains a `{`, a `,` or an escaped quote — all legal in a
JSON string. A checkpoint is UNTRUSTED INPUT; the thing that reads it is a
parser. It is strict by design (no trailing commas, no unquoted keys, no
`NaN`/`Infinity` literals, raw control characters in strings refused), keeps
each number's RAW TEXT so `4.0` is not a shape of 4 and a 2^53+ offset is not
rounded, and refuses DUPLICATE KEYS rather than taking last-wins — two entries
for one tensor name are two disagreeing offsets, and picking one silently is
how a loader reads the wrong bytes.

**Decisions, and what they rejected.**
- **Validate the header against the file before decoding a byte.** Negative
  dims, a backwards offset pair, a byte span that disagrees with
  `dtype × elementCount`, a range past the end of the buffer, and OVERLAPPING
  tensors are each refused BY NAME. REJECTED: trusting the header and letting
  an index throw — that reports a corrupt checkpoint as an
  `ArrayIndexOutOfBoundsException` from inside a decode loop, which is the same
  information with the diagnosis removed. The declared header length is bounded
  (100 MB) BEFORE anything is allocated, so a hostile u64 is a named error and
  not an `OutOfMemoryError`.
- **Dtypes: map four, refuse the rest BY NAME.** F32, F64, BF16 (into
  `HostBf16Storage`, raw patterns, §0.4.455) and I32 map. **F16 is refused** —
  fp16 has no host representation and neither widening nor narrowing it would
  put the checkpoint's own numbers in the tensor. **I64 is refused** — it is a
  `DType` but has no `HostI64Storage`, and narrowing corrupts exactly the token
  ids and position buffers HF stores at that width. **fp8 is refused as the H5
  deferral**, naming the reserved `kvQuantDtype` manifest slot. REJECTED
  throughout: a silent upcast, which produces numbers the producer never wrote.
- **Per-tensor reads, not whole-file.** `SafetensorsFile.load` reads only that
  tensor's `[begin, end)` range. REJECTED: slurping the file and slicing — it
  doubles peak memory at the moment memory is scarce, and a 70B checkpoint is
  the intended caller. REJECTED for v1: `mmap` — right for a large f32
  checkpoint whose storage could be a buffer VIEW, but Tlaloc's host storages
  are Kotlin primitive arrays so the decode copies regardless; a zero-copy
  MemorySegment-backed storage is a NAMED DEFERRAL (the FFM stack exists).
- **The sharded index is IMPLEMENTED, not deferred.**
  `model.safetensors.index.json`'s `weight_map` routes each tensor to a shard;
  shards open LAZILY on first mention and close together, and
  `openCheckpoint(dir)` picks the index or the single file behind one
  `WeightSource`. A `weight_map` value is an untrusted string, so it is required
  to be a BARE filename — an index that can name `../../etc/passwd` is an index
  that can read arbitrary files.
- **`LoadedTensor` does not invent a phantom `Shape`.** A `DTensor`'s shape
  parameter is a compile-time fact and a checkpoint's rank is a runtime one, so
  the caller brands it (`asF32<S>()` / `asBf16` / `asI32` / `asF64`, each
  refusing a dtype mismatch by name). REJECTED: returning `DTensor<Nothing, _>`
  and letting subtyping paper over it — it typechecks and it lies about what
  was proven.

**Parity, and the oracle.** `LlamaSafetensorsParityTest` runs a REAL
safetensors checkpoint written by the reference implementation:
`harness/python/write_llama_safetensors.py` synthesizes `LlamaDecoderPrimal`'s
thirteen parameters and saves them through `safetensors.numpy.save_file` AND
(at bf16) `safetensors.torch.save_file`. Tlaloc's reader loads them; the CPU
baseline pipeline (`recognize → coarsen → decomposeCoarsened`) runs the decode
step through Tlaloc-IREE; `harness/python/run_pytorch_llama.py` — the existing
op-for-op PyTorch mirror (§0.4.289) — supplies the loss on the same numbers.

Two claims, both pinned:
1. **Bit-for-bit read.** The writer records the raw element bit patterns it
   actually wrote at chosen flat indices (both ends, where an offset off-by-one
   shows first, plus a spread through the middle) and the reader must reproduce
   them EXACTLY — `Float.toRawBits()` for F32, the raw 16-bit pattern out of
   `HostBf16Storage` for BF16. No tolerance: reading bytes is not arithmetic.
2. **Decode parity** at 1e-3 relative on the loss, for the f32 checkpoint and
   the bf16 one.

The **bf16 file is the load-bearing half**: its numbers are the f32 ones put
through `torch.bfloat16`, so torch does the rounding and Tlaloc must read the
identical 16-bit patterns — which incidentally pins §0.4.455's
round-to-nearest-even against torch's own conversion. Widening back to f32 is
exact, and the PyTorch reference runs on the widened values, so "an f32 graph
on a bf16 checkpoint computes on the checkpoint's real numbers" is a certified
sentence rather than a hopeful one. A separate assertion requires the two
checkpoints to produce DIFFERENT losses — if bf16 rounding were a no-op, or if
the bf16 read quietly fell back to the f32 file, that is the line that notices.

**Why synthesized weights, not a downloaded TinyLlama.** The bytes of a random
f32 tensor and a trained one are the same kind of bytes, and
`LlamaDecoderPrimal` is not TinyLlama's graph anyway (single layer, single
head, no GQA — its own scope notes say so), so a download would add a network
dependency and a multi-gigabyte footprint to the certification suite without
strengthening either claim. What a real checkpoint WOULD add is HF's tensor
NAMING and the sharded-index form: the index is pinned in
`SafetensorsFileTest`, and mapping HF names onto a model-layer graph is H3's
business, where the manifest is written.

**Why the IREE-CPU lane and not the interpreter or the GPU.** The reference
interpreter's op coverage is deliberately narrow (it does not carry `RSQRT`,
and widening it is AD-engine work, not ingestion work). The IREE-CPU lane and
the PyTorch oracle were already pinned against each other in §0.4.289, so the
ONLY new variable in this test is where the weights came from, and any
disagreement is the reader's. Running the step on the GPU through PJRT is a
SECOND claim with its own floor (~4e-5, never tighter than 1e-4) and belongs
with H3's real decode step, where a logits difference has a top-1 consequence.

**Format-level oracles without any toolchain.** `SafetensorsTest` builds files
BYTE BY BYTE — never through a writer of ours, because a reader tested against
its own writer certifies that the two agree, not that either matches the
format. It covers mixed dtypes in one file, a rank-0 tensor, a zero-element
tensor, signed zero pinned on raw bits, the full 256-exponent bf16 sweep, and
every refusal above. `JsonTest` covers the escapes, the control-character rule,
the duplicate-key refusal and the exact-integer rule.

**Sensitivity check.** The bf16 decode was deliberately byte-swapped before the
parity test was trusted: it failed at `checkBf16Probes` with "Tlaloc read a
different bf16 pattern than torch wrote", then passed again on revert.

**Named deferrals from this slice**: mmap / zero-copy `MemorySegment`-backed
storage; `HostI64Storage` (and with it I64 checkpoints); F16 and fp8 (H5);
loading a real HF Llama end to end, which is a NAME-MAPPING problem
(`model.layers.N.self_attn.q_proj.weight` → graph parameter) and belongs with
H3's model layer; a safetensors WRITER (nothing in the serving path writes
one — the manifest is the artifact, weights are an input); device-side decode
parity (H3); and streaming a checkpoint straight into device buffers, which is
what a serving loader eventually wants and which needs H3's manifest-side
buffer story first.

**Still open in Phase H**: H3 (the `vllm-tlaloc` platform plugin, certified on
CUDA), H4 (KPTX paged-attention kernel + recognizer claiming), H5 (SGLang
variant + KV-quant).

### H3a — the serving artifact and its Python side (§0.4.469)

The first half of gap-list item 6, and the slice where the arc's central
property stops being an intention. **No new op kind, no new attrs, no gradient
math** — so no new AD refusal and no new `KotlinSourceRenderer` arm; nothing
entered the IR. What entered is a LINE: everything upstream of it is Kotlin,
everything downstream is Python + PJRT, and the line is a directory.

**The artifact.** `io.tlaloc.maestro.serving.ServingArtifactWriter.export`
writes, for a family of `DecodeGraphSpec`s:

```
  <artifact>/
    tlaloc-serving.json          the ServingManifest
    bodies/<sha256>.mlir         StableHLO+SDY, content-addressed
    programs/<entryId>.json      one ProgramManifest per entry, unmodified
    weights/…                    pointed at, not necessarily present
```

The manifest carries model shape (including the **KV-pool axis order, stated
and not implied** — four integers where two may be equal is not a layout), the
**bucket ladder** (H1c's named deferral, closed: the artifact states its own
ladder so the Python side does bucket SELECTION without a second copy of
`DecodeBucketPolicy`'s arithmetic), a **weights pointer**, and one entry per
compiled ladder point with its role-tagged input/output slots, its
`donationPairs`, its `entryPoint`, its body's content address, and
`DecodeGraphSpec.executableCacheKey` verbatim.

**`ProgramManifest` was extended by ZERO fields, and that is the decision.**
It describes ONE program; a serving artifact is a deployment of a FAMILY of
programs sharing a model, a ladder and a weights pointer, whose slots
additionally carry a serving ROLE. REJECTED: adding `bucket` / `role` /
`donationPairs` to `ProgramManifest` and `TypeDescriptor` — they are
meaningless to a `program { }` step that runs a training epoch, and a schema
most of whose instances must leave fields null has stopped describing
anything. The composition runs the other way: a `ServingEntry` POINTS AT a
`ProgramManifest` by path, and the exporter cross-checks that the two agree
about the boundary types and the content address, so the redundancy is
*checked* rather than *drifting*.

**Other decisions, and what they rejected.**
- **`verifySignature` runs on every graph at EXPORT time.** The consumer
  cannot call back; a signature mismatch discovered in Python is a shape error
  from inside XLA with the diagnosis removed.
- **Bodies are named by their own SHA-256 and de-duplicated.** REJECTED:
  naming them by entry id — readable, and it throws away the one property that
  lets a loader verify bytes it was handed (`verify_bodies()` on the Python
  side re-hashes them).
- **The entry point is `main`, written that way.** REJECTED: the §0.4.325
  spike's regex rename — a loader that edits a program's text before running
  it is not running the program it was given.
- **One module per entry.** A serving process compiles buckets independently;
  one module with N functions makes every compile pay for every bucket's text.
- **Textual MLIR, JSON manifest.** REJECTED: StableHLO bytecode (better
  transport, a NAMED DEFERRAL) — textual MLIR is what
  `jaxlib.mlir.ir.Module.parse` takes and what §0.4.299 proved jaxlib consumes
  verbatim, and a `grep`-able artifact is worth a lot in the slice that first
  draws this line. REJECTED: protobuf/kotlinx-serialization — `:core`'s strict
  parser (§0.4.468, written because a checkpoint is untrusted input) already
  exists, so the artifact and the checkpoint are read by one discipline.
- **Weights stay EMBEDDED as body constants in v1**, and the pointer says so
  (`embedded: true`, no path). When H3b stages them they become graph
  parameters and the schema does not move. REJECTED: copying a checkpoint into
  the artifact to make the directory "self-contained".

**The Python side.** `harness/python/tlaloc_serve.py` — reads the manifest,
verifies body hashes, selects a bucket off the declared ladder (refusing
over-cap BY NAME rather than clamping, because a clamped context silently
truncates a user's history), pads with the wire convention, compiles through
`jaxlib.mlir` + `backend.compile_and_load` (the §0.4.299 spike pattern,
reused), caches executables by the manifest's `cacheKey`, executes, and slices
the logits back to the real rows so the padding row never leaves the loader.
`run_tlaloc_serve_check.py` is the CLI the certification drives. The padding
constants are duplicated from `DecodePadding` **deliberately and loudly** —
they are a wire convention between two processes — and
`check_padding_constants()` exists so the certification PINS the two copies
together instead of trusting a comment.

**Oracles.** (1) The exported directory describes itself truthfully: every
body hashes to the name it is filed under, every `@main` is there, each
per-entry `ProgramManifest` parses with its OWN parser and agrees, bodies are
one file per DISTINCT program, and two exports of one model are byte-identical
(a content address, not a timestamp). (2) Manifest round-trip and refusal
suite: an entry off the declared ladder, duplicate ladder points, a donation
pair aliasing mismatched types, an unaligned context ladder, a weights pointer
that claims both embedded and a path, an unknown schema version, a missing
field, an unknown slot role — each refused by name. (3) **Export-then-run**:
Kotlin exports; a Python subprocess with no JVM and no gradle in the loop
loads the directory, compiles, runs a decode step of three sequences in a
bucket-4 graph, and its numbers are compared against the host interpreter.

**Two lanes, two floors, and why that is not hedging.** The same artifact runs
on the PJRT **CPU** client at **1e-5** relative and on the PJRT **CUDA** client
on the GB10 at **1e-3** relative. The looseness is MEASURED, not guessed:
XLA-GPU lowers a default-precision f32 `dot_general` through TF32, which puts
this step ~5e-4 relative from the host while the CPU lane sits at ~1e-7 on the
identical bytes — so it is the backend's precision policy, not an emission
bug. A GPU-only oracle at 1e-3 would have been a semantics test with a
thousandfold hole in it; a CPU-only oracle would not have certified what a
deployment runs. This closes H1c's deferred device-side measurement with the
honest answer: **the number is per-backend**. Two claims escape the tolerance
entirely — the pools are poisoned at 1000+index so a misplaced write is three
orders of magnitude out, and the padded row's scratch page is pinned with
exact `==` on BOTH lanes, because an untouched slot is COPIED, not computed.

**Sensitivity check.** Two deliberate mutations before the oracles were
trusted: `tlaloc_serve.py`'s `PADDING_SEQ_LEN` flipped to 0 (both lanes
failed), and the block table sent to Python desynced by one page from the
reference's (both lanes failed). Both passed again on revert.

**Named deferrals from this slice**: buffer DONATION (the manifest carries
`donationPairs`; wiring them into `CompileOptions` is the measurable win of the
next slice); staged weights and the HF name-mapping, i.e. exporting a REAL
Llama decode step rather than the reference mini-graph — **H3b**; the vLLM
platform-plugin classes themselves (`Platform`/`Worker`/`ModelRunner`) and
sampling, which is where H3 actually ends; `precision_config = HIGHEST`
emission for dots that want it, and the top-1 consequence of the TF32 gap on a
real vocabulary; StableHLO bytecode bodies; warm-up policy (which buckets a
deployment compiles at startup, still a tuning question with no answer);
multi-device serving across a mesh; prefill entries (the ladder and the
contract carry `DecodeGraphKind.PREFILL` today, and nothing exports one yet).

**Still open in Phase H**: H3b/H3c (staged weights + a real Llama; the
`vllm-tlaloc` plugin classes), H4 (KPTX paged-attention kernel + recognizer
claiming), H5 (SGLang variant + KV-quant).

### H3b — the `vllm-tlaloc` platform plugin (§0.4.470)

Gap-list item 6's second half. **Read the certification boundary first**,
because it is the whole character of this slice: what is written is the plugin;
what is *certified* is everything under vLLM's API surface; what is *not* is the
two files that import vLLM.

**vLLM was NOT installed, and that is a decision with numbers behind it.** The
attempt ran (`pip install vllm` into `~/.local/venvs/iree`). It RESOLVES on this
box — aarch64, CPython 3.12, vllm 0.29.0 — and it was stopped during the
download phase, before it could touch `site-packages`, once the resolution was
readable. A `--dry-run --report` then stated the closure exactly: **186
packages**, including **torch 2.13.0** (replacing this venv's `torch 2.11.0+cpu`),
**33 CUDA-13 wheels** (`cuda-toolkit 13.0.3`, `nvidia-cublas 13.1.1.3`, …) landing
beside the venv's `jax-cuda12-plugin 0.10.0`, `transformers 5.17.0`, and a numpy
DOWNGRADE 2.4.4 → 2.3.5. That venv is not a scratch environment: it is the one
every certification oracle in this repo runs in — H2's safetensors bf16 parity
reads bytes *torch* wrote, `run_pytorch_llama.py` is the §0.4.289 op-for-op
mirror, and every PJRT lane imports that jax. Replacing the torch under the
oracles, and stacking a CUDA-13 runtime next to a CUDA-12 one in a venv whose
whole job is to run both in one process, is a change to the measurement
apparatus, and it is not this slice's to make unilaterally.
REJECTED: a second venv with vLLM *and* jax in it — it does not avoid the
CUDA-13-beside-CUDA-12 question, it only moves it somewhere nothing else would
notice it breaking.

**So the live-serving certification is a NAMED DEFERRAL with its command.**
When vLLM is installed (ideally in a venv of its own, with jaxlib present and
the CUDA question settled):

```
  pip install -e harness/python          # registers the entry point
  export TLALOC_SERVING_ARTIFACT=<dir written by ServingArtifactWriter>
  python -c "from vllm.platforms import current_platform; print(current_platform)"
  vllm serve <tokenizer/config> --max-num-seqs 4 --max-model-len 4 --block-size 2
```

The first command is the discovery check (vLLM's platform resolution must land
on `TlalocPlatform`); the second is a single-request `generate()`.

**What the plugin is.** `harness/python/vllm_tlaloc/`, plus a `pyproject.toml`
beside it whose only real content is the entry point:

| file | imports vLLM? | certified? |
|---|---|---|
| `__init__.py` — `register()`, `PLATFORM_CLASS_PATH` | no | yes |
| `paging.py` — KV page pool, block tables, slot arithmetic | no | yes |
| `batching.py` — requests → one padded bucket-selected call | no | yes |
| `runner.py` — `TlalocModelRunner`, artifact + pools + sampling | no | yes |
| `platform.py` — `TlalocPlatform` | **yes** | **no** |
| `worker.py` — `TlalocWorker` | **yes** | **no** |

That split is the slice's central engineering decision and not tidiness. A
plugin whose page arithmetic can only be exercised with 186 packages present is
a plugin whose page arithmetic is never exercised. Because the vLLM-facing
classes are adapters and nothing else, the uncertified surface is exactly
"vLLM's API shape" — and the thing underneath it runs against a REAL artifact on
REAL PJRT in this commit.

**Decisions, and what they rejected.**
- **Page 0 is reserved as the padding scratch page and never allocated.**
  `DecodePadding.PADDING_BLOCK` is 0, so every padded row in a bucket reads page
  0; handing it to a live sequence makes an inert row read live KV. Harmless
  *today* (the padded row's logits are sliced off inside the loader) and not
  harmless the moment anything poisons or checksums that page — which H3a's own
  oracle does. One page out of `numBlocks` is the cheap side. REJECTED: a
  per-request padding block, which is a wire constant that varies.
- **The free list is kept SORTED.** REJECTED: a LIFO stack, the usual choice —
  the certification compares against an independently computed expectation, and
  that needs "which page does the newcomer get" to have one answer.
- **`append_token` computes the slot BEFORE advancing `length`**, and allocates
  at most one page. Prefill's many-tokens-at-once is `reserve()`, kept separate
  so the decode path cannot grow by more than a page.
- **The token axis is INDEXED, not flattened away** (`last_token_logits`). The
  contract's logits are `[batch, tokensPerSeq, vocab]` and `tokensPerSeq` is 1
  today; reshaping to `(-1,)` is correct now and silently samples from a
  concatenation of positions the day a prefill entry exists.
- **`check_and_update_config` REFUSES rather than adjusts.** A `--block-size`
  disagreeing with the artifact's compiled `blockSize` is a different KV layout,
  not a preference; `--max-model-len` past the ladder is refused for the same
  reason H3a's loader refuses an over-cap context rather than clamping. It sets
  `worker_cls`, because that is what the hook is for.
- **`determine_available_memory` reports the artifact's pool, not a
  measurement.** vLLM profiles a forward pass to size the KV cache; here the
  pool is a COMPILED shape, and a profile more generous than the truth is a
  silent out-of-bounds write.
- **`get_attn_backend_cls` raises by name**: attention is inside the compiled
  program (`PAGED_ATTENTION`), chosen at export. A config asking for
  FlashAttention gets an answer instead of being quietly ignored.
- **vLLM is not a dependency of the distribution.** It is the HOST. Declaring it
  would let a plain `pip install .` drag the closure above into any venv.
- **A longer-than-one-token prompt is refused BY NAME** — there is no prefill
  entry (H3a's deferral), and a runner that quietly decoded a prompt one token
  at a time would be "working" while doing what no serving system accepts.

**Oracles.** Three lanes, in `VllmPluginContractTest`:
1. **Registration**, needing nothing installed: the entry point in
   `pyproject.toml`, `register()`, and `PLATFORM_CLASS_PATH` all name one class,
   and that class exists in the file the path names. A typo here surfaces in
   vLLM as "no platform found", the least informative error in the system.
2. **The vLLM-free unit lane**: 31 stdlib-only `unittest` cases over the page
   pool, the marshalling, the scheduler-output adapter, sampling and the
   registration metadata — driven from the Kotlin test so the coverage is part
   of `./gradlew test` and not a command somebody remembers.
3. **The plugin-driven decode lane**: Kotlin exports the reference artifact,
   then `TlalocModelRunner` — the class the worker delegates to — runs **two**
   decode steps of three sequences through its own bookkeeping, on the PJRT
   **CPU** client at `1e-5` and the **CUDA** client at `1e-3` (H3a's measured
   per-backend floors; the looseness is XLA-GPU's TF32 dot policy). Two steps
   and not one: a single step cannot tell a KV cache threaded across steps from
   one recomputed, and the pool swap is the runner's most easily-wrong line.

The numbers are checked against the host interpreter, but the **decisions are
pinned exactly on both lanes** — a page id is not a float. Block tables, slot
mapping, positions, `seq_lens` and the chosen bucket are computed in Kotlin from
the RULE and compared to what the plugin did; host-side greedy sampling must
agree with the interpreter's argmax, which it must, because the token it picks
is fed into the next step. Then: two steps at one bucket compile ONE executable
(the manifest's `cacheKey` works) while the batch-of-one lifecycle step compiles
a second (bucket selection really selects); freeing a sequence returns its pages
and the newcomer gets the freed one back (a loop that leaks a page per request
dies of it); and the runner's own history of what it decoded is what vLLM's
scheduler output cannot supply.

**Sensitivity check.** `append_token` was mutated to advance `length` before
computing the slot — the classic off-by-one, and the failure mode that writes
every token one slot past itself. Three lanes failed (the unit lane and both
device lanes); all three passed again on revert.

**House landmine found.** `:maestro:jvmTest` does NOT re-run when only
`harness/python/**` changes — the Python files are not declared task inputs, so
a Python-side edit needs `--rerun` (a mutation check without it reports BUILD
SUCCESSFUL in 600ms on mutated code). This applies to every subprocess
certification in the repo, not only this one.

**Named deferrals from this slice**: the live vLLM lane above (platform
discovery + a real `generate()`); vLLM's own sampler and logprobs (v1 samples
greedily on the host — adapting to vLLM's sampler means materialising logits as
torch CPU tensors, a measurable cost that wants a real vocabulary to measure);
prefill and chunked prefill, refused by name in three places and waiting on a
PREFILL entry in the artifact; buffer donation, still open from H3a and still
the next measurable win; multi-device serving (refused in
`check_and_update_config`); vLLM's prefix caching / RadixAttention, which the
reserved-scratch-page allocator does not model; and the paged-attention KERNEL,
which is H4.

**Still open in Phase H**: H3c (staged weights + a real Llama through the
plugin), H4 (KPTX paged-attention kernel + recognizer claiming), H5 (SGLang
variant + KV-quant).

### H4 — the KPTX paged-attention kernel + claiming (§0.4.471)

Where Tlaloc's differentiator meets the serving path. Helion-style kernel
libraries are **hand-invoked**: the model author writes the call. Here the
author writes `PAGED_ATTENTION`, the claiming pass sees a GB10 under it, and
the fused kernel appears — or does not, on a machine without a KPTX tier, with
the same numbers either way.

**Claiming a first-class op kind.** Every earlier template
(`FlashAttentionKernel`, `RmsNormKernel`, `AttentionKernel`, `RopeKernel`,
`CrossEntropyKernel`) claims an `OpKind.COARSENED` — a compile-time artefact
carrying a `primal_body`, whose fallback is *decomposition* back into that
body. `PAGED_ATTENTION` is not that: it is a real op with an interpreter arm
and a gather-composed emission of its own. So `lowerKernelChoice` grew a
second lane, keyed by **`OpKind`** rather than by a `primal_body` name
(`inferenceRegistry`, `kptxInferenceKernelTemplates`), and its decline path is
**the op itself**. That makes inference-side claiming strictly safer than the
COARSENED kind: there is no "neither annotated nor decomposed" hole to fall
into, and the emitter's new arm says so — a missing descriptor on a
`PAGED_ATTENTION` is the normal case, not the error it is on a COARSENED.
REJECTED: a separate `lowerInferenceKernelChoice` pass (two traversals with
two sets of clone restrictions to keep in step, for one `when` arm), and
wrapping paged attention in a COARSENED to reuse the existing lane (a
`primal_body` no one would ever inline, invented to satisfy a lookup).

**`defaultInferenceKernelTemplates` is EMPTY on purpose.** Claiming emits a
custom_call naming a kernel some runtime must have registered; a default-on
registry would turn every GB10 decode graph into a program XLA cannot link the
moment `:ir` is used without `:runtime-pjrt`. Opt-in per pipeline, the standing
convention for every KPTX template.

**The kernel** (`KptxKernels.pagedAttentionModule`) — the serving sibling of
§0.4.358's dense chain, same skeleton, K and V read *through the block table*:

1. `kptx_paged_scores` — one CTA per (sequence, query head), threads strided
   over the padded context. A live lane `j` resolves
   `block = blockTables[seq, j / blockSize]`, `off = j % blockSize`, dots
   `Q[seq,h,:]` against `K[block, off, kvHead, :]`, scales; a lane at or past
   `seqLen` is written `−inf`.
2. `kptx_paged_softmax` — §0.4.358's row softmax, **extracted and shared
   verbatim** rather than duplicated. The paged form needs exactly that
   program: `exp(−inf − max)` is `0` for any finite max, and a sequence always
   has at least one live lane, so the dead lanes fall out of the softmax on
   their own and no masking arm is needed. The dense module's PTX is unchanged
   (its kernel name and param name are what the helper takes).
3. `kptx_paged_out` — one CTA per output row, threads strided over `headDim`,
   accumulating only over live lanes.

GQA is indexing, not new math: `kvHead = h / (numHeads / numKvHeads)`,
computed per CTA. `seqLens[seq]` is **clamped** to the window width — an
over-long sequence is a scheduler bug, and clamping keeps the kernel inside its
buffers while H1c's bucket policy refuses the over-cap request by name at the
layer that can actually split it.

**What is baked and what is not.** Every dim-derived quantity —
`numHeads`/`headDim`/`blockSize`/`numKvHeads`/`maxBlocksPerSeq` — arrives as a
trailing i32 read from the call frame's *buffer shapes* at dispatch, and
`seqLens` is read from device memory inside the kernel. The one baked value is
`scale`, an f32 immediate in the scores stage, with the module cached per
`(block, scale)`: `scale` is a compile-time literal on the op, not a
dim-derived value, so this is the house specialization-cache pattern and not a
sentinel-dims violation. **Named deferral**: a scale-generic kernel, once
`KptxKernelRegistry.Stage` grows access to the frame's decoded FFI attrs — the
emitted call already carries `scale` in its typed-FFI `backend_config`.

**The module line.** `:kptx` moved from a test-only dependency of
`:runtime-pjrt` to a main-side one. `KptxKernelRegistry` deliberately takes PTX
*text* and so never needed it; claiming does, because the serving path itself
must now register a chain for a name the compiler emitted
(`KptxPagedAttention.register`). `:kptx` is pure-Kotlin PTX construction with
no native surface, and `:ir` stays free of the FFM/CUDA world — the two halves
of claiming live in different modules on purpose, which is the same line H3
drew between the plugin and the work.

**Oracles, and what they found.** At numSeqs 8 / numHeads 8 / numKvHeads 2 /
headDim 64 / blockSize 16 / ctx 128, a permuted block table and ragged lengths
(full window, a single token, mid-page and page-aligned stops):

| pair | worst \|Δ\| |
|---|---|
| kernel vs interpreter's Double paged walk | **1.2e-7** |
| gather-composed emission vs that same walk | **2.44e-4** |
| kernel vs emission | 2.44e-4 |

The gap is **the emission's**, not the kernel's, and the number names its
cause: `2.44e-4 ≈ 2^-12` at magnitude ~1 is TF32's mantissa, which is what XLA
lowers `dot_general` to by default on this device. The kernel's sequential
`fma.rn.f32` chain is within 1e-7 of the Double walk. H1a's smoke pins 6e-8
because its fixture (headDim 4, ctx 6) is too small for XLA to pick a
tensor-core path at all — so this only shows up at a shape with real work in
it. The test therefore pins kernel-vs-oracle at the house 1e-4 floor,
kernel-vs-emission an order looser with the reason stated, and adds the
property that actually matters: **the fused kernel must be at least as close to
the Double oracle as the emission it replaces**.

**The cost, reported honestly.** Per-call floors, same session, executable
already compiled, host round trip included:
**claimed `@kptx_paged_attention` 465 µs, unclaimed gather-composed 310 µs.**
The correctness-tier kernel is **1.5× SLOWER** than the lowering it replaces.
That is the expected §0.4.358 outcome, not a surprise: three f32 scalar loops
with one CTA per (sequence, head) against XLA's tiled tensor-core
`dot_general`s. This slice is the **claiming** milestone — the end-to-end path
from an op in a decode graph to Tlaloc's own PTX running inside an XLA
executable, with the numbers pinned. The warp-specialized pass (the
TLX-informed bf16 GEMM tile ingested in §0.4.357) is the follow-up, and until
it lands **nothing should register this kernel in a serving deployment** —
which is exactly why the default inference registry is empty.

**Named deferrals from this slice**: the performance tier (warp specialization,
shared-memory staging of the page window, one warp per lane group rather than
one CTA per row — the measured 465 µs is the floor this must beat before the
registry is anything but opt-in); a scale-generic kernel (above); bf16 pools,
declined by the template by name rather than silently miscomputed; the
prefill/ragged form, still H1a's deferral; mixed-dtype PJRT input lanes, so the
GPU test still rides `blockTables`/`seqLens` as I32 consts — they do reach the
kernel as real device buffers, XLA materialises a constant, so the dispatch
path under test is the production one; and fusing the adjacent
`KV_CACHE_WRITE` + `PAGED_ATTENTION` pair into one claim, which H1b left
available and which wants the perf tier first.

**House landmine found.** `ptxas` rejects a **non-ASCII byte anywhere in the
PTX file, comments included**, and the failure surfaces as
`CUDA_ERROR_INVALID_PTX` out of `cuModuleLoadData` deep inside an XLA
execution, naming nothing. A typographic `·` in a kernel comment cost one
debugging round trip. Pinned in `PagedAttentionModuleTest` for both chains.

**Still open in Phase H**: H3c (staged weights + a real Llama through the
plugin), the H4 performance tier, H5 (SGLang variant + KV-quant).

### H5 — KV-quant, and the SGLang design entry (§0.4.472)

Two halves, prioritized as the slice's brief said to prioritize them: KV-quant
is a real capability and got the code; SGLang is a second frontend over the
same artifact and got a design record, because **SGLang is not installed and
does not install cheaply here** (neither is vLLM — see H3b; `pip install
sglang` pulls a CUDA torch stack and a compiled kernel library into a venv
whose jaxlib+CUDA pins are what the rest of this arc certifies against). A
plugin package written against an import that is not there, and certified by
nothing, would be a worse artifact than a page that says exactly what the
second frontend costs.

#### (1) KV-quant: the contract, not the directive

**What was there.** §0.4.257's `KvQuantConfig` — an attr stamped on a COARSENED
attention op saying "materialize K and V narrow at runtime", read by a vendor
kernel, backed by no IR, no interpreter arm, and no number anyone could check.
Honest as a kernel hint; useless as a capability. `BackendTarget.kvQuantDtype`
has carried a value since §0.4.258, and what it named was that hint.

**What landed.** The other half — the exact map from a float page pool to
(codes, scales) and back, in three pieces:

- **`KvQuantPool`** (`:ir`, `io.tlaloc.ir.inference`) — the symmetric-absmax
  codec. `scale = absmax / qmax` per group, `code = clamp(roundHalfAway(x/s))`,
  `x̂ = code * s`, with the **error bound derived rather than tuned**:
  `|x − x̂| ≤ scale/2 = absmax/254` for int8, and `maxRoundTripError` recomputes
  it so the tests assert the measurement against the derivation.
- **`OpKind.DEQUANTIZE_KV`** — `(codes, scales) → pool`,
  `out[b,s,h,d] = codes[b,s,h,d] * scales[h]`. Interpreter arm (which
  *delegates to the codec*: one formula, two callers), StableHLO emission
  (convert + `broadcast_in_dim dims = [2]` + multiply), cost-model arm, a
  renderer refusal that points at the codec, and both AD transforms refusing
  **by name**.
- **The manifest slot, fed end to end** — `DecodeModelShape.kvQuant` →
  `ServingModelShape.kvQuant` (`ServingKvQuant`), round-tripped through the
  artifact's JSON.

**Why the scales key on the kv-HEAD axis.** It is the only pool axis that is
neither allocator bookkeeping nor within-vector structure. `numBlocks` and
`blockSize` index *pages*, which the allocator recycles between sequences — a
per-page scale would have to be rewritten on every reassignment, and a decode
step writing one slot would invalidate the scale of every other token sharing
its page. `headDim` is inside a single key vector, where a per-element scale is
an f32 pool with extra steps. Per-head is also what production serving does,
and it is what `KvScaleStrategy.PER_HEAD` has meant since §0.4.257 — this is
the first slice where the meaning is executed. The strategy is **derived from
the scale operand's extent**, never an attr: `[numKvHeads]` is per-head, `[1]`
is per-tensor.

**REJECTED: folding dequantization into `PAGED_ATTENTION`** as two optional
scale operands plus a dtype attr. It reads attractive (one op, one fused
kernel) and it is wrong for this IR: it makes that op's arity a mode flag (5
operands or 7), duplicates the formula inside the arc's most intricate
emission, and hides the quantization from every *other* consumer of a pool
(`KV_CACHE_WRITE`'s read-modify-write, a debug print, a CPU fallback). As its
own kind it is one visible node that CSE shares between a layer's K and V
paths and that H4's claiming lane can claim **as a pair** — which is the
natural shape of the fused follow-on.

**FP8 IS REFUSED BY NAME, not missing.** int8/int4 are *integer-code* formats:
a code is a small integer and the dequantization is one multiply. fp8 is a
*float* format whose code is a bit pattern with its own exponent field, so
`value = code * scale` is simply not its dequantization — an integer-code path
would either store the pattern (making the multiply meaningless) or round
twice, to a different and worse bound than the one derived above. The predicate
is `KvQuantDtype.isIntegerCoded`, and the refusal fires in the codec, the op
parser, `DecodeModelShape`, and at emission. This answers the slice brief's
"fp8 if the dtype story allows": **it does not allow it** — bf16's sealed-DType
work (§0.4.455–458) is precisely the tour fp8 needs, and this is the reasoned
no rather than a half-fp8 that rounds twice and reports a bound it does not
meet.

**The v1 deferral, said out loud IN THE ARTIFACT.** There is no `I8` DType, so
the codes ride an `I32` tensor and **v1 buys the contract, not yet the bytes**.
Rather than leave that to a doc, `ServingKvQuant` carries both facts as
separate fields: `dtype = "int8"` (what the codes *mean*, the accuracy story)
and `codeDtype = "i32"` (what they *ride*, the byte budget). A deployment
sizing a KV pool reads one, a deployment reasoning about answer quality reads
the other, and today they disagree — which is exactly the kind of fact a
manifest exists to carry. When a narrow dtype lands, one field changes and the
emission changes in one place (the `convert`'s operand type).

**Oracles.**

1. *Paged attention over an int8 pool vs the f32 pool, at a derived floor.*
   The same decode step twice — once on f32 pools, once on pools quantized and
   dequantized **in the graph** — with the agreement floor computed from the
   quantization bound and the actual inputs (`scale/2` on V directly, plus the
   softmax-weight wobble the K error induces), not from a tuned tolerance. The
   test also asserts the two runs actually *differ*, so a codec that
   round-tripped exactly could not pass while proving nothing.
2. *A hand-derived vector with exact arithmetic.* Scales chosen so
   `absmax/127` is a power of two (1 and 2), making the expected codes,
   dequantized values and per-element errors all literals a reader checks on
   paper — including three ties that land exactly on the bound, which is where
   the round-half-**away**-from-zero convention is pinned (`kotlin.math.round`
   breaks ties towards +∞, which biases a code set asymmetrically; the
   convention is spelled out in the codec rather than inherited).
3. *The bound is TIGHT.* Over a pseudo-random pool, every element is under
   `scale/2` **and** some element reaches >0.9 of it — a conservative codec
   that rounded everything to zero could not pass.
4. *One formula, two callers*: the in-graph op and the host codec pinned
   elementwise, per-head and per-tensor.
5. *The manifest round trip*, including an artifact written **before** this
   slice (no `kvQuant` field) still reading as unquantized.

**Named deferrals from this half**: the `I8`/`FP8` DTypes and with them a
byte-narrow pool (the whole point of KV-quant in production, and a
bf16-sized piece of work); the **quantized decode-graph signature** — the
`DecodeGraphSpec` slot types and the scale inputs a quantized graph needs at
its boundary, so a plugin can hand a quantized pool to a compiled entry (v1
lands the op, the codec and the manifest field; the graph builder that wires
them into a full decode signature is the tail); a fused
`DEQUANTIZE_KV → PAGED_ATTENTION` claim (H4's lane already has the shape); the
Python side reading `kvQuant` to size its pools; int4 is implemented and
certified but **not** recommended — the measured error is what it is; and
calibration (scales come from the pool's own absmax, with
`quantizeWithScales` taking a caller's better ones).

#### (2) SGLang: what the second frontend actually costs

The precedent is **SGL-JAX** — SGLang's JAX backend, the sibling of the
`tpu-inference` shape §1 chose for vLLM. The integration surface is a
**model runner**: SGLang's Python keeps the scheduler, the radix tree, the
batch formation and the API; the backend supplies a runner that takes a
forward batch and returns logits.

**What SGLang needs that vLLM's plugin already gave us, unchanged:**

- the **artifact and its loader** (H3a): `ServingManifest`, the bucket ladder,
  bodies-by-hash, the PJRT executable cache. This is the whole point of
  manifest-as-artifact — the second frontend reuses it *wholesale*, and
  nothing in it mentions vLLM.
- the **bucket selection** (H1c/H3a): round the batch up, refuse over cap. The
  manifest states the ladder, so the SGLang side performs selection with no
  second copy of `DecodeBucketPolicy`'s arithmetic.
- the **slot binding by role** (`DecodeSlotRole`): SGLang names its tensors
  differently and binds the same roles.
- the **KV pool layout** (`kvPoolAxisOrder`/`kvPoolDims`, and now `kvQuant`).

**What is genuinely SGLang-shaped and is not written:**

- the runner class itself and its registration path (SGLang's backend
  selection is not vLLM's platform-plugin entry point),
- SGLang's **ForwardBatch → slot** adaptation: its scheduler hands a different
  batch object, and `req_to_token` / `token_to_kv_pool` are *its* page
  bookkeeping, not vLLM's `block_tables`. The mapping is mechanical but it is
  the part that must be written against a real installed SGLang, not guessed,
- capture of its **CUDA-graph / overlap-scheduler** expectations, which we do
  not satisfy and do not need to (we execute a compiled PJRT executable).

**RadixAttention and prefix caching stay THEIRS.** We supply graphs; the
radix tree, the prefix match and the eviction policy are the scheduler's, and
integrating *into* SGLang is precisely what buys them. What we would owe the
radix path is a prefill graph that can start from a non-zero context — the
**ragged/chunked-prefill form of `PAGED_ATTENTION`**, which has been H1a's
named deferral since the first slice and is the one IR-level item the SGLang
integration would actually need from this side.

**The cheapness claim, stated so it can be checked rather than assumed**: of
the H3a/H3b Python surface, the loader, the manifest reader, the bucket
selection and the PJRT execution path are frontend-agnostic; what a second
frontend rewrites is the adapter class and its batch translation. That is the
prediction the manifest-as-artifact design makes, and the honest status is
that it is **untested** until an environment carries SGLang.

**Still open in Phase H**: H3c (staged weights + a real Llama through the
plugin), the H4 performance tier, the H5 tails above (narrow DTypes, the
quantized decode-graph signature, the SGLang runner), and the ragged prefill
form that both frontends eventually want.

### H6a — PJRT from Python, with no framework under it (§0.4.475)

**The claim H3a could not make.** "A Python process with no JVM runs the
artifact" has been certified since §0.4.469 — but that process reaches PJRT
through **jaxlib**: `jax._src.xla_bridge` for the backend, `jaxlib.mlir` for
the module, `jaxlib._jax.CompileOptions` for the compile. So the serving
runtime's real dependency list was "a driver, a plugin, and 500 MB of
framework whose CUDA wheel family must not collide with anything else in the
venv". The §0.4.470 refusal (186 packages, torch 2.13, 33 CUDA-13 wheels) and
the §0.4.474 canary are both consequences of that sentence.

`harness/python/tlaloc_pjrt.py` removes the framework from the sentence. It
binds the PJRT C API with **`ctypes` and the standard library only** — no
jax, no jaxlib, no torch, **no numpy** — and it is a mechanical mirror of
`runtime-pjrt/.../ffm/PjrtFfm.kt`, the Kotlin FFM binding that has carried
every PJRT lane here since §0.4.303. Offsets, struct layouts, the
`PJRT_Buffer_Type` codes, the §0.4.333 create-options NamedValues, the
§0.4.304 six-byte `CompileOptionsProto`, the §0.4.459 platform gate: each
carries the § number of the commit that learned it, because re-deriving them
from the header would have meant re-learning the reboot incident and the XLA
Check-fail by repeating them.

**A plugin `.so` can live anywhere**, and that is the whole dependency story:
inside somebody else's jax install, at `/lib/libtpu.so` on a TPU VM, or in a
directory a deployment ships. The path comes from `TLALOC_PJRT_PLUGIN_PATH`
(the same variable `PjrtBinaries` reads JVM-side) or an explicit argument.
Nothing imports the package the file happens to sit in.

**What is bound**: `GetPjrtApi`, `Client_Create` with create_options,
platform name, device enumeration, `Compile`, `BufferFromHostBuffer` for
F32 / **I32** / BF16, `OnDeviceSizeInBytes`, `ToHostBuffer`,
`LoadedExecutable_Execute` with the padded `ExecuteOptions`, `NumOutputs`
through `GetExecutable`, `Event_Await` / `Event_Destroy`, and every Destroy —
each handle a context manager, each `close()` clearing its pointer *before*
the destroy call so a raise cannot become a double free.

**I32 is the one code with no JVM twin.** The Kotlin has never staged an
integer buffer, so `PJRT_Buffer_Type_S32 = 4` was read off the same enum the
others come from — and is *certified*, not assumed: the i32 lane round-trips
a buffer through a real executable and compares exact integers, which a wrong
type code cannot survive. (Mutating it to 5 fails both lanes; that check was
run and reverted.)

**Certification, in two independent halves.** `PjrtCtypesBindingTest`:

1. **The ABI mirror, GPU-less.** The driver's `--layouts` mode reports every
   struct size, the padding-sensitive field offsets, the whole `PJRT_Api`
   offset table, the buffer-type codes, the compile-options bytes, and the
   marshalled create_options array — and the JVM asserts each against
   `PjrtFfm`'s own values and against the bytes
   `PjrtFfm.marshalCreateOptions` writes. This half needs no accelerator,
   which matters because a struct laid out wrong does not fail at the call
   site; it segfaults three calls later. The one interesting difference is
   in our favour: the Kotlin pads `PJRT_ExecuteOptions` after `launch_id` by
   hand because FFM lays fields exactly where you put them, while ctypes
   pads by C's own alignment rule — and the test proves the two land on the
   same 120 bytes.
2. **Real XLA, with jax blocked.** Four graphs (`x+y` f32, a 2×3·3×2 matmul,
   an i32 add, a bf16 identity) compiled and executed through the ctypes path
   on the GB10's CUDA plugin, against `DxirInterpreter`. Floors: **1e-5**
   elementwise, **1e-3** matmul (XLA-GPU's TF32 dot policy — named, not
   tuned), **exact** on i32, **bit-for-bit** on the bf16 patterns, plus the
   staged buffer's on-device size proving 2-byte bf16 storage. Values cross
   the process boundary in both directions as **raw bit patterns**, so no
   claim here passes through a decimal printer.

**The pin.** Before importing anything, the driver installs a
`sys.meta_path` finder that **raises** on `jax`, `jaxlib`, `torch` or
`numpy`, and the entire path runs underneath it. The interpreter is the
oracle venv, where jax 0.10.0 *is* installed (§0.4.474 pins exactly that), so
"no jax was imported" is a claim about restraint rather than absence — the
framework was reachable and the serving surface did not need it. Proving the
same thing in an empty venv would have proven less and cost a venv. And
because an unfired guard certifies nothing, the run ends by deliberately
importing jax and reporting that it was stopped; both halves are asserted.

**Named deferrals.** `tlaloc_serve.py` was **unchanged** at §0.4.475 and
still reached PJRT through jaxlib — re-pointing the H3a loader at this
binding was **H6b**, and it landed in §0.4.476 (below). Also unbound, by
name: `Compile`'s donation options (the manifest
has carried `donationPairs` since H3a), the async `PJRT_Buffer_ReadyEvent`
path, f16 / i8 / fp8 staging, multi-device execute, and the §0.4.461
multi-node create-options — those exist JVM-side so a multi-node client can
be *refused*, and a knob that can only be set to its own refusal is worse
than no knob.

### H6b — the loader runs on it, and the dependency list empties (§0.4.476)

**What changed.** `tlaloc_serve.py` now compiles and executes through
`tlaloc_pjrt`. The H3a surface is intact — manifest read, `verify_bodies`,
`select_bucket` / `entry_for` off the ladder, `run_decode(...)` returning
`(logits, kv_pools)`, the `cacheKey`-keyed executable cache — and every jax
and numpy call underneath it is gone. At module scope the loader imports
`json`, `os`, `hashlib`, `pathlib`, `dataclasses`, `typing` and
`tlaloc_pjrt`. Nothing else.

**The headline, and how it is measured rather than asserted.** The CUDA lane
of `ServingArtifactExportRunTest` runs the whole path — manifest parse, body
hashing, bucket selection, compile, staging, execute, readback — under a
`sys.meta_path` guard that *raises* on `jax`, `jaxlib`, `torch` and `numpy`,
in the oracle venv where jax is genuinely installed. The guard goes up
**before `tlaloc_serve` is imported**, so the module's whole dependency
surface is under test and not merely whatever its methods happened to touch.
`theFullServingPathRunsWithNoFrameworkImported` asserts three separate facts
on that same subprocess — the one that produced the numbers, not a
hello-world beside it: nothing forbidden is in `sys.modules`, nothing was
even *attempted* (so this is not a path that tried and fell back), and the
guard **fired** when deliberately provoked. Without the third, the first two
are equally consistent with a guard nobody installed (§0.4.474's lesson).

`import_guard.py` is that net, factored out of §0.4.475's copy of it. Two
copies of a safety net drift, and the one that drifts is the one nobody is
watching. Its `blocked_attempts` is now snapshotted *before* the
self-provocation, so the field means "what the path under test reached for"
rather than "what the check did to itself".

**The wire format is flat lists.** Without numpy there is no ndarray, so
every host-side tensor is a flat row-major Python sequence and the shape
comes from the manifest — which is where it was authoritative anyway.
`run_decode` returns logits nested to the slot's declared trailing dims
(`[tokensPerSeq, vocab]` per real row, token axis KEPT at 1 so
`last_token_logits` still finds the last position the day a prefill entry
exists) and the pools whole and flat. REJECTED: a tiny ndarray-alike so the
surface *looked* unchanged — a shim that is 5% of numpy is a thing every
caller must learn and cannot trust.

**Mixed dtypes stopped being a squeeze.** Each slot is staged as its own
PJRT buffer of its own declared type: the five I32 index operands are real
device `S32` buffers, the pools and logits `F32`, and `bf16` rides as raw
u16 patterns. `f64`/`i64`/`bool` are refused **by name** rather than
coerced. The manifest always described per-slot dtypes — this honours a
schema that was already right.

**TWO ENGINES, and the honest reason.** `engine="ctypes"` is the serving
engine and the default for every platform that has a plugin file.
`engine="jax"` survives as an **oracle only**, because **jaxlib ships no CPU
PJRT plugin `.so`** — its CPU client is a C++ class inside the jaxlib
extension, so the binding has nothing to `dlopen` for `"cpu"` and the tight
**1e-5 XLA-CPU semantics lane** has no ctypes route on this machine.
REJECTED: deleting it for tidiness, which would have deleted a certified row
to make a sentence shorter. Its imports live inside its methods, so
selecting ctypes imports none of it. `default_engine_for` is the one place
the rule is written, and it is the only thing that changes the day a CPU
plugin (or a TPU VM's `libtpu.so`) exists here.

**The distribution's `dependencies` is now `[]`**, pinned by a test. It said
`numpy` while the loader used ndarrays. An oracle's numpy is something you
install to MEASURE a deployment; leaving it declared makes pip unable to
tell the two apart. `tlaloc_pjrt` and `import_guard` join `tlaloc_serve` in
`py-modules`, because shipping a loader without what it imports installs a
module that cannot import.

**Sensitivity check.** A bare `import numpy` added at `tlaloc_serve`'s
module scope failed *both* the guard test and the CUDA numeric lane, then
was reverted.

**A note on the suite count.** The JUnit total moves 2294 → 2295 — one new
Kotlin test — but the stdlib-only Python lane grew **31 → 42 cases**
(`LoaderIsStandardLibraryOnly`, plus the empty-dependency pin). Those run
inside `VllmPluginContractTest` lane 2 as a subprocess and so contribute one
XML row between them, which is the shape H3b chose and not something this
slice should quietly change. Counting them would have meant inventing a
second way to count.

**Named deferrals, unchanged by this slice.** Buffer donation (still
unwired — every step round-trips whole pools through the host, a
performance fact and not a correctness one), staged weights, prefill,
multi-device execute, and the CPU lane's dependence on the jax oracle.

### H7 — vLLM installed, and the live lane run (§0.4.477)

**The two files that had never executed, executed. One of them was
wrong.**

#### The venv, and why it could be built at all

`~/.local/venvs/vllm`, created from nothing with `python3 -m venv`, holds
**vLLM 0.29.0 + torch 2.13.0+cu130 — 197 distributions, ~7.7 GB, and no
jax**. The oracle venv was not touched: `check_oracle_venv.py` was run in
`~/.local/venvs/iree` before and after and reports the same pins
(jax 0.10.0, torch 2.11.0+cpu, `torch.version.cuda is None`), and
`OracleVenvIntegrityTest` is green in the clean-room run below.

The install resolved and completed on aarch64 under `--only-binary=:all:`,
which was a deliberate choice and not a flag copied from somewhere: a
source build of any one of 197 packages on this box is a stall with no
upper bound, and the slice was time-boxed. Wheels existed for all of them.
The §0.4.470 dry-run's shape was right (186 packages then, 197 now).

**This venv is possible only because of H6.** Before §0.4.475–476 the
loader reached PJRT through jaxlib, so a live vLLM lane would have needed
`jax[cuda12]` installed beside vLLM's 33 CUDA-13 wheels — the exact
collision the H6 rail exists to prevent, one directory over. Since H6 the
serving path is ctypes and the standard library, so the vLLM venv gets
`pip install -e harness/python` (which pulls **nothing**) and
`$TLALOC_PJRT_PLUGIN_PATH` pointed at the `xla_cuda_plugin.so` that
happens to sit in the ORACLE venv's site-packages. Loading a `.so` from
another venv's directory is `dlopen`, not a Python dependency: no code
from that venv is imported, and its `python` is never executed. That
sentence is the whole reason H6 came first.

#### What the live lane does — `run_vllm_live_check.py`

Three stages, in the vLLM venv, against a real artifact on real PJRT-CUDA:

1. **Discovery.** `from vllm.platforms import current_platform` resolves to
   `vllm_tlaloc.platform.TlalocPlatform`. vLLM logs `Platform plugin tlaloc
   is activated`. The entry point, `register()`'s dotted path, the
   `PlatformEnum.OOT` contract and `get_device_name`/`supports_v1`/
   `is_async_output_supported` all hold against the vLLM that is installed
   rather than the one the docstrings were written from.
2. **The config hook**, against **real `vllm.config.CacheConfig` and
   `ParallelConfig`** — `check_and_update_config` sets
   `worker_cls = vllm_tlaloc.worker.TlalocWorker`, takes `block_size` and
   `num_gpu_blocks_override` from the artifact, and all four refusals fire
   by name: `--block-size` disagreeing with the compiled `blockSize`,
   `--max-model-len` past the context ladder, `--max-num-seqs` past the
   batch ladder, `world_size != 1`. A refusal that has only ever met a
   `SimpleNamespace` has never met a pydantic validator; these have.
3. **The worker lane.** `TlalocWorker` through the v1 worker API:
   `init_device`, `load_model`, `determine_available_memory` (384 bytes —
   the compiled pool, computed from the manifest and checked against
   `numBlocks × blockSize × numKvHeads × headDim × 4 × 2 × numLayers` on
   the JVM side), `get_kv_cache_spec`, `initialize_from_config` (accepted
   when it agrees, refused **by name** when vLLM sizes the pool
   differently), then two `execute_model` steps returning a real
   `vllm.v1.outputs.ModelRunnerOutput`, and the chunked-prefill refusal.

#### The claim, and its floor

`VllmLivePluginTest` runs that lane in the **vLLM venv** and H3b's
already-certified `run_vllm_tlaloc_check.py` in the **oracle venv**, over
the same exported artifact and the same request, and compares. The logits
agree **bit-for-bit — `==`, not a tolerance** — across two venvs with
different torch builds, only one of which has jax, on both steps; so do
the sampled tokens, the chosen buckets and the compile count (1: one
bucket, one compile, the cache key did not move between callers). A
tolerance would have been wrong here: both lanes run the same ctypes
engine against the same plugin `.so` on the same device, so the only thing
that differs is the ADAPTER, and any difference at all is the defect.

#### What it found, on its first execution

`KeyError`, step 0. `TlalocWorker.execute_model` builds the batch and
*then* admits the new sequences, so
`decode_requests_from_scheduler_output` asked the runner for the token
history of a request it had not been told about — **the first step of
every server that would ever have started**. The vLLM-free unit lane
passed it because its `last_token_of` stand-in was a dict literal that
happened to have an entry for the new id, and a dict is more forgiving
than a runner. That gap *is* the gap between "written" and "certified",
and it is the argument for the whole slice.

The fix is not a reorder. A new request's feed token is already in the
scheduler output (the prompt's last token), so asking the runner for it
was always a detour through state that need not exist yet; taking it from
the prompt leaves `last_token_of` with one meaning — CACHED requests —
instead of two, and makes the function independent of the worker's
ordering rather than merely compatible with it. An empty prompt is refused
by name rather than indexed off the end. Two regression tests in
`vllm_tlaloc_test.py`: one whose `last_token_of` **raises** like a real
runner and asserts it is never consulted for a new request, one for the
empty prompt.

#### `./gradlew :maestro:exportServingArtifact`

The UNCERTIFIED list's item 3, closed as a side effect of needing it: a
`JavaExec` task over the **jvmMain** runtime classpath (not jvmTest — if
the exporter needed a test fixture, the artifact would be a test fixture,
and H3a's claim that the directory is the deployment would be a claim
about the test source set). `-PoutDir=/abs/path`; six entries written.

#### Deferrals this slice did NOT close, named

- **`LLM.generate()` / `vllm serve` end to end.** Both need a HuggingFace
  `config.json` and a tokenizer for a real model. The only artifact this
  repo can export is `ReferenceDecodeGraph` — a 16-wide-vocabulary LCG toy
  with no tokenizer. This is **H3c** (weight name-mapping + a real Llama),
  already the audit's largest open item, and it is a MODEL-COVERAGE gap,
  not a plugin gap. Fabricating a `config.json` around the reference model
  would have certified the fabrication, so the lane stops at the worker
  API and says so.
- **vLLM's own sampler and logprobs.** `ModelRunnerOutput.logprobs` is
  `None` and sampling is host-side greedy, as H3b decided.
- **The scheduler-output type.** The adapter is duck-typed on purpose (its
  docstring says why: the dataclass is internal and has been renamed
  between versions), so the live lane hands it a stand-in carrying the v1
  field names. Everything the adapter hands its result *to* — `WorkerBase`,
  `ModelRunnerOutput`, the validated config — is real. Driving vLLM's real
  scheduler requires an engine, which requires a model, which is H3c.
- **SGLang** remains a design record; `pip install sglang` now has an
  obvious answer (a third venv) and no one has spent it.

### H3c-1 — a real Llama's weights, by role (§0.4.478)

The first slice of the arc's last open item. H7 stopped at the worker API
because the only exportable artifact was a 16-token LCG toy; this slice
puts a **real 1.1B-parameter Llama checkpoint** on disk and gives the repo
a typed way to ask it for a tensor. It does NOT yet build a decode graph
from those tensors — that is H3c-2 — and it says so rather than implying
otherwise.

**The checkpoint.** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, fetched with
`huggingface_hub` in the **vLLM venv** into
`~/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0` — a cache
directory under `$HOME`, **inside neither venv**. 2.2 GB, single-file
`model.safetensors`, 201 tensors, every one BF16, `format: pt`. Chosen for
three properties, all of which a certification needs and none of which a
"tiny-random" stub has: it is **GQA** (32 heads / 4 KV heads), its MLP is
**rectangular** (2048 -> 5632), and it is **untied** — so the tied case had
to be certified some other way (it was; see below).

**What landed.**

1. `ir/.../inference/HfLlama.kt` (commonMain) — `HfLlamaConfig` (config.json
   through `:core`'s strict `parseJson`), `LlamaWeightRole` /
   `LlamaLayerPart`, and `HfLlamaNames`: the HF-name <-> role **bijection**,
   the role list a config implies, and `expectedDims(role, config)`.
2. `ir/.../inference/HfLlamaCheckpoint.kt` (jvmMain) — the only part that
   touches a filesystem. Opens a directory through §0.4.468's
   `SafetensorsIndex.openCheckpoint`, so **sharding is already handled and
   this file never learns which form it got**; TinyLlama is single-file, a
   70B is thirty shards, and the code above is identical.
3. `harness/python/read_hf_llama_probes.py` — the oracle. Imports torch and
   `safetensors` **on purpose**; it is not serving-path code and runs in the
   vLLM venv.

**THE LAYOUT FACT, verified rather than assumed.** HuggingFace stores every
`nn.Linear` weight **transposed, `[out_features, in_features]`**, because
`F.linear(x, W)` computes `x @ W.T`. The slice was told to verify this
against the real file and did:

| tensor | dims in the file |
|---|---|
| `model.layers.N.self_attn.k_proj.weight` | `[256, 2048]` |
| `model.layers.N.self_attn.v_proj.weight` | `[256, 2048]` |
| `model.layers.N.mlp.gate_proj.weight` | `[5632, 2048]` |
| `model.layers.N.mlp.down_proj.weight` | `[2048, 5632]` |

Those four are **rectangular**, which is the whole point: `q_proj` and
`o_proj` on this model are both `[2048, 2048]` and **cannot distinguish the
two conventions at all**. A certification that used only a square model
would assert nothing, and the test says so in place. `model.embed_tokens.
weight` is likewise no witness — it is a lookup table, not a Linear, so
`[vocab, hidden]` either way.

There is a second, independent confirmation in the tied-embedding case: a
tied `lm_head` reuses the embedding buffer, which **only typechecks under
`[out, in]`**. Under the other convention the head and the table would be
transposes of each other and tying would be a shape error.

**Defaulting rules, which are the knowledge the type exists to hold.**
`num_key_value_heads` absent means MHA (= `num_attention_heads`) — get it
wrong and an MHA checkpoint loads cleanly as GQA with 1/N of its K/V.
`head_dim` **when stated wins over `hidden/heads`**, because Llama-3.2 and
several derivatives decouple them (TinyLlama states none and the quotient
is 64). `rms_norm_eps` defaults to 1e-6, `rope_theta` to 10000.

**Refused BY NAME**, each because the alternative is a model that serves
and is quietly wrong: a scaled `rope_scaling` (both the modern `rope_type`
and the pre-4.43 `type` spelling; `{"rope_type": "default"}` correctly
means no scaling and is not refused), `attention_bias=true` (Llama proper
has none, Qwen2 does, and there are no bias roles), a non-Llama
`architectures[0]` unless the caller passes `strictArchitecture=false`, a
missing shape key rather than a guessed default, an indivisible GQA
grouping, and — the one worth reading twice — **a config that disagrees
with its own file about tying, in both directions**. A tied head is the
embedding table and an untied one is a separately trained matrix; picking
either answer silently moves every logit the model will ever produce.

**Oracles and floors.**

| claim | oracle | floor |
|---|---|---|
| the role<->name map is a bijection covering the model | 201 roles vs the real file's **201 tensors**, `verifyInventory()` returning no unmapped names | exact, total |
| HF stores Linear weights `[out, in]` | `expectedDims` vs the real header for **all 201 tensors**, plus the four rectangular witnesses spelled out | exact dims |
| the config record is the checkpoint's real shape | TinyLlama's own config.json, verbatim, in both the common and jvm tests | exact |
| the reader's bytes are the bytes torch sees | `read_hf_llama_probes.py` (torch 2.13.0+cu130 + `safetensors`, vLLM venv) on six probes incl. the **last element of the largest tensor** | **exact raw bf16 bit patterns, no tolerance** |
| …and every tensor's shape and dtype agrees with torch's own header read | same oracle's `shapes`/`dtypes` tables, all 201 | exact |
| tied embeddings resolve the head to the embedding table | a **hermetic** safetensors checkpoint the test writes byte by byte | `==` on the whole buffer |
| a config/file disagreement about tying is fatal | the same, both directions | by name |
| a transposed projection is caught at load, with the convention named | the same, `k_proj` written `[hidden, kvOut]` | by name, both dims in the message |

**SENSITIVITY CHECK, run and reverted.** `expectedDims(K_PROJ)` was flipped
to `[hiddenSize, kvProjOut]` — i.e. the *other* convention. Three tests went
red: the hermetic transpose case, the real-checkpoint layout lane, and the
torch probe lane. Reverted; the suite is green at the committed state.

**Lane split, deliberate.** The hermetic tests (tiny safetensors files the
test writes itself) run everywhere and gate `./gradlew test` on a fresh
machine. The real-checkpoint lane **self-skips** when
`~/.cache/tlaloc-checkpoints/...` is absent (override with
`TLALOC_HF_LLAMA_CHECKPOINT`); the fetch command is in the test's KDoc. On
THIS machine it ran: 9 tests, **0 skipped**, both real lanes included.

**REJECTED.** `:core`'s `io` package for the mapping — it holds FORMATS and
knows nothing about a transformer. `:maestro` — §0.4.468 already rejected it
for weights, and this is weights. Mocking `WeightSource` in the tests — the
claims are about how a *directory* resolves, and a mock replaces exactly the
thing under test. Loading first and letting the graph builder find the shape
mismatch — by then the tensor is a `FloatArray` with no name attached.
Fabricating a small config to avoid a 2.2 GB download — §0.4.477 refused the
same trade and was right.

**NAMED GAPS, carried forward.**

- **`WeightSource` exposes no `dtype(name)`.** Asking a checkpoint what
  width a tensor is stored at currently means decoding it. The fix is a
  header accessor in `:core` beside the parser that already knows; until
  then `HfLlamaConfig.storageDType` reports `torch_dtype` (what the producer
  INTENDED, not what the bytes ARE) and the test closes the gap by opening
  the same file through `SafetensorsFile` directly.
- **`:ir`'s Test JVM heap is now 2 GB** (Gradle's default is 512 MB). Lane
  B decodes a 131 MB `ShortArray` from a 131 MB read buffer. Probing only
  small tensors would have stayed inside the default and lost the probe that
  matters most — the last element of the largest tensor, which is what fails
  when a `data_offsets` base or a short read is wrong.
- **No graph is built from these weights yet.** Roles in, tensors out. H3c-2
  is `DecodeGraphSpec` + `ServingArtifactWriter` fed from a
  `HfLlamaCheckpoint`, and it is where the transpose stops being a
  documented fact and becomes a matmul.
- **The tokenizer is untouched.** `vllm serve` needs it and `tokenizer.json`
  / `tokenizer.model` were fetched alongside the weights, but nothing here
  reads them.
- **fp16 checkpoints remain refused** by §0.4.468's dtype table. TinyLlama is
  bf16 and loads; a fp16 Llama would need an `F16` DType first.

### H3c-2 — the real Llama becomes a graph (§0.4.479)

The other half of H3c's first leg. §0.4.478 could read a real TinyLlama-1.1B
checkpoint by role and had verified its layout; it computed nothing. This
slice **builds the decode graph from those tensors and certifies its LOGITS
against HuggingFace transformers.**

**What landed, four files.**

1. `ir/.../inference/HfLlamaDecodeGraph.kt` (commonMain) — the builder.
   `embed -> N x (rms_norm -> q/k/v -> RoPE -> KV_CACHE_WRITE x2 ->
   PAGED_ATTENTION -> o_proj -> +residual -> rms_norm -> SwiGLU -> +residual)
   -> final rms_norm -> lm_head`. Pure `DxirBuilder` over an `HfLlamaConfig`;
   touches no file.
2. `ir/.../inference/HfLlamaStagedWeights.kt` (jvmMain) — **the transpose,
   performed.** §0.4.478's `[out, in]` layout fact stops being documentation
   here: the seven Linears per layer and `lm_head` are transposed ONCE,
   host-side, into math layout; the embedding table and the norm gains are
   not, and the predicate that decides is `HfLlamaNames.isTransposedLinear`
   rather than a second list.
3. `harness/python/hf_llama_reference.py` — the oracle. Imports torch and
   transformers ON PURPOSE, runs in the **vLLM venv** (torch 2.13.0+cu130,
   transformers 5.17.0), and is not serving-path code.
4. Contract change: `DecodeGraphSpec.weightSlots`, a new `DecodeSlotRole.WEIGHT`.

**WEIGHTS ARE STAGED INPUTS, NOT BODY CONSTANTS — the slice's one design
decision.** §0.4.469's reference graph bakes its weights in; a real model
cannot. TinyLlama is 1.1e9 parameters and rendered as StableHLO `dense<[...]>`
literals that is **tens of gigabytes of TEXT**, in a file whose whole premise
(H3a) is that it *is* the deployment. Three further reasons it is right rather
than merely feasible: the bucket ladder compiles six or more executables and
constants would put a private copy of every weight in each, while staged
buffers are shared; the weights are already bytes in a safetensors file, so
staging is a read and an upload with no step where they become program text;
and the executable stops depending on weight VALUES, so two checkpoints of one
architecture share a compile. The slots go **after** the KV pools, so
`KV_POOL_INPUT_BASE` and every `donationPairs` entry H3a wrote are untouched —
pinned by a test.

**THE ORACLE, and what exactly was certified.** A **REDUCED slice: the first
TWO layers of the real checkpoint** — TinyLlama's real embedding table, its
real layers 0 and 1, its real final norm and its real untied head. Both sides
are reduced by the same arithmetic (Tlaloc over `config.copy(numLayers = 2)`,
transformers over `cfg.num_hidden_layers = 2`), so the thing compared is a
well-defined 2-layer Llama whose every weight came out of TinyLlama's file.
The reason is cost and it is stated rather than hidden: the reference
interpreter is a triple-loop evaluator and a 22-layer 1.1B step through it is
minutes at a heap that would dominate the suite. The reduced slice exercises
**every op in the decode graph**; the layers it drops are twenty more
instances of shapes already covered.

A 5-token prompt was run as **five single-token decode steps with the KV pools
threaded**, because `PAGED_ATTENTION`'s ragged chunked-prefill form is H1a's
still-open deferral — a PERFORMANCE deferral, not a correctness one.

| claim | oracle | floor |
|---|---|---|
| the real Llama's logits, every position of a 5-step prefill | `hf_llama_reference.py` — `AutoModelForCausalLM`, **fp32 on CPU**, vLLM venv | **1e-5 relative**; measured worst **2.3e-6** |
| …and the server would emit the same text | the same, per position | **argmax exact, and the whole top-5 ORDER exact** |
| the staged transpose is load-bearing | the same graph with `q_proj` staged in the file's `[out, in]` layout (shape-legal: q is square on this model) | the logits MOVE, by > 1e-2 |
| the built graph IS the contract | `DecodeGraphSpec.verifySignature` inside `build()`, and again as a test | exact, per slot |
| every staged Linear slot is the REVERSE of its file dims, every non-Linear identical | hermetic, against `HfLlamaNames.expectedDims` | exact dims, 7 per layer + `lm_head` |
| the RoPE tables are HF's DUPLICATED form | `headDim = 2` makes `inv_freq[0] = 1`, so the angle at position p is exactly p radians — written down, not recomputed | 1e-6, and `cos[p][0] == cos[p][1]` exactly |

**The floor is fp32-vs-fp32 on CPU, deliberately.** bf16 -> f32 is exact
(§0.4.468), and the oracle runs on CPU so TF32 never enters; what remains
between the two sides is ACCUMULATION ORDER — torch's blocked GEMM against a
row-major triple loop over a 2048-wide contraction. 1e-5 relative is a bound
with 4x headroom over the measurement, and a test asserts the worst case is
strictly INSIDE it, so the floor cannot quietly become the measurement.

**A GAP FOUND AND CLOSED ON THE WAY.** `RSQRT` and `SILU` had StableHLO
emission, a cost-model arm, a `TileFusion` entry and a RECOGNIZER ANCHOR each
(`RmsNormRecognizer` anchors on RSQRT, `SwiGLURecognizer` on SILU) — and **no
`DxirInterpreter` arm**. The reference evaluator refused the two ops every
transformer in this repo is built out of. Building a real Llama found it in
one run. Both arms landed, following the house Double-then-narrow convention
of the SQRT/SIGMOID arms they sit between.

That moved a boundary §0.4.273 had pinned. `LlamaDecoderCoarsenedAdTest`'s two
"surfaces interpreter boundary" cases were **inverted, not deleted**: the
Llama primal now evaluates on the host, and the coarsened form is additionally
held to the raw form's loss at 1e-5 relative — the coarseners' central claim
(a bundle computes what it replaced) checked by EVALUATION on this workload
for the first time. Host FD validation of the Llama gradient is now possible
for the first time; that is a named opening, not a promise.

**REJECTED.** A `TRANSPOSE` node per projection in the graph — the weight is a
PARAMETER, so there is no constant to fold and the cost is a full re-layout of
every projection matrix on every decode step. Certifying against a
"tiny-random" Llama to stay inside the old 2 GB test heap — §0.4.478 already
paid for a real checkpoint precisely so the numbers would be a real model's,
and a parity claim against a stub certifies the stub. A one-step sensitivity
check for the transpose — at `seqLens = 1` the softmax has a single term and
is exactly 1.0, so attention returns the just-written V and **Q does not enter
the answer at all**; the first version of that test reported a movement of
exactly 0.0 and was rewritten to run to a real context.

**NAMED GAPS, carried forward.**

- **THE PJRT/DEVICE LANE IS NOT CLAIMED.** `PjrtSession.runOn` is
  single-dtype (F32) and this graph's first five operands are I32; the
  mixed-dtype staging the serving path uses lives in the PYTHON loader, which
  reads a `ServingArtifactWriter` artifact — and that writer does not yet
  carry a weight table. The device lane arrives **with** the artifact.
- **H3c-3, the next slice, named precisely**: teach `ServingManifest` /
  `ServingArtifactWriter` a weight table and `tlaloc_serve.py` to stage it
  from the checkpoint, then re-run this parity on PJRT-CUDA and record ITS
  floor (expect the serving lane's 1e-3, TF32's dot policy). Until then a
  spec with non-empty `weightSlots` must be refused by the exporter rather
  than written as a half-artifact.
- **The tokenizer is still untouched** (§0.4.478's gap, unchanged).
- **`:ir`'s test JVM heap is now 8 GB** (was 2 GB, §0.4.478). Two real layers
  plus the `[32000, 2048]` table and the untied head is ~880 MB of staged f32,
  and each transposed Linear exists twice for the duration of its transpose.
- **The 22-layer lane is not run.** The reduced slice is what is certified and
  the entry says so everywhere it appears.

### H3c-3 — the artifact learns to carry a model, and a real Llama answers (§0.4.480)

The leg §0.4.479 named. Its graph was certified against HuggingFace and could
only run on the JVM interpreter, for one stated reason: **the artifact had
nowhere to put a weight table.** This slice gives it one, teaches the
framework-free loader to bind it, and then runs a real TinyLlama-1.1B on
PJRT-CUDA — where it produces, token for token, what HuggingFace produces.

**What landed.**

1. `ServingWeightsPointer.table: List<ServingWeightFile>` — one entry per
   `WEIGHT` slot, in the spec's own order, each naming a path, dtype, dims,
   byte length and SHA-256.
2. `ServingArtifactWriter.stageWeights` — `weights/NNNN_<slot>.bin`, **raw
   little-endian, dense row-major, no header**: the file IS the operand, in
   the form `PJRT_Client_BufferFromHostBuffer` takes.
3. `tlaloc_pjrt.buffer_from_file` + `RAW_DTYPES` — `readinto` a ctypes buffer
   and hand PJRT the address. **No Python number is ever created for a
   weight**, which is not an optimisation: 1.1e9 Python floats is tens of GB
   of PyObject, so the list-based `buffer_from_host_f32` path is not slow
   here, it is impossible.
4. `ServingArtifact.weight_buffers()` / `verify_weights()` — staged ONCE per
   artifact, held as live device buffers, bound by NAME in `run_decode`.
5. `HfLlamaServingExport` + `./gradlew :maestro:exportLlamaServingArtifact`.
6. `harness/python/run_llama_generate.py` (stdlib + `tlaloc_serve`, the
   deployment side) and `harness/python/hf_llama_greedy_oracle.py` (torch +
   transformers, the vLLM venv, ON PURPOSE — not serving-path code).

**THE ARTIFACT CARRIES STAGED BYTES, AND POINTS AT NOTHING ELSE.** §0.4.469
REJECTED copying a checkpoint into the artifact, and that judgement stands for
the *checkpoint*. What `table` carries is not the checkpoint: it is the
checkpoint **after** §0.4.479's host-side transpose, widened to the graph's
dtype, one file per operand, in call order. The rule the two branches divide
on is **an artifact points at what it did not have to change and carries what
it did** — and the reason is the loader, which has no framework under it
(§0.4.476): asking pure Python to transpose 1.1e9 floats or widen bf16 is
minutes per process start. The JVM already did that work once, at export,
where the code that knows the layout fact lives. `format: "safetensors"` with
an empty table — the loader reading an unmodified checkpoint itself — stays
legal in the schema and is a NAMED DEFERRAL, not a refusal.

**THE CERTIFICATION.** `HfLlamaServingArtifactTest`, in `./gradlew test`,
self-skipping without the checkpoint / the vLLM venv / a plugin `.so`.

| claim | oracle | floor |
|---|---|---|
| a real TinyLlama-1.1B, **all 22 layers**, greedy-decoding "The capital of France is" through an exported artifact on **PJRT-CUDA** | `hf_llama_greedy_oracle.py` — `AutoModelForCausalLM`, **fp32 on CPU**, vLLM venv (torch 2.13.0+cu130, transformers 5.17.0) | **6 of 6 generated token ids EXACTLY equal**; the text is `Paris.\n\n2.` on both sides |
| every `WEIGHT` operand of the compiled entry has a file behind it | the manifest vs the entry's own slot list | **201 = 201**, exact |
| the staged file is the length its dims imply, and the length the manifest states | `Files.size` vs `byteLength` vs `count × 4`, all 201 | exact, three ways |
| the bytes are the bytes the exporter wrote | `verify_weights()` — SHA-256 of all 4196 MiB, re-hashed in the serving process | exact |
| the manifest survives its own wire format | `fromJson(toJson()) == manifest` with a 201-entry table | `==` |
| a half-artifact is refused | a stager for specs with no weight slots; an `embedded` pointer carrying a table | by name, hermetic |

**WHY TOKEN IDS AND NOT A LOGIT TOLERANCE.** XLA-GPU's default f32
`dot_general` policy is TF32 (§0.4.477's serving floor is 1e-3 for exactly
this reason) and the oracle is fp32 on CPU, so a logit tolerance here would be
a number chosen to pass. An argmax is not. Greedy decoding agrees EXACTLY
until the two arithmetics disagree about a top-1, so the honest claim is **the
length of the prefix that agrees** — and the test asserts the full requested
budget and reports the first divergence if one appears, rather than shrinking
the budget until it is comfortable. At 6 tokens on this prompt there is no
divergence to report. **The prompt is never written down as ids**: the oracle
tokenizes it with the checkpoint's own tokenizer and Tlaloc is asked to
continue those. §0.4.477 refused to fabricate a `config.json`; a hand-typed
token id is the same refusal, smaller.

**THE NUMBERS, for scale rather than as a benchmark.** 201 staged weights,
4196 MiB; export (read bf16, widen, transpose, hash, write) **9.0 s**; first
decode step 7.0 s (weight upload + XLA compile of a 22-layer graph); median
step **1.35 s**. That median is NOT a throughput claim and the next paragraph
says why.

**WHAT IS SLOW, AND IT IS THE KNOWN ITEM.** The KV pools still round-trip to
the host every step as flat Python lists — 44 pools × 32768 floats per token,
constructed and unpacked in pure Python. That is buffer DONATION, which has
ridden `donationPairs` since H3a and has been the named "next measurable win"
three times; it is now the dominant cost of a real decode and it is
measurable for the first time. The weights, by contrast, are uploaded once.

**vLLM `LLM.generate()` — ATTEMPTED, AND IT FAILED FOR A NAMED STRUCTURAL
REASON.** With the real artifact exported and `TLALOC_SERVING_ARTIFACT` set,
`LLM(model=<the checkpoint>, max_num_seqs=1, max_model_len=64, block_size=16)`
in the vLLM venv gets past platform discovery and dies inside `EngineCore`
startup on:

```
NotImplementedError: tlaloc: attention is compiled into the serving artifact's
programs (OpKind.PAGED_ATTENTION); there is no runtime-selectable attention
backend to name
```

That is `vllm_tlaloc/platform.py`'s `get_attn_backend_cls`, refusing by name
exactly as §0.4.470 designed it to. The refusal is right about Tlaloc and
wrong about vLLM 0.29.0: the v1 engine core calls that classmethod
**unconditionally** while building the model runner, so it is not an optional
hook a backend may decline. **This is a plugin gap, and now a precisely
scoped one** — H7's entry predicted the remaining work was H3c's model
coverage, and with the model in hand the last obstacle turns out to be one
classmethod. The fix is a design question, not a typo: the plugin must return
a backend CLASS whose `get_kv_cache_shape` agrees with the artifact's compiled
pool layout (`kvPoolAxisOrder`/`kvPoolDims`, already in the manifest) and
whose forward is never called, because `TlalocWorker.execute_model` runs the
compiled program. Writing a stub that lies about its own forward is the sort
of thing this arc has refused all the way through, so it is NAMED as **H3c-4**
rather than improvised at the end of a slice. `run_llama_generate.py` is the
demo path meanwhile, and the runbook says so.

**NAMED GAPS, carried forward.**

- **H3c-4: `get_attn_backend_cls`**, above. It is the only thing between this
  repo and `vllm serve`, and the whole serving stack below it is certified.
- **The jax ORACLE engine refuses staged weights BY NAME.** It exists because
  jaxlib ships no CPU PJRT plugin `.so`; staging a checkpoint through it would
  mean a second, differently-shaped upload path in the one engine the
  deployment never runs. The consequence is stated: **a staged-weight artifact
  has no CPU lane in this loader**, and its oracle is transformers.
- **Buffer donation**, above — now the measured bottleneck.
- **bf16 weight tables.** The writer refuses a non-f32 staged slot by name.
  Halving the artifact and the upload needs the GRAPH to be a bf16 graph
  (the G1 path), not a file-format change.
- **One ladder point.** The demo exports batch 1 × context 64. H1c already
  certified that bucketing does not change the answer; every extra point is
  another full XLA compile of a 22-layer model.
- **The tokenizer is still untouched** (§0.4.478's gap, unchanged, and now
  deliberate: ids in, ids out — text is the frontend's job).
- **`:maestro`'s test JVM heap is now 2 GB** (Gradle's default is 512 MB).
  The export widens, transposes and serialises one tensor at a time; the
  embedding table alone is 262 MiB of f32 plus its byte buffer. It is 2 GB and
  not :ir's 8 GB precisely because `stageWeight` is a CALLBACK — 4196 MiB of
  weights is never resident.

### ARC STATE (§0.4.473, the close-out) — read this first

**THE PATH IS BUILT END TO END, IT EXECUTES, IT HAS RUN UNDER REAL vLLM
SINCE §0.4.477 — AND SINCE §0.4.480 IT SERVES A REAL LLAMA.**

A real TinyLlama-1.1B, all 22 layers, greedy-decodes through an exported
Tlaloc serving artifact on PJRT-CUDA in a process with **no JVM and no
framework**, and produces the **same six token ids** HuggingFace transformers
produces for the same prompt. The three slices that got there: §0.4.478 put
the checkpoint on disk and made it readable BY ROLE; §0.4.479 turned those
tensors into a decode graph certified against transformers at 1e-5; §0.4.480
gave the artifact a staged WEIGHT TABLE and the framework-free loader a way to
bind it, which is what the device lane had been waiting on.

**What is left is ONE CLASSMETHOD.** `vllm serve` / `LLM.generate()` still
fails, and after §0.4.480 attempted it the reason is no longer "no real
model": vLLM 0.29.0's v1 engine core calls `get_attn_backend_cls`
unconditionally, and `vllm_tlaloc/platform.py` refuses it by name because
attention is compiled into the artifact. That is **H3c-4**, scoped in the
§0.4.480 entry.

Nine sections closed the arc on 2026-09-21 (suite **2119 → 2290**); four
more the same day carried it past the framework (**2290 → 2296**):

| § | Slice | What it closed | Suite |
| --- | --- | --- | --- |
| 0.4.464 | ratification | the shape decision (`tpu-inference`, not `torchtpu-vllm`), the gap list, the H1–H5 slicing | 2119 |
| 0.4.465 | H1a | `OpKind.PAGED_ATTENTION` — decode form, gather-composed emission, inference-only refusals | 2119 → 2136 |
| 0.4.466 | H1b | `OpKind.KV_CACHE_WRITE` — one `stablehlo.scatter`, flat slots, `-1` padding | 2136 → 2156 |
| 0.4.467 | H1c | `DecodeGraphSpec` + `DecodeBucketPolicy` + the keyed executable cache | 2156 → 2182 |
| 0.4.468 | H2 | safetensors + a strict JSON parser in `:core`; parity vs a checkpoint torch wrote | 2182 → 2221 |
| 0.4.469 | H3a | `ServingArtifactWriter` + `tlaloc_serve.py` — the line, and no JVM past it | 2221 → 2237 |
| 0.4.470 | H3b | `vllm_tlaloc/` — the plugin, split along "does this file import vLLM" | 2237 → 2241 |
| 0.4.471 | H4 | the KPTX paged-attention kernel and OpKind-keyed claiming | 2241 → 2259 |
| 0.4.472 | H5 | `KvQuantPool` + `OpKind.DEQUANTIZE_KV` + `kvQuant` in the manifest; SGLang priced | 2259 → 2290 |
| 0.4.473 | close-out | this sweep + [SERVING_RUNBOOK.md](SERVING_RUNBOOK.md) — docs only | 2290 |
| 0.4.474 | H6 rail | `OracleVenvIntegrityTest` + [SERVING_RUNBOOK.md §0.1](SERVING_RUNBOOK.md) — the oracle venv is frozen, and now says so out loud | 2290 → 2292 |
| 0.4.475 | H6a | `harness/python/tlaloc_pjrt.py` — the PJRT C API bound from Python with **ctypes alone**, mirroring the FFM runtime; jax blocked by an import guard while it runs | 2292 → 2294 |
| 0.4.476 | H6b | `tlaloc_serve.py` rewired onto that binding — the **whole** serving path runs with no jax, jaxlib, torch or numpy, and the distribution's dependency list is empty | 2294 → 2295 |
| 0.4.477 | H7 | `~/.local/venvs/vllm` (vLLM 0.29.0, its own venv), the live lane run, `platform.py`+`worker.py` CERTIFIED, one real bug found and fixed, `exportServingArtifact` | 2295 → 2296 |
| 0.4.478 | H3c-1 | a REAL TinyLlama-1.1B checkpoint on disk, and `HfLlamaConfig` + `HfLlamaNames` + `HfLlamaCheckpoint` — HF names to roles, with the transposed-`[out, in]` layout VERIFIED against it and the bytes checked against torch | 2296 → 2320 |
| 0.4.479 | H3c-2 | `HfLlamaDecodeGraph` + `HfLlamaStagedWeights` + `DecodeGraphSpec.weightSlots` — the real checkpoint becomes a decode graph, certified against HF transformers at **1e-5 relative with argmax and top-5 exact**; RSQRT/SILU interpreter arms found and closed on the way | 2320 → 2330 |
| 0.4.480 | H3c-3 | `ServingWeightsPointer.table` + `buffer_from_file` + `HfLlamaServingExport` — the artifact carries a staged weight table and a REAL 22-layer TinyLlama serves on PJRT-CUDA, **6/6 generated token ids equal to HuggingFace** | 2330 → 2333 |

**Before touching anything in the next section, read
[SERVING_RUNBOOK.md §0.1](SERVING_RUNBOOK.md).** `~/.local/venvs/iree` is
**frozen oracle infrastructure** — the measurement apparatus for every
cross-language claim in the table below, JVM-side PJRT lanes included
(`PjrtBinaries` resolves `xla_cuda_plugin.so` out of its site-packages).
Every "certify it like this" command in the UNCERTIFIED list creates its
own venv for that reason, and §0.4.474 put a canary in `./gradlew test`
that turns a mutation of the oracle from a silently-moved number into a
red line with the separate-venv recipe attached.

#### CERTIFIED — the claim, its oracle, and its floor

| claim | oracle | floor |
|---|---|---|
| `PAGED_ATTENTION` computes paged attention | a dense contiguous walk over separately-gathered pages + a hand-exact uniform-softmax case with poisoned dead slots | exact structure; elementwise |
| …and XLA agrees | `PjrtPagedAttentionSmokeTest`, real XLA on the GB10 | **6e-8** (fixture too small for a tensor-core path) |
| `KV_CACHE_WRITE` writes where the allocator said | a position-encoded pool (every element equals its own flat index) + an independent pool-major walk | exact |
| …and XLA drops rather than clamps an OOB scatter | `PjrtKvCacheWriteSmokeTest` | **exact** (a write does no arithmetic) |
| the two ops agree what a flat slot is | the round trip: write, then read at `seqLens = 1`, where softmax is exactly 1.0 | bit-for-bit |
| bucketing does not change the answer | the padding invariant: batch-3-in-bucket-4 vs batch-3-in-bucket-3, across batch AND context buckets | **`==`**, reference interpreter |
| the safetensors reader reads the producer's bytes | `write_llama_safetensors.py` records raw bit patterns; f32 via `toRawBits`, bf16 via `HostBf16Storage` | **exact, no tolerance** |
| an f32 graph on a bf16 checkpoint computes the checkpoint's real numbers | `LlamaSafetensorsParityTest` vs `run_pytorch_llama.py` (§0.4.289's op-for-op mirror) through the IREE-CPU lane | **1e-3 relative** on the loss |
| the exported directory describes itself truthfully | every body hashes to its filename; per-entry `ProgramManifest` parses with its own parser and agrees; two exports byte-identical | exact |
| a Python process with no JVM runs the artifact | `ServingArtifactExportRunTest`, export-then-subprocess | **1e-5** PJRT CPU (jax ORACLE engine) · **1e-3** PJRT CUDA (**ctypes engine**, §0.4.476) |
| …and the poisoned pools / scratch page are untouched | same test, both lanes | **`==`** (a copied slot is not a computed one) |
| the plugin's page arithmetic and marshalling are right | `VllmPluginContractTest` lane 2 — 31 stdlib-only `unittest` cases | exact |
| the runner threads a KV cache across steps | lane 3 — **two** decode steps of three sequences on both PJRT clients | 1e-5 / 1e-3 on floats; **exact** on every page id, slot, position and bucket |
| Tlaloc's own PTX can replace the lowering inside an XLA executable | `KptxPagedAttentionKernelTest` vs a Double paged walk | **1.2e-7** kernel · 2.44e-4 emission (= 2^-12, TF32's mantissa) · and the property "the kernel is at least as close to the oracle as the emission it replaces" |
| the KV-quant bound holds and is tight | every element under `scale/2`, some element over 0.9 of it; a hand-derived power-of-two-scale vector checked on paper | **derived**, not tuned |
| every new op refuses in both AD transforms and the renderer | `PagedAttentionTest` / `KvCacheWriteTest` / `DequantizeKvTest`, three refusal cases each | by name |
| the oracle venv every row above is measured against is intact | `OracleVenvIntegrityTest` — `check_oracle_venv.py` in that interpreter, pins asserted on the JVM side; and the policy itself certified against the §0.4.470 mutated inventory | exact versions; `torch.version.cuda is None` |
| the ctypes binding lays the PJRT structs down exactly as the certified FFM binding does | `PjrtCtypesBindingTest.layoutsMirrorTheFfmBinding` vs `PjrtFfm`'s own layouts, offset table, compile-options bytes and marshalled create_options — **no device needed** | exact |
| a Python process with **no framework at all** compiles and executes through PJRT | `PjrtCtypesBindingTest` vs `DxirInterpreter`, four graphs on real XLA-CUDA | **1e-5** elementwise · **1e-3** matmul (TF32) · **exact** i32 · **bit-for-bit** bf16 |
| …and it imported no jax, jaxlib, torch or numpy while doing it | a `sys.meta_path` guard that raises, in the venv where jax *is* installed — and is itself made to fire before the run reports | by name |
| the WHOLE SERVING PATH — manifest, bucket, compile, staging, execute, readback — runs with no framework imported | `ServingArtifactExportRunTest.theFullServingPathRunsWithNoFrameworkImported`, asserted on the same subprocess that produced the CUDA lane's numbers: nothing loaded, nothing even attempted, and the guard shown to fire | by name, three facts |
| and the deployment needs only a plugin `.so` | `vllm-tlaloc`'s `dependencies` is `[]`, pinned by a test; `tlaloc_serve`'s import pulls in no framework, pinned by another; the plugin is found by FILE lookup in three places, never by import | exact |
| the loader's shape arithmetic survives having no ndarray | `LoaderIsStandardLibraryOnly` — flatten/unflatten inverses, `numel`, per-dtype staging, dtype refusal by name, engine defaulting | exact |
| **vLLM's own platform discovery lands on `TlalocPlatform`** | `VllmLivePluginTest`, real vLLM **0.29.0** in `~/.local/venvs/vllm` (its own venv; the oracle untouched) | exact class path; vLLM logs the activation |
| **`check_and_update_config` imposes and refuses against REAL `vllm.config` objects** | same test — `worker_cls` set, `block_size`/`num_gpu_blocks_override` taken from the artifact, and all four refusals (block size, max-model-len, max-num-seqs, world size) fired on a `vllm.config.CacheConfig`/`ParallelConfig` | by name, four messages |
| **`TlalocWorker` runs the v1 worker API and returns vLLM's real `ModelRunnerOutput`** | same test — `load_model`, `determine_available_memory` (= the compiled pool, checked arithmetically JVM-side), `get_kv_cache_spec`, `initialize_from_config` accept **and** refusal, two `execute_model` steps, the chunked-prefill refusal | exact |
| **a REAL TinyLlama-1.1B (22 layers) serves from an exported artifact on PJRT-CUDA** | `hf_llama_greedy_oracle.py` — transformers `AutoModelForCausalLM`, fp32 on CPU, vLLM venv | **6/6 generated token ids `==`**; `Paris.\n\n2.` on both sides |
| …and the 4196 MiB of staged weights in that artifact are the bytes the exporter wrote | `verify_weights()`, SHA-256 re-hashed in the serving process | exact |
| the same artifact called through vLLM and called directly gives the same numbers | the vLLM venv's worker lane vs the oracle venv's `run_vllm_tlaloc_check.py` runner lane — two venvs, two torches, jax in only one | **`==`, bit-for-bit** on every logit, plus sampled tokens, buckets and compile count |

Four sensitivity checks were run before the oracles were trusted, each
mutated then reverted: the bf16 decode byte-swapped (H2), `PADDING_SEQ_LEN`
flipped to 0 and the block table desynced by a page (H3a),
`append_token` advancing `length` before computing the slot (H3b), and a
bare `import numpy` added at `tlaloc_serve`'s module scope (H6b — which
failed BOTH the guard test and the CUDA numeric lane, since the guard goes
up before the loader is imported). Every one of them failed the lane it was
supposed to fail.

#### WRITTEN BUT UNCERTIFIED — and exactly how to certify it

1. ~~**The live vLLM path**~~ — **CERTIFIED (H7, §0.4.477)**, with one
   named remainder. `~/.local/venvs/vllm` exists (vLLM 0.29.0, 197
   distributions, no jax), the oracle venv was not touched, and
   `VllmLivePluginTest` runs discovery + the config hook + the whole v1
   worker API live, matching the runner lane **bit-for-bit across the two
   venvs**. The recipe that worked is [SERVING_RUNBOOK.md §4](SERVING_RUNBOOK.md);
   note that the `jax[cuda12]` in the command block this entry used to
   carry is **not installed and must not be** — H6 removed the need, and
   installing it beside vLLM's CUDA-13 wheels is the collision the rail
   exists to prevent.

   **THE REMAINDER: `LLM.generate()` / `vllm serve` itself — and since
   §0.4.480 it is ONE CLASSMETHOD, not a model gap.** The artifact for a real
   TinyLlama-1.1B exists and serves (see the §0.4.480 entry); pointing vLLM at
   it reaches `EngineCore` startup and dies on
   `vllm_tlaloc/platform.py`'s `get_attn_backend_cls` refusal, which vLLM
   0.29.0 calls unconditionally. That is **H3c-4**. The historical framing
   follows. (§0.4.479
   narrowed it further: the graph exists and is certified; the ARTIFACT that
   carries its staged weights does not — H3c-3.) Both need a
   HuggingFace `config.json` and tokenizer for a real model, and the only
   exportable artifact is the reference LCG toy. That is **H3c**, not a
   plugin gap; see the H7 entry. **§0.4.478 landed H3c-1**: the config and
   the weights are now readable by role from a real TinyLlama-1.1B
   checkpoint (see that entry's oracles). H3c-2 — the decode graph and the
   artifact built FROM those tensors — is what this command still waits on. Certify it by exporting a real Llama and
   running:

   ```bash
   export TLALOC_SERVING_ARTIFACT=<dir written by ServingArtifactWriter>
   export TLALOC_PJRT_PLUGIN_PATH=<a plugin .so>
   vllm serve <the model whose weights that artifact was exported from> \
       --max-num-seqs 4 --max-model-len 4 --block-size 2
   ```
2. **The SGLang runner** — a design record only (§5 H5(2)). `pip install
   sglang` has the same venv problem. The **checkable prediction** is that
   the loader, manifest reader, bucket selection and PJRT execution path are
   reused wholesale and only the adapter class + its `ForwardBatch → slot`
   translation are new. Certify it by writing that adapter against a real
   installed SGLang and re-running the H3b lane-3 shape against it.
3. ~~**The `./gradlew exportServingArtifact` task.**~~ **DONE (§0.4.477).**
   `./gradlew :maestro:exportServingArtifact -PoutDir=/abs/path` — a `JavaExec`
   over the **jvmMain** runtime classpath. H7 needed it: a live-serving
   certification needs a directory a shell can point `$TLALOC_SERVING_ARTIFACT`
   at, and "be inside the export test" is not that.

#### AWAITS THE CLOUD TPU VM

The artifact does not mention CUDA: the bodies are StableHLO+SDY, the
manifest is JSON, and `ServingArtifact.load(..., platform=…)` takes the
platform as a parameter. On a TPU VM, §3 of the runbook is the same commands
with `platform="tpu"` — once G2b turns `PjrtTpuSmokeTest`'s self-skips into
passes ([TPU_BRINGUP.md](TPU_BRINGUP.md); the ordered queue is
[TPU_READINESS_AUDIT.md](TPU_READINESS_AUDIT.md) §5).

What a TPU session owes this arc, specifically:

- **re-run the export-then-run lane on `tpu` and record ITS floor.** The two
  floors above are the PJRT CPU client's and XLA-GPU's TF32 dot policy; a
  TPU's default `dot_general` precision is neither, so neither number
  transfers. This is the single most valuable thing a TPU VM adds here.
- the H3b lane-3 runner steps on TPU (the same two-step KV-threading check),
- the H4 claiming lane **does not apply** — KPTX is PTX, and the decline
  path is the op itself, which is exactly why claiming was built that way.

#### The arc's invariants, verified by grep at the close

Three new `OpKind`s entered the IR in Phase H and no others:
`PAGED_ATTENTION`, `KV_CACHE_WRITE`, `DEQUANTIZE_KV`.

- **All three are in `INFERENCE_ONLY_OP_KINDS`**, each with a rationale in
  its `OpKind` doc comment and in `inferenceOnlyKindRefusal`.
- **All three refuse in BOTH transforms** — `DxirReverseTransform` and
  `DxirForwardTransform` each consult `INFERENCE_ONLY_OP_KINDS` and `error`
  with the named message, and each has a test asserting it.
- **All three refuse BY NAME in `KotlinSourceRenderer`** (arms at lines
  598 / 610 / 625), with a test each. The north star holds: every new op
  either renders or refuses there by name.
- **All three are COMPLETE where they run** — interpreter arm, StableHLO
  emission, cost-model arm. That is the inference-only kind's defining
  difference from a demoted kind, and it is what makes serving executable.
- **No silent gaps**: the refusal messages name the kind, the rationale, and
  the differentiable spelling (FlashAttention/GQA for paged attention,
  SCATTER/SCATTER_ADD for the cache write, MUL-by-scale for dequantize).
- **No gradient math was written anywhere in this arc.** One AD engine.

#### What remains, in the order a next session should take it

0. **H3c-4 — `get_attn_backend_cls`, the last thing between this repo and
   `vllm serve`.** §0.4.480 exported a real Llama artifact, pointed vLLM
   0.29.0 at it, and got past platform discovery into `EngineCore` startup
   before the plugin's deliberate refusal stopped it. The plugin must hand
   vLLM a backend CLASS whose `get_kv_cache_shape` agrees with the manifest's
   `kvPoolAxisOrder`/`kvPoolDims` and whose forward is never reached. See the
   §0.4.480 entry for why that stub is a design question rather than a typo.

1. ~~**H3c-3 — the staged-weight ARTIFACT, and a real Llama through the plugin.**~~
   **DONE (§0.4.480).** What it leaves behind: buffer donation is now the
   measured bottleneck (item 3), and the jax oracle engine refuses staged
   weights by name, so a staged-weight artifact has no CPU lane in the loader.
   The superseded text follows.

   **H3c-3 — the staged-weight ARTIFACT, and a real Llama through the plugin.**
   The largest open item, now precisely scoped by §0.4.479: the decode graph
   and the parity are done; `ServingManifest`/`ServingArtifactWriter` must
   learn a weight table and `tlaloc_serve.py` must stage it, after which the
   PJRT-CUDA lane and `vllm serve` both follow. It was, before that slice, and the one that turns "the path executes" into "the path
   serves". It is a NAME-MAPPING problem plus making weights graph
   parameters instead of body constants. **§0.4.477 raised its value:** the
   plugin, the platform and the worker are now certified against real vLLM,
   so H3c is the *only* thing between this repo and an actual `vllm serve`
   — no longer one of two unknowns.
2. ~~**H6b — re-point `tlaloc_serve.py` at the ctypes binding.**~~ **DONE
   (§0.4.476).** What it leaves behind: the CPU semantics lane still rides the
   jax ORACLE engine, because jaxlib ships no CPU PJRT plugin `.so`. That is a
   fact about jaxlib, not a tail of this work — but it means the phrase "no
   framework" is precise only about the accelerator lanes, and the doc says so
   in both places it appears.
3. **Buffer donation.** `donationPairs` has ridden the manifest since H3a
   and is still unwired into `CompileOptions`. Named the "next measurable
   win" twice; it still is.
4. **The ragged / chunked-prefill `PAGED_ATTENTION` form.** H1a's deferral
   since the first slice, refused by name in three places, and the one
   IR-level item both vLLM's chunked prefill and SGLang's radix path need.
5. **The H4 performance tier** — warp specialization, shared-memory staging
   of the page window. The floor to beat is the measured **465 µs** against
   the lowering's 310 µs. Until it lands, nothing should register this
   kernel in a deployment, which is why the default inference registry is
   empty.
6. **Narrow DTypes (`I8`, and the fp8 tour)** — a bf16-sized piece of work,
   and the difference between KV-quant's contract and its bytes.
7. **`precision_config = HIGHEST`** for dots that want it, and the top-1
   consequence of the TF32 gap on a real vocabulary — which only becomes
   measurable once (1) lands.

Smaller named tails, carried forward so nothing is lost: StableHLO bytecode
bodies; the embeddings entry point (speculative decoding / multimodal); warm-up
policy (which buckets a deployment compiles at startup); multi-device serving
across a mesh; vLLM's own sampler and logprobs; prefix caching / RadixAttention
(theirs, by design); mmap / zero-copy `MemorySegment`-backed weight storage;
`HostI64Storage`; a scale-generic KPTX kernel; the fused `KV_CACHE_WRITE` +
`PAGED_ATTENTION` and `DEQUANTIZE_KV` + `PAGED_ATTENTION` claims; the quantized
decode-graph signature; the Python side reading `kvQuant` to size its pools;
and the training-side vectorized gather/scatter axis+list forms, which this
arc deliberately did not touch.
