# Inference serving — vLLM / SGLang integration audit (2026-09-21)

**Status: PHASE H RATIFIED (Pedro, 2026-09-21) — GO on the H1–H5
slicing in §4.** The arc runs on CUDA (vLLM runs on the GB10); the same
artifacts serve on TPU the day G2b lands. Running record in §5.

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

## 2. The honest gap list

1. **Paged attention** — vLLM's core primitive: attention over a
   block-table-indexed KV page pool. We have FlashAttention/GQA
   recognition and a KPTX attention kernel, but **no paged-KV form**.
   Inference-only ⇒ **no VJP required** (the single biggest scope
   relief): a coarse op + gather-composed reference emission, with
   vendor/KPTX kernel claiming as the follow-on. Biggest single item.
2. **KV-cache ops** — cache-write scatter + page-table gather.
   `SCATTER`/`GATHER` exist in narrow forms; the already-recorded
   "gather/scatter axis+list forms" tail is exactly what is missing.
3. **Decode-graph shapes + bucketing** — single-token decode with
   cache-in/cache-out signatures, compiled per (batch, seq-bucket) and
   cached in `PjrtSession`. Our static-shape story **suffices**: vLLM's
   TPU backend buckets identically (bounded dynamism is their roadmap
   item too).
4. **Weight ingestion** — HF safetensors → `DTensor` loader, then
   decode-graph parity vs the HF reference (`LlamaDecoderPrimal` and the
   PyTorch-agreement harness are the substrate).
5. **Dtypes** — bf16 landed in Phase G (§0.4.455–458, native PJRT BF16
   buffers included); int8/fp8 KV-quant stays a named deferral with a
   manifest slot already reserved.
6. **The plugin itself** — conform to vLLM's platform-plugin API
   (worker + model-runner classes), sampling host-side in v1.

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
