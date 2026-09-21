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
