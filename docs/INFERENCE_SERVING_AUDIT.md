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

**Still open in H1**: gap-list item 2 (KV-cache write scatter + page-table
gather forms) and item 3 (decode-graph shape contract + bucketing).
