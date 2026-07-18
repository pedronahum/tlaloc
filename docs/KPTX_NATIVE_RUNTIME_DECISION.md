# KPTX v3.4 — Glow-style native runtime: go/no-go

**Decision: NO-GO today; conditional GO gated on kernel coverage.**
(§0.4.348 closes the KPTX arc with this analysis; the gate below is
measurable, not aspirational. Recommendation recorded for Pedro's
ratification — reopening is one sentence in a future session.)

## The question

The PyTorch-Glow study (docs/KPTX_PLAN.md) framed the endgame option:
a native runtime — instruction IR + static memory arena + our own
scheduler — around the KPTX kernel library, replacing PJRT-XLA as the
execution engine for coarsened programs. Task v3.4 deferred the
decision until the arc produced evidence. It has.

## Evidence

| # | Finding | Source |
|---|---------|--------|
| 1 | Typed-FFI custom_call floor cost ≈ **135 µs/call** (JVM host round-trip per execution + XLA fusion loss at rms_norm granularity) | §0.4.337, same-session lanes |
| 2 | Custom_call **jitter 0–1.2 ms/call**, run-dependent; decompose lanes stable in the same session | §0.4.337 |
| 3 | Direct `cuLaunchKernel` measures **14–18 µs/launch** on the GB10 (100-launch loop incl. amortized sync) — ~10× below the custom_call floor | §0.4.346 |
| 4 | The XLA lane is already competitive: 2.51 ms/step, within 7% of JAX-GPU; XLA codegens **everything unclaimed** for free | §0.4.308 |
| 5 | Kernel library coverage today: the rms_norm family (fwd + bwd pair). The Llama coarse-op set needs GQA/attention, TransformerMLP/SwiGLU, RoPE, CrossEntropy; matmul stays cuBLASLt by standing decision | §0.4.344, KPTX plan |

## Analysis

A native runtime buys exactly what rows 1–3 quantify: direct launches
(~10× lower dispatch floor), no host-callback jitter, a static memory
plan, no XLA compile times. It costs what row 4 warns: **we forfeit XLA
codegen for every op we have not hand-written** — scheduling, memory
planning, transfers, multi-GPU, and cuBLASLt integration all become
ours. With today's coverage (row 5: one op family), a native runtime
would execute rms_norm beautifully and everything else at
decompose-to-nothing — there is no engine to run the unclaimed 77% of
the recognizer's absorbed ops, let alone the residue.

Meanwhile the 135 µs tax is amortizable *inside* XLA by the route v2
just made cheap: **coarser kernels**. An attention or MLP kernel raises
compute-per-call and cuts calls-per-step; the round-trip stays constant
while its denominator grows. That is the correct next investment, and
it is also exactly the work that closes the coverage gap the native
runtime needs anyway. The two roads share a lane until coverage is
done.

## The gate (measurable)

Reopen as GO when **both** hold:

1. **Coverage** — the DSL kernel library claims the full LlamaDecoder
   coarse-op set (GQA, TransformerMLP/SwiGLU, RmsNorm, RoPE,
   CrossEntropy) through the L3 registry, forward and backward
   (backward claiming design is the standing prerequisite —
   `handleCoarsenedAdjoint` inlines gradient_body today).
2. **Arithmetic** — a measured all-claimed XLA-lane step shows host
   round-trip overhead ≥ ~20% of step time (count calls × 135 µs
   against the step floor). Below that, the native runtime's win
   doesn't pay for owning scheduling + memory + transfers.

Early trigger regardless of the gate: if tcgen05/WGMMA-era kernels hit
a hard constraint in XLA's custom-call path (stream/ordering/TMA
descriptor ownership), the decision reopens immediately.

## What was NOT built, on purpose

No instruction-IR runtime skeleton, no arena allocator, no scheduler
spike. A speculative half-runtime would rot against a gate that may
never open; the kernel library, the DSL, and the launch registry are
deliberately runtime-agnostic (framework owns buffers/streams — the
pyptx lesson), so a future GO starts from a clean seam, not a rewrite.
