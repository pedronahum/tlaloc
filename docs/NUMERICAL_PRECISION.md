# Numerical precision

What each backend computes for f32, f64 and bf16 programs, where a device result
differs from the interpreter's and by how much, and the tolerances the tests use.

## The reference

- **The F32 interpreter** (`DxirInterpreter`) rounds every op's result to f32. A value
  the program casts to bf16 is rounded to bf16 there.
- **The F64 interpreter** (`DxirInterpreterF64`) computes in float64. The op sweep
  (below) checks each op's F32 result against it on f32-rounded inputs.
- **Device results** are compared against the F32 interpreter. They differ for the
  reasons in the next three sections.

## Matrix products

| Spelling | GB10 (XLA CUDA) | TPU |
|---|---|---|
| Plain `MATMUL`, `CONV2D`, linear-algebra dots: no precision attribute | TF32 (10-bit mantissa) | one bf16 pass |
| The emitter's explicit f32 dot algorithm (`F32_DOT_ALGORITHM`: attention, MoE and paged-attention paths) | f32, one pass, f32 accumulation | rewritten to `precision = HIGHEST` (below) |

- **`portableF32Dots`** (on by default in TPU and Metal sessions) replaces the explicit
  algorithm with `precision = [P, P]`. P is `HIGHEST` unless `F32DotPrecision` says
  otherwise. It does not change a plain `MATMUL`, so on the GB10 those stay TF32 with
  it on.
- **A TPU** runs an f32 dot as bf16 passes: `DEFAULT` one, `HIGH` three, `HIGHEST`
  six.
- **Tolerance:** a TF32 product has a relative error near 1e-3. Tests that compare
  matrix-product graphs on a device use 1e-2 (`TestBackend.defaultDotRelTolerance`,
  3e-2 on a TPU; `Tolerances.TF32_DOT`).

## Narrowing and excess precision

- **XLA CUDA folds convert pairs.** By default it folds an f32 → bf16 → f32 pair, so
  the device keeps f32 where the program rounded to bf16 and the interpreter did too.
- **Turning folding off:** `CompileOptions.STRICT_PRECISION` sets
  `xla_allow_excess_precision = false`, which keeps every narrowing; pass it as
  `PjrtSession(compileOptions = CompileOptions.STRICT_PRECISION)`. Every session
  compiles with `CompileOptions.DEFAULT` otherwise, on every target; the tests' TPU
  sessions use `STRICT_PRECISION`. Metal and Cerebras never fold.
- **In tests:** `TestBackend.foldsConvertPairs` says which backend folds. The op
  sweep's `cast_bf16` sample does not claim the device surface for this reason.

## Sums, fusion and autotuning

- **Sums and fusion:** XLA reassociates sums and fuses elementwise chains. Against the
  interpreter, elementwise ops and reductions on the GB10 differ by up to about 4e-5
  relative. Tests never pin a device comparison tighter than 1e-4; the sweep's default
  is 2e-4.
- **Autotuning:** XLA picks GEMM and fusion kernels per compile. Two loads of a serving
  model differed by up to 0.017 in a logit; with autotuning fixed, the logits are
  bit-identical across loads and across steps of interleaved sequences.

## Softmax and cross entropy

- **`LOG_SOFTMAX`** is computed as `(x − M) − log1p(S)`, with S the sum without one
  maximum. It keeps the largest logit's log probability where `log(softmax)` rounds
  it to 0: past a margin of 16.6 in f32, 36.7 in f64 (arXiv 2610.31642).
- **The dominant entry:** the gradients of `SOFTMAX` and `LOG_SOFTMAX` write the entry
  with probability above 1/2 as minus the sum of the others, so they sum to zero.
- **`SOFT_CROSS_ENTROPY`'s student gradient** is built from the logit difference with
  `EXPM1` and `LOG1P`. A student 1e-6 from its teacher gets the gradient within 1e-5
  of its size in f32; the difference of softmaxes is 0.4 % off there.

## bf16

- **Mixed-precision training** (`MIXED_BF16`): bf16 compute, f32 master weights, no
  loss scaling.
- **bf16 serving weights:** each projection rounds its f32 input to bf16 and
  accumulates in f32. With bf16 weights, Qwen3-0.6B's logits come within 3.0e-3 of the
  largest logit of HuggingFace's; the f32-weight tolerance of 2e-3 fails for them.

## Tolerances in tests

`io.tlaloc.testing.Tolerances` (`:test-support`) holds the defaults per surface and
dtype, and per-op overrides, each with its reason.
- **Validation:** every value is positive (checked at load).
- **Strict mode:** `TLALOC_STRICT_TOLERANCES=1` ignores the overrides.
- **Failure messages:** `Tolerances.assertClose` names the worst entry, and the atol
  (over `|expected| ≤ 1`) and rtol (over the rest) that would have passed, times 1.2.

| Comparison | Default | Overrides |
|---|---|---|
| F32 op against F64 on f32-rounded inputs | 2e-5 | TRIGAMMA 1e-4, POLYGAMMA 2e-4, EIGH_V 1e-3 |
| Device (F32) against the F32 interpreter | 2e-4 | 1e-2 with matrix products (TF32); EIGH_V 1e-3 |
| Device (F64) against the F64 interpreter | 1e-10 | |
| Gradient against central differences (F64) | 1e-6 | per sample (`OpSample.gradTolerance`) |

Errors are scaled by `max(1, max |expected|)`.

## The op sweep

`OpCatalogue` (`:test-support`) has an entry for every `OpKind`: samples, and the
surfaces the op is not expected to work on, each with its reason. A new `OpKind`
without an entry fails `OpCatalogueTest`.

The surfaces are run by:
- `OpSweepIrTest` (`:ir`): the F32 interpreter, F64 against F32, reverse mode against
  central differences, forward mode against reverse, vmap against a loop, and
  rendering to Kotlin;
- `OpSweepEmitTest` (`:stablehlo`): StableHLO emission at F32 and F64;
- `OpSweepDeviceTest` (`:runtime-pjrt`): the device against the interpreter.

Excluded surfaces run too. An exclusion that no longer holds fails the sweep, so the
exclusion lists stay true. Each module writes every result to `build/op-sweep/`.
