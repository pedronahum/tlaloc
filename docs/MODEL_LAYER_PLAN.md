# Phase F — the model/optimizer layer (`:nn`)

**Status: COMPLETE (§0.4.434–444, all landed 2026-09-21) — the ratified
scope, as amended (decision 3: the COMPILER route), is fully delivered.**
Ratified by Pedro 2026-09-21 (§0.4.434); amended the same day before any
substrate code landed (§0.4.436); F0–F8 landed in eleven §, one day. The
end-to-end close: an MLP and a conv net TRAIN through capture →
`DxirReverseTransform` → interpreter/GPU with the F3 optimizers, the
captured gradient graph compiles and runs on the GB10 through
`PjrtSession` matching the interpreter at 3.8e-5, and 50 Adam steps track
PyTorch at ~1e-7 relative from a shared init. Deferrals are consolidated
at the end of §4's F8 entry. This
document is the Phase F authority, the
[SPARSE_PARITY_AUDIT.md](SPARSE_PARITY_AUDIT.md) /
[CUSTOM_DERIVATIVES_DESIGN.md](CUSTOM_DERIVATIVES_DESIGN.md) pattern:
scope, design decisions with rejected alternatives, §-sized slicing, and
the running record as slices land. Companion to
[DIFFKT_PARITY_PLAN.md](DIFFKT_PARITY_PLAN.md)'s Phase F entry, which
pointed here the day the book of work reopened for its final phase.

## 1. What DiffKT ships (the Phase 0 audit's closed inventory)

`org.diffkt.model`: `Layer`/`Model`/`Sequential`/`TrainableComponent`
abstractions; layers Dense, Conv2d (+SamePadding), MaxPool2d/AvgPool2d,
BatchNorm (+Training variant), Embedding + EmbeddingBag, Dropout, GRU,
Flatten, ReluLayer, AffineTransform; FanMode initializers; optimizers
SGD, Adam, RMSprop, Momentum, FixedLearningRate. Plus the item parked
here since Phase E ratification: row-sparse embedding gradients.

## 2. Ratified design decisions

1. **A new `:nn` module** (`io.tlaloc.nn`), depending on `:core` +
   `:autograd`. REJECTED: growing `:autograd` (the AD engine stays a
   library the NN layer consumes, not a home for layers); growing
   `:core` (keeps the tensor substrate lean).
2. **Immutable, functional components — DiffKT's own convention.**
   A layer is an immutable value holding its `DTensor` parameters;
   `forward(x)` is pure; a training step returns a NEW model (and new
   optimizer state). REJECTED: PyTorch-style mutable modules — they
   fight the entire stateless design (threefry keys, pure grad
   transforms, the interpreter's value semantics).
3. **The AD route is the COMPILER stack — trace to DXIR, transform,
   execute.** *(AMENDED by Pedro, 2026-09-21, superseding the original
   runtime-tape decision before any code landed on it: "what's the
   whole point of the Kotlin compiler creating the autodiff code?" —
   exactly.)* The model's forward traces ONCE through the `:autograd`
   `Tracer` into a real `DxirFunction` via the existing
   `Tape.toDxirFunction(name, paramIds, returnIds)` — one param per
   (input + parameter tensor), ARBITRARY arity (the 1–4 ceiling is a
   property of the `grad {}` lambda-intrinsic surface only, never of
   the IR). **`DxirReverseTransform` — the compiler's AD — produces the
   gradient function**, and execution is a backend choice: the
   `DxirInterpreter` on host, or coarsening → recognition →
   StableHLO → `PjrtSession` on GPU with cached executables (the
   §0.4.307 amortization) — the §0.4.292 LlamaDecoder
   backward-vs-PyTorch GPU certification is precisely this shape at 13
   parameters. Model graphs are thereby visible to the
   recognizer/coarsener/KPTX pipeline like any other Tlaloc graph.
   Tracer op coverage is extended by TRACE spellings that record
   `TapeEntry`s with the right OpKind/attrs (forward value via the
   `:core` host twins); gradients always come from the registry rules
   through the transform — no gradient math is written in `:nn`.
   REJECTED: the runtime value-tape as the model route (host-only,
   interpreted, sidelines the compiler stack that IS the product; it
   remains a debugging fallback); packing parameters through
   fixed-arity `grad {}` (arity ceiling, unnatural flattening); a
   parallel NN-specific autograd (would fork the certified rule set).
4. **Parameter trees walk functionally.** `Trainable` components expose
   their parameter tensors (with stable keys) and rebuild themselves
   from updated tensors; optimizers are pure functions
   `(params, grads, state) → (params', state')`. Adam/RMSprop/Momentum
   state lives in the optimizer value, keyed like the parameters.
5. **Randomness is threefry-keyed and explicit.** Dropout takes a
   `RandomKey` per call (or a key the layer splits per step);
   initializers take a key and are bit-deterministic. No global RNG
   state anywhere. REJECTED: a seeded-generator convenience — it would
   reintroduce exactly the mutable stream D1 removed.
6. **BatchNorm training stats are functional**: the training-mode
   forward returns (output, updated running stats); inference mode
   reads them. Same convention as optimizer state.
7. **Row-sparse embedding gradients: dense v1.** The tape's embedding
   gradient comes back dense (EMBEDDING_GRAD's scatter into the full
   table); the row-sparse (CSR, Phase-E substrate) form is a recorded
   design in this doc's §4, taken only if a slice lands it honestly —
   otherwise it stays a named deferral, not a blocker.
8. **Oracles.** Layer forwards/gradients pin against hand-computed
   values and the interpreter (the same rules certify both paths);
   optimizers pin against hand-stepped references; the end-to-end
   training story pins against a PyTorch reference run via the
   `harness/python` precedent (§0.4.289's cross-language flow) at
   loss-curve tolerance. Dropout/initializers pin bit-exact against the
   `:core` threefry kernels.
9. **Out of scope for Phase F v1** (recorded, not silent): GPU-resident
   training loops (the tape is a host path; the GPU lane remains the
   emitted-graph story), checkpoint serialization beyond tensor
   round-trip helpers, data-loading utilities, and DiffKT's `Model`
   convenience subclasses beyond what `Sequential` + `Trainable` cover.

## 3. The §-sized slicing (the finalization workflow's task list)

| Slice | Content | Size |
|---|---|---|
| F0 | Deep audit: walk DiffKT `model/` fresh-clone; per-layer op-requirements map vs the tape's current coverage; append the closed inventory + gap table to this doc | 1 § |
| F1 | `:nn` module + `Trainable`/`Sequential` substrate + the trace→DXIR→`DxirReverseTransform` `valueAndGradients(model, loss)` contract (interpreter execution v1; the compiled-GPU step is F8's integration) | 1 § |
| F2 | Dense, Flatten, ReluLayer, AffineTransform + FanMode initializers (threefry, bit-deterministic) | 1 § |
| F3 | Optimizers: FixedLearningRate, SGD, Momentum, RMSprop, Adam — pure `(params, grads, state)` functions, hand-stepped oracles | 1 § |
| F4 | Conv stack: Conv2d (+SamePadding), MaxPool2d, AvgPool2d — Tracer TRACE spellings recording the ops into the captured graph | 1 § |
| F5 | BatchNorm (functional stats) + Dropout (threefry-keyed) | 1 § |
| F6 | Embedding + EmbeddingBag; row-sparse gradient disposition per §2.7 | 1 § |
| F7 | GRU — library-level unroll over time steps (the tape handles unrolled graphs natively) | 1 § |
| F8 | End-to-end training certification: MLP + small conv net to convergence vs PyTorch reference; the compiled-GPU training step (PjrtSession over the captured gradient graph); Phase F close-out sweep | 1–2 § |

Total: ~9–10 honest §.

## 4. Running record

### F0 — §0.4.435: the deep audit (fresh-clone DiffKT walk + tape gap table)

Source of truth: a fresh `--depth 1` clone of facebookresearch/diffkt,
`kotlin/api/src/main/kotlin/org/diffkt/model/` (34 files, ~1900 lines),
walked file-by-file 2026-09-21. Everything below is read off that source,
not remembered. This entry is the reference every F1–F8 agent reads.

#### 4.0.1 The abstraction stack (`Trainable` / `TrainableComponent` / `Layer` / `Model`)

- `Layer<T>` — `operator fun invoke(vararg inputs: DTensor): DTensor`
  plus `getSingleInput` (asserts arity 1). `LayerSingleInput<T>` narrows
  to `invoke(input: DTensor)`. An infix `DTensor.into(layer)` for
  pipelining. `cpu()`/`gpu()` default to identity.
- `Trainable<T>` — the training-capable node:
  `extractTangent(output, extractor: (input, output) -> DTensor): Tangent`
  (the reverse-derivative extraction hook: each trainable tensor asks the
  extractor for its gradient w.r.t. `output`),
  `trainingStep(optim, tangent): T` (returns a NEW instance — DiffKT is
  already functional at the parameter level), and `store(ByteBuffer)` /
  `load(ByteBuffer)` checkpoint hooks.
- `TrainableComponent<T> : Trainable<T>` — a node with children:
  `val trainables: List<Trainable<*>>` plus
  `withTrainables(newList): T` (positional rebuild). Its `trainingStep`
  zips `trainables` against `Tangent(grads: Collection<Tangent>)` and
  rebuilds; `extractTangent` maps over children.
- `TrainableTensor(tensor: DTensor) : Trainable` — the leaf.
  `trainingStep` = `optim.tensorTrainingStep(tensor, tangent.value)`
  wrapped in a new `TrainableTensor`. `extractTangent` =
  `Tangent(extractor(tensor, output))`. `store`/`load` write
  rank + dims + f32 data through an auto-expanding ByteBuffer.
- `TrainableLayer<T>` = `TrainableComponent + Layer`;
  `TrainableLayerSingleInput` adds the single-input narrowing.
- `Model<T> : TrainableComponent<T>` — `abstract val layers: List<Layer<*>>`,
  `trainables` = `layers.filterIsInstance<Trainable<*>>()`,
  `predict(data)` = fold of `invoke` over layers,
  `withTrainables` splices updated trainables back into the layer list
  positionally (non-trainable layers pass through).
- `Sequential(layers)` — same fold, as a `TrainableLayerSingleInput`.
- `LayerWithInferenceMode` — `val inferenceMode: Layer<*>` (BatchNorm
  freezes to `AffineTransform`; Dropout to identity).

Tlaloc mapping (per §2 decisions): the same shape, minus `store`/`load`
beyond tensor round-trip helpers (out of scope v1) and with
`extractTangent` replaced by the tape's id-addressed gradients — F1's
`valueAndGradients(model, loss)` walks the parameter tree, records each
parameter as a tape leaf, and reads gradients back by tape id, so no
extractor protocol is needed.

#### 4.0.2 The optimizer contracts, exactly as DiffKT spells them

`Optimizer<T>` — `train(component, tangent)` = `component.trainingStep(this, tangent)`
then `afterFit()`. Per-tensor hook: `tensorTrainingStep(tensor, gradient): DTensor`.
State is MUTABLE inside the optimizer: a `nextParameter` cursor
incremented per `tensorTrainingStep` call, reset in `afterFit()` —
per-parameter state lives in a `mutableListOf<DTensor>` indexed by visit
order. Our F3 replaces the cursor with keyed state per §2.4 (pure
`(params, grads, state) → (params', state')`), keeping the arithmetic
below bit-identical.

- `FixedLearningRateOptimizer(alpha)`: `t - alpha * g`.
- `SGDOptimizer(initialLearningRate=.001f, weightDecay=0f, momentum=0f)`:
  - momentum == 0: `velocity = g`.
  - first visit of a parameter (iteration 0): `velocity = g` (seeded, stored).
  - else: `v' = momentum * v + (1 - momentum) * g` — NOTE: the
    EMA form, NOT PyTorch's `v' = momentum * v + g`.
  - update: `t - learningRate * v'`.
  - `afterFit()`: `learningRate = initialLearningRate / (1 + weightDecay * iteration)`
    — DiffKT's "weightDecay" is LEARNING-RATE DECAY, not L2. Record it,
    spell it `lrDecay` in `:nn`, and say so in the KDoc (F3).
- `RMSpropOptimizer(alpha=0.005f, beta=0.9f)`:
  - first visit: `ms = g²` (stored). Else `ms' = beta * ms + (1 - beta) * g²`.
  - update: `t - alpha * g / sqrt(ms')` — NO epsilon in DiffKT. F3 keeps
    a literal-parity mode oracle but adds an `eps` param defaulting to 0f.
- `AdamOptimizer` — A PLACEHOLDER in DiffKT (`TODO("Not yet implemented")`,
  both methods). There is NO DiffKT Adam maths to be parity with. F3
  implements standard Kingma–Ba Adam (bias-corrected m̂/v̂,
  `t - lr * m̂ / (sqrt(v̂) + eps)`, defaults lr=1e-3, β₁=.9, β₂=.999,
  eps=1e-8) and oracles it against hand-stepped values + the F8 PyTorch
  harness, not against DiffKT.
- `Momentum.kt` is not an optimizer: it is the EMA helper
  `momentumUpdated(new, momentum) = (1 - momentum) * this + momentum * new`
  (tensor and Float overloads), shared with BatchNorm — momentum
  multiplies the NEW batch statistic (the PyTorch `running_stats`
  convention, momentum=0.1 default).

#### 4.0.3 Initializers (`Initializer` + `FanMode`)

- `gaussian(mean=0f, variance=1f)`: `nextGaussian() * sqrt(variance) + mean`
  per element (java.util.Random). Ours: `normalFloats(key, n)` scaled —
  bit-deterministic threefry, per §2.5.
- `uniform(min=0f, max=1f)`: uniform per element. Ours: `uniformFloats`.
- `kaimingUniform(fanMode, activationGainFactor)`:
  `bound = sqrt(3f / fan) * gain`, uniform in ±bound, where
  `fan = fanSize(shape) * shape.drop(2).product`,
  `FanIn.fanSize = shape[1]`, `FanOut.fanSize = shape[0]` (the PyTorch
  formula on DiffKT's own layouts). Gain constants: Linear/Conv/Sigmoid 1f,
  Tanh 5f/3, Relu sqrt(2f), LeakyRelu(slope) sqrt(2/(1+slope²)).
- Layer defaults: Dense W AND b both `uniform(±sqrt(1/numInputs))`;
  Conv2d filter `kaimingUniform(FanIn, LeakyRelu(sqrt(5f)))` (the PyTorch
  conv default); Embedding/EmbeddingBag `gaussian()`.

#### 4.0.4 The layers, one by one

| Layer | Params (trainable ✓) | Forward | Notes |
|---|---|---|---|
| `Dense(numIn, numOut, random, bias=true, activation=Identity)` | W ✓ `[numIn, numOut]`, b ✓ `[numOut]` (omitted from `trainables` when `bias=false`; b then FloatScalar.ZERO) | `activation(x.matmul(W) + b)`, input rank ≥ 2 | broadcast of b over rows |
| `Conv2d(filterShape, hStride, vStride, activation, paddingStyle, random)` | filter ✓ `[Co, kh, kw, Ci]` | `activation(conv2d(x NHWC, filter, strides, paddingStyle))` | NO bias tensor. PaddingStyle: Valid / Same / Full / Explicit(t,b,l,r). Same (TF SAME): per axis, `in % stride == 0 ? max(k - stride, 0) : max(k - in % stride, 0)` total, split top-heavy-bottom |
| `Conv2dWithSamePadding` | — | `Conv2d` with `PaddingStyle.Same` | subclass sugar |
| `MaxPool2d(poolH, poolW)` | none | window = stride = pool, NHWC rank-4, H %% poolH == 0 required | not trainable |
| `AvgPool2d(poolH, poolW)` | none | ditto, mean per window | |
| `Flatten` | none | `input.flatten(startDim = 1)` → `(N, *)` | object |
| `ReluLayer` | none | `relu(input)` | object |
| `AffineTransform(m ✓, b ✓)` | both TrainableTensor | `m * x + b` elementwise | BatchNorm's frozen form |
| `Activation.{Relu, Identity, Sigmoid, Tanh}` | none | the four objects; Dense/Conv compose them post-op | |
| `Dropout(p)` | none | train: `x * mask`, `mask[i] = rand() > p ? 1/(1-p) : 0f` (inverted scaling at TRAIN time); eval (`inferenceMode`): identity | takes `random` per CALL — our §2.5 key-per-call is the same shape |
| `Embedding(numEmb, embSize, random)` | table ✓ `[V, D]` | `embedding(table, intIndices)`: `(*) → (*, D)` | NO paddingIndex in DiffKT (ours has it in `:core`; keep as an optional `:nn` extra) |
| `EmbeddingBag(numEmb, embSize, reduction, random)` | table ✓ | flatten indices → embed → per bag `slice(start, end)` on axis 0 → `reduction.reduce` → `concat` → `[numBags, D]` | bags = `bagOffsets` (linear starts, last bag runs to end). Reduction shipped: ONLY `Sum` (`sum(0, keepDims=true)`); Mean/Max unimplemented in DiffKT |
| `GRU(numIn, numHidden, random, acc, linearBeforeReset=false)` | 3 or 4 Dense gates ✓ | see below | `RecurrentBase` fold/accMap |

GRU, exactly as DiffKT spells it (input `[batch, seq, numIn]`, batchAxis 0,
seqAxis 1; `initialState` zeros `[1, numHidden]` `expand`ed to batch;
per step `slice(i, i+1, seqAxis).squeeze(seqAxis)`):

- LinearAfterResetGru (the default), gates are whole `Dense` layers with
  their own biases and activations baked in
  (`xh2u`, `xh2r`: `Dense(numIn+numHidden, numHidden, σ)`;
  `xh2n`: `Dense(numIn+numHidden, numHidden, tanh)`):
  ```
  xh = concat(x, h, axis=1)
  u  = xh2u(xh)                        // update gate, σ
  r  = xh2r(xh)                        // reset gate, σ
  n  = xh2n(concat(x, r * h, axis=1))  // candidate, tanh AFTER reset
  h' = (1 - u) * n + u * h
  ```
- LinearBeforeResetGRU (`x2n`: `Dense(numIn, numHidden, Identity)`,
  `h2n`: `Dense(numHidden, numHidden, Identity)`):
  `n = tanh(x2n(x) + r * h2n(h))`, same u/r/h'.
- `AccType.Fold` returns the LAST output `[batch, numHidden]`;
  `AccType.AccMap` concats per-step outputs (each `unsqueeze(seqAxis)`)
  into `[batch, seq, numHidden]`.

BatchNorm, exactly (the momentum convention matters):

- Op: `batchNorm(input rank ≥ 2 …C-last, scaleShift [2, C])`, reduces ALL
  axes but the last: `mean = Σx/n`, `var = Σx²/n − mean²` (biased),
  `out = scale * (x − mean)/sqrt(var + 1e-5f) + shift`,
  scale = `scaleShift[0]`, shift = `scaleShift[1]`. Returns
  `BatchNormResult(result, n, sum, sumOfSquares, mean, variance)`.
- `BatchNormTraining(numFeatures, momentum=0.1f)` (V2, the default):
  running state = (runningN, runningSum, runningSumOfSquares), each
  EMA'd: `running' = (1−momentum)*running + momentum*batchStat`.
  `stats`: `mean = runningSum/runningN`, `var = runningSS/runningN − mean²`.
  scaleShift init: ones `[1,C]` concat zeros `[1,C]`.
- `BatchNormTrainingV1`: running (mean, variance) EMAs, with Bessel's
  correction `var * n/(n−1)` applied to the BATCH variance before the EMA;
  runningVariance init ones.
- `inferenceMode` = `freezeBatchNorm`: `m = scale/sqrt(var + 1e-5f)`,
  `b = shift − m * mean` → `AffineTransform(m, b)`.
- DiffKT mutates the running stats inside `invoke` (their own TODO #172
  flags it); our §2.6 functional `(output, newStats)` return IS
  `batchNormTrainV2`'s actual signature — DiffKT wrote the pure function
  and then wrapped it mutably. We keep the pure function.

#### 4.0.5 The tape today (what `:autograd` already covers)

`TracedOps.kt` records, and `Backward.kt` routes through
`applyRegistryRule` (the real `VjpRegistry` + `DxirInterpreter` bridge):

- Elementwise: ADD SUB MUL DIV NEG RELU SQRT EXP LOG TANH SIGMOID POW
  (same-shape at any rank, `S : Shape` generic), STEP (inline zero arm).
- BROADCAST: scalar→rank-1/2/3 (`broadcastScalar`), rank-1→rank-2 row
  (`broadcastRow`, dims=[1]) and col (`broadcastCol`, dims=[0]),
  rank-1→rank-3 (`broadcastInner`/`broadcastBatch`) — with the
  `broadcast_dimensions` attr carried on the entry (§0.4.85).
- Reductions: SUM and MEAN, FULL-to-scalar only (no axis forms on the
  tape yet — the rules DO support `reduction_dims`, §0.4.415 Phase A1).
- MATMUL: rank-2 `[m,k]×[k,n]` and batched rank-3 (same OpKind).
- Leaves: `traceLeaf` (F32 only — `TapeEntry.value` is a FloatArray),
  `isConstant` leaves skip adjoint materialisation (§0.4.65).
- `Capture.kt` (`Tape.toDxirFunction`) replays a tape as a DxirFunction —
  the export lane if a Phase F model graph ever needs emission; not on
  the F1–F8 critical path.

Bridge mechanics every slice must respect (`Backward.applyRegistryRule`):
one transient `DxirOp` per reverse step, one dedicated `DxirParam` PER
OPERAND POSITION (aliasing-safe), each typed `DxirType(F32, operandDims)`;
entry `attrs` ride onto the transient primal; the env maps param id →
cached FloatArray; each rule contribution is `DxirInterpreter.evalNode`'d
and seeded by tape id. Consequences:

- Ops whose rules reference the primal RESULT re-evaluate the forward
  inside the contribution tree — fine, the interpreter arms exist.
- Integer operands (EMBEDDING indices) ride the F32 env: the
  EMBEDDING_GRAD interpreter arm reads indices via `idx[p].toInt()` with
  NO dtype require, so float-encoded index leaves work. (The forward
  EMBEDDING interpreter arm DOES require I32 — irrelevant here: tape
  forwards run the `:core` host twins, never the interpreter.) Exactness
  cap: float-encoded ints are exact to 2²⁴ — assert `vocab < 16_777_216`
  in the F6 spelling.
- Adding an op kind = (1) a TracedOps spelling that computes the forward
  via the `:core` host twin and records inputs+dims+attrs, (2) the kind
  added to Backward.kt's registry-dispatch arm. Nothing else.

`VjpRegistry` rules already registered (the parity arc's full set —
everything Phase F needs is here): ADD SUB MUL DIV NEG ABS RELU SUM MEAN
MATMUL CONV2D CONV_TRANSPOSE2D AVGPOOL2D MAXPOOL2D RESHAPE MAX MIN
SOFTMAX CONCAT SLICE WHERE COMPARE ZEROS_LIKE PAD DOT POW EXP LOG SIN COS
TAN ATAN LGAMMA DIGAMMA TRIGAMMA POLYGAMMA SIGN SQRT TANH SIGMOID CAST
TRANSPOSE REVERSE GATHER EMBEDDING SPARSE_MATMUL BROADCAST SUM_TO
BROADCAST_LIKE PAD_TO SLICE_AT SLICE_LIKE PAD_LIKE RNG_UNIFORM RNG_NORMAL
CHECK_SHAPE_LIKE. NOT registered: BATCHNORM (the fused op has no VjpRule
— F5 composes batchnorm from primitives on the tape instead; the maths
then rides the existing rules, honouring "no new gradient math in :nn").

#### 4.0.6 THE GAP TABLE — Tracer spellings each slice must add

Attr spellings are the DxirInterpreter's, verified against its arms.
"Route" = forward host twin in `:core` + kind added to Backward dispatch.

| Slice | New spelling | OpKind + attrs | Forward route | Notes |
|---|---|---|---|---|
| F1 | none (library only) | — | — | `valueAndGradients` over N params: generalise Grad.kt's fixed-arity entries to a keyed list (backward() is already id-addressed; pure library code) |
| F2 Dense | none | — | — | `matmul` + `broadcastRow`-plus + relu/sigmoid/tanh all exist. Initializers: pure `:core` `uniformFloats`/`normalFloats` + FanMode math in `:nn` |
| F2 Flatten | `reshape(dims)` | RESHAPE (no attrs; result dims on entry) | pure copy of `value` with new dims | ReshapeRule registered; also gives GRU its squeeze/unsqueeze |
| F3 optimizers | none | — | — | pure host math on DTensor/FloatArray; no tape involvement at all |
| F4 Conv2d | `conv2d(w, strideH, strideW, padding)` | CONV2D; attrs `window_strides`=[sH,sW], `padding`=[[t,b],[l,r]], (`lhs_dilation` `rhs_dilation` `window_reversal` defaulted) | `conv2dGeneral` (NCHW × OIHW) | Conv2dRule registered. LAYOUT DECISION (F4 records it): `:nn` adopts Tlaloc-native NCHW/OIHW; DiffKT's NHWC/[Co,kh,kw,Ci] is a layout transpose of the same maths, and the F8 PyTorch oracle is NCHW-native anyway. SamePadding: compute DiffKT's Same formula in `:nn` and pass explicit padding. `feature_group_count` stays 1 — host twins reject grouped (§0.4.429 named deferral) |
| F4 MaxPool2d | `maxPool2d(windowH, windowW)` | MAXPOOL2D; attrs `window`=[h,w], `window_strides`=window, `padding`=[[0,0],[0,0]] | `maxPool2dGeneral` | MaxPool2dRule registered (tie-splitting convention §0.4.389, differs from XLA select_and_scatter — F8 oracle: avoid exact ties or compare vs PyTorch which splits the same way? PyTorch picks FIRST max — pin oracles on tie-free grids) |
| F4 AvgPool2d | `avgPool2d(windowH, windowW)` | AVGPOOL2D; same attr shape | `avgPool2dGeneral` | AvgPool2dRule registered |
| F5 BatchNorm | axis reductions: `sum(axes)` / `mean(axes)`; channel broadcast: rank-1→rank-4 | SUM / MEAN with attrs `reduction_dims`=axes; BROADCAST with `broadcast_dimensions`=[1] (NCHW channel) | host loops (trivial) | NO fused BATCHNORM on the tape — compose `(x−μ)/√(σ²+ε)·γ+β` from SUB DIV SQRT MUL ADD + the new axis/broadcast spellings; gradient rides existing rules. Rank-4 elementwise ops already generic. Functional stats per §2.6 are `:nn`-level (host EMA on DTensors, no tape) |
| F5 Dropout | none (composition) | — | — | mask = CONSTANT leaf (`isConstant=true`) built from `uniformFloats(key, n)` thresholded at host, values ∈ {0, 1/(1−p)}; forward = existing MUL. Bit-deterministic per key; zero adjoint cost through the constant |
| F6 Embedding | `embedding(tableTracer, indices)` | EMBEDDING; attrs `padding_index` (Int, −1 = none) | `:core` `embedding(table, indices, paddingIndex)` host twin | EmbeddingRule registered (fused EMBEDDING_GRAD dense scatter — the §2.7 dense v1). Indices: I32 DTensor at the API, recorded as a float-encoded CONSTANT leaf (see 4.0.5; assert vocab < 2²⁴). Dense grad = full-table zeros + scattered rows |
| F6 EmbeddingBag | `slice(start, end, axis)`, `concat(list, axis)` | SLICE attrs `start_indices`/`limit_indices`/`strides` (full-rank lists); CONCAT attr `dimension` | host copies | SliceRule/ConcatRule registered. Bag loop is `:nn` library code over embed→slice→sum(axes, keepDims via reshape)→concat, exactly DiffKT's spelling. Sum reduction only (DiffKT parity) |
| F7 GRU | `concat` (from F6), `slice` (from F6), `reshape` (from F2) | — | — | cell = Dense reuse + existing elementwise; unroll is library-level fold, tape handles it natively. `expand` of initialState: broadcastRow or explicit host tile as constant-ok leaf? initialState is a LEAF param-like tensor broadcast to batch — use BROADCAST rank-1→rank-2 (exists) or rank-2 [1,H]→[B,H] via new broadcast_dimensions=[0,1]… F7 decides; smallest honest form is tiling the zeros host-side (initial h carries no gradient when zeros-const) |
| F8 | none | — | — | losses: MSE = existing SUB/MUL/MEAN; cross-entropy = logSoftmax composition needs the F5 axis reductions (sum over class axis) + existing EXP/LOG — or MEAN of per-row NLL assembled at host as constant one-hot MUL. Pin the composition in F8, not a new op |

Backward.kt dispatch additions across F2–F6 (one line each, the
registry-bridge arm): RESHAPE, CONV2D, MAXPOOL2D, AVGPOOL2D, SLICE,
CONCAT, EMBEDDING (+ SUM/MEAN axis forms ride the existing SUM/MEAN arm
— the rules read `reduction_dims` off the carried attrs; the bridge
already propagates attrs since §0.4.85).

#### 4.0.7 `:core` substrate mapping (what each layer consumes)

- Constructors: `Tensors.f32Scalar/f32Vector/f32Matrix/f32Zeros/f32Tensor3/
  f32Tensor4/i32Vector/i32Matrix`; escape hatches `hostF32()`/`hostI32()`.
- Host twins ready and certified by the parity arc: `conv2dGeneral` (+
  `conv2dDataAdjoint`/`conv2dKernelAdjoint` used by the rule bodies),
  `maxPool2dGeneral`/`maxPool2dGrad`, `avgPool2dGeneral`/`avgPool2dGrad`,
  `embedding` (paddingIndex arity-disambiguated form §0.4.409) /
  `embeddingGrad`, `batchNormGeneral` (rank-4 NCHW, eps param — usable
  for F5's INFERENCE-mode fast path and as a forward oracle; not on the
  tape), `softmax`/`logSoftmax`/`crossEntropyLoss`/`nllLoss` (host-side
  oracles for F8's loss composition), matmul/outerProduct/where/clip and
  the full elementwise family.
- `Random.kt`: `RandomKey(k0, k1)` / `fromSeed(Long)`, `split(n)`,
  `foldIn(data)`, `threefryBits`, `uniformFloats(key, n)`,
  `normalFloats(key, n)` (+ cauchy/exponential/chiSquare, not needed by
  Phase F). Initializer keys: `key.split(...)` per parameter in layer
  constructors — F2 fixes the split discipline (one child key per
  parameter tensor, in declaration order) and documents it.
- `SparseTensor` (CSR) + SPARSE_MATMUL rule: the §2.7 row-sparse
  embedding-gradient substrate, NOT taken in v1 (dense EMBEDDING_GRAD).

#### 4.0.8 The F8 PyTorch-oracle pattern (the §0.4.289 precedent)

`harness/python/run_pytorch_llama_grad.py` + `benchmarks/.../
LlamaDecoderIreeVsPytorchTest.kt` is the template: resolve
`~/.local/venvs/iree/bin/python`, probe `python -c "import torch"` via
ProcessBuilder, `assumeTrue` self-skip with an instructive message when
missing; JSON/stdout exchange of inputs and reference outputs; exact
seeds both sides. F8 adds `harness/python/run_pytorch_nn_train.py`:
build the twin MLP + small conv net in PyTorch, load the SAME initial
weights (exported from the Kotlin side, since threefry ≠ torch RNG),
run N SGD/Adam steps, emit per-step losses; the Kotlin test asserts
loss-curve agreement at tolerance and final-weight agreement at a looser
one. Optimizer parity caveats from 4.0.2 (DiffKT-SGD's EMA momentum ≠
PyTorch SGD momentum; RMSprop eps) mean F8 pins PyTorch-formula
optimizers (`:nn` Adam, plain SGD lr-only) for the cross-language run
and keeps DiffKT-formula quirks to hand-stepped Kotlin oracles.

#### 4.0.9 Landmines pinned for later agents

1. DiffKT's Adam is a TODO placeholder — never cite DiffKT as the Adam
   oracle (4.0.2).
2. DiffKT SGD "weightDecay" is LR decay; momentum is the EMA form. Hand
   oracles must use THOSE formulas when claiming DiffKT parity (4.0.2).
3. `momentumUpdated` weights the NEW stat by `momentum` (0.1) — BatchNorm
   running stats follow PyTorch's convention, not Flax's (1−momentum).
4. Tape leaves are F32-only; integer indices ride as float-encoded
   constant leaves, exact below 2²⁴ (4.0.5).
5. Grouped conv (`feature_group_count > 1`): interpreter yes, host twins
   NO (§0.4.429). `:nn` Conv2d v1 is groups=1; a groups param waits on
   the host-twin tail.
6. MAXPOOL2D_GRAD splits ties to ALL winners; PyTorch routes to the
   first. Oracles pin on tie-free grids (quarter-integer values).
7. The pool host twins require divisibility like DiffKT (window=stride,
   H % pool == 0); F4 keeps the same requires.
8. BatchNorm on the tape is a COMPOSITION (no BATCHNORM VjpRule) — do
   not reach for the fused op kind in F5 (4.0.6).
9. `Dropout` mask leaves MUST be `isConstant=true` or the reverse walk
   pays a dead materialisation per entry (§0.4.65 mechanism).

Deferred, by name: row-sparse embedding gradients (§2.7 stands —
dense v1); grouped-conv host twins; EmbeddingBag Mean/Max reductions
(DiffKT ships only Sum; parity is Sum); `store`/`load` checkpointing
beyond tensor round-trip; GPU-resident training (§2.9); DiffKT's
`LinearBeforeResetGRU` DNNL hookup (never existed upstream either).

### F1 — §0.4.437: the `:nn` module + the compiler-route substrate

The first slice on the amended route (§0.4.436), and the loop closes
exactly as decision 3 promised: `nn/src/commonMain` holds component
values and capture bookkeeping, ZERO gradient math, and the 2-layer
Sequential's gradients come out of `DxirReverseTransform` hand-exact.

**What landed.**

- `:nn` (`io.tlaloc.nn`), registered in `settings.gradle.kts`, `api`
  deps on `:core`/`:ir`/`:autograd` (the surface speaks `DTensor`,
  `DxirFunction`, `Tracer` in public types).
- `captureN` in `:autograd`'s `Capture.kt` — the N-ary generalisation
  of `capture`/`capture2`: N tensors → N tape leaves (list order = the
  `DxirFunction`'s positional param order) → `Tape.toDxirFunction`.
  This is the arbitrary-arity entry the amendment named; the gap
  table's F1 row ("none — library only") holds: no new TRACE spelling
  (`sum()` already reduces full-to-scalar), no `Backward.kt` change.
- Components per F0 §4.0.1, erased-Tracer surface: `Layer.forward(x:
  Tracer<Shape>, params: Params)` (single-input v1 — DiffKT's
  `LayerSingleInput` fold), `Trainable` (stable path-like keys in
  declaration order + `withParameters` functional rebuild),
  `TrainableLayer`, `NamedParameter`, `AffineTransform` (`m·x + b`
  elementwise, F0's spelling), `Sequential` (fold; child keys prefixed
  `"<layerIndex>."` — positional like DiffKT's `withTrainables`
  splice). `Params` is a scoped key→leaf-tracer resolver: a container
  narrows it before handing it to a child, so a layer only ever speaks
  its OWN keys.
- The differentiation contract in `Training.kt`: `capture(model,
  inputs, lossFn)` traces the forward once with every input + every
  parameter as a differentiable leaf (inputs first, then parameters in
  key order), requires a scalar loss, converts via `toDxirFunction`,
  and applies `DxirReverseTransform.apply(primal, includeForward =
  true)` ONCE — the gradient function returns `(loss, *grads)` so a
  training step is a single `DxirInterpreter.evalFunction` call.
  `CapturedStep.run(model, inputs)` re-binds host values positionally
  (guarded: parameter keys must equal the captured list), returns
  `StepResult(loss, gradients keyed like the model, inputGradients)`.
  `valueAndGradients` = capture + run, honestly retracing per call.

**Design decisions, with rejections.**

- **Erased `Tracer<Shape>` at the `:nn` boundary.** REJECTED:
  threading phantom shapes through `Layer` — Dense-style shape changes
  make a general interface inexpressible without rank/arity variant
  explosions, for no safety the traced ops' dims checks don't already
  give. DiffKT's own surface is untyped `DTensor`.
- **`Params` resolver over a "traced twin" model.** REJECTED:
  rebuilding the model with Tracer-typed fields per capture — it
  duplicates every layer class; the resolver keeps each layer's
  forward single-sourced. Also REJECTED: DiffKT's `extractTangent`
  extractor protocol — key-addressed gradients from the transform make
  the hook meaningless here (F0 §4.0.1 anticipated this).
- **One combined `(loss, *grads)` evaluation** via `includeForward =
  true`. REJECTED: separate primal + gradient evals per step (two
  walks where one suffices); the standalone primal is still captured and
  exposed — it is the prediction path, the route pin's witness, and
  F8's emission artifact.
- **Caching = caller-held `CapturedStep` in F1.** The pair re-binds
  values per step (params are `DxirParam`s, never baked constants) and
  refuses structure drift by key comparison. RECORDED CONTRACT: F8
  adds the structure-keyed cache in front of the compiled-GPU lane
  (the §0.4.307 amortization); `valueAndGradients` remains the honest
  retrace-per-call convenience until then.

**Oracle story.** Quarter-integer grid throughout, `==` not
tolerance: a 2-layer Sequential's five gradients (4 params + the
input gradient, which the transform returns for free) pinned against
the hand derivation; the ROUTE PIN asserts the captured primal is a
real `DxirFunction` with 5 params — one past the `grad {}` 4-arity
ceiling the IR never had; and the trace-vs-hand-built-DXIR oracle
builds the same single-affine graph through `capture` and through
`DxirBuilder` directly, transforms BOTH, and pins elementwise
equality (plus both against hand values). One bit-exactness lesson
worth keeping: `2·0·(−0.25) = −0.0`, and the oracles assert the
signed zero. Re-bind is pinned too: one `CapturedStep`, two
parameter sets, both losses hand-exact, structure drift refused.

**Deferred, by name:** multi-input `Layer` surface (single-input v1;
F7 revisits if GRU wants it at the interface); automatic
structure-keyed caching (F8, per the recorded contract); everything
in F0's standing deferral list.

### F2 — §0.4.438: the simple layers + the initializer family

**What landed.**

- `Dense` in DiffKT's exact semantics (F0 §4.0.4): `w [numIn, numOut]`,
  `b [numOut]`, forward `activation(x matmul W + b)` with the bias
  riding the §0.4.85 row-broadcast TRACE spelling
  (`broadcast_dimensions = [1]`, whose reverse is the axis-0 SUM every
  bias gradient is); `bias = false` skips the add and keeps `b` out of
  the trainables (DiffKT parks a `FloatScalar.ZERO` placeholder; we keep
  none — same maths). The DiffKT constructor surface is the companion
  `Dense(numInputs, numOutputs, key, bias, activation)` drawing BOTH W
  and b `uniform(±√(1/numInputs))` — DiffKT's own default for both.
- **The key-split discipline, fixed for all of Phase F** (F0 §4.0.7's
  open item): a layer splits its key ONCE into one child per parameter
  tensor in declaration order (`split(2)[0]` → w, `[1]` → b), and
  `bias = false` still consumes the same split, so the drawn W is
  bit-identical with and without a bias.
- `Flatten` (`flatten(startDim = 1)`; rank-2 input passes through
  untouched — DiffKT's flatten is the same no-op view there — and
  rank ≥ 3 records the new `reshape` spelling), `ReluLayer`, and the
  `Activation` objects (Identity/Relu/Sigmoid/Tanh) Dense composes
  post-op. `AffineTransform` had landed in F1.
- Initializers over the D1 threefry streams (`:nn`-level host math,
  zero tape involvement): `uniformInit`/`gaussianInit` (DiffKT's affine
  rescales verbatim, on `uniformFloats`/`normalFloats`), `fanOf` +
  `ActivationGain` (DiffKT's fan and gain constants exactly),
  `kaimingUniformInit` (`bound = √(3/fan)·gain`). `kaimingNormalInit`
  and `xavierUniformInit`/`xavierNormalInit` are RECORDED Tlaloc
  extensions — the standard He/Glorot (PyTorch) formulas, absent from
  DiffKT and never oracled against it.
- The gap table's F2 TRACE spelling: `Tracer.reshape(newDims)` —
  RESHAPE, no attrs, result dims on the entry, forward a pure row-major
  copy; `ReshapeRule` reverses it for free. (Also F7's future
  squeeze/unsqueeze substrate, as the table noted.)
- One interpreter arm: `OpKind.RELU` forward (the STEP mask's twin,
  same `> 0` convention at zero). No interpreted graph ever carried a
  forward RELU before — the value-tape computes forwards host-side and
  ReluRule's contributions are STEP+MUL — but the compiler route
  evaluates the transform's `includeForward` output, which re-emits the
  primal ops. `Backward.kt` untouched, as the amendment promised.

**Design decisions, with rejections.**

- **Dense input is rank-2 `[batch, numInputs]` in v1** — DiffKT accepts
  rank ≥ 2 (its matmul broadcasts leading axes); the traced matmul is
  rank-2/rank-3 today, so the rank-3 Dense input form is a NAMED
  DEFERRAL, not a silent divergence.
- **Kaiming/Xavier uniform variants DELEGATE to `uniformInit`** so
  every uniform initializer shares one bit pattern per (key, bound).
  REJECTED: per-variant draw loops — a bit-drift risk between spellings
  for zero gain.
- **`kaimingNormalInit` scales by `std` directly** rather than routing
  `variance = std²` through `gaussianInit` — `sqrt(std·std)` need not
  round-trip to `std` in f32, and the tests pin bits.

**Oracle story.** Layered, bit-exact at the base: `uniformInit` /
`gaussianInit` pinned `==` against the raw `uniformFloats` /
`normalFloats` draws under the identical affine rescale for a fixed
key; fan factors and gain constants pinned analytically (including the
exact degeneracies `LeakyRelu(0) = Relu`'s gain and `LeakyRelu(1) = 1`);
the Kaiming/Xavier variants pinned against the base initializers at
test-replicated bounds; the Dense factory pins the split discipline and
the `bias = false` W-stability. Gradient certs on the quarter grid,
`==` not tolerance: the Dense+ReluLayer+Dense Sequential (unambiguous
relu masks, no pre-relu zeros) hand-exact through `valueAndGradients` —
every masked-position zero comes out `+0.0` (the interpreter's
accumulator-init absorbs the `−0.0`s the STEP·upstream products
produce); the trace-vs-hand-built-DXIR oracle on a single Dense with
composed Relu (MATMUL → BROADCAST → ADD → RELU → MUL → SUM built both
ways, both through the SAME transform, elementwise equal and both
hand-exact); Flatten alone (the gradient comes back rank-3-shaped —
ReshapeRule's reverse witnessed by the shape itself) and Flatten→Dense
chained hand-exact.

**Deferred, by name:** rank-3 Dense input (above); F0's standing
deferral list unchanged.

### F3 — §0.4.439: the optimizers, pure and functional

**What landed.** `Optimizers.kt` in `:nn` — the decision-4 contract made
concrete: `Optimizer<S>` with `initialState()` and pure
`step(params, grads, state) → OptimizerStep(params', state')`, plus the
model-level `Optimizer.step(model, grads, state) → (model', state')`
convenience over `withParameters`, so a training loop is a fold. State
is keyed like the parameters and created LAZILY — a key absent from the
state map IS DiffKT's "first visit of a parameter", with no visit-order
cursor. The gap table's F3 row held exactly: pure host math on
DTensor/FloatArray, zero tape/trace involvement, zero gradient math.

- `FixedLearningRate(alpha)`: `t − α·g`, stateless (`Unit`). The F0
  audit records no DiffKT default for `alpha`, so none is offered.
- `SGD(initialLearningRate = .001f, lrDecay = 0f, momentum = 0f)`:
  DiffKT-exact — EMA momentum `v' = μ·v + (1−μ)·g` (NOT PyTorch's
  `μ·v + g`), first visit seeds `v = g`, update `t − lr·v'`, and the
  `afterFit()` schedule `lr = lr₀/(1 + lrDecay·completedSteps)`.
  DiffKT's `weightDecay` is spelled `lrDecay`, as F0 §4.0.2 demanded.
  `momentum = 0` stores no velocity slots at all.
- `RMSprop(alpha = .005f, beta = .9f, eps = 0f)`: DiffKT-literal —
  first visit `ms = g²`, else `ms' = β·ms + (1−β)·g²`, update
  `t − α·g/√ms'` with NO epsilon by default; `eps` is the recorded
  Tlaloc extension in PyTorch's placement (`√ms' + ε`), never oracled
  against DiffKT.
- `Adam(lr = 1e-3f, β₁ = .9f, β₂ = .999f, ε = 1e-8f)`: standard
  Kingma–Ba, BIAS-CORRECTED. **The recorded finding the slice asked
  for: DiffKT's `AdamOptimizer` is a `TODO("Not yet implemented")`
  placeholder (F0 §4.0.2), so there is no "does DiffKT bias-correct"
  fact to match — ours bias-corrects per the paper** and its oracle is
  hand-stepped math now, the PyTorch harness in F8. The bias-correction
  step count is global to the state, not per key — the `Trainable`
  contract steps every parameter together (recorded so nobody "fixes"
  it apart).
- **The "Momentum optimizer" of the slice list IS `SGD(momentum = μ)`**:
  DiffKT's `Momentum.kt` is not an optimizer but the EMA helper (the
  other recorded F0 finding), shipped here as `momentumUpdated`
  (Float + tensor overloads, momentum weighting the NEW statistic —
  the PyTorch running-stats convention) for F5's BatchNorm.

**Design decisions, with rejections.** Typed per-optimizer state values
(`SGDState`/`RMSpropState`/`AdamState`, internal constructors) —
REJECTED: one generic slot-map state shared by all optimizers (stringly
slots, no compiler help pairing optimizer to state), and DiffKT's
mutable `nextParameter` visit cursor + `afterFit()` reset (fights
decision 2, breaks on reorder; keyed lazy maps carry the same
semantics). Guards are strict: every parameter must have a gradient of
matching dims AND unknown gradient keys refuse — REJECTED: silently
ignoring extras (typo camouflage). `hostF32()` returns the backing
array, so every output is a fresh allocation; inputs are never written.

**Oracle story.** Hand-stepped references, 3 steps on the 2-param toy
(`w = [1, −2]`, `b = [0.5]`) with distinct quarter-grid gradients per
step so every slot outlives its first-visit seeding. Exact
`assertContentEquals` where the arithmetic is dyadic: FixedLearningRate
all 3 steps, SGD-with-momentum all 3 steps (velocities pinned too),
lrDecay steps 1–2, RMSprop step 1 (`√g² = |g|` on the grid) and every
`ms` slot at every step. 1e-6 against inline DOUBLE-precision hand math
where sqrt/div enter (an independent spelling — no code shared with the
Float path): lrDecay's `lr = 0.5/3` step, RMSprop steps 2–3, Adam all
3 steps plus the pinned step-1 collapse (`m̂ = g`, `v̂ = g²` at `t = 1`
⇒ update `≈ lr·sign(g)`). Equivalence pin: `SGD(momentum = 0)` ==
`FixedLearningRate` bit-exact over 3 steps. The eps test pins BOTH
regimes: `eps > 0` leaves a zero-gradient parameter untouched;
`eps = 0` NaNs on `0/√0` — the DiffKT-literal hazard, pinned as a
finding rather than papered over. Purity/laziness pinned (inputs and
old state unchanged, initial states empty). The integration cert:
`Dense(2, 1, fromSeed(7))` + F1's `valueAndGradients` + `SGD(lr = .1)`
on a quarter-grid batch of 4, loss `mean((y − 1.5)²)` STRICTLY
decreasing across all 10 steps (11 losses compared).

**Deferred, by name:** PyTorch optimizer parity (F8's harness, per
F0 §4.0.8 — DiffKT-formula quirks stay on hand-stepped Kotlin oracles);
F0's standing deferral list unchanged.

### F4 — §0.4.440: the conv stack joins the traced graph and the layer set

**What landed.** The gap table's three F4 TRACE spellings in
`TracedOps.kt`, and the four conv-stack layers in `:nn`'s new
`ConvLayers.kt`. Zero gradient math anywhere, zero `Backward.kt`
changes, zero interpreter changes — the §0.4.385/386/389 fused adjoints
were already waiting behind `Conv2dRule`/`AvgPool2dRule`/`MaxPool2dRule`
and the transform routes the captured graph straight through them.

- `Tracer.conv2d(w, strideH, strideW, padTop, padBottom, padLeft,
  padRight)` — CONV2D over NCHW × OIHW, forward via the certified
  `:core` `conv2dGeneral` host twin, attrs EXACTLY the interpreter's:
  `window_strides = [sH, sW]`, `padding = [[t, b], [l, r]]`, the
  dilation/reversal/group attrs left to their shared defaults. Groups
  stay 1 at the trace level (the host twin the forward routes through
  rejects grouped — §0.4.429; the IR itself supports them).
- `Tracer.maxPool2d(h, w)` / `Tracer.avgPool2d(h, w)` — the classic
  non-overlapping pool (the only form `MaxPool2dRule` v1
  differentiates), attrs all explicit: `window`, `window_strides =
  window`, `padding = [[0,0],[0,0]]`. DiffKT's spatial-divisibility
  require lives in the spelling (F0 landmine 7), so every layer above
  inherits it.
- `Conv2d(filter [Co, Ci, kh, kw], hStride, vStride, activation,
  paddingStyle)` — DiffKT's exact surface (NO bias tensor; activation
  composes post-op; `vStride` strides H, `hStride` W, DiffKT's own
  naming) on **the recorded F4 LAYOUT DECISION: Tlaloc-native
  NCHW/OIHW.** DiffKT's NHWC/`[Co, kh, kw, Ci]` is a layout transpose
  of the same maths, the F8 PyTorch oracle is NCHW-native anyway, and
  the fan is unaffected (`shape[1]·shape.drop(2).product` = `Ci·kh·kw`
  on both layouts). `PaddingStyle` ships all four DiffKT variants —
  Valid / Same / Full / Explicit — and Same is computed AT THE LAYER
  per axis from the input dims (`samePadding`, the TF formula off
  DiffKT's source: `total = in % s == 0 ? max(k − s, 0) : max(k − in %
  s, 0)`, before = total/2, odd unit AFTER) and passed down as explicit
  attrs, so the captured graph only ever carries literal padding. The
  companion draws `kaimingUniform(FanIn, LeakyRelu(√5))` under the F2
  key discipline (`split(1)[0]` for the single parameter).
- `Conv2dWithSamePadding` (the DiffKT subclass sugar, delegating to a
  Same-styled `Conv2d`), `MaxPool2d(poolH, poolW)`, `AvgPool2d(poolH,
  poolW)` — the pools are plain non-trainable `Layer`s.

**Design decisions, with rejections.** Forward values route through
the `:core` host twins (`conv2dGeneral`/`maxPool2dGeneral`/
`avgPool2dGeneral`) — REJECTED: reimplementing the loops inline in
TracedOps (the elementwise-op pattern) — the twins are the certified
engines the rule bodies themselves execute through, so trace-time
forward and gradient-body recomputation share one implementation.
The pool spellings keep DiffKT's divisibility require — REJECTED:
inheriting the engines' floor-division permissiveness (the host and
interpreter handle a remainder, but DiffKT refuses, and parity means
refusing where it refuses). A `groups` parameter on `Conv2d` —
REJECTED for v1: the host twin rejects grouped (§0.4.429 stands);
recorded as available in the IR, not a DiffKT feature.

**Oracle story.** All hand assertions `==` — the grids are
quarter-integer and the arithmetic dyadic. The trace-vs-hand-built-DXIR
oracle on the full net CONV2D(stride 2) → RELU → MAXPOOL2D → SUM:
built through the `:nn` capture AND through `DxirBuilder` with the
interpreter's exact attr spellings, both through the SAME transform,
elementwise `assertContentEquals` (stronger than the slice's 1e-6).
The hand-exact certs: the 1×1×3×3 valid conv through
`valueAndGradients` (`∂L/∂W` = the four block sums, `∂L/∂x` = the
tap-coverage sums); the same conv net's gradients (only the
pool-winning window carries gradient — `∂L/∂W` = its x taps, `∂L/∂x` =
W scattered there); maxpool on the shuffled well-separated 16-distinct
grid under `Σy²` (argmax positions get `2·max`, all else exactly zero,
and the empty `gradients` map pins that pooling is untrainable);
avgpool over two channels (uniform `2·mean/4` splat); the Same-padding
conv (loss, `∂L/∂W`, `∂L/∂x` all hand-exact, and `∂L/∂x`'s dims pin
the output-extent-equals-input-extent contract). `samePadding` itself
is pinned per axis (including `k < in % stride` → zero and the
odd-total after-heavy split), the companion's draw is bit-exact
against `kaimingUniformInit` under the key discipline, and
`withParameters` is functional (original untouched, unknown keys
refuse).

**Deferred, by name:** grouped conv at the layer/trace level (waits on
the §0.4.429 host-twin tail); rhs-dilated ("à-trous") conv at the trace
level (the IR and rule support it; no DiffKT surface asks for it);
F0's standing deferral list unchanged.

### F5 — §0.4.441: BatchNorm desugared, Dropout keyed — and the transform never noticed

**What landed.** The gap table's F5 TRACE spellings in `TracedOps.kt` —
`sum(axes)` / `mean(axes)` (SUM/MEAN with `reduction_dims`, reduced axes
DROPPED, forward loops mirroring the interpreter's §0.4.366 projection
bit-for-bit) and `broadcastAlong(vec, axis)` (rank-1 → rank-N BROADCAST
with `broadcast_dimensions = [axis]`, the §0.4.85 `broadcastRow`
generalised to any receiver rank — the NCHW channel broadcast at
`axis = 1`) — and `BatchNorm.kt` + `Dropout.kt` in `:nn`. Zero gradient
math, zero `Backward.kt` changes, zero rule changes: the §0.4.366
axis-aware SumRule/MeanRule and the §0.4.84 BroadcastRule were already
waiting, exactly as the amendment promised.

- **BatchNorm** (DiffKT `BatchNormTraining` V2, F0 §4.0.4, F0 landmine
  8 honoured): NO fused BATCHNORM kind — the training forward desugars
  `μ = Σx/n`, `σ² = Σx²/n − μ²` (biased), `out = γ·(x − μ)/√(σ² +
  1e-5) + β` into SUM(axes)/DIV/SUB/MUL/SQRT/ADD + the axis broadcast,
  so the transform differentiates THROUGH the batch statistics (the
  full three-term batch-norm gradient falls out of the registry rules).
  Channel axis is AXIS 1 (NCHW, the F4 layout decision; rank-2 `[N, C]`
  is the Dense-stack form), reduction over all other axes. `EPS = 1e-5f`
  is DiffKT's literal, not a knob. Running state is the functional V2
  triple `BatchNormStats(runningN, runningSum, runningSumOfSquares)`
  (decision 6 — `batchNormTrainV2`'s own pure signature), EMA'd through
  F3's `momentumUpdated` (momentum weights the NEW stat, default 0.1f —
  F0 landmine 3): `trainForward` returns the `(output, updatedStats)`
  pair, `updatedStats(batch)` is the host-side step for a held
  `CapturedStep` (stats never enter the trace — they are not
  differentiable state), `withStats` rebuilds. `inferenceMode()` is
  DiffKT's `freezeBatchNorm` — `m = γ/√(σ²+ε)`, `b = β − m·μ` off the
  RUNNING stats — frozen to the new `ChannelAffine(m, b)` layer
  (trainable, like the frozen DiffKT `AffineTransform`'s tensors).
- **Dropout** (F0 §4.0.4): inverted dropout, DiffKT's own comparison —
  `mask[i] = u[i] > p ? 1/(1−p) : 0f` over `uniformFloats(key, n)`,
  eval (`inferenceMode()`) = the new `IdentityLayer`. The mask enters
  the trace as a CONSTANT leaf (`isConstant = true` — the D2 precedent
  and F0 landmine 9), the forward is one existing MUL, and the key is
  part of the layer VALUE (`withKey` re-keys per step — `Layer.forward`
  has no key slot; DiffKT's per-call `random` is the same discipline
  one constructor earlier).

**Design decisions, with rejections.**

- **`ChannelAffine` as its own layer.** REJECTED: widening F1's
  same-shape `AffineTransform` with implicit broadcasting — the trace
  spelling would silently depend on input rank; the explicit
  `broadcast_dimensions = [1]` op is the honest captured-graph form.
- **`inferenceMode()` refuses on fresh stats** (`runningN = 0` → 0/0)
  rather than freezing NaNs — DiffKT would NaN silently; recorded as a
  deliberate divergence in loudness, not maths.
- **`forward` = the TRAINING forward** (what DiffKT's mutating `invoke`
  computes), identical to `trainForward(...).output`; inside a
  Sequential fold the stats update is discarded (a fold cannot return
  per-layer state). Stats threading through containers is a NAMED
  DEFERRAL; standalone use calls `updatedStats` per batch.
- **Stats accumulation order pinned**: `updatedStats`'s host loop folds
  in flat row-major order — the same sequence the traced `sum(axes)`
  spelling and the interpreter's SUM arm use, so the three surfaces
  never drift by a bit.

**Oracle story.** The trace-vs-hand-built-DXIR oracle on all three new
spellings at once (`Σ(broadcastAlong(x.sum([0,2,3])·w + x.mean([0,2,3]),
1) ⊙ x)` built through the Tracer AND through `DxirBuilder` with the
interpreter's exact attrs, both through the SAME transform, elementwise
`==`). BatchNorm gradients (γ, β, x — statistics terms included) against
an independent DOUBLE-precision spelling of the analytic formulas
`dx = (γ/σ)(g − ḡ − x̂·mean(g·x̂))` at 1e-4 (the F3 precedent — ε=1e-5
keeps √ off the dyadic grid); the rank-2 form pins `dβ = n` exactly and
the zero-mean cancellation (`dγ, dx → 0`) at tolerance. Running stats:
two training calls at momentum 0.25 (dyadic), every EMA step
`==`-exact, through BOTH surfaces (`trainForward`'s pair and
`updatedStats`), purity of the original pinned. The freeze: m/b against
the double reference at 1e-6, the frozen forward's loss at 1e-4, and
`ChannelAffine`'s own gradients hand-exact `==` (`dm = Σ_channel x` —
the `broadcast_dimensions = [1]` reverse — `db = n`, `dx = m` splat).
Dropout: mask bit-exact vs the raw `uniformFloats` threshold
(`assertContentEquals`), train-mode input gradient == the mask exactly
(even at the non-dyadic scale 4/3 — upstream ≡ 1), loss = the
flat-order f32 accumulation, eval identity and `p = 0` both exact,
determinism per key and `withKey` re-keying pinned against the stream.

**Deferred, by name:** stats threading through Sequential containers
(above); `BatchNormTrainingV1` (running mean/var EMAs with Bessel's
correction — DiffKT ships V2 as the default; V1 waits for a consumer);
F0's standing deferral list unchanged.

### F6 — §0.4.442: Embedding and EmbeddingBag — the index rides I32, the scatter stays dense

**What landed.** The gap table's F6 TRACE spellings in `TracedOps.kt` —
`embedding(indices, paddingIndex)` (EMBEDDING over (table, indices), the
optional `padding_index` attr recorded only when ≥ 0, forward via the
§0.4.409 `:core` host twins at index rank 1 and 2), `slice(start, end,
axis)` (SLICE, the interpreter's full-rank
`start_indices`/`limit_indices`/`strides` attr spelling, stride-1 —
the only form SliceRule v1 differentiates) and `concat(parts, axis)`
(CONCAT, attr `dimension`) — plus `EmbeddingLayers.kt` in `:nn`. Zero
gradient math, zero `Backward.kt` changes, zero rule changes: the
§0.4.370/409 EmbeddingRule (fused EMBEDDING_GRAD dense scatter — the
ratified §2.7 dense v1) and SliceRule/ConcatRule were already waiting.

- **The dtype plumbing the slice named as in-scope** (F0 §4.0.5 revised
  for the compiler route): `TapeEntry` gains `dtype: DType = F32` — the
  value cache stays a FloatArray for every dtype (the interpreter's own
  float-encoded environment; its EMBEDDING arms read indices via
  `toInt()`), and what the dtype governs is the `DxirType` that
  `Tape.toDxirFunction` stamps on the reproduced node. An I32 index
  leaf (`traceLeafI32`, float-encoding asserted exact below 2²⁴ —
  landmine 4 made loud) therefore comes out an I32-TYPED `DxirParam`,
  the interpreter's EMBEDDING arm accepts the captured graph, and
  `DxirReverseTransform` returns the §0.4.419 ZEROS_LIKE structural
  zero for it — non-differentiable by DTYPE, no `isConstant` flag, no
  float-encoded-constant workaround (the value-tape's F0 §4.0.5 note is
  MOOT on the compiler route). `captureN` takes `List<DTensor<*, *>>`
  and dispatches F32/I32 leaves; `capture`/`CapturedStep.run` bind I32
  inputs float-encoded (re-asserting the 2²⁴ cap per re-bind — new
  step, new indices); RESHAPE and SLICE are dtype-preserving (the
  rank-2 index flatten stays I32 in the captured graph).
- **`Embedding`** (DiffKT `Embedding(numEmbeddings, embeddingSize,
  random)`, table `gaussian()` under the F2 key discipline): `forward`
  requires an I32 input tracer — trace-time, mirroring DiffKT's
  `IllegalArgumentException` on a non-IntTensor — and gathers rank-1
  `[N] → [N, D]` or rank-2 `[B, N] → [B, N, D]`. `paddingIndex` is the
  RECORDED Tlaloc extra (absent from DiffKT's layer, present in the
  `:core` §0.4.409 substrate): exact-zero rows forward, exactly-zero
  table-row gradient (the attr rides the primal onto EMBEDDING_GRAD).
- **`EmbeddingBag`** (DiffKT's exact source shape): `forwardBags(
  indices, bagOffsets, params)` = flatten → embed → per bag
  `slice(start, end, 0)` → `Reduction.reduce` → `concat`, offsets
  linear into the flattened indices, LAST bag runs to the end.
  `Reduction` is DiffKT's sealed companion class with the one member
  DiffKT implements: `Sum` = `sum(0, keepDims = true)`, spelled
  SUM(axes=[0]) + keepdims RESHAPE. The single-input `Layer.forward`
  REFUSES exactly like DiffKT's vararg `invoke` throws;
  `withOffsets(bagOffsets)` is the Layer view for ONE bag structure —
  honest under the F1 caching contract, since offsets are SLICE attrs
  and a different bag structure is a different captured graph anyway.

**Design decisions, with rejections.**

- **Indices as I32-typed PARAMS, never baked constants.** REJECTED:
  recording indices as float-encoded `isConstant` leaves (the
  value-tape's pre-amendment plan) — a constant bakes the batch into
  the graph, killing `CapturedStep` re-binding (pinned: one capture,
  two index batches, both hand-exact); and the I32 param is what makes
  the interpreter's EMBEDDING dtype require pass and the transform's
  §0.4.419 arm fire.
- **`bagOffsets` as a host IntArray, not a traced tensor.** DiffKT
  takes an IntTensor but reads it host-side to drive the slice loop —
  offsets shape the GRAPH (slice attrs), they are not data flowing
  through it. REJECTED: tracing them — a slice bound cannot be a
  runtime value in the captured form.
- **Empty bags refuse** (`start == end`): our SLICE spelling requires a
  positive extent; DiffKT's `slice` would admit the empty bag and Sum
  would produce a zero row. Recorded NARROWING, loud not silent.
- **`hostValues` dispatch in `Training.kt`** rather than widening
  `hostF32` — the F32/I32 encoding decision is the capture layer's, not
  the tensor substrate's.

**Oracle story.** All `==`, quarter grid, through `valueAndGradients`:
collision-count gradients (`dTable[v,:] = count(v)` under `Σout`) and
the weighted form (`dTable[v,:] = Σ_{idx[p]=v} c[p,:]` under
`Σ(out⊙c)` — discriminating positions the count cannot); paddingIndex
rows exactly zero BOTH directions (forward rows through the captured
primal — the interpreter reading `padding_index` off the graph — and
the padded row's gradient); the trace-vs-hand-built-DXIR oracle TWICE
(EMBEDDING→SUM with the hand-built I32 param, and SLICE×2→CONCAT→MUL→
SUM — both routes through the SAME transform, elementwise equal, both
against hand values), with the ROUTE PIN asserting the captured index
param's dtype IS I32; EmbeddingBag Sum hand-exact (bags [0,2),[2,3),
[3,end) with per-bag upstream rows scattering into the right table
rows); the rank-2 flatten (dtype-preserving RESHAPE witnessed by the
rank-2-shaped structural-zero input gradient); the full
Embedding→Flatten→Dense Sequential chain (table gradient arriving
THROUGH the dense weights, `dTable[v,:]` = scattered `wᵀ` rows);
re-bind on a held `CapturedStep`; companion draws bit-exact under the
key discipline; the 2²⁴ cap, the non-I32 refuse, and the DiffKT-parity
Layer-form EmbeddingBag refuse all pinned loud.

**Deferred, by name — ROW-SPARSE embedding gradients (§2.7 stands,
dense v1 ratified), with the worked design recorded:** a post-hoc
dense→CSR conversion of the transform's output is NOT the feature (it
pays the dense materialisation the option exists to avoid) — the
honest form is a `rowSparseGradients` lane where (1) the per-step
sparsity pattern (sorted unique touched vocab rows + per-position row
map) is computed HOST-SIDE from the indices at bind time in
`CapturedStep.run` — indices are runtime values, so the pattern is
per-step and never belongs in the captured graph, exactly like
SPARSE_MATMUL's component-operand convention (§0.4.418) — and (2) the
graph emits a compact `[touched, D]` scatter (an EMBEDDING_GRAD
variant taking the position→compact-row map as an I32 operand; the
existing EMBEDDING_GRAD walk with a remapped vocab extent is the
implementation) whose result `run` wraps with the host pattern into
the Phase-E CSR `SparseTensor` over `[V, D]`; certified against the
dense gradient's nonzero rows, plus a sparse SGD row-update on the
optimizer side. Blocked on nothing, sized ~1 §, taken when a consumer
(the F8 training loop at real vocab sizes) asks. Also deferred:
EmbeddingBag Mean/Max (unimplemented in DiffKT — parity is Sum; F0's
standing item), empty bags (the recorded narrowing above), and F0's
standing deferral list unchanged.

### F7 — §0.4.443: GRU — the unroll is the graph, and BPTT is just the transform

**What landed.** `Gru.kt` in `:nn` — the gap table's F7 row held to the
letter: ZERO new TRACE spellings, zero `Backward.kt` changes, zero rule
changes, zero gradient math. The cell is Dense reuse plus the existing
elementwise/CONCAT/SLICE/RESHAPE spellings; the sequence loop is a plain
Kotlin `for` building the trace; the captured `DxirFunction` is the
UNROLLED graph and `DxirReverseTransform` differentiates it like any
other — backpropagation through time is not a feature of this layer, it
is what reverse-mode ON the unrolled graph already is.

- **The gate equations, exactly as F0 §4.0.4 read them off DiffKT's
  source.** Input `[batch, seq, numInputs]` (batch axis 0, seq axis 1);
  per step DiffKT's own `slice(t, t+1, seqAxis).squeeze(seqAxis)`
  (SLICE + the squeeze RESHAPE); default variant (`LinearAfterResetGru`)
  `xh = concat(x, h, 1)`, `u = xh2u(xh)`, `r = xh2r(xh)`,
  `n = xh2n(concat(x, r·h, 1))`, `h' = (1−u)·n + u·h`, the gates whole
  `Dense` layers with their own biases and baked-in activations
  (σ/σ/tanh); `linearBeforeReset` variant
  `n = tanh(x2n(x) + r·h2n(h))` with Identity-activated `x2n`/`h2n`
  carrying their own biases, same `u`/`r`/`h'`. `AccType.Fold` returns
  the last `[batch, numHidden]`; `AccType.AccMap` concats the per-step
  outputs (each unsqueezed at the seq axis) into `[batch, seq,
  numHidden]`.
- **The initial state is a CONSTANT zeros leaf** `[batch, numHidden]`
  (`isConstant = true`, the F5 mask precedent) — the gap table's own
  recommendation: DiffKT's `initialState` is non-trainable zeros
  `expand`ed to batch, and a zeros constant carries no gradient, so the
  expand needs no traced broadcast. REJECTED: tracing it as a parameter
  (F0's parameter table lists only the gate Denses).
- Parameter keys are gate-prefixed Dense keys in declaration order
  (`xh2u.w`, `xh2u.b`, `xh2r.*`, then `xh2n.*` or `x2n.*`/`h2n.*`) —
  the Sequential-style prefix delegation, `withParameters` grouping by
  gate and delegating to `Dense.withParameters`.
- The companion is the DiffKT surface `GRU(numInputs, numHidden, key,
  acc, linearBeforeReset)`; key discipline lifted to a composite layer:
  ONE `split(3)` (or `split(4)`) into child keys per GATE in declaration
  order, each gate Dense applying the F2 per-parameter `split(2)` below
  that.

**Design decisions, with rejections.**

- **One concrete class covering both candidate variants** (nullable
  gate slots, private primary constructor, two public tensor-level
  constructors). REJECTED: mirroring DiffKT's `RecurrentBase` two-
  subclass hierarchy — under the self-typed `Trainable<T>` a
  variant-choosing factory must return an erased `TrainableLayer<*>`,
  which `capture`'s `M : Layer, M : Trainable<M>` bounds cannot hold.
- **The tensor-level constructors REQUIRE DiffKT's activations**
  (σ/σ/tanh; Identity/Identity for the before-reset pair) — a loud
  parity guard so the audited gate equations cannot be silently
  re-plumbed; pinned by refusal tests.
- **The multi-input `Layer` question F1 parked: not needed.** GRU is
  single-input like DiffKT's own (`getSingleInput`); the F1 deferral
  closes as "no consumer materialised".

**Oracle story.** Two independent oracles per the slice, both on a
2-unit cell (numIn = numHidden = 2, batch 1, quarter-grid weights):
FORWARD losses pinned at 1e-5 against a JUnit-side Float replication
whose same-JVM-ops claim is stated precisely in the test KDoc — it
mirrors the `DxirInterpreter` arms operation for operation (MATMUL's
f32 accumulation in (k, n) order from a zero accumulator, bias ADD
after the matmul, SIGMOID/TANH computed in Double and narrowed — the
arms' own spelling, and what `valueAndGradients`' loss actually is
under `includeForward`), so agreement is expected to the bit and the
1e-5 pin is the slice's contract, not the observed slack. GRADIENTS
pinned at 1e-5 against central finite differences (step 1e-5,
truncation O(1e-10)) over an INDEPENDENT double-precision replication
of the equations — every parameter element AND every input element, at
seq 1 and seq 3 (Fold), seq 3 (AccMap — the concat path), and seq 2
for `linearBeforeReset`. The recorded single-step structural fact: with
h₀ = 0 the reset gate reaches the loss only through `r·h₀`, so its
gradients are EXACT (±)0 — pinned `== 0f`, with the 3-step cert pinning
the gate live (|grad| > 1e-4) once h ≠ 0. The BPTT sanity: Fold's loss
reads only h₃ yet the step-0 input slice carries nonzero gradient
(shape `[1, 3, 2]` pinned too). Companion draws bit-exact against
freshly-drawn Denses under the gatewise split for BOTH variants;
`withParameters` functional (original untouched, pass-through pinned)
and every guard refuses loudly. The trace-vs-hand-built-DXIR spot check
was NOT taken ("if cheap" — the unrolled step is ~20 hand-built ops,
and every constituent spelling already carries its own F2/F5/F6
trace-vs-DXIR pin through the same transform).

**Deferred, by name:** none new. F1's "multi-input Layer surface if GRU
wants it" deferral CLOSES unneeded (above); DiffKT's
`LinearBeforeResetGRU` DNNL hookup stays never-existed-upstream; F0's
standing deferral list unchanged.

### F8 — §0.4.444: the end-to-end training certification, the compiled GPU step, and the close-out

**What landed.** Three certifications and zero new library code — Phase F's
final slice is deliberately all oracle: the F1–F7 stack already contained
everything training needs, and F8 proves it end to end on a fixed
threefry-keyed synthetic task (x ∈ [−1,1)^{16×4} off
`fromSeed(1234)`'s first child stream, target `y = x₀·x₁ + 0.5·x₂ −
0.25·x₃` — the product term keeps the hidden layer honest), MSE loss,
bit-deterministic forever.

- **(1)+(4) The host training certs** (`:nn` `EndToEndTrainingTest`): the
  MLP `Dense(4,8) → Relu → Dense(8,1)` captures ONCE (the F1 caching
  contract exercised — the trained model re-binds through the same
  `CapturedStep` all 200 steps) and Adam(0.05) collapses the loss
  0.30533 → 0.0012239 (pinned < 2e-3 absolute AND < initial/20;
  "monotone-ish" made precise as strictly-decreasing non-overlapping
  25-step window means). The step-0 FD spot check: central differences
  THROUGH the captured loss itself (perturbed `withParameters` +
  `CapturedStep.run`) on 3 elements spanning first-layer weight/bias and
  output-layer weight, at 2e-3 + 2% (h = 1e-2; the pin's error budget is
  derived in the KDoc, not tuned). The conv net `Conv2d([2,1,2,2]) → Relu
  → Flatten → Dense(18,1)` trains 0.04120 → 0.000179 over 40 Adam steps
  (pinned < initial/10 + 10-step window means).
- **(2) THE COMPILED GPU TRAINING STEP** (`:benchmarks`
  `NnMlpGpuTrainingTest`, PJRT/CUDA self-skip ladder) — the amended
  decision-3 payoff: the F1-captured gradient `DxirFunction` — the SAME
  object the host lane interprets — goes `toStablehlo` → PJRT compile →
  GB10 execution via `PjrtSession.runOn`. One step's six outputs (loss +
  input gradient + 4 parameter gradients) pin elementwise against
  `DxirInterpreter` at 1e-4, observed max|diff| 3.8e-5 (f32
  reduction-order noise — the §0.4.292 LlamaDecoder backward cert's own
  band). Then FIVE training steps run with gradients entirely on the GPU
  lane (host-side F3 Adam between them, per the §2.9 scope) against the
  interpreter-lane twin: per-step losses within 1e-4, final parameters
  within 1e-3 — observed 4.6e-5 — and `session.cacheSize == 1` pins that
  all five dispatches reuse ONE compiled executable (params are
  `DxirParam`s, never baked constants — the §0.4.307 amortization
  working exactly as the F1 contract said it would). **No emission gap
  was hit**: every op in the captured MLP gradient graph has an emitter
  arm, so the slice's record-and-pin-host contingency was not needed.
- **(3) PyTorch convergence parity** (`:benchmarks`
  `NnMlpVsPytorchTrainingTest` + `harness/python/run_pytorch_nn_train.py`,
  the §0.4.289 npy-export/subprocess/JSON pattern, torch venv self-skip):
  SAME initial weights (exported — threefry ≠ torch RNG), same data, same
  MSE, 50 identical Adam(0.02) steps (both sides Kingma–Ba bias-corrected
  — the F0 finding that DiffKT's Adam is a TODO placeholder is exactly
  why PyTorch is the oracle). OBSERVED: step-0 rel diff 9.8e-8, steps 0–9
  max 5.9e-7, final 9.3e-8 — the two f32 trajectories track each other
  ~1e-7 relative for the whole run, far tighter than the slice's
  float-order-divergence contingency anticipated. Pins sit 2–4 orders
  above observed (1e-5 / 1e-4 / 1e-2 + both finals < 0.05) to absorb
  torch-version drift; the observed numbers are recorded in the test KDoc
  so a regression to "merely within tolerance" stays visible.

**Design decisions, with rejections.**

- **Targets enter the trace as a CONSTANT leaf** (`y.constant(targets,
  dims)`) — the fixed-task form: full-batch training re-binds x and the
  parameters, the target rides the graph. REJECTED for v1: the target as
  a second capture INPUT — `capture` is single-input (F1's DiffKT
  `LayerSingleInput` fold), and widening it belongs to the mini-batch
  story (deferral below), not to this certification.
- **The GPU lane unpacks outputs host-side in the test**, mirroring
  `CapturedStep.run`'s indexing, rather than growing a
  `CapturedStep.runOn(session)` in `:nn`. REJECTED: an `:nn` →
  `:runtime-pjrt` dependency — the module boundary decision (execution is
  a BACKEND CHOICE the caller makes) survives its first consumer; the
  ~10-line unpacking is the certified recipe for anyone who wants the
  convenience wrapper later.
- **The F1 caching-contract question resolved**: the "structure-keyed
  cache in front of the compiled-GPU lane" the F1 record promised IS
  `PjrtSession`'s MLIR-text-keyed executable cache (structurally
  identical captures produce identical MLIR and hit one slot — pinned by
  `cacheSize == 1`), plus the caller-held `CapturedStep`. An automatic
  :nn-level capture cache keyed on model structure was NOT built —
  named deferral, taken when a consumer holds many structures at once.

**Oracle story.** Four independent oracles converge on the same stack:
hand-pinned convergence bounds on a bit-deterministic task (host), the FD
check (derivative-free, through the captured loss), the interpreter
(certified by the whole parity arc) against XLA-on-GB10 elementwise, and
PyTorch (a foreign AD + foreign runtime) tracking the full 50-step
trajectory from shared bits. The GPU and PyTorch certs self-skip cleanly
on hosts missing CUDA or torch, and say what to install.

**Phase F consolidated deferral list** (the close-out sweep — every item
named across F0–F8, none silent):

1. Row-sparse embedding gradients — dense v1 ratified (§2.7); the worked
   CSR design is recorded in the F6 entry, sized ~1 §, taken when a
   consumer at real vocab sizes asks.
2. Grouped conv at the layer/trace level — waits on the §0.4.429
   grouped-conv host-twin tail (`feature_group_count` lives in the IR).
3. Rhs-dilated ("à-trous") conv at the trace level — IR + rule support
   it; no DiffKT surface asks (F4).
4. Rank-3 Dense input — DiffKT accepts rank ≥ 2; traced matmul is
   rank-2/3 (F2).
5. Mini-batch training with per-step targets — needs the multi-input
   capture surface (target as a re-bindable input instead of a baked
   constant) or per-batch re-capture; the F8 certs are full-batch
   fixed-task by design.
6. Automatic structure-keyed capture cache in `:nn` — resolved to
   caller-held `CapturedStep` + `PjrtSession`'s MLIR-keyed executable
   cache (above); an automatic layer waits for a multi-structure
   consumer.
7. BatchNorm stats threading through `Sequential` (F5) and
   `BatchNormTrainingV1` (F5 — V2 is DiffKT's default; V1 waits for a
   consumer).
8. EmbeddingBag Mean/Max reductions (DiffKT ships only Sum — parity IS
   Sum) and empty bags (recorded narrowing: we refuse loudly, F6).
9. `store`/`load` checkpointing beyond tensor round-trip helpers, data
   loaders, GPU-RESIDENT optimizer state (the F8 GPU lane is
   gradients-on-device + host optimizer, the ratified §2.9 scope).
10. DiffKT's `LinearBeforeResetGRU` DNNL hookup — never existed upstream.

**End state.** Phase F is COMPLETE per the ratified + amended scope:
every DiffKT `model/` abstraction, layer, initializer and optimizer has
its Tlaloc form in `:nn` (34 upstream files' semantics, audited and
matched or recorded-divergent by name), the AD route is the compiler
stack end to end, and the training loop runs certified on host and GPU
with PyTorch-parity evidence. With Phase F closed, the DiffKT parity
book of work has NO remaining open phase — see DIFFKT_PARITY_PLAN.md's
end-state header.
