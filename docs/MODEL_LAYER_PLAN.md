# Phase F — the model/optimizer layer (`:nn`)

**Status: RATIFIED (Pedro, 2026-09-21) — GO on the scope below.** This
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
3. **The AD route is the `:autograd` runtime tape.** Models carry
   arbitrarily many parameter tensors; the `grad {}` intrinsics are
   fixed-arity by design (1–4 params) and stay the compiler path for
   research code. The tape's `applyRegistryRule` bridge already routes
   backward through the REAL `VjpRegistry` via a scratch DxirBuilder +
   env walk — so every op with a VjpRule + interpreter arm can join the
   tape by adding a `Tracer` forward spelling. That is the whole
   extension mechanism; no new gradient math is written in `:nn`.
   REJECTED: packing parameters through fixed-arity `grad {}` (arity
   ceiling, unnatural flattening); a parallel NN-specific autograd
   (would fork the rule set the whole book of work certified).
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
| F1 | `:nn` module + `Trainable`/`Sequential` substrate + tape-backed `valueAndGradients(model, loss)` contract | 1 § |
| F2 | Dense, Flatten, ReluLayer, AffineTransform + FanMode initializers (threefry, bit-deterministic) | 1 § |
| F3 | Optimizers: FixedLearningRate, SGD, Momentum, RMSprop, Adam — pure `(params, grads, state)` functions, hand-stepped oracles | 1 § |
| F4 | Conv stack: Conv2d (+SamePadding), MaxPool2d, AvgPool2d — Tracer spellings over the existing rules/interpreter arms | 1 § |
| F5 | BatchNorm (functional stats) + Dropout (threefry-keyed) | 1 § |
| F6 | Embedding + EmbeddingBag; row-sparse gradient disposition per §2.7 | 1 § |
| F7 | GRU — library-level unroll over time steps (the tape handles unrolled graphs natively) | 1 § |
| F8 | End-to-end training certification: MLP + small conv net to convergence vs PyTorch reference; Phase F close-out sweep of this doc + the parity plan | 1–2 § |

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
