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

(Filled per slice as the workflow lands them.)
