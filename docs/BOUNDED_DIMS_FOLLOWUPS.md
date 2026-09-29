# Bounded dimensions: open issues

Known gaps in bounded dimensions (`DimBound`, `Bounded<B>`, `boundedProgram`,
`BoundedProgramExport`, `tlaloc_bounded.py`, the Triton backend's bounded mode), to
address later. The design is in [design/bounded-dims.md](design/bounded-dims.md) and the
history of the work in [work-log/bounded-dims-progress.md](work-log/bounded-dims-progress.md).

## 1. Bounded training in `:nn`'s `capture`

`boundedProgram` works on raw tracers. A model built from `:nn` layers and trained with
`capture(model, inputs) { ... }` compiles one step per input shape, as before: nothing in
`:nn` knows about bounds. This is the largest gap between what the feature promises
("training workloads with varying sequence lengths") and what a user of the layers gets.

- To do: a bounded variant of `capture` (input specs with bounded axes, the valid mask
  reachable from the loss, one captured step per bucket), running through `PjrtSession`
  like `BoundedExecutor`. The masking of attention (`MultiHeadAttention`) and of the losses
  over padded positions has to come with it.
- Starting point: `PjrtBoundedTrainingTest`, which does this by hand for a masked MSE.

## 2. One compile per bound: bounded dynamism inside the program

Buckets cost one compile per bucket and up to 2x the work per request (13 % slower
execution in the mixed-length measurement). The spike in the design doc found that a
program with static inputs and `stablehlo.set_dimension_size` inside runs on XLA GPU and
sums exactly the first `n` elements; a `#stablehlo.bounds` parameter compiles but the PJRT
C API cannot create a buffer for it.

- To do: define which DXIR ops honour a dynamic size and what a padded region holds after
  each (XLA's dynamic padder semantics), implement that in the interpreter, check it on
  PJRT and IREE op by op, and emit one program per bound. Buckets stay the fallback.

## 3. Triton bounded mode: batching, several outputs, GPU memory

The bounded mode runs one request per execution, through host memory, with one output.
A rank-0 input or output is refused by `TritonModelRepository` (Triton needs a reshape
the mode does not read).

- To do, when a serving use needs it: group concurrent requests of one bucket into one
  execution (a batch axis is itself a bounded axis), read inputs in place from CUDA shared
  memory as the generic mode does, allow several outputs (a training step), and accept
  `dims: [1]` with `reshape: { shape: [ ] }` for scalars.
- Not run: gRPC with `tritonclient` only ran from a scratch venv; this machine has no
  `tritonclient` installed (see the progress log for the pinned `cuda-python` 12.9).

## 4. Compile-time checks: bounds from other modules, sizes inside `grad {}`

`BOUNDED_DIM_EXCEEDED` reads `max` from the delegating constructor call of a bound
declared in the module being compiled. A bound from a compiled dependency has no
constructor arguments in FIR, so its sizes are checked at run time only. Sizes inside a
`grad {}` lambda are `-1` in the lowering, so only constant sizes at tensor-building calls
are checked.

- To do: carry the bound in class metadata (an annotation with the value, checked
  against the constructor argument) so dependencies are checked too; consider carrying
  bounds on `DxirType` so the lowering can reject a `concat` or `reshape` that provably
  leaves the bound.

## 5. Padding is checked at bucket edges, not at every size

The exporter compares padded and exact results in the interpreter at 1 and at the first
and last size of every bucket, on seeded random inputs (integer inputs are 0). A program
whose padding error appears only at some interior size, or only for particular integer
inputs, passes.

- To do: optionally check every size for small bounds, and let an I32 spec carry a value
  range so index inputs are exercised.

## 6. `BoundedProgram` thread-safety and trace retention

The trace cache is a plain map: not safe from several threads, and exact-size runs keep
one trace per distinct size for the program's lifetime. `:autograd` is common code with no
lock available there.

- To do: a JVM-side synchronized wrapper or a bounded cache, or document that serving and
  training loops use bucketed runs only (one trace per bucket).

## 7. Three readers of one manifest schema

`BoundedManifest.kt`, `tlaloc_bounded.validate` and `bounded_mode.cc ReadManifest` each
implement the `tlaloc-bounded-v1` rules. The review of the branch found them out of step
once (mask shapes, required keys, the recorded padding check).

- To do: a shared conformance fixture set (valid and invalid manifests) that all three
  readers are tested against.

## 8. The language-model exporter's ladders

`HfServingExport` already buckets by batch and context (`DecodeBucketPolicy`); its ladders
are not derived from a `DimBound`. Low value on its own; worth doing if the two export
paths are unified.
