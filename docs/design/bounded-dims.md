# Bounded symbolic dimensions

Status: design for branch `feat/bounded-dims`. Progress and decisions taken while
building it are in [../work-log/bounded-dims-progress.md](../work-log/bounded-dims-progress.md).

## Goal

A user declares an axis whose size varies at run time but never exceeds a bound,
for example a sequence axis of at most 4,096 positions. Tlaloc checks the bound
where it can at compile time, runs the program at any size up to the bound, and
exports it so that a server handles every size up to the bound without compiling
per request.

## What exists today

### Types

- A dimension in the type is a `ShapeAtom`: `Sym`, `Lit<N>`, `Mul`, `Add`, or
  `Named<IndexName, atom>` (`core/.../Shape.kt`). No atom carries a size. `Lit<Int>`
  is a marker class, not the number. Sizes live only at run time, in
  `DTensor.dims: IntArray`.
- The K2 plugin (`FirLambdaToDxirLowering.resolveParamType`) keeps the rank and
  the simple names of `Named` index objects, and writes every dimension as `-1`.
  Positional atoms are dropped. Gradient code synthesized by the plugin runs
  `io.tlaloc.core.ops` host functions that read `dims` at run time, so a
  `grad { }` function already runs at any size.
- Named-axis mismatches in `contract` are rejected by Kotlin's own overload
  resolution (a shared type variable), and also by the plugin (`NAMED_INDEX_MISMATCH`)
  inside `grad { }`. Plugin diagnostics are reported on the source of the whole
  intrinsic call. `TENSOR_SHAPE_MISMATCH` cannot fire for plugin-lowered code,
  because every dimension there is `-1`.
- Negative-compilation tests exist: each test writes a `Main.kt`, runs
  `K2JVMCompiler` in-process with the plugin jar, and asserts on the collected
  diagnostics and their positions (`TlalocCompileTimeErrorTest`,
  `DiagnosticSourcePositionTest`).

### IR, StableHLO, interpreter

- `DxirType(dtype, dims, axisNames)`. `-1` is the only non-concrete dimension;
  runtime-extent ops (`SUM_TO`, `PAD_TO`, `BROADCAST_LIKE`, ...) read the size from a
  template operand where a gradient rule needs it.
- The StableHLO emitter writes static shapes only (`TLALOC_EMIT_CONTRACT.md`:
  "`?` dynamic dims are out of scope"). Nothing in the repository emits `?`,
  `#stablehlo.bounds`, `set_dimension_size` or `get_dimension_size`.
- `DxirInterpreter.evalFunction` needs concrete dimensions and checks each input's
  length against its parameter type.
- Concrete DXIR for a user program comes from the capture API
  (`capture`, `captureN`), which traces the program on example tensors and records
  their dimensions.

### Serving: how varying lengths and batch sizes are handled

Every compiled program has static shapes. Variation is handled by **buckets**:

- **Language models** (`HfServingExport`, `ServingArtifactWriter`, manifest
  `tlaloc-serving-v1/v2/v3`). `DecodeBucketPolicy` builds a batch ladder (powers
  of two up to `maxBatch`, `maxBatch` appended) and a context ladder (powers of
  two from `minContext` to `maxContext`, each rounded up to a multiple of
  `blockSize`, `maxContext` appended). Each (batch, context) point is one decode
  entry and, up to `prefillMaxBatch`, one prefill entry. A request over the top
  bucket is refused, not clamped.
  - Decode pads missing rows with token 0, position 0, block 0, `seqLen` 1 and slot
    -1 (the KV write is dropped); rows are bit-identical across buckets in the
    interpreter (`DecodeGraphPaddingInvariantTest`).
  - Prefill rows are right-aligned: padding tokens first (slot -1), each token with
    its own position; the causal length is `position + 1`, so padding is never read.
  - Selection: the Python runtime (`tlaloc_serve.select_bucket`) rounds batch and
    context up to ladder points and does an exact lookup; the Triton backend
    (`sequence_mode.cc`) takes the cheapest entry by `batch * context` that covers
    the request; the vLLM plugin goes through `tlaloc_serve`.
  - The CLI default (`ExportLlamaServingArtifact`) is a single context bucket
    unless a context ladder is given.
- **Other StableHLO under Triton** (`tlaloc_backend.cc`, no manifest). The
  `artifact` parameter lists one file per shape; a request runs on the file whose
  input shapes it matches exactly, or, with `max_batch_size > 0`, on the smallest
  batch that holds its rows, zero-padded along dim 0. No padding along any other
  dimension. There is no Kotlin exporter for these families; the example files
  are written by `ExportTritonExamples`.
- **User programs** (capture API, `runOnPjrt`): one program per concrete shape.
  `PjrtSession` caches executables by the StableHLO text, so every new length is a
  new compile.

Compile counts are observable: `tlaloc_serve.ServingArtifact.compile_count` (Python,
lazy per entry), `PjrtSession.cacheSize` (JVM), and the per-body "compiled ... in N
ms" log lines of the Triton backend (eager at load).

## Strategy: buckets, not dynamic shapes

### Bounded dynamic StableHLO: what was checked

A spike against the XLA CUDA PJRT plugin on the GB10 (the plugin the three runtimes
use), through the repository's ctypes binding (`tlaloc_pjrt.py`):

| Program | Compile | Run |
|---|---|---|
| `tensor<?xf32>` parameter (unbounded) | refused: "Unbounded dynamism is disabled" | — |
| `tensor<?xf32, #stablehlo.bounds<8>>` parameter | compiles | refused for every host buffer: the executable expects a 36-byte `f32[<=8]` (data plus a size word); `PJRT_Client_BufferFromHostBuffer` creates static `f32[n]` buffers only, and the C API has no call that creates a dynamic-shaped buffer |
| static `tensor<8xf32>` + `tensor<i32>` parameters, `set_dimension_size` inside, `exp`, sum | compiles | sums exactly the first `n` elements for n = 1, 3, 8 (values within f32 rounding of `n·e^0.5`) |

So bounded dynamism at the program boundary does not work through the PJRT C API
that the Python runtime, the vLLM plugin and the Triton backend all use. Dynamism
inside a program with static inputs does run on XLA GPU, for the one reduction
checked. Using it would mean one compile per bound instead of one per bucket, but it
needs, before it can be relied on: interpreter semantics that match XLA's dynamic
padder for every op Tlaloc emits (what the padded region holds after each op, which
ops honour the dynamic size), the same checks on IREE, and a policy for what an
output's padded region contains. None of that is verified, so it is not used in
this run. It is the obvious follow-up (see "Not in this run").

### Decision

Export one static program per bucket. A runtime picks, for each bounded dimension,
the smallest bucket that holds the request, pads the inputs, runs, and slices the
outputs. Programs whose result depends on how many positions are real (a sum, a
softmax, an attention over the bounded axis) read a **valid mask** or **valid
length** input that the runtime fills. The exporter checks, in the interpreter, that
each bucket's padded result equals the exact-size result, and refuses the export
when it does not.

## Type-level API

In `:core`, package `io.tlaloc.core`, marked `@ExperimentalTlalocApi`:

```kotlin
abstract class DimBound(val max: Int)          // max >= 1
class Bounded<B : DimBound> : ShapeAtom        // an axis of size 1..B.max

object MaxSeq : DimBound(4096)                 // the user's bound

val x: DTensor<Rank2<Named<SeqLen, Bounded<MaxSeq>>, Named<Hidden, Sym>>, F32> = ...
```

- `Bounded<B>` is an ordinary `ShapeAtom`, so it goes anywhere an atom goes:
  positionally (`Rank1<Bounded<MaxSeq>>`) or inside `Named`.
- **The bound object is the dimension's identity.** Within one program, every axis
  typed `Bounded<MaxSeq>` has the same run-time size. Two axes that vary
  independently use two bound objects, even with the same `max`. This is what lets a
  runtime pick one bucket per bound, and it is checked at run time.
- Nothing existing changes: `Sym`, `Lit`, `Named`, the rank classes and the
  plugin's type mapping are untouched. The plugin already maps an unrecognised atom
  to a `-1` dimension, so `grad { }` over bounded types lowers as it does for `Sym`.
- Why an object and not a number in the type: Kotlin has no integer type
  parameters. A marker object with a constructor argument is readable by the plugin
  (the delegating constructor call is in the FIR of a source-declared object) and at
  run time (the object instance).

### Specs read from the type

Export and bucketed runs need each input's axes. They are read from the type, so the
type is the only place the bound is written:

```kotlin
val xSpec = specOf<Rank2<Named<SeqLen, Bounded<MaxSeq>>, Named<Hidden, Sym>>>(F32, 16)
//                                                                        fixed sizes ^
```

`specOf` (JVM, `:autograd`) is `inline reified`; it walks `typeOf<S>()`, takes each
`Bounded<B>` axis's `B` object instance, and consumes one fixed size per other axis.

## Compile-time checks

Kotlin's type checker already does part of it, because `DTensor`'s shape parameter is
invariant and `contract` and `matmul` share type variables between their operands:

- mixing `Bounded<MaxSeq>` with `Bounded<MaxOther>`, or with `Sym`, in `matmul` or a
  named `contract` is a type mismatch at the call's file, line and column. Tests pin
  this.
- The elementwise operators do not share one: `BroadcastOps.kt` declares
  `<S1, S2> DTensor<S1>.plus(DTensor<S2>): DTensor<Shape>` (and `minus`, `times`, `div`),
  so any two shapes type-check and broadcasting is checked at run time. The plugin
  checks these calls (below).

The plugin adds a FIR checker (`TlalocBoundedDimChecker`) for what the type checker
cannot see, because it involves numbers:

| Diagnostic | Where | When |
|---|---|---|
| `BOUNDED_DIM_EXCEEDED` (error) | a call whose result type has a `Bounded<B>` axis | the size argument for that axis is an integer constant greater than `B.max` or less than 1. Covers the `Tensors` factories (`f32Matrix`, `f32MatrixOf`, `f32Zeros`, `f32Tensor3`, ...) and `specOf`'s fixed sizes landing on a bounded axis |
| `BOUNDED_AXIS_MISMATCH` (error) | the broadcasting `plus`, `minus`, `times`, `div` of `io.tlaloc.core.ops` | two axes aligned from the right carry different bound objects. A bound against an unbounded atom is allowed: it may be a size-1 axis that broadcasts |
| `BOUNDED_DIM_INVALID` (error) | `object X : DimBound(n)` | `n` is an integer constant less than 1 |
| `BOUNDED_SPEC_ARITY` (error) | `specOf<S>(dtype, fixedSizes...)` | the number of fixed sizes is not the number of unbounded axes of `S`, or a constant fixed size is below 1 |

- Reported with `reporter.reportOn(expression.source, ...)`, the call's file, line and
  column, as the existing shape diagnostics are.
- The checker reads `B.max` from the delegating constructor call of a
  **source-declared** bound object. For a bound compiled in another module the
  number is not in FIR (constructor arguments are not part of class metadata), so
  the compile-time check is skipped for it, and the run-time check still applies.
  This is a stated limit, not a silent pass: the run-time check is what enforces the
  bound.
- A size that is not a compile-time constant is checked at run time only.

## Running bounded programs (interpreter)

`BoundedProgram` in `:autograd`:

```kotlin
val encoder = boundedProgram(
    name = "masked_mean",
    inputs = listOf(xSpec),
    output = specOf<Rank1<Named<Hidden, Sym>>>(F32, 16),
) { xs, ctx ->
    val x = xs[0]                                // [len or bucket, 16]
    val m = ctx.validMask(MaxSeq)                // [len or bucket], 1 real, 0 padding
    ...
}

encoder.run(listOf(x))                  // interpreter, exact size
encoder.runBucketed(listOf(x), ladders) // interpreter, padded to the bucket, then sliced
encoder.capture(mapOf(MaxSeq to 128))   // DXIR at one size (what export lowers)
```

- `run` checks the inputs against the specs (fixed axes equal, bounded axes in
  `1..max`, every axis of one bound the same size), traces the program at the exact
  sizes and evaluates it with `DxirInterpreter`. Traces are cached per size.
- `ctx.validMask(B)` and `ctx.validLength(B)` are extra parameters of the traced
  function (`F32 [size]` of ones and zeros, and `F32` scalar). At exact size the mask is
  all ones.
- `runBucketed` pads each input with zeros along its bounded axes to the bucket,
  fills the masks, runs the bucket's trace, and slices each output's bounded axes back
  to the real size.
- `checkPadding(sizes)` compares `runBucketed` with `run` on seeded random inputs at
  each given size and reports the largest difference. The export calls it.

## Export

`BoundedProgramExport.export(program, dir, ladders)` in `:maestro` (JVM):

```
<artifact>/
  tlaloc-bounded.json         the manifest
  bodies/<sha256>.mlir        one StableHLO module per bucket, entry `main`
  programs/<entryId>.json     one ProgramManifest per bucket
```

- Ladders: per bound, powers of two from `min(16, max)` up to `max`, with `max`
  appended, as `DecodeBucketPolicy` builds its context ladder; or an explicit ladder
  (strictly ascending, top element equal to `max`). Several bounds give the cross
  product.
- Before writing, the exporter runs `checkPadding` at the edges of every bucket
  (1, each bucket's smallest and largest size) and refuses the export, naming the
  size, output and difference, when a padded result differs from the exact one by
  more than the tolerance (default `1e-5` relative to the largest magnitude).

### Manifest `tlaloc-bounded-v1`

A new file name and a new version, not a new version of `tlaloc-serving.json`:
the serving manifest describes a language model (heads, KV pools, roles), and a
bounded program has none of that. The two share conventions: strict JSON through
`:core`'s `parseJson`, content-addressed bodies, a `ProgramManifest` per entry.

```json
{
  "schemaVersion": "tlaloc-bounded-v1",
  "name": "masked_mean",
  "bounds": [{"name": "MaxSeq", "max": 4096, "buckets": [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]}],
  "padding": {"value": 0.0},
  "inputs": [
    {"name": "x0", "role": "DATA", "dtype": "f32", "axes": [{"bound": "MaxSeq"}, {"size": 16}]},
    {"name": "validMask_MaxSeq", "role": "VALID_MASK", "dtype": "f32", "axes": [{"bound": "MaxSeq"}], "bound": "MaxSeq"}
  ],
  "outputs": [{"name": "y0", "dtype": "f32", "axes": [{"size": 16}]}],
  "entries": [
    {"id": "MaxSeq16", "sizes": {"MaxSeq": 16}, "entryPoint": "main",
     "bodyPath": "bodies/<sha>.mlir", "bodyHash": "<sha>", "programPath": "programs/MaxSeq16.json"}
  ],
  "paddingCheck": {"sizes": {"MaxSeq": [1, 16, 17, 32]}, "maxDifference": 0.0, "tolerance": 1e-5}
}
```

- Input roles: `DATA` (a user tensor, padded with `padding.value`), `VALID_MASK`
  (`[bucket]`, 1 for a real position), `VALID_LENGTH` (scalar, the real size). The
  order of `inputs` is the order of `@main`'s parameters.
- A reader refuses an unknown `schemaVersion`, an unknown role and an unknown axis
  form by name. Readers of this manifest also refuse unknown top-level keys, so a
  later field cannot be ignored by an old reader without a version bump (the
  serving-manifest readers ignore unknown keys today; that is not changed here).
- Backward compatibility: nothing about `tlaloc-serving.json` or `ProgramManifest`
  changes, so every existing artifact and fixture reads as before. The existing
  manifest tests are the evidence.

## Runtimes

| Runtime | In this run |
|---|---|
| Python (`harness/python/tlaloc_bounded.py`, new) | Load, bucket selection, padding, masks, slicing, lazy compile per bucket with `compile_count`; stdlib only, like `tlaloc_serve.py` |
| `tlaloc_serve.ServingArtifact.load` (and so the vLLM plugin) | Refuses a directory holding `tlaloc-bounded.json` and no `tlaloc-serving.json` by name, pointing at `tlaloc_bounded` |
| `TritonModelRepository` (Kotlin) | Writes a bounded artifact as a bounded-mode model (`bounded_manifest`, `max_batch_size: 0`, `-1` for bounded dims) |
| Triton backend (C++) | Bounded mode (`triton/backend/bounded_mode.cc`): strict manifest reader, signatures checked at load, every body compiled at load; per request the same bucket choice, padding, masks and slicing as the Python runtime, on the host path. One output only. A `serving_manifest` pointing at `tlaloc-bounded.json` is refused for its `schemaVersion` |

## Training steps

`program.valueAndGrad(wrt)` turns a program with one scalar output (a loss) into a program
whose outputs are the loss and its gradients with respect to the inputs in `wrt`. Each
bucket's trace is differentiated by `DxirReverseTransform` with the masks and lengths as
input-only parameters. A gradient with respect to an input with bounded axes has that
input's shape and is sliced like any output; a gradient with respect to a fixed-shape input
(a weight) sums over the real positions when the loss masks the padded ones, which
`checkPadding` verifies on every output.

`runAll` and `runBucketedAll` take a `BoundedExecutor`; the default is the interpreter, and
`{ t, args -> session.runOn(t.function, args, t.cacheKey) }` runs each trace on a
`PjrtSession`, which compiles each bucket once. A Kotlin training loop over batches of
varying length uses that; an exported training step is an artifact like any other, with
several outputs.

## Testing plan

- `:core`: `Bounded` and `DimBound` construction; `max < 1` refused.
- `:compiler-plugin`, negative compilation through the existing in-process harness:
  a literal size over the bound, at the call's line and column; a literal below 1;
  a size at the bound compiles; an invalid `DimBound`; mixing two bounds in `+`
  (plugin) and in `matmul` and `contract` (type checker) at the call's position;
  `grad { }` over a `Bounded` parameter compiles and gives the same gradient as the
  `Sym` version at several sizes.
- `:autograd`: `specOf` reads bounds and fixed sizes, and refuses a wrong count;
  `run` refuses a size over the bound, a size of 0, and two sizes for one bound;
  for every size 1..bound of a small bound, `runBucketed` equals `run` equals a
  hand-written static-shape program, for a row-wise program, a masked reduction and
  a masked softmax; `checkPadding` catches an unmasked sum.
- `:maestro`: export writes one body per bucket; manifest round trip; unknown version,
  role and key refused; an unmasked program's export refused; every existing
  serving-manifest test unchanged.
- Python (stdlib `unittest`, driven from a Kotlin test as `vllm_tlaloc_test` is):
  bucket selection, padding, mask, slicing, refusals; `tlaloc_serve` refuses a bounded
  artifact.
- GPU (skips by name without a PJRT plugin): a Kotlin test exports a bounded
  artifact and runs every size up to the bound through `tlaloc_bounded.py` on
  PJRT-CUDA, compared with the interpreter's exact-size result; it records the
  compile count.
- LLM serving: this run does not change `HfServingExport`, `ServingArtifactWriter` or
  `tlaloc_serve`'s execution path. The greedy-id tests
  (`HfLlamaServingArtifactTest`, `ServingArtifactExportRunTest`) run unchanged as the
  check.
- Compile counts: a mixed-length workload (seeded lengths in 1..512) through the
  Python runtime: one compile per distinct length for per-shape programs, against one
  per bucket touched for the bounded export. Numbers go in the work log.

## Not in this run

- Bounded dynamic StableHLO (`#stablehlo.bounds`, `set_dimension_size`). Evidence
  above; not verified for the interpreter, IREE, or the full op set.
- Deriving the LLM exporter's context or batch ladder from a `DimBound`. The LLM path
  already buckets; the ladders stay as they are.
- vLLM support (vLLM serves language models only; its loader refuses a bounded artifact).
- Triton: several outputs, dynamic batching of bounded requests, GPU-memory inputs read in
  place (bounded mode goes through the host, one request per execution).
- Bounds on `grad { }`'s DXIR (`DxirType` gets no bound field); the plugin checks
  constant sizes at call sites, not shapes inside a lambda.
- Arithmetic on bounds (`concat` of two bounded axes giving a bound of the sum).
- Compile-time checks for bounds declared in another compiled module (run-time only).
- IREE runs of bounded exports.
