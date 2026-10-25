# Dynamic shapes

XLA compiles a program for fixed shapes, and so does Tlaloc: every program a device
runs has static shapes. A sequence length, a batch size or a context that varies from
call to call is handled by compiling a few sizes (buckets), padding each input to the
smallest bucket that holds it, and telling the program which positions are real.

| Path | Compiles |
|---|---|
| `grad { }` | nothing at run time; one synthesized gradient serves every shape |
| `capture` and `PjrtSession.runOn` | once per distinct input shape |
| `boundedProgram` (`:autograd`, experimental) | once per bucket |
| LLM serving artifacts | once per (batch, context) bucket of `DecodeBucketPolicy` |

200 requests with 163 distinct lengths took 163 compiles and 264 s at their exact
lengths, and 6 compiles and 7.6 s with buckets from 16 to 512; execution was 13 %
slower from the padding (`docs/work-log/bounded-dims-progress.md`).

## Bounded axes

An axis can carry an upper bound in its type. The bound is an object:

<!-- doc-test: dyn-bound -->

```kotlin
object MaxSeq : DimBound(512)

typealias Tokens = DTensor<Rank2<Named<SeqLen, Bounded<MaxSeq>>, Named<Hidden, Sym>>, F32>
```

The object is the axis's identity: every axis typed `Bounded<MaxSeq>` in one program
has the same size, so two independent axes need two objects. The compiler plugin
refuses a constant size outside `1..max` (`Tensors.f32Zeros<Bounded<MaxSeq>, Sym>(600, 4)`
is a compile error) and broadcasting arithmetic between axes with different bounds.
Inside `grad { }` a bounded axis is symbolic, like `Sym`.

## Bounded programs

`boundedProgram` declares a program over bounded inputs. The body reads the valid
positions from its context: `ctx.validMask(bound)` is 1 at real positions and 0 at
padding, `ctx.validLength(bound)` the real size as a scalar.

<!-- doc-test: dyn-program -->

```kotlin
val meanOverTokens = boundedProgram(
    "masked_mean",
    listOf(specOf<Rank2<Bounded<MaxSeq>, Sym>>(F32, 3)),   // [<= 512, 3]
    specOf<Rank1<Sym>>(F32, 3),
) { xs, ctx ->
    val x = xs[0]
    val masked = x * x.broadcastAlong<Shape>(ctx.validMask(MaxSeq), 0)
    masked.sum<Rank1<Sym>>(intArrayOf(0)) / ctx.validLength(MaxSeq)
}

val ladders = BucketLadders.powersOfTwo(meanOverTokens.bounds)   // 16, 32, …, 512
val y = meanOverTokens.runBucketed(listOf(x), ladders)            // x is [5, 3]: runs the 16 bucket
```

- `run(inputs)` traces and runs at the exact sizes (one trace per size);
  `runBucketed(inputs, ladders)` pads to the bucket, runs, and slices the result back.
- A `BoundedExecutor` decides where a trace runs: the interpreter by default, or a
  device, `BoundedExecutor { t, args -> session.runOn(t.function, args, t.cacheKey) }`,
  which compiles each bucket once.
- `valueAndGrad(wrt)` turns a scalar program into a training step: it returns the
  loss and the gradients of the inputs `wrt`, bucketed the same way. Masks and
  lengths get no gradient.
- Padding is zeros, so a sum is right without a mask and a mean or a softmax is not.
  `checkPadding(ladders)` runs the program at the edges of every bucket, padded and
  unpadded, and reports the largest difference.

### Exporting and serving

`BoundedProgramExport.export(program, dir)` (`:maestro`) checks the padding (and
refuses to write when the program ignores it), then writes one StableHLO program per
bucket combination with a `tlaloc-bounded.json` manifest. Two runtimes serve it:

- Python: `harness/python/tlaloc_bounded.py`, `BoundedArtifact.load(dir).run(...)`;
  it compiles each bucket the first time a request needs it.
- Triton: the backend's bounded mode compiles every bucket when the model loads; one
  output, one request per execution.

## LLM serving

A serving artifact has one decode program per (batch, context) bucket of its
`DecodeBucketPolicy` (powers of two up to the caps, contexts aligned to the KV block
size) and prefill programs up to a batch. A request runs in the smallest bucket that
holds it; padded rows write no KV entries. A size above the caps is refused, never
clamped. [SERVING_ARCHITECTURE.md](SERVING_ARCHITECTURE.md) has the details.

## Limits

- No dynamic StableHLO: bounded `tensor<?>` dimensions are not emitted, so one program
  per bucket is the cost of a bound.
- `capture` in `:nn` has no bounded form, so models built from layers compile once per
  shape, and `MultiHeadAttention` and the losses take no mask.
- Bounded programs are f32 only, run no `vmap`, and have no arithmetic on bounds
  (a `concat` of two bounded axes).
- Only bounds declared in the module being compiled are checked at compile time.

The design is in [design/bounded-dims.md](design/bounded-dims.md); the open items are in
[FOLLOWUPS.md](FOLLOWUPS.md).
