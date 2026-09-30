# per-example-gradients — `vmap { grad { } }` on a small classifier

A two-layer network, `logits = tanh(x · W1) · W2`, with a cross-entropy loss, over a
batch of 16 examples. `vmap` turns the gradient of one example into the gradients of
every example at once, one per row of a new leading `Batch` axis; the plugin writes the
batched code at compile time.

```
[1] per-example gradients: W1 [16, 4, 8], W2 [16, 8, 3]
    per-example gradient norms:
    #0  1.31699   #1  0.81525   #2  0.85913   #3  0.61830
    #4  1.09589   #5  0.94980   #6  1.36638   #7  1.24971
    #8  1.63262   #9  0.66777   #10 1.08422   #11 0.75244
    #12 0.59461   #13 0.55601   #14 1.21916   #15 1.11684
    largest difference from the Double reference: 1.02e-07 of the largest entry
    example 5, dL/dW1[2][3]: compiled -0.0096516, reference -0.0096516, finite difference -0.0096516
[2] gradient of the mean loss over the batch: W1 4 x 8
    largest difference from the mean of the per-example gradients: 7.80e-09
```

```kotlin
val g1 = vmap2(batchAxis(Batch), Batched, Batched) { x: X, y: Y ->
    grad { w: W1 -> crossEntropyLoss((x matmul w).tanh() matmul w2, y).toFloat() }(w1)
}
// g1: (DTensor<Rank3<Named<Batch, Sym>, Sym, Named<Feature, Sym>>, F32>,
//      DTensor<Rank3<Named<Batch, Sym>, Sym, Named<Class, Sym>>, F32>)
//     -> DTensor<Rank3<Named<Batch, Sym>, Named<Feature, Sym>, Named<Hidden, Sym>>, F32>
```

- `x` and `y` are one example's features and one-hot label; `Batched` puts the batch
  axis in front of both. `w1` and `w2` are values the lambda captures, shared by every
  example.
- The batch axis is `Named<Batch, Sym>` in the types, so passing a tensor batched along
  another axis does not compile.
- `grad { w -> vmap2(...) { x, y -> loss }(xs, ys).mean() }` differentiates the batched
  loss; [2] checks it against the mean of the per-example gradients.

Every number in [1] is checked against `referenceGradients` in `Main.kt`: the same
network's backpropagation written out in plain Kotlin `Double`, which is itself checked
against a central finite difference of the loss. `vmap` is experimental
(`@OptIn(ExperimentalTlalocApi::class)`); the design is in
[docs/design/vmap.md](../../docs/design/vmap.md).

## Run

```bash
./gradlew publishToMavenLocal -x test               # at the repo root: vmap is not in 0.1.0-alpha02
./gradlew -p examples/per-example-gradients run
```

Needs a JDK 25 and nothing else.
