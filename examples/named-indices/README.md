# named-indices — axis names the compiler checks

**What it shows.** In NumPy, PyTorch and JAX a tensor's axes are numbered, so
`a @ b` type-checks whenever the inner lengths happen to agree. Transpose a
weight matrix by accident, swap a batch axis for a sequence axis, and you find
out at runtime — if you are lucky.

In Tlaloc an axis carries a *name* in its Kotlin type:

```kotlin
DTensor<Rank2<Named<Batch, Sym>, Named<SeqLen, Sym>>, F32>
//            ^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^
//            axis 0 is "batch"  axis 1 is "seq"
```

`contract` is declared so the contracted axes must share a name, which turns a
shape bug into an overload-resolution failure in **Kotlin's own type checker**
— a red squiggle in the IDE, no plugin diagnostic and no runtime check
involved.

The example does two contractions and points at a third that does not compile:

1. **rank-2** — `(Batch × SeqLen) · (SeqLen × Hidden) → (Batch × Hidden)`.
2. **rank-4** — the `QKᵀ` core of attention, with *two* batching axes (`Batch`,
   `Heads`) and one contracting axis (`Dim`). This is the real transformer
   shape; nothing is reshaped down to 2-D first.
3. **the compile error** — a commented-out block at the bottom of `Main.kt`
   that contracts a `SeqLen` axis against a `Vocab` axis. Uncomment it and the
   build fails before a single float is allocated.

## Running it

This is a standalone Gradle project. It resolves Tlaloc from **mavenLocal**,
exactly as your own project would, so publish first:

```bash
# from the repo root
./gradlew publishToMavenLocal
./gradlew -p examples/named-indices run
```

## Expected output

Verbatim, from this machine (GB10, JDK 25, Kotlin 2.3.20):

```
[1] rank-2 contraction   (Batch x SeqLen) . (SeqLen x Hidden) -> (Batch x Hidden)
    dims: [2, 4]  (expected [2, 4])
    [0] 3.800  4.400  5.000  5.600
    [1] 8.300  9.800  11.300  12.800

[2] rank-4 attention core  Q(B,H,Tq,D) . Kt(B,H,D,Tk) -> S(B,H,Tq,Tk)
    dims: [1, 2, 4, 4]  (expected [1, 2, 4, 4])
    --- head 0 ---
    q=0  0.1436  0.1504  0.1572  0.1641
    q=1  0.3779  0.4004  0.4229  0.4453
    q=2  0.6123  0.6504  0.6885  0.7266
    q=3  0.8467  0.9004  0.9541  1.0078
    --- head 1 ---
    q=0  3.2998  3.3691  3.4385  3.5078
    q=1  4.0342  4.1191  4.2041  4.2891
    q=2  4.7686  4.8691  4.9697  5.0703
    q=3  5.5029  5.6191  5.7354  5.8516

[3] the program that does NOT compile: see the block at the bottom
    of src/main/kotlin/Main.kt — uncomment it and run again.
named-indices OK
```

## What to read in the source

All of it is in [`src/main/kotlin/Main.kt`](src/main/kotlin/Main.kt):

| Look at | For |
|---|---|
| `rank2Contraction()` | How a named tensor is built, and which axis survives a contraction |
| `object KeySeqLen : IndexName` | Defining your own axis name — three lines |
| `attentionScores(...)` | The rank-4 signature: two batching axes, one contracting axis, spelled in types |
| the commented block in `main()` | The program that is *supposed* to fail; uncomment it |

## Notes

- **No compiler plugin here.** This example depends on `io.tlaloc:core` only.
  Named axes are enforced by Kotlin generics, so the K2 plugin is not in the
  loop — its job is lowering `grad { }` bodies (see
  [`../quickstart/`](../quickstart/)). That separation was the design goal, not
  a limitation; see `docs/audits/named_indices_audit.md`.
- When the rank-4 contraction lowers to StableHLO it becomes a
  `stablehlo.dot_general` whose `lhs_batching_dims = [0, 1]` and
  `lhs_contracting_dims = [3]` are *derived from the names*, not hand-written.
