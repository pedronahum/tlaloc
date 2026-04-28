# Named-index examples (Layer 1 §0.4.241+)

Reference snippets demonstrating Tlaloc's named-index type system.

These files document the public Kotlin surface introduced in Layer 1. They
are **documentation-grade**, not part of the Gradle build — copy into a
project that depends on `io.tlaloc:core` to run.

| File | Purpose |
|------|---------|
| [`NamedMatmulExample.kt`](NamedMatmulExample.kt) | Basic Rank-2 × Rank-2 named contraction over a shared `SeqLen` axis. |
| [`AttentionForwardExample.kt`](AttentionForwardExample.kt) | **Rank-4 QK^T** attention-score forward pass, two batching axes (`Batch`, `Heads`) + one contracting axis (`Dim`). Lands as of Layer 1.5 §0.4.242+. |
| Shape-mismatch (below) | A program that **does not compile** because two operands carry disjoint named axes. |

## Shape-mismatch example (compile error by design)

```kotlin
import io.tlaloc.core.Batch
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.Hidden
import io.tlaloc.core.Named
import io.tlaloc.core.Rank2
import io.tlaloc.core.SeqLen
import io.tlaloc.core.Sym
import io.tlaloc.core.Vocab
import io.tlaloc.core.ops.contract

fun main() {
    val a: DTensor<Rank2<Named<Batch, Sym>, Named<SeqLen, Sym>>, F32> = TODO()
    val b: DTensor<Rank2<Named<Vocab, Sym>, Named<Hidden, Sym>>, F32> = TODO()
    val c = a contract b   // <-- compile error
}
```

The Kotlin compiler reports a type-mismatch at the `contract` call site
because **no overload of `contract` matches**: the rank-2 overload requires
the LHS's axis-1 and RHS's axis-0 to share the *same* `IndexName` singleton
(the `NameK` type variable). Here `SeqLen` ≠ `Vocab`, so the overload
cannot resolve.

The error rendering is Kotlin's own — usually "Type mismatch: expected
`DTensor<Rank2<Named<SeqLen, Sym>, Named<*, *>>, F32>`, got
`DTensor<Rank2<Named<Vocab, Sym>, Named<Hidden, Sym>>, F32>`" — and points
directly at the call site, no plugin diagnostic involved. See the Refined
Option A discussion in `docs/audits/named_indices_audit.md` for why this
runs through Kotlin's native type checker rather than via a custom K2
extension.

## Running the runtime examples

```bash
# From a project that depends on io.tlaloc:core (see core/build.gradle.kts):
kotlin examples/named-indices/NamedMatmulExample.kt
```

Both compiling examples use only the runtime path — the K2 plugin is not
required to type-check the named contraction (Refined Option A's whole
point), it's only required to lower contraction calls inside `grad { }`
lambdas.
