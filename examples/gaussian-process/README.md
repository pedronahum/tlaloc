# gaussian-process — fitting kernel hyperparameters through a Cholesky solve

A Gaussian process with a squared-exponential kernel, fitted to 24 noisy samples
of `sin(x)` by gradient descent on the negative log marginal likelihood. The
likelihood — the kernel matrix, a solve through its Cholesky factor and its
log-determinant — is written inside `grad3 { }`, and the plugin differentiates it
at compile time.

```
[1] gradient at ℓ = 3, σf = 0.5, σn = 0.5
                 d/dlog ℓ    d/dlog σf    d/dlog σn
    compiled      6.587054841     -3.178274637      7.513186760
    finite diff   6.587054841     -3.178274637      7.513186759
    largest difference: 8.33e-11 of the largest entry

[2] Adam on θ = (log ℓ, log σf, log σn)
    step        ℓ       σf       σn        nll
       0    3.000    0.500    0.500    17.9309
      50    1.587    0.842    0.114    -5.1862
     100    1.693    0.881    0.117    -5.2408
     150    1.693    0.874    0.116    -5.2420
     200    1.693    0.874    0.116    -5.2420
     250    1.693    0.874    0.116    -5.2420
     300    1.693    0.874    0.116    -5.2420

[3] fitted: ℓ = 1.693, σf = 0.874, σn = 0.116 (the data's noise is 0.1)
    |gradient| at the end: 7.68e-06
    nll: 17.9309 → -5.2420
```

(The printed nll includes the constant `(n/2)·log 2π`; the lambda leaves it out.)

```kotlin
typealias Matrix = DTensor<Rank2<Sym, Sym>, F64>
typealias Vector = DTensor<Rank1<Sym>, F64>

val nllAndGrad = grad3 { theta: Vector, d: Matrix, y: Matrix ->
    val ell = theta[0]
    val sf = theta[1]
    val sn = theta[2]
    val k = (d * (-0.5 * (-2.0 * ell).exp())).exp() * (2.0 * sf).exp() +
        d.identityLike() * (2.0 * sn).exp()
    val alpha = k.solveSpd(y)
    0.5 * (y * alpha).sum().toDouble() + 0.5 * k.logDetSpd().toDouble()
}
```

The example runs in double precision (`F64`), the usual choice for a Gaussian process: K's
condition number grows as the noise σn shrinks, and the fitted σn is about 0.1. Changing
`F64` to `F32` in the two type aliases (and the literals and `toDouble()` to their F32
spellings) gives the same fit with a gradient 3.8e-7 away from the finite differences
instead of 8.3e-11.

`d` holds the squared distances `(xᵢ − xⱼ)²` and `y` the observations as an
`n×1` matrix. They are lambda parameters because a `grad { }` body reads
tensors only through its parameters; their gradients are computed and ignored.

`solveSpd` and `logDetSpd` are lowered to `cholesky`, `triangularSolve` and
elementwise ops. The derivative comes from Murray's rule for the Cholesky factor
and implicit differentiation of the triangular solves, not from differentiating
the factorization's loops.

Act [1] checks the compiled gradient against central differences of
`nllReference`, the same formula in plain Kotlin `Double` with its own Cholesky
and no Tlaloc. The example stops if they differ by more than 1e-8 of the largest
entry.

## Run

The linear-algebra ops and F64 under `grad { }` are not in the `0.1.0-alpha02`
release on Maven Central, so this example needs Tlaloc built from this checkout:

```bash
./gradlew publishToMavenLocal -x test
./gradlew -p examples/gaussian-process run
```

To keep `~/.m2` as it is, pass the same `-Dmaven.repo.local=<dir>` to both
commands. Needs a JDK 25; runs on the CPU in about a second.
