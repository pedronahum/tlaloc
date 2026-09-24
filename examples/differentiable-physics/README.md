# differentiable-physics — gradient descent through a simulator

A ball has to go through a hoop 4.6 m away and 3.05 m up. There is air drag, so
there is no formula to invert. Tlaloc differentiates the **simulator** — a `for`
loop integrating Newton's laws — and walks downhill until the shot goes in.

```
    |                        oo ooo oooo ooo o                 
    |                 o ooo o                 oooo o           
    |             oo o                              ooo##      
    |         oo o                                             
    |      oo  ... ... ... ... ...                             
    |   oo ...                    . ....                       
    |oo                                 . ..                   
    |                                       ...                
    |                                           ...            
    +----------------------------------------------------------

    first guess:   35.52° at 6.400 m/s  —  misses by 2.13 m
    learned throw: 48.91° at 8.266 m/s  —  0.0000 m from the centre of the rim

    SWISH.
```

**What it shows.** This is the capability that has nothing to do with machine
learning: you can differentiate *a program*, including its loops, and use the
derivative to solve for the program's inputs. No model, no dataset, no training
— a physics simulation and the chain rule.

```kotlin
const val G = 9.81f; const val DRAG = 0.24f; const val DT = 0.025f; const val STEPS = 38
const val X0 = 0.0f; const val Y0 = 2.0f; const val HOOP_X = 4.6f; const val HOOP_Y = 3.05f

val dMiss = grad2 { angle: Float, speed: Float ->
    var x = X0
    var y = Y0
    var vx = speed * angle.cos()
    var vy = speed * angle.sin()
    for (i in 0 until STEPS) {
        vx = vx - DRAG * vx * DT              // drag
        vy = vy - (G + DRAG * vy) * DT        // gravity + drag
        x = x + vx * DT
        y = y + vy * DT
    }
    (x - HOOP_X) * (x - HOOP_X) + (y - HOOP_Y) * (y - HOOP_Y)   // squared miss
}
```

Those names are `const val`s declared outside the lambda, including the loop's trip
count. [What the body may reference](#what-the-body-may-reference) lists what
folds and what still refuses.

That loop is differentiated **at compile time**. The K2 plugin lowers the lambda
to Tlaloc IR, applies φ-calculus coarsening to the loop (the [OOPSLA 2021
method](https://doi.org/10.1145/3485507) — a constant trip count unrolls into
straight-line code symbolic differentiation can close over), reverse-transforms
it, and synthesizes the gradient into this method's bytecode. At runtime
`dMiss` is an ordinary Kotlin function of two floats: no tape, no graph object,
nothing recording what the loop did.

## The derivative is a file you can read

Because the build passes `dumpGradSourceDir`, the compiler also writes out the
gradient it derived, as Kotlin. For this simulator that is **832 lines, 823
operations** — the chain rule carried back through 38 timesteps:

```kotlin
fun valueAndGrad2_body_grad(angle: DTensor<ScalarShape, F32>, speed: DTensor<ScalarShape, F32>)
        : Triple<DTensor<ScalarShape, F32>, DTensor<ScalarShape, F32>, DTensor<ScalarShape, F32>> {
    val v3: DTensor<ScalarShape, F32> = Tensors.f32Scalar(2.0f) // %3 = const : f32
    val v4: DTensor<ScalarShape, F32> = angle.cos() // %4 = COS(%0)
    val v5: DTensor<ScalarShape, F32> = (speed * v4) // %5 = MUL(%1, %4)
    val v6: DTensor<ScalarShape, F32> = angle.sin() // %6 = SIN(%0)
    ...
    val v1445: DTensor<ScalarShape, F32> = (v1438 + v1444) // %1445 = ADD(%1438, %1444)
    return Triple(v741, v1445, v1441)
}
```

Nobody writes that by hand, and no other autodiff framework will show it to you.
The full file lands in `build/gradients/main/` after a build.

## Nothing here is taken on trust

`simulate()` in `Main.kt` is a second, independent transcription of the same
physics in plain `kotlin.math`, with no Tlaloc in it. Act `[1]` central-differences
that function and compares it against the compiled gradient at four points; the
example `check`s the worst relative disagreement and refuses to print a result if
it exceeds 2e-2. On this machine it is **8.401e-04**.

The final verdict is also independent: whether the ball is inside the rim radius
is judged by `simulate()`, not by the loss the optimizer was minimising.

## Running it

Standalone Gradle project resolving Tlaloc from **mavenLocal**, exactly as your
own project would, so publish first. **No GPU, no dataset, no network.**

```bash
# from the repo root
./gradlew publishToMavenLocal
./gradlew -p examples/differentiable-physics run
```

## What the body may reference

A captured reference the compiler can resolve to a **compile-time constant** is
folded into the lowered IR as exactly the constant an inline literal would have
produced, so the simulator above reads like the `simulate()` transcription it is
checked against. Concretely, what folds is:

- any `const val`, wherever it is declared (top level, file level, or in a
  companion / named `object`) — including `STEPS` as a loop trip count;
- a `const val` whose own initializer is constant arithmetic (`const val HALF_DT =
  DT / 2.0f`), resolved through the Kotlin compiler's own constant evaluator;
- a top-level `val`, or a `val` local to the enclosing function, whose initializer
  the compiler can fold. These are single-assignment with a fixed initializer, so
  the value folded is the value the lambda would have read.

**The interesting result is that nothing in this example's output changed.** The
derivative the compiler writes into `build/gradients/main/` is *byte-identical* to the
one it wrote from the literal body — same 832 lines, same 823 operations, same
`md5 5d0c2b9704f12a2863938df08f2cae61`. That is the design working: the fold emits
the same `DxirConst` the literal path emits, so the reverse transform, the
φ-calculus coarsening and the synthesized bytecode cannot tell the two spellings
apart. Only the dump's *filename* moved, because the `valueAndGrad2` call is three
lines further down the file.

### A captured runtime value

A value the compiler cannot know — the result of a
call, a parameter of the enclosing function — becomes a **trailing parameter of
the derived gradient**, which the plugin binds at the call site by reading the
very declaration the lambda closed over. The derivation still happens at compile
time; only the value arrives at run time. That is what act `[5]` of this example
does:

```kotlin
private fun aimingSolver(hoopX: Float, hoopY: Float): (Float, Float) -> Pair<Float, Float> =
    grad2 { angle: Float, speed: Float ->
        ...
        (x - hoopX) * (x - hoopX) + (y - hoopY) * (y - hoopY)
    }
```

`aimingSolver(4.6f, 3.05f)` and `aimingSolver(6.75f, 3.05f)` are two closures over
**one** compiled derivative, and the example finite-differences both of them
against the plain-Kotlin simulator before printing anything.

**The arity does not change.** A captured value is an input, never a
differentiation target: `grad2` above still returns a `Pair` of two gradients —
`d/dangle` and `d/dspeed` — and not one per captured hoop coordinate. The
derivative the compiler writes for it says so out loud — four parameters in,
two gradients out (`build/gradients/main/Main_kt_234_5_grad2.kt`, after a build):

```kotlin
fun grad2_body_grad(
    angle: DTensor<ScalarShape, F32>, speed: DTensor<ScalarShape, F32>,
    hoopX: DTensor<ScalarShape, F32>, hoopY: DTensor<ScalarShape, F32>,
): Pair<DTensor<ScalarShape, F32>, DTensor<ScalarShape, F32>> {
```

### What is still refused

```kotlin
var gain = 2.0f                              // a `var`
val GAIN = readConfig()                      // a top-level property
class Box { val gain = readConfig() }        // a member property
val t: DTensor<Rank1<Sym>, F32> = load()     // a captured tensor
```

A `var` has no single value to bind: the gradient is derived where the lambda is
written and runs where it is called. A top-level or member property is read
through a *getter call*, not a local read, so there is no declaration for the
synthesized gradient to close over. A captured tensor's shape would have to come
from a call-site slot it does not have. Each is refused by name:

```
e: Tlaloc could not lower this lambda at compile time: captured value 'gain' is
   not a compile-time constant (it is a `var`, and a captured runtime value must
   be immutable — the gradient is derived where the lambda is written but runs
   where it is called, so a mutable capture has no single value to bind. Declare
   'gain' as a `val`) — ...
```

Try one. It is the house rule in action — an unsupported case says so, loudly,
instead of quietly falling back to something slower that would still have produced
a number. The refusal is a compile **error** and the build stops. With
`tlaloc { strictLowering.set(false) }` in `build.gradle.kts` it is a warning, and
the program throws `IllegalStateException` the first time it calls `dMiss`.

## A limitation this example removed

The printer renders an op only when `:core` has a tensor function for it, which
is why the derivative above can read `angle.cos()`. `ABS`, `RSQRT`, `GELU` and
`SILU` have none yet; a gradient containing one still compiles, and its dump
prints `dump SKIPPED` with the op's name instead of source.

## Expected output

Verbatim, from this machine (GB10 / aarch64, JDK 25, Kotlin 2.4.20):

```
Tlaloc differentiable physics — a free throw, solved by differentiating the simulator

    release  (0.0, 2.0) m         hoop  (4.60, 3.05) m
    gravity  9.81 m/s²             linear drag  0.24 /s
    38 Euler steps of 0.025 s   =   0.95 s of flight

    The unknowns are the launch angle and speed. With drag inside the loop there
    is no formula to invert, so we differentiate the loop and walk downhill.

[1] is the compiler's derivative the real derivative?

       angle    speed         d/dangle    finite diff         d/dspeed    finite diff
       0.620    6.400       -17.556320     -17.555487        -2.353585      -2.355096
       0.850    8.300        -0.353795      -0.353808         0.048524       0.048565
       1.050    6.500        15.031733      15.031807        -2.304254      -2.303359
       1.200    9.000        36.211536      36.212093         1.755917       1.754598

    worst relative disagreement: 8.401e-04
    -> the compiler differentiated 38 timesteps of Newtonian motion, correctly.

[2] the throw, learned

    first guess: 35.52° at 6.400 m/s — misses by 2.13 m

     step      angle     speed     miss (m)
        0     35.52°     6.400       2.1346
       30     48.83°     8.248       0.0186
       60     48.88°     8.266       0.0028
       90     48.90°     8.266       0.0008
      120     48.91°     8.266       0.0002
      150     48.91°     8.266       0.0001
      180     48.91°     8.266       0.0000
      210     48.91°     8.266       0.0000
      240     48.91°     8.266       0.0000

    learned throw: 48.91° at 8.266 m/s
    ball at step 38: (4.600, 3.050)      hoop: (4.60, 3.05)
    0.0000 m from the centre of the rim (rim radius 0.23 m)

    SWISH.

[3] the first guess ('.'), what the gradient found ('o'), the rim ('#')

    |                                                          
    |                        oo ooo oooo ooo o                 
    |                 o ooo o                 oooo o           
    |             oo o                              ooo##      
    |         oo o                                             
    |      oo  ... ... ... ... ...                             
    |   oo ...                    . ....                       
    |oo                                 . ..                   
    |                                       ...                
    |                                           ...            
    |                                              ..          
    |                                                .         
    |                                                          
    |                                                          
    |                                                          
    +----------------------------------------------------------
     0 m                                                5.2 m

[4] the derivative of the simulator, as the compiler wrote it

    Main_kt_133_23_valueAndGrad2.kt — 832 lines, 823 operations.
    Nobody wrote this by hand: it is the chain rule carried back through
    38 timesteps of the loop above, and it is ordinary Kotlin over `:core`.

    // Generated by DxirFunction.toKotlinSource() from function 'valueAndGrad2_body_grad'.
    // Each `val` is one DXIR op, bound to its certified `:core` host-twin call;
    // the trailing comment names the SSA node it renders.
    import io.tlaloc.core.*
    import io.tlaloc.core.ops.*
    
    fun valueAndGrad2_body_grad(angle: DTensor<ScalarShape, F32>, speed: DTensor<ScalarShape, F32>): Triple<DTensor<ScalarShape, F32>, DTensor<ScalarShape, F32>, DTensor<ScalarShape, F32>> {
        val v3: DTensor<ScalarShape, F32> = Tensors.f32Scalar(2.0f) // %3 = const : f32
        val v4: DTensor<ScalarShape, F32> = angle.cos() // %4 = COS(%0)
        val v5: DTensor<ScalarShape, F32> = (speed * v4) // %5 = MUL(%1, %4)
        val v6: DTensor<ScalarShape, F32> = angle.sin() // %6 = SIN(%0)
        val v7: DTensor<ScalarShape, F32> = (speed * v6) // %7 = MUL(%1, %6)
        val v10: DTensor<ScalarShape, F32> = Tensors.f32Scalar(0.24f) // %10 = const : f32
        val v11: DTensor<ScalarShape, F32> = (v10 * v5) // %11 = MUL(%10, %5)
    ... 815 lines ...
        val v1445: DTensor<ScalarShape, F32> = (v1438 + v1444) // %1445 = ADD(%1438, %1444)
        return Triple(v741, v1445, v1441)
    }

    the whole file: <repo>/examples/differentiable-physics/build/gradients/main/Main_kt_133_23_valueAndGrad2.kt

[5] the same derivative, aimed somewhere else at run time

    The hoop in `aimingSolver` is a parameter of the enclosing function, not a
    `const val`. The lambda reads it; the gradient is still derived at compile
    time and the value is bound where the function is called.

    hoop                 d/dangle    finite diff         d/dspeed    finite diff
    4.60, 3.05         -17.556320     -17.555487        -2.353585      -2.355096
    6.75, 3.05          -4.018332      -4.014647        -5.316582      -5.315873

    worst relative disagreement: 9.180e-04
    -> one compiled derivative, two targets, both right.

differentiable-physics OK
```

## What to read in the source

| Look at | For |
|---|---|
| [`src/main/kotlin/Main.kt`](src/main/kotlin/Main.kt) | The `grad2 { }` block with the simulator inside it, and `simulate()`, the independent transcription it is checked against |
| [`build.gradle.kts`](build.gradle.kts) | `dumpGradSourceDir` — the two lines that make the derivative a file |
| `build/gradients/main/*.kt` | The derivative itself, after a build |

## Related

- [`readable-gradients/`](../readable-gradients/) — the same printed-derivative
  idea, taken further: the printed file is compiled *without* the plugin and
  checked bit-for-bit against the compiled gradient.
- [`gpu-training/`](../gpu-training/) — the same compiler, now deriving the
  gradient of a neural network and training it on a GPU.
