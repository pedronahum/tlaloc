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
val dMiss = grad2 { angle: Float, speed: Float ->
    var x = 0.0f
    var y = 2.0f
    var vx = speed * angle.cos()
    var vy = speed * angle.sin()
    for (i in 0 until 38) {
        vx = vx - 0.24f * vx * 0.025f             // drag
        vy = vy - (9.81f + 0.24f * vy) * 0.025f   // gravity + drag
        x = x + vx * 0.025f
        y = y + vy * 0.025f
    }
    (x - 4.6f) * (x - 4.6f) + (y - 3.05f) * (y - 3.05f)   // squared miss
}
```

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
The full file lands in `build/gradients/` after a build.

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

## A limitation you will meet immediately

Every number inside the `grad2 { }` body is a literal rather than one of the
`const val`s at the top of the file. That is not style — the lambda is lowered
to Tlaloc IR, and a reference out of that scope is **refused by name**:

```
e: Tlaloc could not lower this lambda at compile time: reference to symbol outside
   the lowering scope: /X0
```

Swap `0.0f` for `X0` and read the refusal. It is worth doing once: it is the
house rule in action — an unsupported case says so, loudly, instead of quietly
falling back to something slower that would still have produced a number.

Since §0.4.499 that refusal is an **error**, not a warning, and the build stops.
It used to be a warning, and the program then threw `IllegalStateException` the
first time it called `dMiss` — the same information, one run later. If you want
that late failure back, pass
`-P plugin:io.tlaloc.plugin:strictLowering=false`.

## A limitation this example removed

Writing it turned one up. `angle.cos()` always *differentiated* correctly, but
the gradient **printer** had no `:core` tensor twin for `COS`, so act `[4]`
refused:

```
dump SKIPPED — the gradient synthesised fine, but it has no honest Kotlin
rendering: COS at %4 — no `:core` tensor host twin exists for this unary
(the bmm-precedent twin gap)
```

The first draft dodged it by parameterising on the release velocity components
instead of an angle. That was the wrong fix: the honest one was to add the
missing `DTensor.sin()` / `DTensor.cos()` twins to `:core` and teach the printer
their spelling (§0.4.496), which is why the derivative above now reads
`angle.cos()`. `ABS`, `RSQRT`, `GELU` and `SILU` are still in that gap and still
refuse by name.

## Expected output

Verbatim, from this machine (GB10 / aarch64, JDK 25, Kotlin 2.3.20):

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

    Main_kt_128_23_valueAndGrad2.kt — 832 lines, 823 operations.
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

    the whole file: /home/pedro/programming/tlaloc/examples/differentiable-physics/build/gradients/Main_kt_128_23_valueAndGrad2.kt

differentiable-physics OK
```

## What to read in the source

| Look at | For |
|---|---|
| [`src/main/kotlin/Main.kt`](src/main/kotlin/Main.kt) | The `grad2 { }` block with the simulator inside it, and `simulate()` — the independent transcription that keeps it honest |
| [`build.gradle.kts`](build.gradle.kts) | `dumpGradSourceDir` — the two lines that make the derivative a file |
| `build/gradients/*.kt` | The derivative itself, after a build |

## Related

- [`readable-gradients/`](../readable-gradients/) — the same printed-derivative
  idea, taken further: the printed file is compiled *without* the plugin and
  checked bit-for-bit against the compiled gradient.
- [`gpu-training/`](../gpu-training/) — the same compiler, now deriving the
  gradient of a neural network and training it on a GPU.
