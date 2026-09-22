# mnist — the benchmark everybody knows, with a gradient nobody wrote

```
    predicted 7, label 7           predicted 2, label 2           predicted 1, label 1
    predicted 0, label 0           predicted 4, label 4           predicted 1, label 1

    9,366 of 10,000 test images correct  =  93.66 %
```

**What it shows.** The real MNIST — 60,000 handwritten digits, downloaded and
parsed from the original idx files — trained by a gradient that `:nn`'s `capture`
derived from the forward pass with the compiler's own reverse-mode transform.
There is no `.backward()` in this example, no `zero_grad()`, and no hand-written
derivative. Act `[4]` prints test digits as ASCII next to the model's verdict, so
you can check the claim with your eyes rather than with a number.

```kotlin
val model0 = Sequential(Dense(784, 128, keys[0]), ReluLayer, Dense(128, 10, keys[1]))

// Traced once. `step.gradient` is the gradient graph DxirReverseTransform derived.
val step = capture(model0, listOf(x0), name = "mnist_mlp") { logits ->
    val target = logits.constant<Shape>(trainY, intArrayOf(trainCount, 10))
    val residual = logits - target
    (residual * residual).mean()
}

repeat(600) {
    val out = step.unpack(lane.run(step.gradient, step.bind(listOf(trainX), model)))
    val (nextModel, nextState) = optimizer.step(model, out.gradients, state)
    model = nextModel; state = nextState
}
```

On a GB10 that loop runs **600 steps in 8.3 s (13.8 ms/step)** and compiles
**one** executable for all 600 dispatches — the weights enter the graph as
parameters, not as baked constants, so nothing recompiles when they change.

## Two lanes, one program

| | Steps | Images | Wall clock | Test accuracy |
|---|---|---|---|---|
| **GPU** (PJRT/XLA, CUDA) | 600 | 4,096 | 8.28 s | **93.66 %** on all 10,000 |
| **Host** (JVM interpreter) | 40 | 512 | 19.85 s | 82.50 % on 1,000 |

`TLALOC_EXAMPLE_LANE=host` forces the second. It is the same captured gradient
running on a different backend — the interpreter is a correctness engine, not a
fast CPU backend, so the example shortens the run rather than pretending.

## Two honest notes

**The loss is squared error against one-hot targets, not cross-entropy.**
`:core` has `crossEntropyLoss`, but there is no traced spelling of it on
`Tracer` today — nor of a `max` reduction, which a numerically stable
log-sum-exp needs. Squared error uses only ops that exist. It costs a point or
two of accuracy, and saying so is cheaper than a silently unstable example.

**It trains full-batch over a 4,096-image subset, not streaming mini-batches.**
The targets ride into the captured graph as a constant leaf, and a baked
constant cannot be re-bound per step. Streaming mini-batches through this graph
would silently keep training against the first batch's labels — so the example
does the thing that is actually correct, and `Main.kt` says why at the point
where you would otherwise write the bug.

## Running it

Standalone Gradle project resolving Tlaloc from **mavenLocal**, so publish first:

```bash
# from the repo root
./gradlew publishToMavenLocal
./gradlew -p examples/mnist run

# the CPU lane, on any laptop
TLALOC_EXAMPLE_LANE=host ./gradlew -p examples/mnist run
```

**First run downloads MNIST** (~11 MB) into `~/.cache/tlaloc-datasets/mnist/`
and reuses it forever after. No network? The example prints a named `SKIP`, the
`curl` line to fetch the files by hand, and exits `0` — never a stack trace.

## Expected output

Verbatim, from this machine (GB10 / aarch64, CUDA, JDK 25, Kotlin 2.3.20), with
XLA's own log lines filtered out:

```
Tlaloc on MNIST — 60,000 handwritten digits, one captured gradient
[0] the data
    train 60,000 images    test 10,000 images
    28x28 greyscale, scaled to [0,1]; no other preprocessing
    cache: /home/pedro/.cache/tlaloc-datasets/mnist
[1] the model and the captured gradient
    Dense(784 -> 128) -> ReLU -> Dense(128 -> 10)
    4 parameter tensors, 101,770 scalars
    weights initialised from threefry seed 20260922 (bit-exact against JAX's stream)
    forward : 11 IR nodes
    gradient: 27 IR nodes, derived by DxirReverseTransform
[2] training on GPU via PJRT/XLA (CUDA)
      step          loss
         0      0.112445
        75      0.017152
       150      0.010211
       225      0.007002
       300      0.005089
       375      0.003867
       450      0.003017
       525      0.002512
       599      0.002052
    600 full-batch steps over 4,096 images in 8.43 s  (14.05 ms/step)
    loss 0.112445 -> 0.002052
    compiled executables after training: 1
    (one program, 600 dispatches — the weights are graph PARAMETERS)
[3] held-out accuracy
    9,373 of 10,000 test images correct  =  93.73 %
[4] what it actually sees
                                                                                              
                                                                                              
                                            +%%%%%%#.                             .%-         
                                          .%%#.   #%=                             =%          
          #%%%%%*********.                 .     #%%:                            :%:          
                : :::: %%-                     :%%%:                            .%#           
                      %@:                     *%%+                              =%:           
                    :%%:                     *%%=                              =@+            
                    #%:                     #%%=                               %#             
                  :%%:                     %%%                                +%+             
                 #%#.                      %%%%%%%%+++%%%%%%%=               :%%:             
               .#%=                         ====+%%%+==.                     #%=              
              :%%%.                                                                           
              =%#                                                                             
    predicted 7, label 7           predicted 2, label 2           predicted 1, label 1        
                                                                                              
                                                                                              
                 +%#.                                                                         
                *%%%-                        =%       ++                          %%%         
              %%%%%%%%#.                    :%=       =%                         #%%:         
            :#%%%%+::=%%-                  +%.        @%                        =%%*          
           .%%%*      -%%%.               +%-       .%%:                       :%%%           
           .%%        .#@%+               %+        +%+                        +%%            
           #%%       :%%%%.               +%#----=+*%%-                       :%%+            
           #%%     *#%%%*                   .::::  +%%                        %%#             
           -%%%%%%%%%%%+                           =%%                       -%%              
            :+%%%%%%-                              =%%                       #%*              
                                                   *:                         +-              
                                                                                              
    predicted 0, label 0           predicted 4, label 4           predicted 1, label 1        
mnist OK
```

The GPU numbers are not bit-reproducible: XLA autotunes its GEMM kernels at
compile time, so the loss drifts in the last decimals between runs. The host
lane is deterministic.

## What to read in the source

| Look at | For |
|---|---|
| [`src/main/kotlin/Main.kt`](src/main/kotlin/Main.kt) | `capture` once, the fold that trains, and the two honest notes at the point they matter |
| [`src/main/kotlin/Mnist.kt`](src/main/kotlin/Mnist.kt) | Downloading and parsing idx — 70 lines, nothing Tlaloc-specific |
| [`src/main/kotlin/Lanes.kt`](src/main/kotlin/Lanes.kt) | The whole backend choice: interpreter or GPU, behind one 3-method interface |

## Related

- [`gpu-training/`](../gpu-training/) — the same machinery on a synthetic task
  small enough to see the decision boundary drawn as ASCII art.
- [`differentiable-physics/`](../differentiable-physics/) — the same compiler,
  differentiating a physics simulator instead of a network.
