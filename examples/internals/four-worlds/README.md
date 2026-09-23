# four-worlds — where your code lives, and what may cross between

**What it shows.** Tlaloc splits a distributed program into four scopes, each
one an ordinary Kotlin receiver type:

| World | What belongs there | How you enter it |
|---|---|---|
| **Kernel** | the maths — elementwise ops, matmuls, reductions | the body lambda of `program { }` |
| **Orchestration** | building Kernel bodies into shippable artifacts | `Tlaloc.program(...)` |
| **Program** | composing artifacts into a graph | `Tlaloc.workflow { }` |
| **Cluster** | running the graph on real hardware | each step's StableHLO artifact, run by a Maestro `Tlaloc` step |

Two consequences, both visible in this example:

- **What crosses a step boundary is never a raw tensor.** It is a
  `BufferHandle<T, M>`, carrying both the value's type and the mesh `M` it
  lives on. Feed a step a handle of the wrong shape or the wrong mesh and
  Kotlin's own checker rejects the call site — no runtime guard, no lint rule.
  [`src/shapeError/kotlin/WorldErrors.kt`](src/shapeError/kotlin/WorldErrors.kt)
  is a real file that proves it, and `./gradlew -p examples/internals/four-worlds
  shapeError` is the command that makes the compiler say so.
- **The scopes are a design, and one of their claims turned out not to hold.**
  This README used to say that `program` cannot be called from inside a Kernel
  body. Writing the failing case down as a file the build compiles showed that
  it *can*: the `@WorldScope` DslMarker shadows an **implicit** outer receiver
  in a nested builder, and `Tlaloc.program` names its receiver explicitly, which
  DslMarker never blocks. The separation you can rely on today is the typed
  handle above. Keeping the claim would have been cheaper than checking it.

- **The whole taxonomy is opt-in, on purpose.** The four scopes
  and `BufferHandle` carry `@ExperimentalTlalocApi`, a `@RequiresOptIn(ERROR)`
  marker, so both source files here start with
  `@file:OptIn(ExperimentalTlalocApi::class)` and would not compile without it.
  The reason is the bullet above plus the design's own KDoc, which scopes it to
  "v1 keeps each op single-scope" and names Kotlin's context parameters as where
  a multi-scope op would send every signature in the table. `grad`, the op
  surface and `:nn` are **not** marked — see
  [docs/GETTING_STARTED.md, section 4b](../../../docs/GETTING_STARTED.md#4b-experimentaltlalocapi-the-provisional-part-of-the-surface).

The run prints: one step built and executed; two steps composed with a handle
flowing between them (and the recorded edge); and each step's StableHLO
artifact, which a Maestro `Tlaloc` step runs on a cluster.

## Running it

This is a standalone Gradle project resolving Tlaloc from **mavenLocal**, so
publish first:

```bash
# from the repo root
./gradlew publishToMavenLocal
./gradlew -p examples/internals/four-worlds run
```

## Expected output

Verbatim, from this machine (GB10, JDK 25, Kotlin 2.3.20). The body hash is
content-addressed over the emitted StableHLO, so it is stable across runs:

```
[1] one step, built by `program { }`
    name:              activate_and_score
    body hash:         8d772d28b944a644855c530be9106ababf64c6fd0d583f23534cf77b68675c19
    body size:         455 bytes of StableHLO
    inputs:            [TypeDescriptor(dtype=f32, dims=[5], axisNames=[])]
    outputs:           [TypeDescriptor(dtype=f32, dims=[], axisNames=[])]
    mesh requirement:  Mesh0
    result:            35.0   (relu([1,-2,3,-4,5])^2 summed = 1+9+25)

[2] two steps, composed by `workflow { }`
    workflow:  activate_then_score
    steps:     [activate, score]
    edges:     [activate -> score (None)]
    result:    35.0   (same answer, two artifacts)
    (reshard kind is None because both steps sit on Mesh0 with the same
     axis names; a mesh change or a transpose records an explicit edge.)

[3] the artifacts a Maestro `Tlaloc` step would run
    activate: 212 bytes of StableHLO, bodyHash 1597799e2a376bf503ab76910d7dba696f99a02dd5bc09d36ee311456ce4ef4b
    score: 335 bytes of StableHLO, bodyHash 44d9225a49de4895ce151d24ac25d78e09bc2cd75751f63469e74417501ed624

[4] the program that does NOT compile

    fun wrongHandleType() {
        val matrixStep = Tlaloc.program(
            "matrix", Tensors.f32Matrix<Sym, Sym>(2, 3, FloatArray(6)), Mesh0,
        ) { m -> m.sum() }
        Tlaloc.workflow("mismatched") {
            val activated = step(activate, seed(input, Mesh0))
            step(matrixStep, activated)   // <-- compile error, by design
        }
    }

    `activate` yields a Rank1 handle; `matrixStep` wants a Rank2 one. What
    crosses a step boundary is a typed BufferHandle<T, M>, so the wrong one
    does not type-check. Watch it:

        ./gradlew -p examples/internals/four-worlds shapeError

    e: WorldErrors.kt:44:26 Argument type mismatch: actual type is
       'BufferHandle<DTensor<Rank1<Sym>, F32>, Mesh0>', but
       'BufferHandle<DTensor<Rank2<Sym, Sym>, F32>, Mesh0>' was expected.

    A SECOND claim used to live here, commented out: that calling
    `Tlaloc.program` from inside a Kernel body would not resolve. Turning
    these comments into a file the build actually compiles showed that it
    DOES resolve, so the claim is gone. The @WorldScope DslMarker shadows
    an IMPLICIT outer receiver inside a nested builder; `Tlaloc.program`
    names its receiver explicitly, and DslMarker never blocks that. The
    world separation you can rely on today is the typed handle above.

four-worlds OK
```

## What to read in the source

All of it is in [`src/main/kotlin/Main.kt`](src/main/kotlin/Main.kt):

| Look at | For |
|---|---|
| `singleStepProgram()` | `program { }`: the Kernel body, the artifact it produces (hash, size, manifest), and running it through its shim |
| `input.handleOn(Mesh0)` | How a tensor from outside the system becomes a `BufferHandle` |
| `twoStepWorkflow()` | `workflow { }`: `seed(...)`, `step(...)`, and the handle typing that makes the composition check itself |
| `wf.edges` | The reshard metadata Tlaloc records per edge — `None` here, `Mesh`/`Transpose` when the placement actually changes |
| the commented block in `main()` | The two failures: wrong handle type, and calling an orchestration op from a kernel body |

## Notes

- **No compiler plugin here.** `program { }` captures its body with the
  `:autograd` runtime tracer, so this example depends on `core` + `autograd` +
  `maestro` and nothing else. The plugin's role in the four worlds is
  compile-time scope discipline, which is what the commented-out block
  exercises.
- **No cluster is contacted.** The example prints each step's StableHLO body
  and manifest hash; a Maestro workflow's `Tlaloc` step is what would run them.
  Nothing in this example needs a network.
- The value falls out of the workflow because `step(...)` runs each artifact's
  shim in process as it composes.
