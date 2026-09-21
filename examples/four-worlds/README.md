# four-worlds — where your code lives, and what may cross between

**What it shows.** Tlaloc splits a distributed program into four scopes, each
one an ordinary Kotlin receiver type:

| World | What belongs there | How you enter it |
|---|---|---|
| **Kernel** | the maths — elementwise ops, matmuls, reductions | the body lambda of `program { }` |
| **Orchestration** | building Kernel bodies into shippable artifacts | `Tlaloc.program(...)` |
| **Program** | composing artifacts into a graph | `Tlaloc.workflow { }` |
| **Cluster** | running the graph on real hardware | the emitted Maestro descriptor |

Two consequences, both visible in this example:

- **Calling into the wrong world is a compile error.** `program` is an
  extension on `OrchestrationScope`, so it cannot be called from inside a
  Kernel body — there is no orchestration receiver in there. No runtime guard,
  no lint rule.
- **What crosses a step boundary is never a raw tensor.** It is a
  `BufferHandle<T, M>`, carrying both the value's type and the mesh `M` it
  lives on. Feed a step a handle of the wrong shape or the wrong mesh and
  Kotlin's own checker rejects the call site.

The run prints: one step built and executed; two steps composed with a handle
flowing between them (and the recorded edge); and the Maestro JSON descriptor a
real cluster ingests.

## Running it

This is a standalone Gradle project resolving Tlaloc from **mavenLocal**, so
publish first:

```bash
# from the repo root
./gradlew publishToMavenLocal
./gradlew -p examples/four-worlds run
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

[3] the Maestro descriptor a cluster would ingest (2167 chars, first 240):
    {"properties":{"owner":"tlaloc"},"workflow":{"id":"tlaloc_activate_then_score","name":"activate_then_score","steps":[{"step":{"id":"activate","type":"Kubernetes","params":{"image":{"value":"tlaloc-runtime:0.0.1","type":"STRING"},"tlaloc_art...

[4] the two programs that do NOT compile: see the block at the bottom
    of src/main/kotlin/Main.kt — uncomment either and run again.
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
- **No cluster is contacted.** `MaestroDescriptor.emit` produces the JSON; a
  real Maestro instance would be the thing that consumes it and launches each
  step's artifact as a Kubernetes job. Nothing in this example needs a network.
- The value falls out of the workflow because `step(...)` runs each artifact's
  shim in process as it composes. `StubExecutor`, which older snippets used for
  this, is deprecated and deliberately not used here.
