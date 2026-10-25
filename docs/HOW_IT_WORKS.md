# How Tlaloc runs a program

What happens between the Kotlin you write and the numbers that come back: when
derivatives are taken, when a program is compiled, where tensors live, and what
crosses between the host and a device.

## Two ways to a derivative

| | `grad { }` | `capture(...)` (`:nn`, `:autograd`) |
|---|---|---|
| Taken | at compile time, by the K2 compiler plugin | at run time, by tracing the model once |
| Becomes | Kotlin bytecode in your class files | a `DxirFunction` (the gradient program) |
| Runs | on the JVM, over host tensors | in the host interpreter, or on a device through `PjrtSession` |
| Shapes | symbolic: one gradient serves every shape | concrete: the traced shapes |
| Read it | `tlaloc { dumpGradSource.set(true) }` | `step.gradSource()` |

### `grad { }`

The plugin lowers the lambda to DXIR, Tlaloc's tensor IR, while Kotlin compiles your
file. It coarsens loops and branches, takes the reverse-mode derivative, and writes the
result back as ordinary Kotlin IR that replaces the call. At run time there is no
plugin, no interpreter and no device: the gradient is a function in your jar that calls
`io.tlaloc.core.ops` on host tensors. Values the lambda captures (immutable local
`Float`, `Double`, `Int` or `Long` values) become parameters read at the call.

When a step fails the build fails at the call's file and line, unless
`strictLowering` is off; then the call compiles, and throws when it runs.

### `capture`

`capture(model, inputs) { prediction -> loss }` runs the model once on tracers, records
the program, and derives its gradient. The result, a `CapturedStep`, holds both as
`DxirFunction`s with the traced shapes:

<!-- doc-test: how-capture -->

```kotlin
val step = capture(model, listOf(x)) { prediction -> (prediction * prediction).mean() }

val out = step.run(model, listOf(x))   // loss and gradients, in the host interpreter
val text = step.gradient.toStablehlo() // the program a device compiles
```

Capture once and run every step; only the values change. A new input shape, or a
target baked in as a constant, needs a new capture and, on a device, a new compile.

## On a device: `PjrtSession`

A `PjrtSession` (`:runtime-pjrt`) loads a PJRT plugin through the JDK's foreign
function interface (no JNI, no Python), creates a client on one device and compiles
programs for it.

<!-- doc-test: how-session -->

```kotlin
val inputs = listOf(x.hostF32()) + step.parameterTensors(model).map { it.hostF32() }

PjrtSession(target = PjrtTarget.Cuda).use { session ->
    val first = session.runOn(step.gradient, inputs)   // compiles, then runs
    val again = session.runOn(step.gradient, inputs)   // runs the same executable
    println(session.cacheSize)                         // 1
}
```

### What a session compiles

Before compiling, a session lowers TPU kernel ops (on other targets an op that allows
it is replaced by its reference program; one that does not is refused by name), emits
StableHLO, and on TPU and Metal rewrites the explicit
f32 dot algorithm into a portable precision. It runs no pattern recognition and no
kernel selection; those are separate passes you call yourself
([CUSTOM_KERNELS.md](CUSTOM_KERNELS.md)).

A program is compiled the first time it is run, or ahead of time with
`prepare(fn)`. Executables are kept for the session's life, keyed by the StableHLO text,
so any change to the text (a shape, a dtype, a constant) is a new compile. A
`cacheKey` argument skips re-emitting the text on later calls; the key must then
determine the program.

With `TLALOC_PJRT_CACHE_DIR` set, compiled executables are also kept on disk and
loaded by later processes ([PERFORMANCE.md](PERFORMANCE.md#seeing-what-was-compiled)).

### Calls are synchronous

Every call returns when the device has finished. An upload returns once the device
has the data; an execution returns once its results are complete; a download returns
with the values.

### What crosses to the host

| Call | Moves |
|---|---|
| `runOn`, `runOnHost`, `runOnF64`, `runOnBf16` | every input up, every output down, on every call |
| `bufferFromHostF32`, `bufferFromHostI32`, … | one array up; the `PjrtBuffer` stays on the device |
| `executeOn(fn, buffers)`, `executeStablehlo(text, buffers)` | nothing; the outputs are new device buffers |
| `PjrtBuffer.toFloatArray(n)`, … | one buffer down |

A training loop over `runOn` moves every parameter up and every gradient down on every
step. Keeping them on the device is the difference between 25.78 ms and 2.22 ms a step
for a small transformer ([the case study](case-studies/training-step-on-device.md)).

### Buffers and donation

A `PjrtBuffer` belongs to whoever created it; the session does not track it. Close
buffers before the session. A closed buffer is refused by name (`PjrtBuffer used after
close()`), as is a buffer of a closed session.

A program can hand an input's memory to an output: `toStablehlo(outputAliases =
mapOf(param to result))` marks the parameter `tf.aliasing_output`, and PJRT then writes
the result into the input's buffer. The input is consumed by the call; close it and do
not pass it again. Only `executeStablehlo` runs such text; `runOn`, `executeOn` and
`prepare` emit without aliases, so they never donate.
`CapturedStep.withDeviceAdamW(optimizer)` builds a training step whose parameters and
moments are donated this way, with the learning rate and bias corrections as inputs, so
a schedule never recompiles.

### Device memory

> [!NOTE]
> With no options, the CUDA plugin reserves 75 % of GPU memory when a client is created.
> On a machine with unified memory, such as the GB10, that is host RAM, and every
> session's client reserves its own. Tlaloc therefore always passes options: no
> preallocation (the pool grows as needed and is freed when the session closes) and at
> most half the memory.

| Variable | Default | |
|---|---|---|
| `TLALOC_PJRT_MEMORY_FRACTION` | `0.5` | the most the client may use, in (0, 1] |
| `TLALOC_PJRT_PREALLOCATE` | `false` | reserve the fraction at creation |

`session.memoryStats()` returns the bytes in use and the peak. XLA's autotuning can need
more memory at a program's first compile than the program needs to run.

### Threads

Every session method may be called from several threads. Calls of the same executable
run one at a time; different executables do not wait for each other. `close()` waits
for calls in flight.

## Other backends

| Backend | Compiles | Notes |
|---|---|---|
| `DxirInterpreter` | nothing | f32, op by op; the reference the device tests compare with |
| `DxirInterpreterF64` | nothing | computes in Double, rounds f32 and bf16 nodes after each op |
| `runOnPjrt(fn, inputs)` | on every call | a new client per call (about 0.5 s); for one-off runs |
| `runOnIree(fn, inputs)` | on every call | `iree-compile` and `iree-run-module` as subprocesses |

## Precision by backend

| | f32 dot | f64 |
|---|---|---|
| JVM (`grad { }`), interpreters | f32 | Double |
| PJRT CUDA (GB10) | TF32 for a plain `MATMUL` | Double; dots about 9× slower than f32 |
| PJRT Metal, Cerebras | not measured | computed in f32 |
| PJRT TPU | one bf16 pass | not run |

[NUMERICAL_PRECISION.md](NUMERICAL_PRECISION.md) has the details and the tolerances
the tests use.
