# FAQ

### Where does `grad { }` run?

On the JVM. The compiler plugin writes the gradient as Kotlin while your code compiles,
and it runs on host tensors like any other function. To run a gradient on a GPU, capture
the model (`capture` in `:nn`) and run the captured program with a `PjrtSession`.
[HOW_IT_WORKS.md](HOW_IT_WORKS.md) explains both paths.

### Does it need Python?

Not at run time. The PJRT CUDA plugin is a shared library that ships in JAX's CUDA
wheel, so the usual way to get it is `pip install "jax[cuda12]"` into a virtual
environment ([GETTING_STARTED.md](GETTING_STARTED.md#5-running-on-a-gpu)); Tlaloc loads
the library and nothing else from it. A served model needs a PJRT plugin and a driver:
the Python serving runtime calls the PJRT C API through ctypes, with no JAX, PyTorch or
NumPy.

### Can I use it from Java?

The libraries are Java 21 bytecode, except the GPU runtimes (25). `grad { }` needs the Kotlin compiler plugin, so
gradients are written in Kotlin; a model exported for serving loads and runs from plain
Java ([`examples/java-inference`](../examples/java-inference/)).

### Which Kotlin and JDK versions?

Kotlin 2.4.20 to 2.4.29 for `0.1.0-alpha02` (the plugin uses compiler internals, so other
versions are refused at compile time). JDK 25 to build code that uses `grad { }`, JDK 21
to run it, and JDK 25 for GPU execution.

### Which hardware has it run on?

NVIDIA GPUs through PJRT and IREE, Apple-silicon GPUs through MetalHLO, the Cerebras
simulator, and the CPU interpreter. Nothing has run on a TPU or on more than one GPU.
[CAPABILITIES.md](CAPABILITIES.md#where-the-tests-have-been-executed) lists where each
test suite has been executed.

### Why do GPU results differ from the interpreter's?

On the GB10 a plain f32 matrix multiply runs at TF32 precision, and XLA fuses and
reorders sums; the tests allow 1e-2 relative for a dot there.
[NUMERICAL_PRECISION.md](NUMERICAL_PRECISION.md) has what each backend computes.

### Why does my program compile again?

A device executable is compiled for exact shapes and is keyed by the program's text, so
a new input shape, or a value baked into the program as a constant, is a new compile.
Bucket varying sizes ([DYNAMIC_SHAPES.md](DYNAMIC_SHAPES.md)), and keep compiled
executables across processes with `TLALOC_PJRT_CACHE_DIR`
([PERFORMANCE.md](PERFORMANCE.md#seeing-what-was-compiled)).

### My training loop on the GPU is slow.

Check what crosses the host each step. `runOn` uploads every input and downloads every
output; with AdamW on the host that is the whole model twice a step. Keeping the state on
the device made a small transformer's step 11.6× faster
([case study](case-studies/training-step-on-device.md)).

### Can I read the derivative?

Yes. `tlaloc { dumpGradSource.set(true) }` prints the gradient of each `grad { }` as
Kotlin, and `step.gradSource()` prints a captured model's
([READABLE_REVERSE.md](READABLE_REVERSE.md)).

### Can I write my own derivative or kernel?

`customVjp` and `customJvp` replace the derivative of a function inside `grad { }`;
KPTX kernels and Mosaic kernels replace operations on a GPU or a TPU
([CUSTOM_KERNELS.md](CUSTOM_KERNELS.md)).

### Several GPUs or hosts?

Tlaloc emits and differentiates sharded programs and collectives; its runtime executes
on one device, and they have run on several virtual CPU devices in the tests only
([DISTRIBUTED.md](DISTRIBUTED.md)).

### Is the API stable?

No: this is an alpha, and names and signatures can change between alphas.
[COMPATIBILITY.md](COMPATIBILITY.md) says what may change, and what is marked
`@ExperimentalTlalocApi`.

### Where do I report a problem?

Open an issue in the repository; [CONTRIBUTING.md](../CONTRIBUTING.md) has how to
contribute, and [SECURITY.md](../SECURITY.md) how to report a vulnerability.
[FOLLOWUPS.md](FOLLOWUPS.md) lists the issues already known.
