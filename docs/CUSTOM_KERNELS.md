# Custom derivatives and kernels

Three ways to replace what Tlaloc would generate: a derivative rule of your own, a
GPU kernel written in Kotlin (KPTX), and a TPU kernel (Mosaic). Each says where it
runs.

| | Runs on | Status |
|---|---|---|
| `customVjp`, `customJvp`, `customVjpJvp` | the JVM (`grad { }`) and the interpreters | tested; not emitted to StableHLO |
| KPTX kernels as XLA custom calls | NVIDIA GPUs through PJRT | tested on the GB10; opt in per program |
| Mosaic kernels (`MOSAIC_KERNEL`) | TPU | checked on the CPU; never run on a TPU |

## Your own derivative

`customVjp(f, vjp)` inside `grad { }` makes the reverse mode use `vjp` for `f`.
`vjp(u, x)` receives the cotangent of `f`'s result and `f`'s input, and returns the
cotangent of the input:

<!-- doc-test: kern-custom-vjp -->

```kotlin
val g = grad { x: DTensor<Rank1<Sym>, F32> ->
    val clippedSquare = customVjp(
        { t: DTensor<Rank1<Sym>, F32> -> t * t },
        { u: DTensor<Rank1<Sym>, F32>, t: DTensor<Rank1<Sym>, F32> -> u * 3.0f },  // not 2t: the rule wins
    )
    clippedSquare(x).sum().toFloat()
}
g(Tensors.f32Vector(floatArrayOf(1f, 2f, 3f)))   // [3.0, 3.0, 3.0]
```

| Function | Supplies | For |
|---|---|---|
| `customVjp(f, vjp)`, `customVjp2` | the reverse rule | `grad`, `valueAndGrad` |
| `customJvp(f, jvp)`, `customJvp2` | the forward rule, `jvp(x, dx)` | `jvp`, `jacobian` |
| `customVjpJvp(f, vjp, jvp)`, `customVjpJvp2` | both | all of the above, and `hessian` |

- Both lambdas are written at the call. The result may be held in a local `val` and
  applied in the same lambda; it may not be passed on.
- Values captured from outside must be compile-time constants, and `vjp` must be
  straight-line code.
- Asking for the mode the rule does not supply (`jvp { }` over a `customVjp`) is a
  build error that names it.
- A stop-gradient is `customVjp({ it }, { u, _ -> u * 0f })`.
- `checkCustomVjp(f, vjp, at)` compares a rule with central differences on the host
  (a rule that is deliberately not the derivative, such as a stop-gradient, fails it).
- The rule is kept as one coarsened op whose gradient is your body. StableHLO emission
  refuses it, so a custom rule runs on the JVM and in the interpreters, not on a device.

## KPTX: GPU kernels in Kotlin

`:kptx` builds PTX from Kotlin: a kernel is a list of PTX instructions, each checked
against an ISA table when it is added, and emitted as PTX text that the CUDA driver
compiles when the module is loaded (no CUDA toolkit at run time).

<!-- doc-test: kern-kptx -->

```kotlin
val scale = PtxKernelTemplate("scale_by_two") { _ ->
    val xPtr = param(".u64", "x_ptr")
    val outPtr = param(".u64", "out_ptr")
    val nP = param(".u32", "n")
    val p = pred()
    val (i, n) = List(2) { r32() }
    val (a, b) = List(2) { r64() }
    val v = f32()
    val done = label("done")
    inst("mov.u32", i, tidX)
    inst("ld.param.u32", n, mem(nP))
    inst("setp.ge.u32", p, i, n)
    inst("bra", done, guard = p)
    inst("ld.param.u64", a, mem(xPtr))
    inst("cvta.to.global.u64", a, a)
    inst("mul.wide.u32", b, i, imm(4))
    inst("add.u64", a, a, b)
    inst("ld.global.f32", v, mem(a))
    inst("add.f32", v, v, v)
    inst("ld.param.u64", a, mem(outPtr))
    inst("cvta.to.global.u64", a, a)
    inst("add.u64", a, a, b)
    inst("st.global.f32", mem(a), v)
    place(done)
    inst("ret")
}
val ptx = scale.specialize(arch = "sm_75").emitPtx()
```

Registered with the PJRT plugin, it is an XLA custom call (this one uses only the
thread index, so one block of 256 threads covers up to 256 elements):

<!-- doc-test: kern-kptx-run -->

```kotlin
KptxKernelRegistry.registerKernel(
    PjrtBinaries.pluginPath!!,
    "scale_by_two",
    KptxKernelRegistry.LaunchConfig(
        ptx = ptx,
        entryName = "scale_by_two",
        grid = { KptxKernelRegistry.Dim3(1) },
        block = KptxKernelRegistry.Dim3(256),
        trailingI32Params = { args -> intArrayOf(args[0].dims.fold(1L) { a, d -> a * d }.toInt()) },  // n
    ),
)
val call = """
    func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
      %0 = stablehlo.custom_call @scale_by_two(%arg0) {api_version = 4 : i32} : (tensor<8xf32>) -> tensor<8xf32>
      return %0 : tensor<8xf32>
    }
""".trimIndent()
val doubled = PjrtSession(target = PjrtTarget.Cuda).use { session ->
    session.bufferFromHostF32(FloatArray(8) { it.toFloat() }, listOf(8)).use { x ->
        session.executeStablehlo(call, listOf(x)).single().use { it.toFloatArray(8) }
    }
}
// [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
```

`KptxKernels` has the library's kernels (RMSNorm forward and backward, RoPE, cross
entropy, attention, paged attention), each a `PtxKernelTemplate` specialized by shape
and architecture.

A kernel becomes an XLA custom call in two steps:

1. **Register it** with the PJRT plugin before compiling a program that calls it:
   `KptxKernelRegistry.registerKernel(pluginPath, name, LaunchConfig(ptx, entryName,
   grid, block, …))`. The kernel's parameters are the call's inputs, then its outputs,
   then the trailing `i32`s the config computes. Launches go on XLA's stream; the
   registry allocates nothing. `PjrtFfiRegistry.registerExecuteHandler` registers any
   Kotlin function as a typed-FFI handler the same way, and `registerNativeLibrary` a
   native library's kernels.
2. **Call it** from a program. `lowerKernelChoice(fn, target, registry)` gives each
   coarsened op (from the pattern recognizers: attention, RMSNorm, RoPE, cross entropy)
   to a `KernelTemplate` that may claim it for the target; a claimed op is emitted as
   `stablehlo.custom_call @name`, an unclaimed one is inlined. `KernelTarget` names the
   device (`NVIDIA_GB10`, `NVIDIA_H100`, `GOOGLE_TPU_V6E`, …). Hand-written StableHLO
   can call a registered target directly through `PjrtSession.executeStablehlo`.

`PjrtSession` does not run `lowerKernelChoice`; the KPTX templates are in no default
registry. The KPTX paged attention kernel is 1.4–1.9× faster than XLA's lowering at
Llama-3-8B shapes and slower at TinyLlama's, and the ratio moved between 0.90 and 2.19
within an hour, so it is registered only on request
([design/KPTX_PAGED_PERF.md](design/KPTX_PAGED_PERF.md)). Each custom call costs a
dispatch, and XLA cannot fuse across it: a LlamaDecoder forward with its RMSNorms
claimed took 1,177 µs against 906 µs decomposed, about 135 µs per custom call
([plans/KPTX_PLAN.md](plans/KPTX_PLAN.md)).

Kernels are claimed for the forward pass only: the reverse transform inlines a
coarsened op's gradient body, so no backward op is left to claim.

## Mosaic kernels for TPU

`DxirEmitter.mosaicKernel(operands, kernel, reference)` adds a `MOSAIC_KERNEL` op: a
serialized Mosaic module (exported from Pallas, or emitted by Tlaloc's
`MosaicRmsNorm`) that a TPU session compiles as `tpu_custom_call`. On other targets the
op runs its `reference` program when created with `referenceFallback = true`, and is
refused by name otherwise; the interpreter runs the reference. The custom call Tlaloc
writes is checked against `jax.export`'s, and Pallas's interpret mode against numpy,
on the CPU; no Mosaic kernel has run on a TPU. Neither transform differentiates the op.
[design/TPU_MEGAKERNELS.md](design/TPU_MEGAKERNELS.md) has the kernels and what is left.
