package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirFunction
import io.tlaloc.runtime.pjrt.ffm.PjrtFfm
import io.tlaloc.stablehlo.toStablehlo
import java.lang.foreign.Arena

/**
 * Compile + dispatch target for [runOnPjrt]. Mirrors `IreeTarget` but for the
 * PJRT-XLA backend.
 *
 *   - [LlvmCpu] — XLA's `cpu` backend (LLVM-based codegen). Cheap to compile,
 *     useful for correctness checks / no-GPU hosts.
 *   - [Cuda]    — XLA's `cuda` backend on NVIDIA GPUs. The default; the
 *     plugin resolved by [PjrtBinaries] is `xla_cuda_plugin.so`.
 *
 * §0.4.305 — the underlying plugin is a single shared library that decides
 * its `platform_name` at Client_Create time. The same plugin .so serves both
 * targets (XLA's PJRT plugin can register for multiple backends). The enum
 * is kept in case future Tlaloc work needs separate plugins for CPU vs CUDA
 * (e.g. a TPU plugin), but v1 just uses the bundled CUDA plugin for both.
 */
enum class PjrtTarget(val platform: String) {
    LlvmCpu(platform = "cpu"),
    Cuda(platform = "cuda"),
}

/**
 * High-level dxir → PJRT-XLA bridge. Mirrors [io.tlaloc.runtime.iree.runOnIree]'s
 * API shape:
 *
 *   inputs / outputs flow as flat row-major [FloatArray]s sized by
 *   `param.type.elementCount` / `return.type.elementCount`.
 *
 * §0.4.305 reimplemented this on top of the §0.4.303/§0.4.304 FFM bindings —
 * no Python subprocess, no `.npy` files crossing process boundaries, no
 * `pjrt_dispatch.py` script. The dispatch path is pure-Kotlin via
 * `java.lang.foreign` straight into the bundled `xla_cuda_plugin.so`.
 *
 * Pipeline:
 *  1. Validate input arity + sizes against [DxirFunction.params].
 *  2. Resolve PJRT plugin path via [PjrtBinaries.pluginPath].
 *  3. Open a confined Arena scoped to this call.
 *  4. PjrtFfm.load → createClient → addressableDevices.first.
 *  5. Compile [fn.toStablehlo].
 *  6. Stage each input as a PjrtBuffer via bufferFromHostF32.
 *  7. Execute, await device-complete event.
 *  8. Read each output as FloatArray via PjrtBuffer.toFloatArray.
 *
 * v1 limitations (still applicable post-FFM):
 *   - F32 only.
 *   - Per-call client + executable creation (~500 ms of XLA service init).
 *     For per-iteration timing, use [PjrtFfm] / [PjrtClient] directly so the
 *     client + compiled executable persist across many invocations.
 *
 * Per-call cost note: the §0.4.302 subprocess facade paid ~5 s of cold JAX
 * import per call; this FFM path pays ~500 ms (XLA service init + CUDA
 * context create). For a long-lived dispatch loop, hoist the
 * `Arena.ofShared() → PjrtFfm.load → createClient → compile` setup outside
 * the loop and reuse [PjrtLoadedExecutable.execute] inside.
 */
fun runOnPjrt(
    fn: DxirFunction,
    inputs: List<FloatArray>,
    target: PjrtTarget = PjrtTarget.Cuda,
    @Suppress("UNUSED_PARAMETER") timeoutSeconds: Long = 300L,
): List<FloatArray> {
    require(fn.params.size == inputs.size) {
        "runOnPjrt: param count ${fn.params.size} != input count ${inputs.size}"
    }
    for ((i, p) in fn.params.withIndex()) {
        require(p.type.dtype == F32) {
            "runOnPjrt: param '${p.name}' dtype is ${p.type.dtype}; v1 only supports F32"
        }
        val expected = p.type.elementCount.toInt()
        require(inputs[i].size == expected) {
            "runOnPjrt: param '${p.name}' expects size $expected (type ${p.type}) " +
                "but received input of size ${inputs[i].size}"
        }
    }
    for ((i, r) in fn.returns.withIndex()) {
        require(r.type.dtype == F32) {
            "runOnPjrt: return[$i] dtype is ${r.type.dtype}; v1 only supports F32"
        }
    }

    val plugin = PjrtBinaries.pluginPath
        ?: error("PJRT plugin not resolved; set TLALOC_PJRT_PLUGIN_PATH or `pip install jax[cuda12]` into ~/.local/venvs/iree")

    val mlir = fn.toStablehlo("")

    Arena.ofConfined().use { arena ->
        val api = PjrtFfm.load(plugin, arena)
        api.createClient().use { client ->
            val devices = client.addressableDevices()
            require(devices.isNotEmpty()) { "PJRT client has no addressable devices for $target" }
            val device = devices.first()

            client.compile(mlir).use { exec ->
                require(exec.numOutputs == fn.returns.size) {
                    "runOnPjrt: PJRT executable reports numOutputs=${exec.numOutputs}; " +
                        "DxirFunction declares ${fn.returns.size} returns"
                }

                // Upload inputs.
                val inputBuffers = fn.params.zip(inputs).map { (p, arr) ->
                    client.bufferFromHostF32(device, arr, p.type.dims)
                }
                try {
                    val outputs = exec.execute(inputBuffers, device)
                    try {
                        return outputs.zip(fn.returns).map { (buf, ret) ->
                            buf.toFloatArray(ret.type.elementCount.toInt())
                        }
                    } finally {
                        outputs.forEach { it.close() }
                    }
                } finally {
                    inputBuffers.forEach { it.close() }
                }
            }
        }
    }
}
