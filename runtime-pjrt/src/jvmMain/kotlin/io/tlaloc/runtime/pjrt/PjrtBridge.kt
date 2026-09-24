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
 *   - [LlvmCpu] — XLA's `cpu` backend (LLVM-based codegen). **Currently
 *     unsupported on the FFM path** because OpenXLA does not ship a
 *     standalone `pjrt_plugin_xla_cpu.so` artifact and JAX's CPU PJRT
 *     impl is in-process Python. Plan: build a CPU PJRT plugin from XLA
 *     source (`xla/pjrt/c/pjrt_c_api_cpu.{cc,h}`) as a separate
 *     deliverable; the FFM bindings here are CPU/CUDA-agnostic and will
 *     pick it up by setting `TLALOC_PJRT_PLUGIN_PATH` once it exists.
 *   - [Cuda]    — XLA's `cuda` backend on NVIDIA GPUs. The default;
 *     the plugin resolved by [PjrtBinaries] is `xla_cuda_plugin.so`
 *     (bundled with `pip install jax[cuda12]`).
 *   - [Tpu]     — Google TPUs via libtpu's PJRT plugin.
 *     Plugin resolution, platform-name expectation, create-options
 *     gating and the self-skipping smoke suite are tested; execution on
 *     a TPU has not been certified (see docs/TPU_BRINGUP.md). Plugin resolved by
 *     [PjrtBinaries.tpuPluginPath]: `TLALOC_PJRT_PLUGIN_PATH` (when it
 *     names a tpu-shaped .so) or the libtpu default install locations —
 *     the PyPI `libtpu` wheel ships `site-packages/libtpu/libtpu.so`,
 *     and TPU VM images carry `/lib/libtpu.so` (older images also spell
 *     the plugin `pjrt_c_api_tpu_plugin.so`). `PJRT_Client_PlatformName`
 *     for libtpu is `"tpu"` — the smoke suite asserts exactly that.
 *
 * The single-plugin model is XLA's design — each PJRT plugin .so registers
 * for one platform string at Client_Create time. CPU + CUDA + TPU each
 * needs its own plugin .so.
 */
enum class PjrtTarget(val platform: String) {
    LlvmCpu(platform = "cpu"),
    Cuda(platform = "cuda"),
    Tpu(platform = "tpu"),
}

/**
 * High-level dxir → PJRT-XLA bridge. Mirrors [io.tlaloc.runtime.iree.runOnIree]'s
 * API shape:
 *
 *   inputs / outputs flow as flat row-major [FloatArray]s sized by
 *   `param.type.elementCount` / `return.type.elementCount`.
 *
 * Built on the [PjrtFfm] bindings: no Python subprocess and no `.npy` files
 * crossing process boundaries. The dispatch path is pure-Kotlin via
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
 * Limitations:
 *   - F32 only.
 *   - Per-call client + executable creation (~500 ms of XLA service init).
 *     For per-iteration timing, use [PjrtFfm] / [PjrtClient] directly so the
 *     client + compiled executable persist across many invocations.
 *
 * Per-call cost note: each call pays ~500 ms (XLA service init + CUDA
 * context create). For a long-lived dispatch loop, hoist the
 * `Arena.ofShared() → PjrtFfm.load → createClient → compile` setup outside
 * the loop and reuse [PjrtLoadedExecutable.execute] inside.
 */
fun runOnPjrt(
    fn: DxirFunction,
    inputs: List<FloatArray>,
    target: PjrtTarget = PjrtTarget.Cuda,
): List<FloatArray> {
    // §0.4.459 (G2a) — named refusal: this one-shot path resolves the
    // CUDA-family plugin ([PjrtBinaries.pluginPath]) and always passes GPU
    // allocator create-options, both wrong for a TPU client. The TPU lane
    // is PjrtSession(plugin = PjrtBinaries.tpuPluginPath, target = Tpu).
    require(target != PjrtTarget.Tpu) {
        "runOnPjrt: PjrtTarget.Tpu is not supported on the one-shot runOnPjrt path — " +
            "use PjrtSession(plugin = PjrtBinaries.tpuPluginPath!!, target = PjrtTarget.Tpu) " +
            "(see docs/TPU_BRINGUP.md)"
    }
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

    val plugin = PjrtBinaries.requireCudaPlugin("runOnPjrt")

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
