package io.tlaloc.runtime.pjrt

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirFunction
import io.tlaloc.stablehlo.toStablehlo
import java.nio.file.Files

/**
 * High-level dxir → PJRT-XLA bridge. Mirrors `runOnIree`'s API shape:
 *
 *   inputs / outputs flow as flat row-major [FloatArray]s sized by
 *   `param.type.elementCount` / `return.type.elementCount`.
 *
 * Pipeline:
 *  1. Validate input arity + sizes against [DxirFunction.params].
 *  2. Emit StableHLO MLIR via [DxirFunction.toStablehlo].
 *  3. Stage MLIR + each input as `.npy` on disk.
 *  4. Spawn `pjrt_dispatch.py` (PJRT-XLA backend, target = [target]).
 *  5. Read output `.npy` files into [FloatArray]s.
 *
 * v1 limitations:
 *   - F32 only.
 *   - Recompile per call (each invocation pays python + jax import + JIT).
 *     Use a benchmark wrapper that runs many iterations inside one
 *     subprocess for per-iteration timing.
 *   - Output shape inference is hardcoded to f32 + the function's declared
 *     return types. Does not validate against actual on-disk shapes from
 *     the dispatcher; a shape mismatch surfaces later as a size-mismatch
 *     error in the FloatArray decoder.
 */
fun runOnPjrt(
    fn: DxirFunction,
    inputs: List<FloatArray>,
    target: PjrtTarget = PjrtTarget.Cuda,
    timeoutSeconds: Long = 300L,
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

    val mlir = fn.toStablehlo("")
    val module = PjrtRuntime.compile(mlir, target)

    val workDir = Files.createTempDirectory("tlaloc-pjrt-bridge-")
    workDir.toFile().deleteOnExit()
    val inputPaths = fn.params.zip(inputs).mapIndexed { i, (p, arr) ->
        val target = workDir.resolve("input_${"%03d".format(i)}_${p.name}.npy")
        NpyWriter.writeFloat32(target, arr, p.type.dims)
        target.toFile().deleteOnExit()
        target
    }

    val outputPaths = PjrtRuntime.invoke(
        module = module,
        function = fn.name,
        inputs = inputPaths,
        nOutputs = fn.returns.size,
        timeoutSeconds = timeoutSeconds,
    )

    return outputPaths.zip(fn.returns).map { (path, ret) ->
        val arr = NpyWriter.readFloat32(path)
        val expected = ret.type.elementCount.toInt()
        require(arr.size == expected) {
            "runOnPjrt: output for return type ${ret.type} arrived with " +
                "${arr.size} floats; expected $expected"
        }
        arr
    }
}
