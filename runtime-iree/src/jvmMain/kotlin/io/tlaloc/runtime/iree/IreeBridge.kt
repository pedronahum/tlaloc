package io.tlaloc.runtime.iree

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.stablehlo.toStablehlo
import java.nio.file.Files

/**
 * High-level bridge from a [DxirFunction] to IREE-CPU dispatch via [IreeRuntime].
 * Mirrors the API shape of [io.tlaloc.ir.passes.DxirInterpreter.evalFunction] —
 * inputs/outputs flow as flat row-major [FloatArray]s sized by `type.elementCount` —
 * so call sites can drop-in swap the interpreter for native IREE dispatch.
 *
 * Pipeline:
 *  1. Validate input arity + sizes against [DxirFunction.params].
 *  2. Emit StableHLO MLIR via [DxirFunction.toStablehlo].
 *  3. Compile via [IreeRuntime.compile] (subprocess `iree-compile`); the module's
 *     temporary directory is deleted when the call returns.
 *  4. Marshal each input to IREE's textual `<shape>xf32=<v0>,<v1>,…` form.
 *  5. Invoke via [IreeRuntime.invoke] (subprocess `iree-run-module`).
 *  6. Parse each output's `<shape>xf32=…` line back to a flat [FloatArray].
 *
 * v1 limitations (relax in §0.4.288+):
 *  - F32 only. F64 / I32 / I64 / Bool tensors will throw at the validation step.
 *  - Element values that round-trip as `nan` / `inf` aren't supported (Float.toFloat
 *    rejects those literals; IREE's textual writer emits them as unquoted `nan`/`inf`).
 *  - Compiles from scratch on every call. For repeated dispatch with fixed [fn],
 *    factor out the [IreeRuntime.compile] step yourself and cache the [IreeModule].
 */
fun runOnIree(
    fn: DxirFunction,
    inputs: List<FloatArray>,
    target: IreeTarget = IreeTarget.LlvmCpu,
    timeoutSeconds: Long = 300L,
    useNpyInputs: Boolean = false,
): List<FloatArray> {
    require(fn.params.size == inputs.size) {
        "runOnIree: param count ${fn.params.size} != input count ${inputs.size}"
    }
    for ((i, p) in fn.params.withIndex()) {
        require(p.type.dtype == F32) {
            "runOnIree: param '${p.name}' dtype is ${p.type.dtype}; v1 only supports F32 — " +
                "promote at the call site or extend formatInput/parseOutput for new dtypes"
        }
        val expected = p.type.elementCount.toInt()
        require(inputs[i].size == expected) {
            "runOnIree: param '${p.name}' expects size $expected (type ${p.type}) " +
                "but received input of size ${inputs[i].size}"
        }
    }
    for ((i, r) in fn.returns.withIndex()) {
        require(r.type.dtype == F32) {
            "runOnIree: return[$i] dtype is ${r.type.dtype}; v1 only supports F32"
        }
    }

    val mlir = fn.toStablehlo("")
    IreeRuntime.compile(mlir, target, timeoutSeconds = timeoutSeconds).use { module ->
        // §0.4.294 — for medium / large configs, the textual `<shape>xf32=v0,v1,…`
        // form blows past flagfile-parse limits (medium = ~24 MB raw f32 → ~240 MB
        // textual). When [useNpyInputs] is true, write each input to a temp `.npy`
        // file (NPY 1.0 little-endian f32) and pass `--input=@<path>` instead.
        // iree-run-module / iree-benchmark-module accept this directly.
        val npyDir = if (useNpyInputs) Files.createTempDirectory("tlaloc-iree-bridge-npy-") else null
        try {
            val inputArgs: List<String> = if (npyDir != null) {
                fn.params.zip(inputs).mapIndexed { i, (p, arr) ->
                    val file = npyDir.resolve("input_${"%03d".format(i)}_${p.name}.npy")
                    NpyWriter.writeFloat32(file, arr, p.type.dims)
                    "@$file"
                }
            } else {
                fn.params.zip(inputs).map { (p, arr) -> formatInput(p.type, arr) }
            }
            val rawOutputs = IreeRuntime.invoke(
                module, function = fn.name, inputs = inputArgs, timeoutSeconds = timeoutSeconds,
            )

            require(rawOutputs.size == fn.returns.size) {
                "runOnIree: iree-run-module returned ${rawOutputs.size} results; expected ${fn.returns.size}"
            }
            return fn.returns.zip(rawOutputs).map { (retNode, raw) -> parseOutput(retNode.type, raw) }
        } finally {
            npyDir?.let { IreeRuntime.deleteRecursively(it) }
        }
    }
}

/** `<dim0>x<dim1>x…xf32=v0,v1,…` for rank ≥ 1; `f32=v` for scalars. */
internal fun formatInput(type: DxirType, arr: FloatArray): String {
    require(type.elementCount.toInt() == arr.size) {
        "formatInput: type elementCount ${type.elementCount} != FloatArray size ${arr.size}"
    }
    val shapePrefix = if (type.dims.isEmpty()) "" else type.dims.joinToString("x") + "x"
    return shapePrefix + "f32=" + arr.joinToString(",")
}

/**
 * Parse an `iree-run-module` result line. Formats observed on IREE 3.11.0:
 *   - rank-0: `f32=3` (single value after `=`)
 *   - rank-1: `4xf32=11 22 33 44` (space-separated values)
 *   - rank-2+: `2x3xf32=[2 4 6][8 10 12]`, `2x2x2xf32=[[2 4][6 8]][[10 12][14 16]]`
 *     (nested brackets per axis, leaves space-separated)
 *
 * Strategy: drop the `…f32=` prefix, strip every `[`/`]`, split on whitespace and
 * commas, parse each token as Float. Validate the count against [type.elementCount]
 * so a parse mismatch surfaces a precise diagnostic instead of a silent miscompare.
 */
internal fun parseOutput(type: DxirType, raw: String): FloatArray {
    val eqIdx = raw.indexOf('=')
    require(eqIdx >= 0) { "parseOutput: missing '=' in IREE output line: '$raw'" }
    val valuePart = raw.substring(eqIdx + 1)
    val flat = valuePart.replace('[', ' ').replace(']', ' ')
    val tokens = flat.split(Regex("[\\s,]+")).filter { it.isNotEmpty() }
    val expected = type.elementCount.toInt()
    require(tokens.size == expected) {
        "parseOutput: expected $expected values for type $type; got ${tokens.size} tokens from '$raw'"
    }
    return FloatArray(expected) { idx ->
        val tok = tokens[idx]
        tok.toFloatOrNull()
            ?: error("parseOutput: token '$tok' (index $idx) is not a parseable Float in '$raw'")
    }
}
