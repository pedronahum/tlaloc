package io.tlaloc.ir

import io.tlaloc.core.ExperimentalTlalocApi
import io.tlaloc.ir.recognizer.kernel.MosaicKernel

/**
 * The [OpKind.MOSAIC_KERNEL] attribute convention, shared by the interpreter,
 * the lowering pass, the StableHLO emitter and function validation.
 *
 * ```
 *   MOSAIC_KERNEL(operand_0 … operand_N-1) → (result_0 … result_K-1)
 *   attrs:
 *     mosaic_kernel       MosaicKernel   required
 *     reference           DxirFunction   required: params typed like the
 *                                        operands, returns like the results
 *     reference_fallback  Boolean        optional, default false
 *     vjp                 DxirFunction   optional: (operands…, results…, one
 *                                        cotangent per result) → one gradient
 *                                        per operand, typed like it
 * ```
 *
 * A kernel with a `vjp` is differentiable in reverse mode: the reverse transform inlines the
 * `vjp`, which may call further kernels (a flash attention's backward kernels). The `vjp` is the
 * author's statement of the gradient, as a custom VJP is; without one the kind stays
 * inference-only.
 *
 * Every alias in [MosaicKernel.inputOutputAliases] must name a real operand
 * and result of the same type.
 */
@ExperimentalTlalocApi
object MosaicKernelAttrs {
    const val KERNEL: String = "mosaic_kernel"
    const val REFERENCE: String = "reference"
    const val REFERENCE_FALLBACK: String = "reference_fallback"
    const val VJP: String = "vjp"

    /** The validated attributes of one MOSAIC_KERNEL op. */
    data class Parsed(
        val kernel: MosaicKernel,
        val reference: DxirFunction,
        val referenceFallback: Boolean,
        val vjp: DxirFunction? = null,
    )

    /** Parse and validate [op]; every refusal starts with [layer]. */
    fun parse(op: DxirOp, layer: String): Parsed {
        require(op.op == OpKind.MOSAIC_KERNEL) {
            "$layer: MosaicKernelAttrs.parse called on ${op.op} (a compiler bug)"
        }
        val kernel = op.attrs[KERNEL]
        require(kernel is MosaicKernel) {
            "$layer: MOSAIC_KERNEL op id=${op.id} requires a '$KERNEL' attr of type MosaicKernel; " +
                "got ${kernel?.let { it::class.simpleName }}"
        }
        val name = kernel.kernelName
        val reference = op.attrs[REFERENCE]
        require(reference is DxirFunction) {
            "$layer: MOSAIC_KERNEL '$name' (op id=${op.id}) requires a '$REFERENCE' attr of type " +
                "DxirFunction — the decomposition the interpreter and non-TPU targets run; " +
                "got ${reference?.let { it::class.simpleName }}"
        }
        val fallback = op.attrs[REFERENCE_FALLBACK] ?: false
        require(fallback is Boolean) {
            "$layer: MOSAIC_KERNEL '$name' '$REFERENCE_FALLBACK' must be a Boolean; got ${fallback::class.simpleName}"
        }
        require(reference.params.size == op.operands.size) {
            "$layer: MOSAIC_KERNEL '$name' has ${op.operands.size} operands but its reference takes " +
                "${reference.params.size} params"
        }
        require(reference.returns.size == op.types.size) {
            "$layer: MOSAIC_KERNEL '$name' has ${op.types.size} results but its reference returns " +
                "${reference.returns.size} values"
        }
        for ((i, p) in reference.params.withIndex()) {
            require(p.type == op.operands[i].type) {
                "$layer: MOSAIC_KERNEL '$name' operand $i is ${op.operands[i].type} but reference " +
                    "param $i is ${p.type}"
            }
        }
        for ((i, r) in reference.returns.withIndex()) {
            require(r.type == op.types[i]) {
                "$layer: MOSAIC_KERNEL '$name' result $i is ${op.types[i]} but reference return $i is ${r.type}"
            }
        }
        for ((operand, output) in kernel.inputOutputAliases) {
            require(operand in op.operands.indices && output in op.types.indices) {
                "$layer: MOSAIC_KERNEL '$name' aliases operand $operand to result $output, but the op " +
                    "has ${op.operands.size} operands and ${op.types.size} results"
            }
            require(op.operands[operand].type == op.types[output]) {
                "$layer: MOSAIC_KERNEL '$name' aliases operand $operand (${op.operands[operand].type}) " +
                    "to result $output (${op.types[output]}); an in-place result must have its operand's type"
            }
        }
        val vjp = op.attrs[VJP]
        if (vjp != null) {
            require(vjp is DxirFunction) { "$layer: MOSAIC_KERNEL '$name' '$VJP' must be a DxirFunction; got ${vjp::class.simpleName}" }
            val n = op.operands.size
            val k = op.types.size
            require(vjp.params.size == n + 2 * k && vjp.returns.size == n) {
                "$layer: MOSAIC_KERNEL '$name' vjp must take $n operands, $k results and $k cotangents and return $n gradients; " +
                    "it takes ${vjp.params.size} and returns ${vjp.returns.size}"
            }
            val want = op.operands.map { it.type } + op.types + op.types
            for ((i, p) in vjp.params.withIndex()) {
                require(p.type == want[i]) { "$layer: MOSAIC_KERNEL '$name' vjp param $i is ${p.type}, expected ${want[i]}" }
            }
            for ((i, r) in vjp.returns.withIndex()) {
                require(r.type == op.operands[i].type) { "$layer: MOSAIC_KERNEL '$name' vjp return $i is ${r.type}, expected ${op.operands[i].type}" }
            }
        }
        return Parsed(kernel, reference, fallback, vjp as DxirFunction?)
    }
}

/**
 * Emit an [OpKind.MOSAIC_KERNEL]: [kernel] on a TPU, [reference] everywhere
 * else it is allowed to run. Result types are [reference]'s return types;
 * take results past the first with [DxirOp.result].
 *
 * @param referenceFallback when false (the default), lowering for a non-TPU
 *   target refuses by name instead of substituting [reference]. The
 *   interpreter always evaluates [reference].
 * @param vjp the kernel's gradient (see [MosaicKernelAttrs]); without it the op
 *   cannot be differentiated.
 */
@ExperimentalTlalocApi
fun DxirEmitter.mosaicKernel(
    operands: List<DxirNode>,
    kernel: MosaicKernel,
    reference: DxirFunction,
    referenceFallback: Boolean = false,
    vjp: DxirFunction? = null,
): DxirOp {
    val attrs = buildMap<String, Any> {
        put(MosaicKernelAttrs.KERNEL, kernel)
        put(MosaicKernelAttrs.REFERENCE, reference)
        if (referenceFallback) put(MosaicKernelAttrs.REFERENCE_FALLBACK, true)
        if (vjp != null) put(MosaicKernelAttrs.VJP, vjp)
    }
    val op = opMulti(OpKind.MOSAIC_KERNEL, operands, reference.returns.map { it.type }, attrs)
    MosaicKernelAttrs.parse(op, "mosaicKernel")
    return op
}
