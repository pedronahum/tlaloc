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
 * ```
 *
 * Every alias in [MosaicKernel.inputOutputAliases] must name a real operand
 * and result of the same type.
 */
@ExperimentalTlalocApi
object MosaicKernelAttrs {
    const val KERNEL: String = "mosaic_kernel"
    const val REFERENCE: String = "reference"
    const val REFERENCE_FALLBACK: String = "reference_fallback"

    /** The validated attributes of one MOSAIC_KERNEL op. */
    data class Parsed(
        val kernel: MosaicKernel,
        val reference: DxirFunction,
        val referenceFallback: Boolean,
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
        return Parsed(kernel, reference, fallback)
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
 */
@ExperimentalTlalocApi
fun DxirEmitter.mosaicKernel(
    operands: List<DxirNode>,
    kernel: MosaicKernel,
    reference: DxirFunction,
    referenceFallback: Boolean = false,
): DxirOp {
    val attrs = buildMap<String, Any> {
        put(MosaicKernelAttrs.KERNEL, kernel)
        put(MosaicKernelAttrs.REFERENCE, reference)
        if (referenceFallback) put(MosaicKernelAttrs.REFERENCE_FALLBACK, true)
    }
    val op = opMulti(OpKind.MOSAIC_KERNEL, operands, reference.returns.map { it.type }, attrs)
    MosaicKernelAttrs.parse(op, "mosaicKernel")
    return op
}
