package io.tlaloc.nn

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirParam
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInline
import kotlin.math.pow

/**
 * A training step whose optimizer runs inside the compiled program: one function computes the
 * loss and gradients ([CapturedStep.gradient]) and applies [AdamW] to every trained parameter,
 * so a step moves no parameters, moments or gradients between host and device.
 *
 * [function] takes, in order:
 *   - the step's inputs and targets ([CapturedStep.inputCount] of them),
 *   - the trained parameters ([CapturedStep.parameterKeys] order), then the frozen ones,
 *   - the first moments and the second moments, one per trained parameter,
 *   - four f32 scalars: the learning rate, the bias corrections `1 − β₁ᵗ` and `1 − β₂ᵗ`, and
 *     the decay factor `1 − lr·weightDecay` ([scalars] computes them for step t).
 *
 * It returns the loss, the updated parameters, the updated first moments and the updated second
 * moments. [outputAliases] maps each parameter and moment to its updated value, so a PJRT
 * session that donates them updates them in place
 * (`toStablehlo(outputAliases = ...)`, `tf.aliasing_output`).
 *
 * The scalars are operands, not constants: a learning-rate schedule and the step count never
 * change the program, so one compile serves the whole run. β₁, β₂ and ε are constants.
 *
 * The update is [AdamW]'s, operation for operation: `m' = β₁m + (1−β₁)g`,
 * `v' = β₂v + ((1−β₂)g)g`, `θ' = θ·k − (lr·(m'/c₁)) / (√(v'/c₂) + ε)`, with `k` the decay factor
 * for the keys [AdamW.decay] selects and 1 for the others.
 */
class DeviceAdamWStep internal constructor(
    function: DxirFunction,
    outputAliases: Map<Int, Int>,
    step: CapturedStep,
    val optimizer: AdamW,
) : DeviceOptimizerStep(
    function, outputAliases, step,
    stateSlots = momentSlots(step, "m") + momentSlots(step, "v"),
    scalarNames = listOf("learning_rate", "bias_correction1", "bias_correction2", "decay_factor"),
) {
    override fun scalars(t: Int): FloatArray = scalars(t, optimizer.learningRate)

    /** The four scalar operands for step [t] (1-based), computed as the host [AdamW] does. */
    fun scalars(t: Int, learningRate: Float = optimizer.learningRate): FloatArray {
        require(t >= 1) { "DeviceAdamWStep.scalars: the step count starts at 1, got $t" }
        return floatArrayOf(
            learningRate,
            1f - optimizer.beta1.pow(t),
            1f - optimizer.beta2.pow(t),
            1f - learningRate * optimizer.weightDecay,
        )
    }

    /** Index of the first moment parameter in [function]. */
    val firstMomentOffset: Int get() = stateOffset
}

/** This step with [optimizer] applied inside the compiled program; see [DeviceAdamWStep]. */
fun CapturedStep.withDeviceAdamW(optimizer: AdamW): DeviceAdamWStep {
    val g = gradient
    val nIn = inputCount
    val nP = parameterKeys.size
    require(g.returns.size == 1 + nIn + nP) {
        "withDeviceAdamW: the gradient function returns ${g.returns.size} values; expected the loss, " +
            "$nIn input gradients and $nP parameter gradients"
    }
    var next = DxirInline.maxId(g) + 1
    fun id() = next++

    val scalarT = DxirType(F32, emptyList())
    val paramTypes = (0 until nP).map { g.params[nIn + it].type }
    val m = paramTypes.mapIndexed { i, t -> DxirParam(id(), "m_${parameterKeys[i]}", t) }
    val v = paramTypes.mapIndexed { i, t -> DxirParam(id(), "v_${parameterKeys[i]}", t) }
    val lr = DxirParam(id(), "learning_rate", scalarT)
    val c1 = DxirParam(id(), "bias_correction1", scalarT)
    val c2 = DxirParam(id(), "bias_correction2", scalarT)
    val shrink = DxirParam(id(), "decay_factor", scalarT)

    val body = ArrayList<DxirNode>(g.body)
    fun emit(kind: OpKind, operands: List<DxirNode>, type: DxirType, attrs: Map<String, Any> = emptyMap()): DxirNode =
        DxirOp(id(), kind, operands, attrs, type).also { body += it }
    fun splat(s: DxirNode, type: DxirType): DxirNode =
        if (type.dims.isEmpty()) s
        else emit(OpKind.BROADCAST, listOf(s), type, mapOf("broadcast_dimensions" to emptyList<Int>()))
    fun constant(x: Float): DxirNode = DxirConst(id(), x, scalarT).also { body += it }

    val b1 = constant(optimizer.beta1)
    val oneMinusB1 = constant(1f - optimizer.beta1)
    val b2 = constant(optimizer.beta2)
    val oneMinusB2 = constant(1f - optimizer.beta2)
    val eps = constant(optimizer.eps)

    val newParams = ArrayList<DxirNode>(nP)
    val newM = ArrayList<DxirNode>(nP)
    val newV = ArrayList<DxirNode>(nP)
    for (i in 0 until nP) {
        val t = paramTypes[i]
        val theta = g.params[nIn + i]
        val grad = g.returns[1 + nIn + i]
        fun mul(a: DxirNode, b: DxirNode) = emit(OpKind.MUL, listOf(a, b), t)
        fun add(a: DxirNode, b: DxirNode) = emit(OpKind.ADD, listOf(a, b), t)
        val mi = add(mul(splat(b1, t), m[i]), mul(splat(oneMinusB1, t), grad))
        val vi = add(mul(splat(b2, t), v[i]), mul(mul(splat(oneMinusB2, t), grad), grad))
        val mHat = emit(OpKind.DIV, listOf(mi, splat(c1, t)), t)
        val vHat = emit(OpKind.DIV, listOf(vi, splat(c2, t)), t)
        val denom = add(emit(OpKind.SQRT, listOf(vHat), t), splat(eps, t))
        val stepSize = emit(OpKind.DIV, listOf(mul(splat(lr, t), mHat), denom), t)
        val decayed = if (optimizer.decay(parameterKeys[i])) mul(theta, splat(shrink, t)) else theta
        newParams += emit(OpKind.SUB, listOf(decayed, stepSize), t)
        newM += mi
        newV += vi
    }

    val fn = DxirFunction(
        name = "${g.name}_adamw",
        params = g.params + m + v + listOf(lr, c1, c2, shrink),
        body = body,
        returns = listOf(g.returns[0]) + newParams + newM + newV,
        meshes = g.meshes,
    )
    val frozenCount = frozenKeys.size
    val aliases = HashMap<Int, Int>()
    for (i in 0 until nP) {
        aliases[nIn + i] = 1 + i
        aliases[nIn + nP + frozenCount + i] = 1 + nP + i
        aliases[nIn + nP + frozenCount + nP + i] = 1 + 2 * nP + i
    }
    return DeviceAdamWStep(fn, aliases, this, optimizer)
}
