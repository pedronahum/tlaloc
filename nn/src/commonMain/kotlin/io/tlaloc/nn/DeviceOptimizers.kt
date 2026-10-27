package io.tlaloc.nn

import io.tlaloc.core.Bool
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
 * A training step whose optimizer runs inside the compiled program ([withDeviceAdamW],
 * [withDeviceSGD], [withDeviceMuon]). [function] takes, in order:
 *   - the step's inputs and targets ([CapturedStep.inputCount] of them),
 *   - the trained parameters ([CapturedStep.parameterKeys] order), then the frozen ones,
 *   - the optimizer state, one tensor per [stateSlots] entry, all zeros before the first step,
 *   - the f32 scalars [scalarNames] ([scalars] computes them for step t).
 *
 * It returns the loss, the updated parameters and the updated state in [stateSlots] order.
 * [outputAliases] maps each parameter and state tensor to its updated value, so a PJRT session
 * that donates them updates them in place (`toStablehlo(outputAliases = ...)`).
 */
abstract class DeviceOptimizerStep internal constructor(
    val function: DxirFunction,
    val outputAliases: Map<Int, Int>,
    val step: CapturedStep,
    val stateSlots: List<StateSlot>,
    val scalarNames: List<String>,
) {
    /** One state tensor: [name] for the trained parameter at [parameterIndex], of [type]. */
    class StateSlot(val name: String, val parameterIndex: Int, val type: DxirType)

    /** The scalar operands for step [t] (1-based) at the optimizer's own learning rate. */
    abstract fun scalars(t: Int): FloatArray

    /** Index of the first state tensor in [function]. */
    val stateOffset: Int get() = step.inputCount + step.parameterKeys.size + step.frozenKeys.size

    /** Index of the first scalar in [function]. */
    val scalarOffset: Int get() = stateOffset + stateSlots.size

    /** The state before the first step: zeros, one array per [stateSlots] entry. */
    fun initialState(): List<FloatArray> = stateSlots.map { FloatArray(it.type.elementCount.toInt()) }
}

internal fun momentSlots(step: CapturedStep, name: String): List<DeviceOptimizerStep.StateSlot> =
    step.parameterKeys.indices.map { i ->
        DeviceOptimizerStep.StateSlot(name, i, step.gradient.params[step.inputCount + i].type)
    }

/**
 * Appends an optimizer update to a gradient function: fresh ids above the gradient's, new
 * parameters for the state and the scalars, and the ops of the update.
 */
internal class UpdateBuilder(val step: CapturedStep) {
    val g: DxirFunction = step.gradient
    val nIn: Int = step.inputCount
    val nP: Int = step.parameterKeys.size
    private var next = DxirInline.maxId(g) + 1
    val body = ArrayList<DxirNode>(g.body)
    val scalarT = DxirType(F32, emptyList())

    init {
        require(g.returns.size == 1 + nIn + nP) {
            "device optimizer: the gradient function returns ${g.returns.size} values; expected the loss, " +
                "$nIn input gradients and $nP parameter gradients"
        }
    }

    fun id() = next++
    fun param(name: String, type: DxirType) = DxirParam(id(), name, type)
    fun theta(i: Int): DxirNode = g.params[nIn + i]
    fun grad(i: Int): DxirNode = g.returns[1 + nIn + i]
    fun typeOf(i: Int): DxirType = g.params[nIn + i].type

    fun emit(kind: OpKind, operands: List<DxirNode>, type: DxirType, attrs: Map<String, Any> = emptyMap()): DxirNode =
        DxirOp(id(), kind, operands, attrs, type).also { body += it }
    fun constant(x: Float): DxirNode = DxirConst(id(), x, scalarT).also { body += it }
    fun splat(s: DxirNode, type: DxirType): DxirNode =
        if (type.dims.isEmpty()) s
        else emit(OpKind.BROADCAST, listOf(s), type, mapOf("broadcast_dimensions" to emptyList<Int>()))
    fun add(a: DxirNode, b: DxirNode) = emit(OpKind.ADD, listOf(a, b), a.type)
    fun sub(a: DxirNode, b: DxirNode) = emit(OpKind.SUB, listOf(a, b), a.type)
    fun mul(a: DxirNode, b: DxirNode) = emit(OpKind.MUL, listOf(a, b), a.type)
    fun div(a: DxirNode, b: DxirNode) = emit(OpKind.DIV, listOf(a, b), a.type)

    /** The function: the gradient's parameters, then [state], then [scalars]; it returns the
     * loss, [newParams] and [newState], with aliases from each parameter and state tensor. */
    fun build(
        suffix: String,
        state: List<DxirParam>,
        scalars: List<DxirParam>,
        newParams: List<DxirNode>,
        newState: List<DxirNode>,
    ): Pair<DxirFunction, Map<Int, Int>> {
        val fn = DxirFunction(
            name = "${g.name}_$suffix",
            params = g.params + state + scalars,
            body = body,
            returns = listOf(g.returns[0]) + newParams + newState,
            meshes = g.meshes,
        )
        val stateOffset = nIn + nP + step.frozenKeys.size
        val aliases = HashMap<Int, Int>()
        for (i in 0 until nP) aliases[nIn + i] = 1 + i
        for (k in state.indices) aliases[stateOffset + k] = 1 + nP + k
        return fn to aliases
    }
}

/**
 * [SGD] inside the compiled program; see [withDeviceSGD]. With momentum the state is one
 * velocity per trained parameter and the scalars are the learning rate and the two velocity
 * weights `a`, `b` of `v' = a·v + b·g`. [scalars] gives `a = 0, b = 1` at the first step, the
 * host [SGD]'s first-visit seed `v = g`, and `a = μ, b = 1 − μ` after it. Without momentum there
 * is no state and the one scalar is the learning rate.
 */
class DeviceSgdStep internal constructor(
    function: DxirFunction,
    outputAliases: Map<Int, Int>,
    step: CapturedStep,
    val optimizer: SGD,
) : DeviceOptimizerStep(
    function, outputAliases, step,
    stateSlots = if (optimizer.momentum == 0f) emptyList() else momentSlots(step, "velocity"),
    scalarNames = if (optimizer.momentum == 0f) listOf("learning_rate") else listOf("learning_rate", "velocity_decay", "gradient_weight"),
) {
    /** The scalars for step [t]: the learning rate `lr₀ / (1 + lrDecay·(t − 1))`, as [SGD] anneals it. */
    override fun scalars(t: Int): FloatArray =
        scalars(t, optimizer.initialLearningRate / (1f + optimizer.lrDecay * (t - 1)))

    fun scalars(t: Int, learningRate: Float): FloatArray {
        require(t >= 1) { "DeviceSgdStep.scalars: the step count starts at 1, got $t" }
        val mu = optimizer.momentum
        return when {
            mu == 0f -> floatArrayOf(learningRate)
            t == 1 -> floatArrayOf(learningRate, 0f, 1f)
            else -> floatArrayOf(learningRate, mu, 1f - mu)
        }
    }
}

/** This step with [optimizer] applied inside the compiled program; see [DeviceSgdStep]. */
fun CapturedStep.withDeviceSGD(optimizer: SGD): DeviceSgdStep {
    val b = UpdateBuilder(this)
    val momentum = optimizer.momentum != 0f
    val v = if (momentum) (0 until b.nP).map { b.param("v_${parameterKeys[it]}", b.typeOf(it)) } else emptyList()
    val lr = b.param("learning_rate", b.scalarT)
    val scalars = if (momentum) listOf(lr, b.param("velocity_decay", b.scalarT), b.param("gradient_weight", b.scalarT)) else listOf(lr)
    val newParams = ArrayList<DxirNode>()
    val newV = ArrayList<DxirNode>()
    for (i in 0 until b.nP) {
        val t = b.typeOf(i)
        val vi = if (momentum) {
            b.add(b.mul(b.splat(scalars[1], t), v[i]), b.mul(b.splat(scalars[2], t), b.grad(i))).also { newV += it }
        } else b.grad(i)
        newParams += b.sub(b.theta(i), b.mul(b.splat(lr, t), vi))
    }
    val (fn, aliases) = b.build("sgd", v, scalars, newParams, newV)
    return DeviceSgdStep(fn, aliases, this, optimizer)
}

/**
 * [Muon] inside the compiled program; see [withDeviceMuon]. The state is a momentum buffer per
 * parameter Muon updates, then the first and the second moments of the parameters its
 * [Muon.fallback] AdamW updates. The scalars are Muon's learning rate and decay factor
 * `1 − lr·weightDecay`, then AdamW's four ([DeviceAdamWStep]).
 */
class DeviceMuonStep internal constructor(
    function: DxirFunction,
    outputAliases: Map<Int, Int>,
    step: CapturedStep,
    stateSlots: List<StateSlot>,
    val optimizer: Muon,
) : DeviceOptimizerStep(
    function, outputAliases, step, stateSlots,
    scalarNames = listOf("muon_learning_rate", "muon_decay_factor", "learning_rate", "bias_correction1", "bias_correction2", "decay_factor"),
) {
    override fun scalars(t: Int): FloatArray = scalars(t, optimizer.learningRate, optimizer.fallback.learningRate)

    fun scalars(t: Int, muonLearningRate: Float, adamLearningRate: Float): FloatArray {
        require(t >= 1) { "DeviceMuonStep.scalars: the step count starts at 1, got $t" }
        val a = optimizer.fallback
        return floatArrayOf(
            muonLearningRate,
            1f - muonLearningRate * optimizer.weightDecay,
            adamLearningRate,
            1f - a.beta1.pow(t),
            1f - a.beta2.pow(t),
            1f - adamLearningRate * a.weightDecay,
        )
    }

    /** The trained parameters (by index) that Muon updates; AdamW updates the others. */
    val muonParameters: List<Int> get() = stateSlots.filter { it.name == "momentum" }.map { it.parameterIndex }
}

/**
 * This step with [optimizer] applied inside the compiled program; see [DeviceMuonStep]. The
 * Newton–Schulz iteration is unrolled: [Muon.nsSteps] iterations of three matrix products per
 * parameter.
 */
fun CapturedStep.withDeviceMuon(optimizer: Muon): DeviceMuonStep {
    val b = UpdateBuilder(this)
    val muonIdx = (0 until b.nP).filter { optimizer.updatesWithMuon(parameterKeys[it], b.typeOf(it).dims.toIntArray()) }
    val adamIdx = (0 until b.nP).filter { it !in muonIdx }
    val buf = muonIdx.map { b.param("momentum_${parameterKeys[it]}", b.typeOf(it)) }
    val m = adamIdx.map { b.param("m_${parameterKeys[it]}", b.typeOf(it)) }
    val v = adamIdx.map { b.param("v_${parameterKeys[it]}", b.typeOf(it)) }
    val scalars = listOf("muon_learning_rate", "muon_decay_factor", "learning_rate", "bias_correction1", "bias_correction2", "decay_factor")
        .map { b.param(it, b.scalarT) }
    val (muonLr, muonShrink, lr, c1, c2, shrink) = scalars

    val newParams = arrayOfNulls<DxirNode>(b.nP)
    val newBuf = ArrayList<DxirNode>()
    val mu = optimizer.momentum
    val oneMinusMu = b.constant(1f - mu)
    val muC = b.constant(mu)
    for ((k, i) in muonIdx.withIndex()) {
        val t = b.typeOf(i)
        val g = b.grad(i)
        // buf' = buf + (1 − μ)(g − buf); the update is g + μ(buf' − g) with Nesterov, buf' without.
        val bi = b.add(buf[k], b.mul(b.splat(oneMinusMu, t), b.sub(g, buf[k])))
        newBuf += bi
        val u = if (optimizer.nesterov) b.add(g, b.mul(b.splat(muC, t), b.sub(bi, g))) else bi
        val o = newtonSchulz(b, u, optimizer)
        val scale = b.constant(optimizer.lrScale(t.dims[0], t.dims[1]))
        val stepLr = b.emit(OpKind.MUL, listOf(muonLr, scale), b.scalarT)
        newParams[i] = b.sub(b.mul(b.theta(i), b.splat(muonShrink, t)), b.mul(b.splat(stepLr, t), o))
    }
    val a = optimizer.fallback
    val b1 = b.constant(a.beta1)
    val oneMinusB1 = b.constant(1f - a.beta1)
    val b2 = b.constant(a.beta2)
    val oneMinusB2 = b.constant(1f - a.beta2)
    val eps = b.constant(a.eps)
    val newM = ArrayList<DxirNode>()
    val newV = ArrayList<DxirNode>()
    for ((k, i) in adamIdx.withIndex()) {
        val t = b.typeOf(i)
        val g = b.grad(i)
        val mi = b.add(b.mul(b.splat(b1, t), m[k]), b.mul(b.splat(oneMinusB1, t), g))
        val vi = b.add(b.mul(b.splat(b2, t), v[k]), b.mul(b.mul(b.splat(oneMinusB2, t), g), g))
        val denom = b.add(b.emit(OpKind.SQRT, listOf(b.div(vi, b.splat(c2, t))), t), b.splat(eps, t))
        val stepSize = b.div(b.mul(b.splat(lr, t), b.div(mi, b.splat(c1, t))), denom)
        val decayed = if (a.decay(parameterKeys[i])) b.mul(b.theta(i), b.splat(shrink, t)) else b.theta(i)
        newParams[i] = b.sub(decayed, stepSize)
        newM += mi
        newV += vi
    }
    val slots = muonIdx.map { DeviceOptimizerStep.StateSlot("momentum", it, b.typeOf(it)) } +
        adamIdx.map { DeviceOptimizerStep.StateSlot("m", it, b.typeOf(it)) } +
        adamIdx.map { DeviceOptimizerStep.StateSlot("v", it, b.typeOf(it)) }
    val (fn, aliases) = b.build("muon", buf + m + v, scalars, newParams.map { it!! }, newBuf + newM + newV)
    return DeviceMuonStep(fn, aliases, this, slots, optimizer)
}

private operator fun <T> List<T>.component6(): T = this[5]

/** [Muon.orthogonalized] on the device: the same steps as the host form. */
private fun newtonSchulz(b: UpdateBuilder, u: DxirNode, optimizer: Muon): DxirNode {
    val (rows, cols) = u.type.dims
    fun transpose(x: DxirNode): DxirNode {
        val d = x.type.dims
        return b.emit(OpKind.TRANSPOSE, listOf(x), DxirType(F32, listOf(d[1], d[0])), mapOf("permutation" to listOf(1, 0)))
    }
    fun matmul(x: DxirNode, y: DxirNode) =
        b.emit(OpKind.MATMUL, listOf(x, y), DxirType(F32, listOf(x.type.dims[0], y.type.dims[1])))
    val tall = rows > cols
    var x = if (tall) transpose(u) else u
    val t = x.type
    val norm = b.emit(OpKind.SQRT, listOf(b.emit(OpKind.SUM, listOf(b.mul(x, x)), b.scalarT)), b.scalarT)
    val eps = b.constant(optimizer.eps)
    val clamped = b.emit(
        OpKind.WHERE,
        listOf(b.emit(OpKind.COMPARE, listOf(norm, eps), DxirType(Bool, emptyList()), mapOf("direction" to "GT")), norm, eps),
        b.scalarT,
    )
    x = b.div(x, b.splat(clamped, t))
    val ca = b.constant(optimizer.nsA)
    val cb = b.constant(optimizer.nsB)
    val cc = b.constant(optimizer.nsC)
    repeat(optimizer.nsSteps) {
        val gram = matmul(x, transpose(x))
        val gt = gram.type
        val poly = b.add(b.mul(b.splat(cb, gt), gram), b.mul(b.splat(cc, gt), matmul(gram, gram)))
        x = b.add(b.mul(b.splat(ca, t), x), matmul(poly, x))
    }
    return if (tall) transpose(x) else x
}
