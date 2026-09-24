/**
 * The GRU layer, gate equations EXACTLY as DiffKT's
 * source defines them: input
 * `[batch, seq, numInputs]` (batch axis 0, seq axis 1), initial hidden state
 * zeros expanded to the batch, per time step
 *
 * ```
 * xh = concat(x, h, axis = 1)
 * u  = xh2u(xh)                          // update gate, Dense(…, σ)
 * r  = xh2r(xh)                          // reset gate,  Dense(…, σ)
 * n  = xh2n(concat(x, r * h, axis = 1))  // candidate,   Dense(…, tanh) — reset BEFORE the linear
 * h' = (1 − u) * n + u * h
 * ```
 *
 * (the default `LinearAfterResetGru`), or DiffKT's `linearBeforeReset` variant
 * `n = tanh(x2n(x) + r * h2n(h))` with `x2n`/`h2n` Identity-activated Denses
 * carrying their own biases, same `u`/`r`/`h'`. `AccType.Fold` returns the
 * LAST step's `[batch, numHidden]`; `AccType.AccMap` concatenates the per-step
 * outputs (each unsqueezed at the seq axis) into `[batch, seq, numHidden]`.
 *
 * The implementation is a LIBRARY-LEVEL
 * unroll. Per time step the gate math is traced ops (SLICE → RESHAPE squeeze,
 * CONCAT, the Dense matmul/broadcast/add, sigmoid/tanh, elementwise MUL/SUB/
 * ADD — every spelling already on the tape), the sequence loop
 * is a plain Kotlin `for` building the trace, and the captured `DxirFunction`
 * is the UNROLLED graph — `DxirReverseTransform` differentiates it like any
 * other graph, so backpropagation through time is not a feature, it is what
 * reverse-mode ON the unrolled graph ALREADY IS. Zero gradient math in this
 * file and zero new TRACE spellings.
 *
 * The initial hidden state is a CONSTANT zeros leaf `[batch, numHidden]`
 * (`isConstant = true`, as for the dropout mask): DiffKT's
 * `initialState` is non-trainable zeros `[1, numHidden]` `expand`ed to batch,
 * and tiling the zeros host-side is equivalent — a
 * zeros constant carries no gradient, so the expand never needs a traced
 * broadcast. REJECTED: tracing `initialState` as a parameter — DiffKT
 * lists ONLY the gate Denses as trainable.
 */
package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.split
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.concat
import io.tlaloc.autograd.constant
import io.tlaloc.autograd.minus
import io.tlaloc.autograd.plus
import io.tlaloc.autograd.reshape
import io.tlaloc.autograd.slice
import io.tlaloc.autograd.tanh
import io.tlaloc.autograd.times

/**
 * DiffKT's `GRU(numInputs, numHidden, random, acc, linearBeforeReset)`, as ONE
 * concrete class covering both candidate-gate variants. REJECTED: mirroring
 * DiffKT's two-subclass hierarchy (`LinearAfterResetGru` /
 * `LinearBeforeResetGRU` under `RecurrentBase`) — under the self-typed
 * `Trainable<T : Trainable<T>>` contract a variant-choosing factory would have
 * to return an erased `TrainableLayer<*>`, which `capture`'s
 * `M : Layer, M : Trainable<M>` bounds cannot hold; the flag-plus-nullable-
 * gates value keeps the factory's return type concrete and the maths
 * identical. The gates are whole [Dense] layers with their own biases and
 * activations baked in, exactly like the source; the tensor-level
 * constructors REQUIRE DiffKT's activations (σ/σ/tanh, and Identity for the
 * before-reset pair) so the gate equations stay DiffKT's — a loud guard,
 * not a modelling freedom.
 *
 * Parameter keys are gate-prefixed Dense keys in declaration order
 * (`"xh2u.w"`, `"xh2u.b"`, `"xh2r.w"`, … — the Sequential-style prefix
 * delegation), so the gradients come back from the transform addressed per
 * gate tensor like every other `:nn` layer.
 */
class GRU private constructor(
    val xh2u: Dense,
    val xh2r: Dense,
    val xh2n: Dense?,
    val x2n: Dense?,
    val h2n: Dense?,
    val acc: AccType,
) : TrainableLayer<GRU> {

    /** DiffKT's `RecurrentBase` accumulation variants. */
    enum class AccType {
        /** Return the LAST step's output `[batch, numHidden]`. */
        Fold,

        /** Concat every step's output into `[batch, seq, numHidden]`. */
        AccMap,
    }

    /** The default variant: candidate `xh2n(concat(x, r*h))`, reset before the linear. */
    constructor(xh2u: Dense, xh2r: Dense, xh2n: Dense, acc: AccType = AccType.Fold) :
        this(xh2u, xh2r, xh2n, null, null, acc)

    /** The `linearBeforeReset` variant: candidate `tanh(x2n(x) + r * h2n(h))`. */
    constructor(xh2u: Dense, xh2r: Dense, x2n: Dense, h2n: Dense, acc: AccType = AccType.Fold) :
        this(xh2u, xh2r, null, x2n, h2n, acc)

    val linearBeforeReset: Boolean = xh2n == null
    val numHidden: Int = xh2u.w.dims[1]
    val numInputs: Int = xh2u.w.dims[0] - numHidden

    init {
        require(numInputs > 0) {
            "GRU: xh2u must be Dense(numInputs + numHidden, numHidden) with numInputs > 0 " +
                "(got w dims ${xh2u.w.dims.toList()})"
        }
        require(xh2u.activation == Activation.Sigmoid && xh2r.activation == Activation.Sigmoid) {
            "GRU: the update and reset gates are sigmoid-activated Denses in DiffKT " +
                "(got xh2u=${xh2u.activation}, xh2r=${xh2r.activation})"
        }
        require(xh2r.w.dims.contentEquals(xh2u.w.dims)) {
            "GRU: xh2r must share xh2u's shape ${xh2u.w.dims.toList()} (got ${xh2r.w.dims.toList()})"
        }
        if (xh2n != null) {
            require(xh2n.activation == Activation.Tanh) {
                "GRU: the candidate gate is a tanh-activated Dense in DiffKT (got ${xh2n.activation})"
            }
            require(xh2n.w.dims.contentEquals(xh2u.w.dims)) {
                "GRU: xh2n must share xh2u's shape ${xh2u.w.dims.toList()} (got ${xh2n.w.dims.toList()})"
            }
        } else {
            val xn = requireNotNull(x2n)
            val hn = requireNotNull(h2n)
            require(xn.activation == Activation.Identity && hn.activation == Activation.Identity) {
                "GRU (linearBeforeReset): x2n and h2n are Identity-activated Denses in DiffKT — " +
                    "the tanh wraps the sum (got x2n=${xn.activation}, h2n=${hn.activation})"
            }
            require(xn.w.dims.contentEquals(intArrayOf(numInputs, numHidden))) {
                "GRU (linearBeforeReset): x2n must be Dense($numInputs, $numHidden) " +
                    "(got w dims ${xn.w.dims.toList()})"
            }
            require(hn.w.dims.contentEquals(intArrayOf(numHidden, numHidden))) {
                "GRU (linearBeforeReset): h2n must be Dense($numHidden, $numHidden) " +
                    "(got w dims ${hn.w.dims.toList()})"
            }
        }
    }

    /** Gate name → Dense, in the declaration order the parameter keys follow. */
    private val gates: List<Pair<String, Dense>> = buildList {
        add("xh2u" to xh2u)
        add("xh2r" to xh2r)
        xh2n?.let { add("xh2n" to it) }
        x2n?.let { add("x2n" to it) }
        h2n?.let { add("h2n" to it) }
    }

    override val parameters: List<NamedParameter> =
        gates.flatMap { (name, gate) ->
            gate.parameters.map { NamedParameter("$name.${it.key}", it.tensor) }
        }

    override fun withParameters(updated: Map<String, DTensor<*, F32>>): GRU {
        val gateNames = gates.map { it.first }
        val perGate = HashMap<String, MutableMap<String, DTensor<*, F32>>>()
        for ((key, tensor) in updated) {
            val dot = key.indexOf('.')
            require(dot in 1 until key.length - 1) {
                "GRU.withParameters: key '$key' is not '<gate>.<param>' (gates: $gateNames)"
            }
            val gate = key.substring(0, dot)
            require(gate in gateNames) {
                "GRU.withParameters: unknown gate '$gate' in key '$key' (known: $gateNames)"
            }
            perGate.getOrPut(gate) { HashMap() }[key.substring(dot + 1)] = tensor
        }
        fun rebuilt(name: String, gate: Dense?): Dense? =
            if (gate == null) null else perGate[name]?.let { gate.withParameters(it) } ?: gate
        return GRU(
            rebuilt("xh2u", xh2u)!!,
            rebuilt("xh2r", xh2r)!!,
            rebuilt("xh2n", xh2n),
            rebuilt("x2n", x2n),
            rebuilt("h2n", h2n),
            acc,
        )
    }

    override fun forward(x: Tracer<Shape>, params: Params): Tracer<Shape> {
        require(x.rank == 3) {
            "GRU: input must be rank-3 [batch, seq, numInputs] (got dims ${x.dims.toList()})"
        }
        val batch = x.dims[0]
        val seq = x.dims[1]
        require(x.dims[2] == numInputs) {
            "GRU: input width ${x.dims[2]} does not match numInputs $numInputs"
        }
        // The zeros-constant initial state (see the file KDoc): DiffKT's
        // `initialState` expand, tiled host-side, carrying no gradient.
        var h: Tracer<Shape> = x.constant(FloatArray(batch * numHidden), intArrayOf(batch, numHidden))
        val perStep = if (acc == AccType.AccMap) ArrayList<Tracer<*>>(seq) else null
        for (t in 0 until seq) {
            // DiffKT's per-step read: slice(t, t+1, seqAxis).squeeze(seqAxis).
            val xt: Tracer<Shape> = x.slice<Shape>(t, t + 1, 1).reshape(intArrayOf(batch, numInputs))
            h = step(xt, h, params)
            perStep?.add(h.reshape<Shape>(intArrayOf(batch, 1, numHidden)))
        }
        return perStep?.let { concat<Shape>(it, 1) } ?: h
    }

    private fun scoped(params: Params, gate: String) = Params { key -> params["$gate.$key"] }

    /** One time step — DiffKT's gate equations, verbatim. */
    private fun step(xt: Tracer<Shape>, h: Tracer<Shape>, params: Params): Tracer<Shape> {
        val xh: Tracer<Shape> = concat(listOf(xt, h), 1)
        val u = xh2u.forward(xh, scoped(params, "xh2u"))
        val r = xh2r.forward(xh, scoped(params, "xh2r"))
        val n = if (xh2n != null) {
            val xrh: Tracer<Shape> = concat(listOf(xt, r * h), 1)
            xh2n.forward(xrh, scoped(params, "xh2n"))
        } else {
            (x2n!!.forward(xt, scoped(params, "x2n")) + r * h2n!!.forward(h, scoped(params, "h2n"))).tanh()
        }
        return (1f - u) * n + u * h
    }

    companion object {
        /**
         * The DiffKT constructor surface. Key discipline (the `:nn` layer convention
         * lifted to a composite layer): the GRU splits its key ONCE into one
         * child per GATE in declaration order — `split(3)` → xh2u/xh2r/xh2n,
         * or `split(4)` → xh2u/xh2r/x2n/h2n — and each gate Dense applies its
         * own `split(2)` per-parameter discipline below that, so every drawn
         * tensor is bit-determined by (key, position in the gate list).
         * The Denses draw DiffKT's own default `uniform(±√(1/numInputs))`.
         */
        operator fun invoke(
            numInputs: Int,
            numHidden: Int,
            key: RandomKey,
            acc: AccType = AccType.Fold,
            linearBeforeReset: Boolean = false,
        ): GRU {
            require(numInputs > 0 && numHidden > 0) {
                "GRU: numInputs=$numInputs and numHidden=$numHidden must be positive"
            }
            return if (!linearBeforeReset) {
                val keys = key.split(3)
                GRU(
                    Dense(numInputs + numHidden, numHidden, keys[0], activation = Activation.Sigmoid),
                    Dense(numInputs + numHidden, numHidden, keys[1], activation = Activation.Sigmoid),
                    Dense(numInputs + numHidden, numHidden, keys[2], activation = Activation.Tanh),
                    acc,
                )
            } else {
                val keys = key.split(4)
                GRU(
                    Dense(numInputs + numHidden, numHidden, keys[0], activation = Activation.Sigmoid),
                    Dense(numInputs + numHidden, numHidden, keys[1], activation = Activation.Sigmoid),
                    Dense(numInputs, numHidden, keys[2]),
                    Dense(numHidden, numHidden, keys[3]),
                    acc,
                )
            }
        }
    }
}
