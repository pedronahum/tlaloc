package io.tlaloc.autograd

import io.tlaloc.core.DTensor
import io.tlaloc.core.DType
import io.tlaloc.core.DimBound
import io.tlaloc.core.ExperimentalTlalocApi
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.ScalarShape
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.core.hostI32
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.passes.DxirInterpreter
import kotlin.math.abs
import kotlin.math.max

/** One axis of a [TensorSpec]: a fixed size, or a size bounded by a [DimBound]. */
@ExperimentalTlalocApi
sealed interface AxisSpec {
    data class Fixed(val size: Int) : AxisSpec {
        init {
            require(size >= 1) { "AxisSpec.Fixed: size must be at least 1, got $size" }
        }
        override fun toString(): String = size.toString()
    }

    data class Bounded(val bound: DimBound) : AxisSpec {
        override fun toString(): String = "<=${bound.boundName}(${bound.max})"
    }
}

/**
 * The dtype and axes of one input or output of a [BoundedProgram]. On the JVM, `specOf<S>(...)`
 * builds one from a tensor type, so the bound is written only in the type.
 */
@ExperimentalTlalocApi
class TensorSpec(val dtype: DType, val axes: List<AxisSpec>) {
    init {
        require(dtype == F32 || dtype == I32) {
            "TensorSpec: dtype ${dtype.name} is not supported by bounded programs (F32 and I32 only)"
        }
    }

    val rank: Int get() = axes.size

    /** The bounds this spec uses, in axis order, without repeats. */
    val bounds: List<DimBound> get() = axes.filterIsInstance<AxisSpec.Bounded>().map { it.bound }.distinct()

    /** The dims at the given size of each bound. */
    fun dimsAt(sizes: Map<DimBound, Int>): IntArray = IntArray(rank) { i ->
        when (val a = axes[i]) {
            is AxisSpec.Fixed -> a.size
            is AxisSpec.Bounded -> sizes[a.bound] ?: error("TensorSpec: no size given for bound ${a.bound}")
        }
    }

    override fun toString(): String = "${dtype.name}${axes.joinToString(",", "[", "]")}"
}

/**
 * The per-bound bucket ladders a bucketed run or an export pads to. Each ladder is strictly
 * ascending and ends at its bound's `max`.
 */
@ExperimentalTlalocApi
class BucketLadders(ladders: Map<DimBound, List<Int>>) {
    val ladders: Map<DimBound, List<Int>> = ladders.mapValues { it.value.toList() }

    init {
        for ((bound, ladder) in this.ladders) {
            require(ladder.isNotEmpty()) { "BucketLadders: the ladder of $bound is empty" }
            require(ladder.zipWithNext().all { (a, b) -> a < b }) {
                "BucketLadders: the ladder of $bound is not strictly ascending: $ladder"
            }
            require(ladder.first() >= 1) { "BucketLadders: the ladder of $bound starts below 1: $ladder" }
            require(ladder.last() == bound.max) {
                "BucketLadders: the ladder of $bound ends at ${ladder.last()}, not at the bound ${bound.max}; " +
                    "a size between them would have no bucket"
            }
        }
    }

    fun ladder(bound: DimBound): List<Int> =
        ladders[bound] ?: throw IllegalArgumentException("BucketLadders: no ladder for bound $bound")

    /** The smallest bucket of [bound]'s ladder that holds [size]. */
    fun bucketFor(bound: DimBound, size: Int): Int {
        require(size in 1..bound.max) { "BucketLadders: size $size is outside 1..${bound.max} of $bound" }
        return ladder(bound).first { it >= size }
    }

    companion object {
        /**
         * Powers of two from `min(minBucket, max)` up to each bound's `max`, with `max` appended
         * when it is not a power of two: the rule `DecodeBucketPolicy` uses for its context ladder.
         */
        fun powersOfTwo(bounds: Collection<DimBound>, minBucket: Int = 16): BucketLadders {
            require(minBucket >= 1) { "BucketLadders.powersOfTwo: minBucket must be at least 1" }
            return BucketLadders(
                bounds.associateWith { b ->
                    val out = ArrayList<Int>()
                    var v = 1
                    while (v < minOf(minBucket, b.max)) v *= 2
                    while (v < b.max) { out += v; v *= 2 }
                    out += b.max
                    out
                },
            )
        }
    }
}

/** The extra inputs a program may ask for, filled by whoever runs it. */
@ExperimentalTlalocApi
enum class BoundedInputRole { DATA, VALID_MASK, VALID_LENGTH }

/**
 * What the program body can ask about the bounded sizes. At an exact-size run the mask is all
 * ones; at a bucketed run it is 1 for real positions and 0 for padding.
 */
@ExperimentalTlalocApi
class BoundedContext internal constructor(
    private val tape: Tape,
    private val traceSizes: Map<DimBound, Int>,
    private val realSizes: Map<DimBound, Int>,
    private val known: Set<DimBound>,
) {
    internal val requested = ArrayList<Pair<BoundedInputRole, DimBound>>()
    internal val leaves = ArrayList<Tracer<Shape>>()

    /** `F32 [size]` of [bound]: 1 at a real position, 0 at a padded one. An extra program input. */
    fun validMask(bound: DimBound): Tracer<Shape> = leaf(BoundedInputRole.VALID_MASK, bound) { real, traced ->
        DTensor<Shape, F32>(HostF32Storage(FloatArray(traced) { if (it < real) 1f else 0f }), intArrayOf(traced), F32)
    }

    /** `F32` scalar: the real size of [bound]. An extra program input. */
    @Suppress("UNCHECKED_CAST")
    fun validLength(bound: DimBound): Tracer<ScalarShape> = leaf(BoundedInputRole.VALID_LENGTH, bound) { real, _ ->
        DTensor<Shape, F32>(HostF32Storage(floatArrayOf(real.toFloat())), intArrayOf(), F32)
    } as Tracer<ScalarShape>

    private fun leaf(
        role: BoundedInputRole,
        bound: DimBound,
        value: (real: Int, traced: Int) -> DTensor<Shape, F32>,
    ): Tracer<Shape> {
        require(bound in known) { "BoundedContext: $bound is not a bound of this program's inputs" }
        val i = requested.indexOf(role to bound)
        if (i >= 0) return leaves[i]
        val t = tape.traceLeaf(value(realSizes.getValue(bound), traceSizes.getValue(bound)))
        requested += role to bound
        leaves += t
        return t
    }
}

/** One traced size of a [BoundedProgram]: the function and what its parameters are. */
@ExperimentalTlalocApi
class BoundedTrace internal constructor(
    val sizes: Map<DimBound, Int>,
    val function: DxirFunction,
    /** One per parameter of [function], in order: the role, and the bound for a mask or length. */
    val parameters: List<Pair<BoundedInputRole, DimBound?>>,
    val outputDims: IntArray,
)

/** The largest difference [BoundedProgram.checkPadding] saw, and where. */
@ExperimentalTlalocApi
data class PaddingReport(
    val sizesChecked: List<Map<DimBound, Int>>,
    val maxDifference: Float,
    val worstSizes: Map<DimBound, Int>?,
    val tolerance: Float,
) {
    val passed: Boolean get() = worstSizes == null || maxDifference <= tolerance
}

/**
 * A program over tensors with bounded axes (docs/design/bounded-dims.md). The body is traced
 * once per size assignment, on zeros of the traced shape, and cached; a run evaluates the trace
 * in the reference interpreter.
 *
 * Every axis of one bound has the same size within a call; a run refuses inputs that disagree.
 */
@ExperimentalTlalocApi
class BoundedProgram(
    val name: String,
    val inputs: List<TensorSpec>,
    val output: TensorSpec,
    private val body: (List<Tracer<Shape>>, BoundedContext) -> Tracer<*>,
) {
    /** Every bound of the inputs, in first-use order. The output may use only these. */
    val bounds: List<DimBound> = inputs.flatMap { it.bounds }.distinct()

    init {
        require(inputs.isNotEmpty()) { "BoundedProgram '$name': at least one input is required" }
        require(bounds.isNotEmpty()) { "BoundedProgram '$name': no input has a bounded axis" }
        val names = bounds.groupBy { it.boundName }.filterValues { it.size > 1 }.keys
        require(names.isEmpty()) {
            "BoundedProgram '$name': two different bounds are named $names; an artifact names bounds by " +
                "their object's simple name, so they must differ"
        }
        val stray = output.bounds - bounds.toSet()
        require(stray.isEmpty()) {
            "BoundedProgram '$name': the output uses bounds $stray that no input has, so no run can size them"
        }
        require(output.dtype == F32) { "BoundedProgram '$name': the output must be F32" }
    }

    private val traces = HashMap<Map<DimBound, Int>, BoundedTrace>()

    /** How many distinct size assignments have been traced. */
    val traceCount: Int get() = traces.size

    /** The sizes of each bound in [tensors], checked against the specs. */
    fun sizesOf(tensors: List<DTensor<*, *>>): Map<DimBound, Int> {
        require(tensors.size == inputs.size) {
            "BoundedProgram '$name': ${tensors.size} inputs given, ${inputs.size} expected"
        }
        val sizes = LinkedHashMap<DimBound, Int>()
        for ((k, t) in tensors.withIndex()) {
            val spec = inputs[k]
            require(t.dtype == spec.dtype) {
                "BoundedProgram '$name': input $k is ${t.dtype.name}, the spec says ${spec.dtype.name}"
            }
            require(t.dims.size == spec.rank) {
                "BoundedProgram '$name': input $k has rank ${t.dims.size}, the spec $spec has rank ${spec.rank}"
            }
            for ((i, axis) in spec.axes.withIndex()) {
                val d = t.dims[i]
                when (axis) {
                    is AxisSpec.Fixed -> require(d == axis.size) {
                        "BoundedProgram '$name': input $k axis $i is $d, the spec fixes it at ${axis.size}"
                    }
                    is AxisSpec.Bounded -> {
                        require(d in 1..axis.bound.max) {
                            "BoundedProgram '$name': input $k axis $i is $d, outside 1..${axis.bound.max} of ${axis.bound}"
                        }
                        val prev = sizes.putIfAbsent(axis.bound, d)
                        require(prev == null || prev == d) {
                            "BoundedProgram '$name': ${axis.bound} is $prev on one axis and $d on input $k axis $i; " +
                                "every axis of one bound has the same size"
                        }
                    }
                }
            }
        }
        return sizes
    }

    /** The trace at [sizes] (one size per bound), traced on first use. */
    fun trace(sizes: Map<DimBound, Int>): BoundedTrace {
        require(sizes.keys == bounds.toSet()) {
            "BoundedProgram '$name': sizes for ${sizes.keys} given, the program's bounds are $bounds"
        }
        for ((b, n) in sizes) require(n in 1..b.max) { "BoundedProgram '$name': size $n is outside 1..${b.max} of $b" }
        return traces.getOrPut(sizes.toMap()) { traceAt(sizes) }
    }

    private fun traceAt(sizes: Map<DimBound, Int>): BoundedTrace {
        val tape = Tape()
        val leaves = inputs.map { spec ->
            val dims = spec.dimsAt(sizes)
            val n = dims.fold(1) { a, b -> a * b }
            when (spec.dtype) {
                I32 -> tape.traceLeafI32(DTensor<Shape, I32>(HostI32Storage(IntArray(n)), dims, I32))
                else -> tape.traceLeaf(DTensor<Shape, F32>(HostF32Storage(FloatArray(n)), dims, F32))
            }
        }
        val ctx = BoundedContext(tape, sizes, sizes, bounds.toSet())
        val out = body(leaves, ctx)
        require(out.tape === tape) {
            "BoundedProgram '$name': the result does not come from this program's inputs"
        }
        val want = output.dimsAt(sizes)
        require(out.dims.contentEquals(want)) {
            "BoundedProgram '$name': at sizes ${sizes.describe()} the result is ${out.dims.toList()}, the output " +
                "spec $output says ${want.toList()}"
        }
        // Parameters come out in tape order, which is creation order: the data leaves first,
        // then each mask or length in the order the body first asked for it.
        val params = leaves.map { BoundedInputRole.DATA to null as DimBound? } +
            ctx.requested.map { (role, b) -> role to b }
        val fn = tape.toDxirFunction(name, paramIds = leaves.map { it.id } + ctx.leaves.map { it.id }, returnIds = listOf(out.id))
        check(fn.params.size == params.size) { "BoundedProgram '$name': ${fn.params.size} params traced, ${params.size} expected" }
        return BoundedTrace(sizes.toMap(), fn, params, out.dims.copyOf())
    }

    /** Runs at the inputs' exact sizes in the reference interpreter. */
    fun run(tensors: List<DTensor<*, *>>): DTensor<Shape, F32> {
        val sizes = sizesOf(tensors)
        val t = trace(sizes)
        val values = argumentValues(t, tensors.map { it.floatValues() }, sizes)
        val out = DxirInterpreter.evalFunction(t.function, values).single()
        return DTensor(HostF32Storage(out), t.outputDims.copyOf(), F32)
    }

    /**
     * Pads the inputs to the smallest bucket of each bound, runs that bucket's trace in the
     * reference interpreter, and slices the result back to the real sizes. This is what an
     * exported artifact's runtime does.
     */
    fun runBucketed(tensors: List<DTensor<*, *>>, ladders: BucketLadders): DTensor<Shape, F32> {
        val sizes = sizesOf(tensors)
        val buckets = sizes.mapValues { (b, n) -> ladders.bucketFor(b, n) }
        val t = trace(buckets)
        val padded = tensors.mapIndexed { k, x ->
            padTo(x.floatValues(), x.dims, inputs[k].dimsAt(buckets))
        }
        val values = argumentValues(t, padded, sizes)
        val out = DxirInterpreter.evalFunction(t.function, values).single()
        val real = output.dimsAt(sizes)
        return DTensor(HostF32Storage(sliceTo(out, t.outputDims, real)), real, F32)
    }

    /** The argument list of [t] for data values [data] and real [sizes]. */
    internal fun argumentValues(t: BoundedTrace, data: List<FloatArray>, sizes: Map<DimBound, Int>): List<FloatArray> {
        val dataIt = data.iterator()
        return t.parameters.map { (role, bound) ->
            when (role) {
                BoundedInputRole.DATA -> dataIt.next()
                BoundedInputRole.VALID_MASK -> {
                    val traced = t.sizes.getValue(bound!!)
                    val real = sizes.getValue(bound)
                    FloatArray(traced) { if (it < real) 1f else 0f }
                }
                BoundedInputRole.VALID_LENGTH -> floatArrayOf(sizes.getValue(bound!!).toFloat())
            }
        }
    }

    /**
     * Compares [runBucketed] with [run] on seeded random inputs at each size assignment of
     * [sizes], and reports the largest difference relative to `max(1, max |exact|)`. F32 inputs
     * are uniform in [-1, 1]; I32 inputs are 0.
     */
    fun checkPadding(
        ladders: BucketLadders,
        sizes: List<Map<DimBound, Int>> = edgeSizes(ladders),
        tolerance: Float = 1e-5f,
        seed: Long = 7,
    ): PaddingReport {
        var worst = 0f
        var worstAt: Map<DimBound, Int>? = null
        var state = seed
        fun next(): Float {
            state = state * 6364136223846793005L + 1442695040888963407L
            return ((state ushr 40).toInt() / (1 shl 24).toFloat()) * 2f - 1f
        }
        for (s in sizes) {
            val tensors: List<DTensor<*, *>> = inputs.map { spec ->
                val dims = spec.dimsAt(s)
                val n = dims.fold(1) { a, b -> a * b }
                when (spec.dtype) {
                    I32 -> DTensor<Shape, I32>(HostI32Storage(IntArray(n)), dims, I32)
                    else -> DTensor<Shape, F32>(HostF32Storage(FloatArray(n) { next() }), dims, F32)
                }
            }
            val exact = run(tensors).hostF32()
            val bucketed = runBucketed(tensors, ladders).hostF32()
            val scale = max(1f, exact.maxOfOrNull { abs(it) } ?: 0f)
            var d = 0f
            for (i in exact.indices) {
                val e = abs(exact[i] - bucketed[i]) / scale
                if (!(e <= d)) d = if (e.isNaN() || exact[i].isNaN() != bucketed[i].isNaN()) Float.POSITIVE_INFINITY else e
            }
            if (d > worst || worstAt == null && d > 0f) { worst = d; worstAt = s }
        }
        return PaddingReport(sizes, worst, if (worst > 0f) worstAt else null, tolerance)
    }

    /**
     * The size assignments [checkPadding] uses by default: for each bound, 1 and the first and
     * last size of every bucket; the cross product when it has at most 64 points, otherwise
     * each bound's list with the other bounds at their largest bucket.
     */
    fun edgeSizes(ladders: BucketLadders): List<Map<DimBound, Int>> {
        val perBound = bounds.associateWith { b ->
            val ladder = ladders.ladder(b)
            (listOf(1) + ladder.flatMapIndexed { i, top -> listOf(if (i == 0) 1 else ladder[i - 1] + 1, top) }).distinct()
        }
        val product = perBound.values.fold(1L) { a, l -> a * l.size }
        if (product <= 64) {
            var acc = listOf(emptyMap<DimBound, Int>())
            for ((b, l) in perBound) acc = acc.flatMap { m -> l.map { m + (b to it) } }
            return acc
        }
        val tops = bounds.associateWith { it.max }
        return perBound.flatMap { (b, l) -> l.map { tops + (b to it) } }.distinct()
    }

    companion object {
        /** Zero-pads a row-major array of [dims] to [target] (every target dim at least the source's). */
        fun padTo(values: FloatArray, dims: IntArray, target: IntArray): FloatArray {
            require(dims.size == target.size && dims.indices.all { dims[it] <= target[it] }) {
                "padTo: ${dims.toList()} does not fit in ${target.toList()}"
            }
            if (dims.contentEquals(target)) return values.copyOf()
            val out = FloatArray(target.fold(1) { a, b -> a * b })
            copyBlock(values, dims, out, target, dims)
            return out
        }

        /** The leading [target] block of a row-major array of [dims]. */
        fun sliceTo(values: FloatArray, dims: IntArray, target: IntArray): FloatArray {
            require(dims.size == target.size && dims.indices.all { target[it] <= dims[it] }) {
                "sliceTo: ${target.toList()} is not inside ${dims.toList()}"
            }
            if (dims.contentEquals(target)) return values.copyOf()
            val out = FloatArray(target.fold(1) { a, b -> a * b })
            copyBlock(values, dims, out, target, target)
            return out
        }

        private fun copyBlock(src: FloatArray, srcDims: IntArray, dst: FloatArray, dstDims: IntArray, block: IntArray) {
            val rank = block.size
            if (rank == 0) { dst[0] = src[0]; return }
            if (block.any { it == 0 }) return
            val srcStride = strides(srcDims)
            val dstStride = strides(dstDims)
            val idx = IntArray(rank)
            val inner = block[rank - 1]
            while (true) {
                var s = 0
                var d = 0
                for (a in 0 until rank - 1) { s += idx[a] * srcStride[a]; d += idx[a] * dstStride[a] }
                src.copyInto(dst, d, s, s + inner)
                var a = rank - 2
                while (a >= 0) {
                    idx[a]++
                    if (idx[a] < block[a]) break
                    idx[a] = 0
                    a--
                }
                if (a < 0) return
            }
        }

        private fun strides(dims: IntArray): IntArray {
            val s = IntArray(dims.size)
            var acc = 1
            for (i in dims.indices.reversed()) { s[i] = acc; acc *= dims[i] }
            return s
        }
    }
}

/** Builds a [BoundedProgram]. [body] gets one tracer per input, erased to `Tracer<Shape>`. */
@ExperimentalTlalocApi
fun boundedProgram(
    name: String,
    inputs: List<TensorSpec>,
    output: TensorSpec,
    body: (List<Tracer<Shape>>, BoundedContext) -> Tracer<*>,
): BoundedProgram = BoundedProgram(name, inputs, output, body)

private fun DTensor<*, *>.floatValues(): FloatArray = when (dtype) {
    I32 -> @Suppress("UNCHECKED_CAST") (this as DTensor<*, I32>).hostI32().let { a -> FloatArray(a.size) { a[it].toFloat() } }
    else -> @Suppress("UNCHECKED_CAST") (this as DTensor<*, F32>).hostF32()
}

private fun Map<DimBound, Int>.describe(): String = entries.joinToString(", ", "{", "}") { "${it.key.boundName}=${it.value}" }
