package io.tlaloc.testing

import io.tlaloc.core.DType
import io.tlaloc.core.ExperimentalTlalocApi
import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirForwardTransform
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirInterpreterF64
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.ir.passes.DxirVmapTransform
import kotlin.math.abs
import kotlin.random.Random

/**
 * A place an op has to work, and the check the sweep runs there.
 *
 * - [INTERPRETER]: the F32 interpreter runs the sample and its results are finite.
 * - [INTERPRETER_F64]: the F64 interpreter agrees with the F32 one.
 * - [VJP]: the reverse-mode gradient of `Σ y ⊙ w` agrees with central differences
 *   (in F64 when the op has an F64 interpreter arm).
 * - [JVP]: the forward-mode tangent of that loss equals `⟨gradient, v⟩`.
 * - [VMAP]: the batched function equals the function applied to each example.
 * - [RENDER]: the sample and its gradient render as Kotlin source.
 * - [EMIT]: the sample emits as StableHLO at F32, and so does its gradient where
 *   [VJP] is claimed.
 * - [EMIT_F64]: the same at F64.
 * - [DEVICE]: the sample (and its gradient where [VJP] is claimed) on a PJRT device
 *   agrees with the F32 interpreter.
 */
enum class Surface { INTERPRETER, INTERPRETER_F64, VJP, JVP, VMAP, RENDER, EMIT, EMIT_F64, DEVICE }

/** How the values of one argument are drawn. */
sealed interface Gen {
    fun draw(r: Random, n: Int): DoubleArray

    data class Uniform(val lo: Double = -1.0, val hi: Double = 1.0) : Gen {
        override fun draw(r: Random, n: Int) = DoubleArray(n) { lo + (hi - lo) * r.nextDouble() }
    }

    /** Integers in `[lo, until)`, for index arguments. */
    data class Ints(val lo: Int, val until: Int) : Gen {
        override fun draw(r: Random, n: Int) = DoubleArray(n) { r.nextInt(lo, until).toDouble() }
    }

    /** A symmetric positive definite `n×n` matrix: `B·Bᵀ + n·I`. */
    data class Spd(val n: Int) : Gen {
        override fun draw(r: Random, n: Int): DoubleArray {
            val b = DoubleArray(this.n * this.n) { r.nextDouble() - 0.5 }
            return DoubleArray(this.n * this.n) { k ->
                val i = k / this.n
                val j = k % this.n
                (0 until this.n).sumOf { b[i * this.n + it] * b[j * this.n + it] } + if (i == j) this.n.toDouble() else 0.0
            }
        }
    }

    /** These values, in order. */
    class Fixed(val values: DoubleArray) : Gen {
        override fun draw(r: Random, n: Int): DoubleArray {
            require(n == values.size) { "Gen.Fixed: ${values.size} values for $n elements" }
            return values.copyOf()
        }
    }
}

/**
 * One argument of a sample. Float arguments take the dtype the sweep runs at (F32 or
 * F64); an [integer] argument is always I32 and gets no gradient.
 */
data class Arg(val name: String, val dims: List<Int>, val gen: Gen = Gen.Uniform(), val integer: Boolean = false) {
    val size: Int get() = dims.fold(1, Int::times)

    fun type(float: DType): DxirType = DxirType(if (integer) I32 else float, dims)
}

/**
 * One configuration of an op: its arguments and a body that builds `y = op(args)`.
 * [body] gets the float dtype and the argument nodes and returns a float result.
 */
class OpSample(
    val name: String,
    val args: List<Arg>,
    /** Relative tolerance of the [Surface.VJP] check against central differences, in F64. */
    val gradTolerance: Double = 1e-6,
    /** The sample contains matrix products (TF32 on GPUs; see [Tolerances]). */
    val usesDot: Boolean = false,
    /** Central-difference step relative to `max(1, |x|)`; default 1e-6 in F64, 1e-2 in F32. */
    val fdStep: Double? = null,
    /** Surfaces this sample is not expected to work on, beyond its entry's. */
    val excluded: Map<Surface, String> = emptyMap(),
    val body: DxirBuilder.(DType, List<DxirNode>) -> DxirNode,
)

/**
 * An [OpKind]'s place in the sweep: its samples and the surfaces where it is not
 * expected to work, each with the reason. A surface absent from [excluded] is claimed.
 * [samples] may be empty only when [noSample] says where the op is tested instead.
 */
class OpEntry(
    val kind: OpKind,
    val samples: List<OpSample>,
    val excluded: Map<Surface, String> = emptyMap(),
    val noSample: String? = null,
)

/** The functions the checks run for one sample at one float dtype. */
class SampleFunctions(val sample: OpSample, val dtype: DType) {
    /** `args → y`. */
    val primal: DxirFunction = DxirBuilder.function("${sample.name}_${dtype}") {
        val ps = sample.args.map { param(it.name, it.type(dtype)) }
        listOf(sample.body(this, dtype, ps))
    }

    val resultType: DxirType = primal.returns.single().type

    /** `(args, w) → Σ y ⊙ w`, `w` an input only. */
    val loss: DxirFunction = DxirBuilder.function("${sample.name}_${dtype}_loss") {
        val ps = sample.args.map { param(it.name, it.type(dtype)) }
        val y = sample.body(this, dtype, ps)
        val w = param("w", y.type)
        val yw = op(OpKind.MUL, listOf(y, w), y.type)
        listOf(op(OpKind.SUM, listOf(yw), DxirType(dtype, emptyList())))
    }

    val floatArgs: List<Int> = sample.args.indices.filter { !sample.args[it].integer }

    /** The gradient of [loss] for every argument (integer ones get zeros). */
    val gradient: DxirFunction by lazy { DxirReverseTransform.apply(loss, inputOnlyTrailingParams = 1) }

    /** The arguments' values for [seed], then `w`. */
    fun inputs(seed: Int = 1): List<DoubleArray> {
        val r = Random(seed * 7919 + sample.name.hashCode())
        val args = sample.args.map { it.gen.draw(r, it.size) }
        val n = resultType.dims.fold(1, Int::times)
        return args + listOf(DoubleArray(n) { r.nextDouble() * 2 - 1 })
    }
}

fun List<DoubleArray>.toFloats(): List<FloatArray> = map { a -> FloatArray(a.size) { a[it].toFloat() } }

/** Evaluates [fn] in the interpreter of [dtype]; results as Doubles. */
fun interpret(fn: DxirFunction, dtype: DType, inputs: List<DoubleArray>): List<DoubleArray> =
    if (dtype == F64) DxirInterpreterF64.evalFunction(fn, inputs)
    else DxirInterpreter.evalFunction(fn, inputs.toFloats()).map { a -> DoubleArray(a.size) { a[it].toDouble() } }

/** Largest `|a − b|` over `max(1, max |b|)`. */
fun scaledError(a: DoubleArray, b: DoubleArray): Double {
    require(a.size == b.size) { "sizes ${a.size} and ${b.size}" }
    if (a.isEmpty()) return 0.0
    val scale = maxOf(1.0, b.maxOf { abs(it) })
    return a.indices.maxOf { i ->
        if (a[i].isNaN() && b[i].isNaN()) 0.0 else if (a[i] == b[i]) 0.0 else abs(a[i] - b[i])
    } / scale
}

class SweepFailure(message: String) : AssertionError(message)

/**
 * The checks that need only `:ir`. A check returns normally when the surface works
 * and throws when it does not.
 */
@OptIn(ExperimentalTlalocApi::class)
object OpChecks {

    fun interpreter(s: OpSample) {
        val f = SampleFunctions(s, F32)
        val out = interpret(f.primal, F32, f.inputs().dropLast(1))
        if (out.any { a -> a.any { !it.isFinite() } }) throw SweepFailure("non-finite result")
    }

    fun interpreterF64(s: OpSample) {
        val f64 = SampleFunctions(s, F64)
        val f32 = SampleFunctions(s, F32)
        val x = f64.inputs().dropLast(1)
        // F32 runs on the F32-rounded inputs, so the difference is the op's own rounding.
        val rounded = x.map { a -> DoubleArray(a.size) { a[it].toFloat().toDouble() } }
        val want = interpret(f64.primal, F64, rounded).single()
        val got = interpret(f32.primal, F32, rounded).single()
        val err = scaledError(got, want)
        val tol = Tolerances.relative(s, F32, Surface.INTERPRETER_F64)
        if (err > tol) throw SweepFailure("F32 against F64: scaled error $err > $tol")
    }

    /** The dtype gradient checks run at: F64 when its interpreter has the op. */
    fun gradientDtype(s: OpSample): DType = try {
        val f = SampleFunctions(s, F64)
        interpret(f.primal, F64, f.inputs().dropLast(1))
        F64
    } catch (_: Throwable) {
        F32
    }

    fun vjp(s: OpSample) {
        val dt = gradientDtype(s)
        val f = SampleFunctions(s, dt)
        val x = f.inputs()
        val grads = interpret(f.gradient, dt, x)
        for (ai in f.floatArgs) {
            val base = x[ai]
            val fd = DoubleArray(base.size) { i ->
                val h = (s.fdStep ?: if (dt == F64) 1e-6 else 1e-2) * maxOf(1.0, abs(base[i]))
                fun at(v: Double): Double {
                    val xs = x.toMutableList()
                    xs[ai] = base.copyOf().also { it[i] = v }
                    return interpret(f.loss, dt, xs).single()[0]
                }
                (at(base[i] + h) - at(base[i] - h)) / (2 * h)
            }
            val got = grads[ai]
            val err = scaledError(got, fd)
            val tol = if (dt == F64) s.gradTolerance else maxOf(s.gradTolerance, 2e-2)
            if (err > tol) {
                throw SweepFailure("d ${s.args[ai].name} ($dt): scaled error $err > $tol against central differences")
            }
        }
    }

    fun jvp(s: OpSample) {
        val dt = gradientDtype(s)
        val f = SampleFunctions(s, dt)
        val x = f.inputs()
        val grads = interpret(f.gradient, dt, x)
        val r = Random(17)
        val v = x.indices.map { i ->
            val float = i < s.args.size && !s.args[i].integer
            DoubleArray(x[i].size) { if (float) r.nextDouble() * 2 - 1 else 0.0 }
        }
        val fwd = DxirForwardTransform.apply(f.loss)
        val out = interpret(fwd, dt, x + v)
        val tangent = out[1].single()
        val want = f.floatArgs.sumOf { ai -> grads[ai].indices.sumOf { grads[ai][it] * v[ai][it] } }
        val err = abs(tangent - want) / maxOf(1.0, abs(want))
        val tol = if (dt == F64) 1e-9 else 1e-4
        if (err > tol) throw SweepFailure("tangent $tangent against ⟨gradient, v⟩ $want ($dt)")
    }

    fun vmap(s: OpSample) {
        val dt = gradientDtype(s)
        val f = SampleFunctions(s, dt)
        val batch = 3
        val examples = (1..batch).map { f.inputs(seed = it).dropLast(1) }
        val batched = DxirVmapTransform.apply(f.primal, List(s.args.size) { true }, batch)
        val stacked = s.args.indices.map { i -> examples.flatMap { it[i].asIterable() }.toDoubleArray() }
        val got = interpret(batched, dt, stacked).single()
        val want = examples.flatMap { interpret(f.primal, dt, it).single().asIterable() }.toDoubleArray()
        val err = scaledError(got, want)
        val tol = if (dt == F64) 1e-12 else 1e-5
        if (err > tol) throw SweepFailure("batched against per example ($dt): scaled error $err > $tol")
    }
}

/** What one check found for one sample. */
data class SweepResult(
    val kind: OpKind,
    val sample: String,
    val surface: Surface,
    val expected: Boolean,
    val error: String?,
) {
    val passed: Boolean get() = error == null

    /** A claimed surface that failed, or an excluded one that works (the exclusion is stale). */
    val unexpected: Boolean get() = passed != expected

    override fun toString(): String = when {
        expected && !passed -> "$kind/$sample on $surface: $error"
        !expected && passed -> "$kind/$sample on $surface works; remove its exclusion"
        else -> "$kind/$sample on $surface: ${if (passed) "ok" else "excluded"}"
    }
}

object OpSweep {

    /**
     * Runs [check] on every sample of every catalogue entry for [surface] and returns
     * the results. Excluded surfaces are run too, so an exclusion that no longer holds
     * is reported.
     */
    fun run(
        surface: Surface,
        entries: Collection<OpEntry> = OpCatalogue.entries.values,
        check: (OpSample) -> Unit,
    ): List<SweepResult> = entries.flatMap { e ->
        e.samples.map { s ->
            val error = try {
                check(s)
                null
            } catch (t: Throwable) {
                if (t is VirtualMachineError) throw t
                (t.message ?: t::class.simpleName ?: "error").lineSequence().first().take(400)
            }
            SweepResult(e.kind, s.name, surface, surface !in e.excluded && surface !in s.excluded, error)
        }
    }

    /**
     * Throws listing every unexpected result. Every result is also written to
     * `build/op-sweep/<surface>.txt` of the running module.
     */
    fun assertExpected(results: List<SweepResult>) {
        if (results.isNotEmpty()) {
            val dir = java.io.File("build/op-sweep").apply { mkdirs() }
            java.io.File(dir, "${results.first().surface.name.lowercase()}.txt")
                .writeText(results.joinToString("\n", postfix = "\n"))
        }
        val bad = results.filter { it.unexpected }
        if (bad.isNotEmpty()) {
            throw SweepFailure(
                "${bad.size} of ${results.size} results differ from the catalogue:\n" + bad.joinToString("\n"),
            )
        }
    }

    /** The default check for each surface that needs only `:ir`. */
    fun irCheck(surface: Surface): (OpSample) -> Unit = when (surface) {
        Surface.INTERPRETER -> OpChecks::interpreter
        Surface.INTERPRETER_F64 -> OpChecks::interpreterF64
        Surface.VJP -> OpChecks::vjp
        Surface.JVP -> OpChecks::jvp
        Surface.VMAP -> OpChecks::vmap
        else -> error("$surface needs a module other than :ir")
    }
}
