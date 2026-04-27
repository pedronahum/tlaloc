package io.tlaloc.benchmarks

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.ir.passes.PhiCalculus

/**
 * §0.4.222 — head-to-head harness Phase 1, scaffolding.
 *
 * Per `docs/HEAD_TO_HEAD_HARNESS_PLAN.md`'s Phase 1 plan, this is the
 * JVM-side scaffold for measuring Tlaloc's gradient-evaluation throughput
 * + numerical baseline on each paper benchmark, in preparation for the
 * cross-framework comparison in Phase 2 (gated on PyTorch / JAX toolchain
 * availability).
 *
 * **What this harness measures (Phase 1)**:
 *  - **Forward value** at a fixed input — the numerical baseline that
 *    Phase 2's PyTorch / JAX runs will be compared against.
 *  - **Gradient values** at the same fixed input (per-input partials).
 *  - **Per-iteration latency** of the gradient evaluation through
 *    [DxirInterpreter] — median, min, p99 nanoseconds over a measured
 *    window after a warmup window.
 *
 * **What this harness does NOT measure (yet)**:
 *  - Native-code throughput. [DxirInterpreter] is a JVM-side dxir
 *    interpreter, not a compiled runtime. The numbers here reflect
 *    interpreter overhead + JVM JIT effects, NOT what a future IREE CPU
 *    backend (M3) will produce. Phase 1's purpose is to establish the
 *    measurement scaffold; Phase 3 will be re-measured against the M3
 *    runtime once it ships.
 *  - PyTorch / JAX comparison. Gated on toolchain availability per the
 *    plan; the JVM harness writes its result for later JSON-based
 *    cross-framework aggregation.
 */
data class HeadToHeadResult(
    val benchmark: String,
    val forwardValue: Float,
    val gradientValues: List<Float>,
    val warmupIterations: Int,
    val measuredIterations: Int,
    val medianNanos: Long,
    val minNanos: Long,
    val p99Nanos: Long,
) {
    /**
     * §0.4.222 — minimal JSON serialisation. No external deps; just enough
     * structure to enable Phase 2's JSON-based cross-framework comparison.
     * Floats are formatted with 6 significant digits (sufficient for f32
     * tolerance).
     */
    fun toJsonString(): String = buildString {
        append("{\n")
        append("  \"benchmark\": \"$benchmark\",\n")
        append("  \"forwardValue\": ${"%.6g".format(forwardValue)},\n")
        append("  \"gradientValues\": [")
        gradientValues.forEachIndexed { i, v ->
            if (i > 0) append(", ")
            append("%.6g".format(v))
        }
        append("],\n")
        append("  \"warmupIterations\": $warmupIterations,\n")
        append("  \"measuredIterations\": $measuredIterations,\n")
        append("  \"medianNanos\": $medianNanos,\n")
        append("  \"minNanos\": $minNanos,\n")
        append("  \"p99Nanos\": $p99Nanos\n")
        append("}")
    }
}

/**
 * Each inhabitant supplies the primal + fixed inputs. The harness handles
 * coarsening (`PhiCalculus.apply`), reverse-mode AD (`DxirReverseTransform.apply`),
 * forward eval, gradient eval, and the timing loop.
 */
interface HeadToHeadBenchmark {
    val name: String
    fun primal(): DxirFunction
    fun fixedInputs(): List<FloatArray>

    /**
     * Run the per-benchmark protocol: coarsen → reverse-AD → forward eval
     * → gradient eval → timing loop. Returns a [HeadToHeadResult] with the
     * baseline values + latency statistics.
     *
     * @param warmup iterations to discard before measurement (handles JIT
     *   tier-up; default 200 per the plan's Phase 1 methodology).
     * @param measured iterations counted in the timing window (default 800).
     */
    fun runBaseline(warmup: Int = 200, measured: Int = 800): HeadToHeadResult {
        val primal = primal()
        val coarsened = PhiCalculus.apply(primal)
        val grad = DxirReverseTransform.apply(coarsened)
        val inputs = fixedInputs()

        val forwardOut = DxirInterpreter.evalFunction(primal, inputs)
        require(forwardOut.size == 1 && forwardOut[0].size == 1) {
            "$name: harness expects scalar forward output (got returns=${forwardOut.size}, " +
                "first-element size=${forwardOut[0].size})"
        }
        val forwardValue = forwardOut[0][0]

        val gradOut = DxirInterpreter.evalFunction(grad, inputs)
        require(gradOut.size == inputs.size) {
            "$name: gradient function should return one value per input " +
                "(got ${gradOut.size}, expected ${inputs.size})"
        }
        val gradientValues = gradOut.map { it[0] }

        repeat(warmup) {
            DxirInterpreter.evalFunction(grad, inputs)
        }
        val timings = LongArray(measured)
        for (i in 0 until measured) {
            val t0 = System.nanoTime()
            DxirInterpreter.evalFunction(grad, inputs)
            timings[i] = System.nanoTime() - t0
        }
        timings.sort()
        val median = timings[measured / 2]
        val min = timings[0]
        val p99 = timings[(measured * 99) / 100]

        return HeadToHeadResult(
            benchmark = name,
            forwardValue = forwardValue,
            gradientValues = gradientValues,
            warmupIterations = warmup,
            measuredIterations = measured,
            medianNanos = median,
            minNanos = min,
            p99Nanos = p99,
        )
    }
}

/**
 * §0.4.222 — first harness inhabitant: QWOP avatar-step (full primal from
 * §0.4.220 integration test). Fixed inputs `m* = 0.1` for all four muscles —
 * the same safe-no-clamp point used in the integration test, so the baseline
 * numerics are reproducible across firings.
 */
object QwopAvatarStepHarness : HeadToHeadBenchmark {
    override val name = "qwop-avatar-step"
    override fun primal() = Qwop.avatarStepPrimal()
    override fun fixedInputs() = listOf(
        floatArrayOf(0.1f), floatArrayOf(0.1f),
        floatArrayOf(0.1f), floatArrayOf(0.1f),
    )
}
