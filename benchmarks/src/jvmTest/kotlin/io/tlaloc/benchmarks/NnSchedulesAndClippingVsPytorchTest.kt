package io.tlaloc.benchmarks

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.nn.CosineDecay
import io.tlaloc.nn.ExponentialDecay
import io.tlaloc.nn.GradientClipping
import io.tlaloc.nn.StepDecay
import io.tlaloc.testing.Goldens
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.502 (Tier 2 item 7) — **the PyTorch oracle for the §0.4.502 training
 * utilities.** The repository pins its optimizers against PyTorch
 * (`NnMlpVsPytorchTrainingTest`); the schedules and the clipping get the same
 * treatment, at the same bar, through the same §0.4.289 pattern (subprocess →
 * JSON).
 *
 * WHAT WAS OBSERVED, torch 2.11.0+cpu on this GB10:
 *  - `StepDecay` vs `StepLR`, `ExponentialDecay` vs `ExponentialLR`: agreement
 *    at 1e-6 relative over 12 steps.
 *  - `CosineDecay` vs `CosineAnnealingLR`: agreement at 1e-6 relative over
 *    `[0, T_max]` — **and a REAL DIVERGENCE past `T_max`**, which this test
 *    asserts rather than avoids: torch's schedule is periodic and climbs back
 *    (k=11 reads 0.00245, k=13 reads 0.0206 after reaching 0 at k=10), while
 *    `CosineDecay` clamps at [CosineDecay.finalLearningRate] forever. That is
 *    the documented decision in `CosineDecay`'s KDoc, and it is certified here
 *    as a difference, not smoothed over.
 *  - `byGlobalNorm` vs `clip_grad_norm_`: the norm agrees exactly (5.5 on the
 *    fixture); the clipped VALUES differ by torch's `+1e-6` denominator guard,
 *    ~1.3e-7 relative on this fixture. Tlaloc is compared to the exact analytic
 *    ratio at 1e-6 and to torch at 1e-5, and the measured gap to torch is
 *    printed so a future change in either direction is visible.
 *  - `byValue` vs `clip_grad_value_`: EXACT. Neither has an epsilon.
 *
 * Without the oracle interpreter or torch, PyTorch's side is its recording
 * (test-support Goldens, suite `schedules-torch`). Installs nothing (see
 * `OracleVenvIntegrityTest`).
 */
class NnSchedulesAndClippingVsPytorchTest {

    // The frozen fixtures, duplicated verbatim from the script.
    private val stepLr0 = 0.1f
    private val stepGamma = 0.5f
    private val stepSize = 3
    private val expLr0 = 0.2f
    private val expGamma = 0.9f
    private val cosLr0 = 0.1f
    private val cosTMax = 10
    private val grads: Map<String, DTensor<*, F32>> = mapOf(
        "g0" to Tensors.f32Vector<Sym>(floatArrayOf(1f, -2f, 3f)),
        "g1" to Tensors.f32Vector<Sym>(floatArrayOf(0.5f, 4f)),
    )
    private val maxNorm = 1f
    private val valueLimit = 0.75f

    /** PyTorch's values, live with torch installed, else the recording (test-support Goldens). */
    private fun oracle(): JsonObject {
        val script = Path.of("..", "harness", "python", "run_pytorch_schedules_clipping.py")
            .toAbsolutePath().normalize()
        val json = Goldens.script("schedules-torch", "schedules_clipping", script, null, "", "torch") { python ->
            val dir = Files.createTempDirectory("tlaloc-sched-oracle-")
            try {
                val out = dir.resolve("out.json")
                val proc = ProcessBuilder(python, script.toString(), out.toString()).redirectErrorStream(true).start()
                val log = proc.inputStream.readBytes().decodeToString()
                assertTrue(proc.waitFor(300, TimeUnit.SECONDS), "oracle timed out\n$log")
                assertEquals(0, proc.exitValue(), "oracle failed:\n$log")
                Files.readString(out)
            } finally {
                dir.toFile().deleteRecursively()
            }
        }
        return parseJson(json) as JsonObject
    }

    private fun doubles(v: JsonArray): List<Double> =
        v.elements.map { (it as JsonNumber).value }

    private fun relClose(expected: Double, actual: Float, tol: Double, tag: String): Double {
        val diff = abs(expected - actual.toDouble())
        val rel = diff / maxOf(1e-12, maxOf(abs(expected), abs(actual.toDouble())))
        assertTrue(rel <= tol, "$tag: torch $expected vs tlaloc $actual (relative $rel > $tol)")
        return rel
    }

    @Test
    fun theThreeSchedulesPytorchAlsoShipsAgreeWithTorchToOneInAMillion() {
        val root = oracle()
        val schedules = root.obj("schedules")
        var worst = 0.0

        val step = StepDecay(stepLr0, stepGamma, stepSize)
        for ((k, want) in doubles(schedules.arr("step")).withIndex()) {
            worst = maxOf(worst, relClose(want, step.at(k), 1e-6, "StepLR k=$k"))
        }
        val exp = ExponentialDecay(expLr0, expGamma)
        for ((k, want) in doubles(schedules.arr("exponential")).withIndex()) {
            worst = maxOf(worst, relClose(want, exp.at(k), 1e-6, "ExponentialLR k=$k"))
        }
        val cos = CosineDecay(cosLr0, cosTMax)
        val cosine = doubles(schedules.arr("cosine"))
        for (k in 0..cosTMax) {
            worst = maxOf(worst, relClose(cosine[k], cos.at(k), 1e-6, "CosineAnnealingLR k=$k"))
        }
        println(
            "[nn-sched-oracle] torch ${(root["torch"] as io.tlaloc.core.io.JsonString).value}: " +
                "worst relative disagreement over 34 scheduled rates = $worst",
        )
        assertTrue(worst < 1e-6, "worst relative disagreement $worst")
    }

    /**
     * The published NEGATIVE result: past `T_max`, torch and Tlaloc disagree on
     * purpose. Asserted so the divergence cannot quietly disappear in either
     * direction — if torch ever stopped restarting, or `CosineDecay` ever
     * stopped clamping, this fails and someone decides again.
     */
    @Test
    fun cosineAnnealingDivergesFromTorchPastTMaxExactlyAsDocumented() {
        val root = oracle()
        val cosine = doubles(root.obj("schedules").arr("cosine"))
        assertTrue(cosine.size > cosTMax + 1, "the fixture must sweep past T_max: ${cosine.size}")
        val cos = CosineDecay(cosLr0, cosTMax)
        assertEquals(0.0, cosine[cosTMax], "torch reaches the floor at T_max")
        for (k in (cosTMax + 1) until cosine.size) {
            assertTrue(
                cosine[k] > 1e-4,
                "torch's CosineAnnealingLR is expected to CLIMB past T_max; k=$k read ${cosine[k]}",
            )
            assertEquals(0f, cos.at(k), "Tlaloc's CosineDecay must hold at the floor past T_max")
        }
        println(
            "[nn-sched-oracle] past T_max=$cosTMax torch climbs to ${cosine.last()} " +
                "while CosineDecay holds at 0.0 — the documented clamp",
        )
    }

    @Test
    fun gradientClippingAgreesWithTorchOnTheNormAndOnBothClipShapes() {
        val root = oracle()
        val c = root.obj("clipping")

        // The norm: formula-independent, so exact agreement is the bar.
        val torchNorm = (c["totalNorm"] as JsonNumber).value
        relClose(torchNorm, GradientClipping.globalNorm(grads), 1e-6, "global norm")

        val clipped = GradientClipping.byGlobalNorm(grads, maxNorm)
        val analytic = c.arr("analyticClipped")
        val torchClipped = c.arr("torchClipped")
        var worstAnalytic = 0.0
        var worstTorch = 0.0
        for ((i, key) in listOf("g0", "g1").withIndex()) {
            val mine = clipped.getValue(key).hostF32()
            val wantAnalytic = doubles(analytic[i] as JsonArray)
            val wantTorch = doubles(torchClipped[i] as JsonArray)
            for (j in mine.indices) {
                worstAnalytic = maxOf(
                    worstAnalytic,
                    relClose(wantAnalytic[j], mine[j], 1e-6, "analytic $key[$j]"),
                )
                worstTorch = maxOf(
                    worstTorch,
                    relClose(wantTorch[j], mine[j], 1e-5, "torch $key[$j]"),
                )
            }
        }
        println(
            "[nn-clip-oracle] by-global-norm: worst relative gap to the exact ratio " +
                "$worstAnalytic, to torch's (norm + 1e-6) form $worstTorch",
        )
        // The divergence is real and small: closer to the analytic answer than
        // to torch, which is the claim byGlobalNorm's KDoc makes.
        assertTrue(worstAnalytic < worstTorch, "Tlaloc must sit closer to the exact ratio than to torch")

        // By value has no epsilon on either side: exact.
        val byValue = GradientClipping.byValue(grads, valueLimit)
        val wantValue = c.arr("valueClipped")
        for ((i, key) in listOf("g0", "g1").withIndex()) {
            val mine = byValue.getValue(key).hostF32()
            val want = doubles(wantValue[i] as JsonArray)
            for (j in mine.indices) {
                assertEquals(
                    want[j].toFloat().toRawBits(),
                    mine[j].toRawBits(),
                    "clip_grad_value_ $key[$j]: torch ${want[j]} vs tlaloc ${mine[j]}",
                )
            }
        }
    }
}
