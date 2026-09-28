package io.tlaloc.maestro.serving

import io.tlaloc.autograd.BoundedProgram
import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.matmul
import io.tlaloc.autograd.BucketLadders
import io.tlaloc.autograd.boundedProgram
import io.tlaloc.autograd.broadcastAlong
import io.tlaloc.autograd.div
import io.tlaloc.autograd.embedding
import io.tlaloc.autograd.minus
import io.tlaloc.autograd.plus
import io.tlaloc.autograd.softmax
import io.tlaloc.autograd.specOf
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.tanh
import io.tlaloc.autograd.times
import io.tlaloc.core.Bounded
import io.tlaloc.core.DTensor
import io.tlaloc.core.DimBound
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.Rank1
import io.tlaloc.core.Rank2
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.hostF32
import io.tlaloc.core.hostI32
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonBool
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.fail

object MaxSeqRun : DimBound(64)
object MaxRowsRun : DimBound(4)
object MaxColsRun : DimBound(16)

/**
 * A bounded-program artifact served by `harness/python/tlaloc_bounded.py` on a PJRT plugin, at
 * every size up to the bound, against the reference interpreter's exact-size result.
 *
 * Needs a PJRT CUDA plugin; without one the Python side exits 2 and each test prints `[skip]`
 * and returns, as the other serving lanes do.
 */
class BoundedArtifactRunTest {

    private val hidden = 4

    private fun maskedMean() = boundedProgram(
        "masked_mean",
        listOf(specOf<Rank2<Bounded<MaxSeqRun>, Sym>>(F32, hidden)),
        specOf<Rank1<Sym>>(F32, hidden),
    ) { xs, ctx ->
        val x = xs[0]
        (x * x.broadcastAlong<Shape>(ctx.validMask(MaxSeqRun), 0)).sum<Rank1<Sym>>(intArrayOf(0)) /
            ctx.validLength(MaxSeqRun)
    }

    private fun maskedSoftmax() = boundedProgram(
        "masked_softmax",
        listOf(specOf<Rank1<Bounded<MaxSeqRun>>>(F32)),
        specOf<Rank1<Bounded<MaxSeqRun>>>(F32),
    ) { xs, ctx -> (xs[0] + (ctx.validMask(MaxSeqRun) - 1f) * 1e9f).softmax() }

    @Test
    fun `masked mean at every size 1 to 64 on PJRT matches the interpreter, with one compile per bucket`() {
        val program = maskedMean()
        val ladders = BucketLadders.powersOfTwo(program.bounds, minBucket = 8)
        assertEquals(listOf(8, 16, 32, 64), ladders.ladder(MaxSeqRun))
        val inputs = (1..MaxSeqRun.max).map { n -> listOf(f32(intArrayOf(n, hidden), n)) }
        val bucketed = serve(program, ladders, inputs) ?: return
        assertEquals(4, bucketed.compileCount, "one compile per bucket")
        assertMatchesInterpreter(program, inputs, bucketed)

        // The same program with one body per size: the static-shape programs a server would
        // otherwise compile. Their results agree with the bucketed ones.
        val perSize = serve(program, BucketLadders(mapOf(MaxSeqRun to (1..MaxSeqRun.max).toList())), inputs) ?: return
        assertEquals(MaxSeqRun.max, perSize.compileCount)
        for ((i, r) in bucketed.results.withIndex()) assertClose(perSize.results[i].second, r.second, "size ${i + 1} bucketed vs per-size", 1e-5f)
        println("[bounded] masked_mean 1..64 on ${bucketed.platform}: ${bucketed.compileCount} compiles in " +
            "%.2f s bucketed; ${perSize.compileCount} compiles in %.2f s one-per-size".format(bucketed.compileSeconds, perSize.compileSeconds))
    }

    @Test
    fun `masked softmax at every size 1 to 64 on PJRT matches the interpreter`() {
        val program = maskedSoftmax()
        val inputs = (1..MaxSeqRun.max).map { n -> listOf(f32(intArrayOf(n), 3 * n)) }
        val run = serve(program, BucketLadders.powersOfTwo(program.bounds, minBucket = 8), inputs) ?: return
        assertMatchesInterpreter(program, inputs, run)
    }

    @Test
    fun `two bounds and an I32 input on PJRT match the interpreter at every size pair`() {
        val vocab = 7
        val program = boundedProgram(
            "embed_two_bounds",
            listOf(
                specOf<Rank2<Sym, Sym>>(F32, vocab, hidden),
                specOf<Rank1<Bounded<MaxColsRun>>>(I32),
                specOf<Rank2<Bounded<MaxRowsRun>, Bounded<MaxColsRun>>>(F32),
            ),
            specOf<Rank2<Bounded<MaxRowsRun>, Sym>>(F32, hidden),
        ) { xs, ctx ->
            val emb = xs[0].embedding<Shape>(xs[1])                                     // [cols, hidden]
            val masked = emb * emb.broadcastAlong<Shape>(ctx.validMask(MaxColsRun), 0)
            val w = xs[2]                                                             // [rows, cols]
            (w.tanh() matmulShape masked)                                             // [rows, hidden]
        }
        val table = f32(intArrayOf(vocab, hidden), 11)
        val inputs = (1..MaxRowsRun.max).flatMap { r ->
            (1..MaxColsRun.max).map { c ->
                listOf(
                    table,
                    DTensor<Shape, I32>(HostI32Storage(IntArray(c) { (it * 5 + r) % vocab }), intArrayOf(c), I32),
                    f32(intArrayOf(r, c), r * 31 + c),
                )
            }
        }
        val ladders = BucketLadders(mapOf(MaxRowsRun to listOf(2, 4), MaxColsRun to listOf(4, 16)))
        val run = serve(program, ladders, inputs) ?: return
        assertEquals(4, run.compileCount)
        // The dot runs in TF32 on the GPU (XLA's default for an f32 dot_general without a
        // precision attribute), so the band is TF32's, not 1e-4: on the GB10 the bucketed
        // results are within 6.4e-4 of the interpreter, and the one-body-per-size results
        // within 4.5e-4 of the bucketed ones (a different contraction length, a different
        // accumulation). In f32, in the interpreter, the export's padding check found the
        // bucketed and exact results equal.
        val vsInterpreter = assertMatchesInterpreter(program, inputs, run, tol = 2e-3f)
        val perSize = serve(
            program,
            BucketLadders(mapOf(MaxRowsRun to (1..MaxRowsRun.max).toList(), MaxColsRun to (1..MaxColsRun.max).toList())),
            inputs,
        ) ?: return
        assertEquals(MaxRowsRun.max * MaxColsRun.max, perSize.compileCount)
        var vsPerSize = 0f
        for ((i, r) in run.results.withIndex()) {
            vsPerSize = maxOf(vsPerSize, assertClose(perSize.results[i].second, r.second, "case $i bucketed vs per-size", 2e-3f))
        }
        println("[bounded] embed_two_bounds on ${run.platform}: largest difference against the interpreter " +
            "$vsInterpreter, bucketed against one body per size $vsPerSize (relative to max(1, |y|))")
    }

    @Test
    fun `the python runtime's unit lane passes`() {
        val python = resolvePython() ?: run { println("[skip] no python3 for tlaloc_bounded_test"); return }
        val pb = ProcessBuilder(python, "-m", "unittest", "tlaloc_bounded_test", "-v")
        pb.directory(harness().toFile())
        pb.redirectErrorStream(true)
        val proc = pb.start()
        val out = proc.inputStream.bufferedReader().readText()
        if (!proc.waitFor(120, TimeUnit.SECONDS)) { proc.destroyForcibly(); fail("tlaloc_bounded_test timed out") }
        assertEquals(0, proc.exitValue(), out)
        val ran = Regex("Ran (\\d+) tests").find(out)?.groupValues?.get(1)?.toInt() ?: 0
        assertTrue(ran >= 11, "the unit lane ran $ran cases\n$out")
    }

    @Test
    fun `the Triton model writer refuses a bounded artifact by name`() {
        val dir = Files.createTempDirectory("tlaloc-bounded-triton")
        try {
            BoundedProgramExport.export(maskedSoftmax(), dir.resolve("a"), BucketLadders.powersOfTwo(listOf(MaxSeqRun), minBucket = 32))
            val e = kotlin.test.assertFailsWith<IllegalArgumentException> {
                TritonModelRepository.write(dir.resolve("a"), dir.resolve("repo"), "bounded")
            }
            assertTrue("is a bounded-program artifact" in e.message!! && "tlaloc_bounded.py" in e.message!!, e.message)
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    // ------------------------------------------------------------------ harness

    private class Served(
        val platform: String,
        val results: List<Pair<List<Int>, FloatArray>>,
        val compileCount: Int,
        val compileSeconds: Double,
    )

    private fun f32(dims: IntArray, salt: Int): DTensor<Shape, F32> {
        val n = dims.fold(1) { a, b -> a * b }
        return DTensor(HostF32Storage(FloatArray(n) { ((it * 37 + salt * 13) % 23 - 11) / 7f }), dims, F32)
    }

    /** Returns the largest difference seen, relative to `max(1, max |want|)`. */
    private fun assertMatchesInterpreter(
        program: BoundedProgram,
        inputs: List<List<DTensor<*, *>>>,
        served: Served,
        tol: Float = 1e-4f,
    ): Float {
        var worst = 0f
        for ((i, case) in inputs.withIndex()) {
            val want = program.run(case)
            val (dims, got) = served.results[i]
            assertEquals(want.dims.toList(), dims, "case $i dims")
            // GPU against interpreter: f32 reduction order differs; 1e-4 is the house floor
            // for f32 arithmetic without a dot.
            worst = maxOf(worst, assertClose(want.hostF32(), got, "case $i (${case.map { it.dims.toList() }})", tol))
        }
        return worst
    }

    private fun assertClose(want: FloatArray, got: FloatArray, what: String, tol: Float): Float {
        assertEquals(want.size, got.size, "$what: size")
        val scale = maxOf(1f, want.maxOf { abs(it) })
        var worst = 0f
        for (i in want.indices) {
            val d = abs(want[i] - got[i]) / scale
            assertTrue(d <= tol, "$what [$i]: want ${want[i]} got ${got[i]}")
            worst = maxOf(worst, d)
        }
        return worst
    }

    /** Exports [program] with [ladders], serves [inputs] through Python on CUDA; null when skipped. */
    private fun serve(program: BoundedProgram, ladders: BucketLadders, inputs: List<List<DTensor<*, *>>>): Served? {
        val python = resolvePython() ?: run { println("[skip] no python3 for the bounded-artifact lane"); return null }
        val dir = Files.createTempDirectory("tlaloc-bounded-run")
        try {
            val artifact = dir.resolve("artifact")
            BoundedProgramExport.export(program, artifact, ladders)
            val request = dir.resolve("request.json")
            Files.writeString(request, requestJson(inputs))
            val output = dir.resolve("out.json")
            val pb = ProcessBuilder(
                python, harness().resolve("run_bounded_check.py").toString(),
                "--artifact", artifact.toString(), "--request", request.toString(), "--output", output.toString(),
            )
            pb.redirectErrorStream(true)
            val proc = pb.start()
            val log = proc.inputStream.bufferedReader().readText()
            if (!proc.waitFor(600, TimeUnit.SECONDS)) { proc.destroyForcibly(); fail("run_bounded_check timed out\n$log") }
            val out = if (Files.exists(output)) parseJson(Files.readString(output)) as JsonObject else null
            when (proc.exitValue()) {
                0 -> Unit
                2 -> { println("[skip] bounded-artifact lane: ${out?.get("error") ?: log.takeLast(400)}"); return null }
                else -> fail("run_bounded_check failed (${proc.exitValue()}): ${out?.get("error")}\n${out?.get("traceback")}\n$log")
            }
            out!!
            val guard = out.obj("guard")
            assertTrue((guard["loaded_forbidden"] as? JsonArray)?.elements?.isEmpty() ?: true, "a framework was imported: $guard")
            assertEquals(JsonBool(true), out["ok"])
            val results = out.arr("results").elements.map { r ->
                r as JsonObject
                r.arr("dims").asIntList("dims") to
                    r.arr("values").elements.map { (it as JsonNumber).value.toFloat() }.toFloatArray()
            }
            return Served(
                out.str("platform"),
                results,
                (out["compileCount"] as JsonNumber).asInt("compileCount"),
                (out["compileSeconds"] as JsonNumber).value,
            )
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    private fun requestJson(cases: List<List<DTensor<*, *>>>): String = buildString {
        append("{\"cases\":[")
        cases.forEachIndexed { i, case ->
            if (i > 0) append(',')
            append("{\"inputs\":[")
            case.forEachIndexed { k, t ->
                if (k > 0) append(',')
                @Suppress("UNCHECKED_CAST")
                val values = if (t.dtype == I32) (t as DTensor<*, I32>).hostI32().joinToString(",")
                else (t as DTensor<*, F32>).hostF32().joinToString(",") { it.toString() }
                append("{\"values\":[").append(values).append("],\"dims\":").append(t.dims.joinToString(",", "[", "]")).append('}')
            }
            append("]}")
        }
        append("]}")
    }

    private fun harness(): Path = Path.of("..", "harness", "python").toAbsolutePath().normalize()

    private fun resolvePython(): String? {
        System.getenv("TLALOC_TORCH_PYTHON")?.let { if (Files.isExecutable(Path.of(it))) return it }
        val home = System.getProperty("user.home")
        if (home != null) {
            val venv = Path.of(home, ".local", "venvs", "iree", "bin", "python")
            if (Files.isExecutable(venv)) return venv.toString()
        }
        return listOf("/usr/bin/python3", "/usr/local/bin/python3").firstOrNull { Files.isExecutable(Path.of(it)) }
    }
}

/** `[r, c] x [c, h] -> [r, h]` on erased tracers. */
@Suppress("UNCHECKED_CAST")
private infix fun Tracer<Shape>.matmulShape(other: Tracer<Shape>): Tracer<Shape> =
    ((this as Tracer<Rank2<Sym, Sym>>) matmul (other as Tracer<Rank2<Sym, Sym>>)) as Tracer<Shape>
