package io.tlaloc.benchmarks

import io.tlaloc.core.HostBf16Storage
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.SafetensorsFile
import io.tlaloc.core.io.parseJson
import io.tlaloc.runtime.iree.IreeBinaries
import io.tlaloc.runtime.iree.runOnIree
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * §0.4.468 (Phase H2) — weight ingestion certified end to end: a real
 * safetensors checkpoint, written by the REFERENCE implementation (Python
 * `safetensors`), read by Tlaloc's Kotlin reader, run through the decode
 * graph, and compared against PyTorch on the same numbers.
 *
 * # Why this is the oracle the reader needs
 *
 * `core/.../SafetensorsTest` builds files byte by byte and owns the FORMAT
 * claim. It cannot, by construction, catch a case where Tlaloc's reading of
 * the format and the producer's writing of it are both self-consistent and
 * different — an endianness assumption, a data_offsets base, a row-major
 * convention. Only bytes produced by the reference implementation can. So
 * this test never writes a safetensors file: `harness/python/
 * write_llama_safetensors.py` does, through `safetensors.numpy.save_file`
 * and `safetensors.torch.save_file`.
 *
 * # The two claims, and the two files
 *
 * 1. **Bit-for-bit read**: the script records the raw element bit patterns it
 *    wrote at chosen flat indices (`probes.json`); the reader must reproduce
 *    them exactly — `Float.toRawBits()` for F32, the raw 16-bit pattern out
 *    of [HostBf16Storage] for BF16. No tolerance: reading bytes is not
 *    arithmetic.
 * 2. **Decode parity**: the weights the reader returns drive
 *    [LlamaDecoderPrimal]'s decode step, and the loss must agree
 *    with `harness/python/run_pytorch_llama.py` — the existing op-for-op
 *    PyTorch mirror (§0.4.289) — on the same weights, at 1e-3 relative
 *    (FP32 reduction order differs between two backends).
 *
 * Execution runs on the SAME lane §0.4.289 certified: the CPU baseline
 * pipeline (recognize -> coarsen -> decomposeCoarsened) through Tlaloc-IREE.
 * That is deliberate experimental design — the lane and the oracle are
 * already pinned against each other, so the only new variable in this test is
 * WHERE THE WEIGHTS CAME FROM, and any disagreement is the reader's.
 * REJECTED: the reference interpreter (its op coverage is deliberately
 * narrow — it does not carry RSQRT, and widening it is AD-engine work, not
 * ingestion work). REJECTED for this slice: running the step on the GPU
 * through PJRT — worth doing, but device-vs-host float agreement is a
 * SECOND claim with its own floor (~4e-5) and it belongs with H3's real
 * decode step, where a logits difference has a top-1 consequence.
 *
 * The BF16 file is the load-bearing half of both. Its numbers are the F32
 * ones put through `torch.bfloat16`, so: the 16-bit patterns Tlaloc reads
 * must equal the ones torch wrote (which incidentally pins §0.4.455's
 * round-to-nearest-even against torch's), the widening back to f32 is exact,
 * and PyTorch runs on the widened values — so a bf16 checkpoint feeding an
 * f32 graph is certified to be reading the checkpoint's real numbers.
 *
 * # What is NOT certified here (named, not hidden)
 *
 * - A REAL trained checkpoint. Weights are synthesized: the audit's §5 H2
 *   entry states why (the bytes of a random tensor and a trained one are the
 *   same bytes, and `LlamaDecoderPrimal` is not TinyLlama's graph anyway —
 *   single layer, single head, no GQA). HF's tensor NAMING and the sharded
 *   index are pinned in `core/.../SafetensorsFileTest`; mapping HF names onto
 *   a model-layer graph is H3's business.
 * - Device execution. This runs Tlaloc-IREE on the CPU. The GPU decode step
 *   is H3, where the manifest carries the weights.
 *
 * Self-skips when IREE, the venv, torch, safetensors or numpy are missing.
 */
class LlamaSafetensorsParityTest {

    private val config = LlamaDecoderConfig.tiny

    private fun resolvePython(): String? {
        System.getenv("TLALOC_TORCH_PYTHON")?.let { if (Files.isExecutable(Path.of(it))) return it }
        val home = System.getProperty("user.home") ?: return null
        val venv = Path.of(home, ".local", "venvs", "iree", "bin", "python")
        return if (Files.isExecutable(venv)) venv.toString() else null
    }

    private fun canImport(python: String, module: String): Boolean {
        val pb = ProcessBuilder(python, "-c", "import $module")
        pb.redirectErrorStream(true)
        return runCatching {
            val p = pb.start()
            p.waitFor(30, TimeUnit.SECONDS) && p.exitValue() == 0
        }.getOrElse { false }
    }

    private fun script(name: String): Path =
        Path.of("..", "harness", "python", name).toAbsolutePath().normalize()

    private fun run(vararg cmd: String) {
        val pb = ProcessBuilder(*cmd)
        pb.redirectErrorStream(true)
        val p = pb.start()
        val out = p.inputStream.readBytes().toString(StandardCharsets.UTF_8)
        assertTrue(p.waitFor(300, TimeUnit.SECONDS), "subprocess timed out: ${cmd.toList()}")
        assertEquals(0, p.exitValue(), "subprocess failed: ${cmd.toList()}\n$out")
    }

    /** The parameter order [LlamaDecoderPrimal.build] declares. */
    private val paramOrder = listOf(
        "x_in", "labels", "theta",
        "q_w", "k_w", "v_w", "out_w",
        "gate_w", "up_w", "down_w", "lm_head_w",
        "eps_attn", "eps_mlp",
    )

    @Test
    fun aRealSafetensorsCheckpointReadsBitExactlyAndDecodesLikePyTorch() {
        assumeTrue(IreeBinaries.available, "iree-compile / iree-run-module not resolved; skipping")
        val python = resolvePython()
        assumeTrue(python != null, "no venv python; skipping")
        assumeTrue(canImport(python!!, "torch"), "torch missing; skipping")
        assumeTrue(canImport(python, "safetensors"), "safetensors missing; skipping")
        assumeTrue(canImport(python, "numpy"), "numpy missing; skipping")

        val dir = Files.createTempDirectory("tlaloc-h2-parity")
        run(
            python, script("write_llama_safetensors.py").toString(),
            "--out-dir", dir.toString(),
            "--tokens", config.tokens.toString(),
            "--dmodel", config.dModel.toString(),
            "--dff", config.dFf.toString(),
            "--vocab", config.vocab.toString(),
        )

        val probes = parseJson(
            String(Files.readAllBytes(dir.resolve("probes.json")), StandardCharsets.UTF_8),
        ) as JsonObject

        val fn = llamaCpuBaselinePipeline()
        assertEquals(paramOrder, fn.params.map { it.name }, "param order drifted from the primal")

        // ---- claim 1+2 on the F32 checkpoint ---------------------------
        val f32Loss = SafetensorsFile.open(dir.resolve("llama_tiny.safetensors")).use { f ->
            assertEquals(paramOrder.toSet(), f.names, "checkpoint holds the wrong tensor set")
            assertEquals("pt", f.metadata["format"], "metadata did not survive the header")
            checkF32Probes(f, probes.obj("f32"))
            val inputs = paramOrder.map { name ->
                val t = f.load(name)
                assertTrue(t.storage is HostF32Storage, "$name is not f32 storage")
                assertEquals(expectedDims(name), t.dims.toList(), "$name has the wrong shape")
                t.toF32Array()
            }
            runOnIree(fn, inputs).single().single()
        }

        // ---- claim 1+2 on the BF16 checkpoint --------------------------
        val bf16Loss = SafetensorsFile.open(dir.resolve("llama_bf16.safetensors")).use { f ->
            checkBf16Probes(f, probes.obj("bf16"))
            val inputs = paramOrder.map { name ->
                val t = f.load(name)
                assertTrue(t.storage is HostBf16Storage, "$name is not bf16 storage")
                assertEquals(expectedDims(name), t.dims.toList(), "$name has the wrong shape")
                // Widening bf16 -> f32 is EXACT, so an f32 graph on a bf16
                // checkpoint computes on the checkpoint's real numbers.
                t.toF32Array()
            }
            runOnIree(fn, inputs).single().single()
        }

        // The two checkpoints are genuinely different numbers — if bf16
        // rounding were a no-op (or if the bf16 read secretly fell back to
        // the f32 file) this assertion is the one that notices.
        assertTrue(
            f32Loss != bf16Loss,
            "the f32 and bf16 checkpoints produced the identical loss $f32Loss — " +
                "bf16 rounding did nothing, which means the bf16 path is not being exercised",
        )

        // ---- the PyTorch oracle, once per checkpoint -------------------
        val f32Ref = torchLoss(python, dir.resolve("f32"), dir.resolve("loss_f32.json"))
        val bf16Ref = torchLoss(python, dir.resolve("bf16"), dir.resolve("loss_bf16.json"))

        assertRel(f32Ref, f32Loss, "f32 checkpoint decode step")
        assertRel(bf16Ref, bf16Loss, "bf16 checkpoint decode step")
    }

    private fun torchLoss(python: String, inputsDir: Path, out: Path): Float {
        run(
            python, script("run_pytorch_llama.py").toString(),
            "--inputs-dir", inputsDir.toString(), "--output", out.toString(),
        )
        val o = parseJson(String(Files.readAllBytes(out), StandardCharsets.UTF_8)) as JsonObject
        return (o["loss"] as JsonNumber).value.toFloat()
    }

    private fun assertRel(reference: Float, got: Float, what: String) {
        val denom = max(1e-6f, abs(reference))
        val rel = abs(reference - got) / denom
        assertTrue(
            rel < 1e-3f,
            "$what: Tlaloc loss $got vs PyTorch $reference (relative ${rel}) exceeds 1e-3",
        )
    }

    private fun expectedDims(name: String): List<Int> = when (name) {
        "x_in", "theta" -> listOf(config.tokens, config.dModel)
        "labels" -> listOf(config.tokens, config.vocab)
        "q_w", "k_w", "v_w", "out_w" -> listOf(config.dModel, config.dModel)
        "gate_w", "up_w" -> listOf(config.dModel, config.dFf)
        "down_w" -> listOf(config.dFf, config.dModel)
        "lm_head_w" -> listOf(config.dModel, config.vocab)
        "eps_attn", "eps_mlp" -> listOf(config.tokens, 1)
        else -> error("unknown param $name")
    }

    private fun checkF32Probes(f: SafetensorsFile, probes: JsonObject) {
        for (name in paramOrder) {
            val p = probes.obj(name)
            val data = (f.load(name).storage as HostF32Storage).data
            assertEquals(p.arr("shape").asIntList("shape"), f.load(name).dims.toList(), "$name shape")
            assertEquals((p["count"] as JsonNumber).asInt("count"), data.size, "$name element count")
            val idx = p.arr("indices").asIntList("indices")
            val bits = p.arr("bits").asIntList("bits")
            for (i in idx.indices) {
                assertEquals(
                    bits[i], data[idx[i]].toRawBits(),
                    "$name[${idx[i]}]: Tlaloc read a different f32 bit pattern than Python wrote",
                )
            }
        }
    }

    private fun checkBf16Probes(f: SafetensorsFile, probes: JsonObject) {
        for (name in paramOrder) {
            val p = probes.obj(name)
            val data = (f.load(name).storage as HostBf16Storage).data
            assertEquals((p["count"] as JsonNumber).asInt("count"), data.size, "$name element count")
            val idx = p.arr("indices").asIntList("indices")
            val bits = p.arr("bits").asIntList("bits")
            for (i in idx.indices) {
                assertEquals(
                    bits[i].toShort(), data[idx[i]],
                    "$name[${idx[i]}]: Tlaloc read a different bf16 pattern than torch wrote",
                )
            }
        }
    }
}
