package io.tlaloc.runtime.pjrt

import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.kernel.lowerMosaicKernels
import io.tlaloc.stablehlo.mosaic.MosaicRmsNorm
import io.tlaloc.stablehlo.toStablehlo
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * What the TPU kernel payloads promise that a CPU host can check: the
 * payloads load, re-emit JAX's backend_config byte for byte, their DXIR
 * references agree with the numpy references, the Kotlin Mosaic emitter
 * still writes the checked-in text, and lowering picks the custom call on a
 * TPU and the reference (or a refusal) elsewhere.
 */
class TpuKernelFixtureCpuTest {

    private val fixtures by lazy { TpuKernelFixtures.NAMES.map { TpuKernelFixtures.load(it) } }

    /** Checked-in payloads must not name the machine they were exported on:
     * not in any file, and not inside a Mosaic body's debug locations. */
    @Test
    fun payloadsNameNoLocalPath() {
        val markers = listOf("/home/", "worktrees", "/Users/", ".py")
        val dir = Path.of(TpuKernelFixtures::class.java.getResource("/tpu-kernels")!!.toURI())
        val files = Files.walk(dir).use { s -> s.filter { Files.isRegularFile(it) }.toList() }
        assertTrue(files.size >= TpuKernelFixtures.NAMES.size * 3, "found only ${files.size} fixture files")
        for (f in files) {
            val texts = mutableListOf(String(Files.readAllBytes(f), Charsets.ISO_8859_1))
            if (f.fileName.toString() == "manifest.json") {
                val body = Regex("\"body_base64\": \"([^\"]+)\"").find(texts[0])!!.groupValues[1]
                texts += String(java.util.Base64.getDecoder().decode(body), Charsets.ISO_8859_1)
            }
            for (t in texts) for (m in markers) {
                assertFalse(m in t, "${dir.relativize(f)} contains '$m' (a local path)")
            }
        }
    }

    @Test
    fun payloadsRecordTheirToolchain() {
        for (f in fixtures) {
            assertEquals(9L, f.manifest["stable_mosaic_version"], "${f.name}: Mosaic serialization version")
            assertEquals("0.10.0", f.versions["jaxlib"], "${f.name}: jaxlib that wrote the payload")
            assertTrue((f.manifest["body_base64"] as String).startsWith("TUzvUg"), "${f.name}: body is MLIR bytecode (ML\\xefR)")
        }
    }

    @Test
    fun pallasBackendConfigReEmitsByteForByte() {
        for (f in fixtures.filter { it.source == "pallas" }) {
            assertEquals(
                f.manifest["exported_backend_config"],
                f.kernel().backendConfigJson(),
                "${f.name}: MosaicKernel.backendConfigJson differs from what jax.export wrote",
            )
        }
    }

    @Test
    fun referencesMatchNumpy() {
        for (f in fixtures) {
            val got = DxirInterpreter.evalFunction(f.program(referenceFallback = false), f.inputs.map { it.values() })
            assertEquals(f.expected.size, got.size, "${f.name}: result count")
            for (i in got.indices) {
                val d = TpuKernelFixtures.maxAbsDiff(got[i], f.expected[i])
                println("[tpu-kernels] ${f.name} result $i: interpreter vs numpy max|diff| = $d")
                assertTrue(d <= f.tolerance, "${f.name} result $i: interpreter differs from numpy by $d > ${f.tolerance}")
            }
        }
    }

    @Test
    fun kotlinMosaicEmitterWritesTheCheckedInText() {
        val outDir = Path.of("build", "kmosaic")
        Files.createDirectories(outDir)
        val texts = listOf(
            "rmsnorm_f32" to MosaicRmsNorm.Dtype.F32,
            "rmsnorm_bf16" to MosaicRmsNorm.Dtype.BF16,
        ).map { (fixture, dtype) ->
            // Shapes and eps come from the Pallas payload the Kotlin one mirrors.
            val base = TpuKernelFixtures.load(fixture)
            val dims = base.inputs[0].shape
            val text = MosaicRmsNorm.emit(rows = dims[0], hidden = dims[1], dtype = dtype, eps = base.eps)
            // The exporter reads these to (re)generate the _kmosaic payloads.
            Files.writeString(outDir.resolve("$fixture.mlir"), text)
            fixture to text
        }
        for ((fixture, text) in texts) {
            val f = TpuKernelFixtures.load("${fixture}_kmosaic")
            assertEquals(
                f.mosaicText, text,
                "${f.name}: MosaicRmsNorm no longer writes the text the payload was serialized from; " +
                    "rerun harness/python/export_tpu_kernels.py --kmosaic-dir runtime-pjrt/build/kmosaic",
            )
        }
    }

    @Test
    fun tpuTargetEmitsTheCustomCallWithTheBody() {
        for (f in fixtures) {
            val mlir = lowerMosaicKernels(f.program(referenceFallback = false), KernelTarget.GOOGLE_TPU_V5E).toStablehlo("")
            assertTrue("stablehlo.custom_call @tpu_custom_call" in mlir, "${f.name}: $mlir")
            assertTrue((f.manifest["body_base64"] as String) in mlir, "${f.name}: body missing")
            assertTrue("kernel_name = \"${f.manifest["kernel_name"]}\"" in mlir, "${f.name}: kernel_name")
            val aliased = (f.manifest["input_output_aliases"] as Map<*, *>).isNotEmpty()
            assertEquals(aliased, "output_operand_aliases" in mlir, "${f.name}: output_operand_aliases")
            if (f.outputTypes.size > 1) {
                assertTrue(":${f.outputTypes.size} = stablehlo.custom_call" in mlir, "${f.name}: multi-result")
            }
        }
    }

    @Test
    fun otherTargetsRunTheReferenceOrRefuse() {
        for (f in fixtures) {
            val cpu = lowerMosaicKernels(f.program(referenceFallback = true), PjrtTarget.LlvmCpu.kernelTarget).toStablehlo("")
            assertFalse("custom_call" in cpu, "${f.name}: a CPU program must not name tpu_custom_call")
            val e = assertFailsWith<IllegalStateException> {
                lowerMosaicKernels(f.program(referenceFallback = false), PjrtTarget.Cuda.kernelTarget)
            }
            assertTrue("'${f.manifest["kernel_name"]}'" in e.message!!, e.message)
        }
    }
}
