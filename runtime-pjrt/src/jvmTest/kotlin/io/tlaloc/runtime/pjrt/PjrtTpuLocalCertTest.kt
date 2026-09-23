package io.tlaloc.runtime.pjrt

import io.tlaloc.runtime.pjrt.ffm.PjrtClientOptions
import io.tlaloc.runtime.pjrt.ffm.PjrtFfm
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * §0.4.459 (G2a) — the LOCAL half of TPU bring-up, certified on a machine
 * with no TPU: plugin resolution rules, the platform gate on client
 * create-options, and the hand-encoded CompileOptionsProto's field
 * numbers. Everything here runs everywhere (no GPU, no plugin, no TPU);
 * the on-hardware claims live in [PjrtTpuSmokeTest] and stay skipped
 * until a Cloud TPU VM runs them (G2b — see docs/TPU_BRINGUP.md).
 */
class PjrtTpuLocalCertTest {

    @Test
    fun tpuTargetPlatformStringIsTpu() {
        // libtpu registers platform "tpu" at Client_Create; the enum's
        // platform string is the contract the smoke suite asserts against.
        assertEquals("tpu", PjrtTarget.Tpu.platform)
    }

    @Test
    fun envResolutionRefusesNonTpuShapedPluginName() {
        // TLALOC_PJRT_PLUGIN_PATH is generic: on CUDA hosts it names
        // xla_cuda_plugin.so, which must never leak into the TPU lane.
        val dir = Files.createTempDirectory("tlaloc-tpu-res")
        try {
            val cuda = dir.resolve("xla_cuda_plugin.so")
            Files.createFile(cuda)
            val resolved = PjrtBinaries.resolveTpuPlugin(
                envValue = cuda.toString(), virtualEnv = null, home = null,
            )
            // The env-named file exists but is not tpu-shaped, so whatever
            // (if anything) resolves must come from the default locations,
            // never from that env value.
            assertNotEquals(cuda, resolved, "cuda-shaped env plugin leaked into the TPU lane")
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun envResolutionAcceptsTpuShapedPluginNames() {
        val dir = Files.createTempDirectory("tlaloc-tpu-res")
        try {
            for (name in listOf("libtpu.so", "pjrt_c_api_tpu_plugin.so")) {
                val plugin = dir.resolve(name)
                Files.createFile(plugin)
                assertEquals(
                    plugin,
                    PjrtBinaries.resolveTpuPlugin(
                        envValue = plugin.toString(), virtualEnv = null, home = null,
                    ),
                    "tpu-shaped env plugin '$name' should win resolution",
                )
                Files.delete(plugin)
            }
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun defaultResolutionFindsLibtpuInVenvSitePackages() {
        // The PyPI libtpu wheel layout: <venv>/lib/python3.*/site-packages/
        // libtpu/libtpu.so — the documented default install location.
        val venv = Files.createTempDirectory("tlaloc-tpu-venv")
        try {
            val so = venv.resolve("lib").resolve("python3.12")
                .resolve("site-packages").resolve("libtpu").resolve("libtpu.so")
            Files.createDirectories(so.parent)
            Files.createFile(so)
            assertEquals(
                so,
                PjrtBinaries.resolveTpuPlugin(
                    envValue = null, virtualEnv = venv.toString(), home = null,
                ),
                "libtpu.so in the venv site-packages should resolve without the env var",
            )
        } finally {
            venv.toFile().deleteRecursively()
        }
    }

    @Test
    fun tpuSessionRefusesGpuAllocatorOptionsByName() {
        // The §0.4.333 memory_fraction/preallocate options are the GPU
        // plugin's; handing them to a TPU client must refuse loudly and
        // BEFORE any FFM work (the plugin path here does not exist — if
        // the refusal fired late, we'd see a library-load error instead).
        val ex = assertFailsWith<IllegalArgumentException> {
            PjrtSession(
                plugin = Path.of("/nonexistent/libtpu.so"),
                target = PjrtTarget.Tpu,
                options = PjrtClientOptions(memoryFraction = 0.5f, preallocate = false),
            )
        }
        assertTrue(
            "GPU" in (ex.message ?: ""),
            "refusal must name the GPU-options mismatch; got: ${ex.message}",
        )
    }

    @Test
    fun cudaSessionRefusesNullOptionsByName() {
        // The other direction of the gate: null options on a CUDA target
        // would revive the §0.4.333 preallocation incident.
        val ex = assertFailsWith<IllegalArgumentException> {
            PjrtSession(
                plugin = Path.of("/nonexistent/xla_cuda_plugin.so"),
                target = PjrtTarget.Cuda,
                options = null,
            )
        }
        assertTrue(
            "preallocates 75% of GPU memory" in (ex.message ?: ""),
            "refusal must name the preallocation it prevents; got: ${ex.message}",
        )
    }

    @Test
    fun compileOptionsProtoDecodesToSingleReplicaSinglePartition() {
        // The 6-byte hand-encoded CompileOptionsProto, decoded structurally.
        // Field numbers verified against xla/pjrt/proto/compile_options.proto
        // (openxla/xla main, 2026-09-21): executable_build_options = 3,
        // num_replicas = 4, num_partitions = 5. Protobuf wire format is
        // backend-agnostic — a TPU compile parses these bytes identically.
        val b = PjrtFfm.COMPILE_OPTIONS_PROTO_BYTES
        assertEquals(6, b.size)
        assertEquals(((3 shl 3) or 2).toByte(), b[0], "field 3, wire type LEN (executable_build_options)")
        assertEquals(4.toByte(), b[1], "embedded ExecutableBuildOptionsProto length")
        assertEquals(((4 shl 3) or 0).toByte(), b[2], "field 4, varint (num_replicas)")
        assertEquals(1.toByte(), b[3], "num_replicas = 1")
        assertEquals(((5 shl 3) or 0).toByte(), b[4], "field 5, varint (num_partitions)")
        assertEquals(1.toByte(), b[5], "num_partitions = 1")
    }
}
