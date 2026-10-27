package io.tlaloc.runtime.pjrt.ffm

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class CompileOptionsTest {

    private fun hex(b: ByteArray) = b.joinToString(" ") { "%02X".format(it) }

    @Test
    fun defaultIsTheSixBytesEveryCompileUsed() {
        assertContentEquals(PjrtFfm.COMPILE_OPTIONS_PROTO_BYTES, CompileOptions.DEFAULT.toProtoBytes())
        assertEquals("1A 04 20 01 28 01", hex(CompileOptions.DEFAULT.toProtoBytes()))
    }

    @Test
    fun spmdOverFourDevicesEncodesPartitionsAssignmentSpmdAndShardy() {
        // executable_build_options (3) {
        //   num_replicas 1, num_partitions 4, use_spmd_partitioning true,
        //   device_assignment (9) { replica_count 1, computation_count 4,
        //     computation_devices (3) { replica_device_ids [0] } ... [1], [4], [5] },
        //   use_shardy_partitioner (19) true }
        val bytes = CompileOptions.spmd(listOf(0, 1, 4, 5)).toProtoBytes()
        assertEquals(
            "1A 1F 20 01 28 04 30 01 4A 14 08 01 10 04 1A 02 08 00 1A 02 08 01 1A 02 08 04 1A 02 08 05 98 01 01",
            hex(bytes),
        )
    }

    @Test
    fun overridesAreMapEntriesInNameOrder() {
        val opts = CompileOptions.DEFAULT
            .withOverride("xla_b", CompileOptions.OptionValue.Int64(300))
            .withOverride("xla_a", CompileOptions.OptionValue.Flag(false))
        // field 7 { key "xla_a", value { bool_field false } } then { key "xla_b", value { int_field 300 } }
        assertEquals(
            "1A 04 20 01 28 01 " +
                "3A 0B 0A 05 78 6C 61 5F 61 12 02 10 00 " +
                "3A 0C 0A 05 78 6C 61 5F 62 12 03 18 AC 02",
            hex(opts.toProtoBytes()),
        )
    }

    @Test
    fun textAndRealOverridesAndOptimizationLevel() {
        val opts = CompileOptions(optimizationLevel = CompileOptions.EffortLevel.O1)
            .withOverride("t", CompileOptions.OptionValue.Text("SAFE"))
            .withOverride("r", CompileOptions.OptionValue.Real(0.5))
        val h = hex(opts.toProtoBytes())
        // optimization_level is field 24, varint: tag C0 01, EFFORT_O1 = 19 = 0x13.
        assertTrue(h.startsWith("1A 07 20 01 28 01 C0 01 13"), h)
        // double 0.5 = 0x3FE0000000000000, little endian after tag 0x21.
        assertTrue("21 00 00 00 00 00 00 E0 3F" in h, h)
        assertTrue("0A 04 53 41 46 45" in h, h)
    }

    @Test
    fun xlaGpuFixesTurnOffV2GemmFusionsAndTritonForFp8Dots() {
        val f32 = "%d = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x2xf32>) -> tensor<4x2xf32>"
        val fp8 = "%d = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<4x32xf8E4M3FN>, tensor<32x2xf8E4M3FN>) -> tensor<4x2xf32>"
        val off = CompileOptions.OptionValue.Flag(false)
        assertEquals(mapOf("xla_gpu_experimental_gemm_fusion_v2" to off), CompileOptions.DEFAULT.withXlaGpuFixes(f32).overrides)
        assertEquals(
            mapOf("xla_gpu_experimental_gemm_fusion_v2" to off, "xla_gpu_enable_triton_gemm" to off),
            CompileOptions.DEFAULT.withXlaGpuFixes(fp8).overrides,
        )
        // An override the caller set is kept.
        val on = CompileOptions.OptionValue.Flag(true)
        val chosen = CompileOptions.DEFAULT.withOverride("xla_gpu_experimental_gemm_fusion_v2", on).withOverride("xla_gpu_enable_triton_gemm", on)
        assertEquals(chosen.overrides, chosen.withXlaGpuFixes(fp8).overrides)
    }

    @Test
    fun refusesAssignmentsThatDoNotMatchThePartitions() {
        assertFailsWith<IllegalArgumentException> { CompileOptions(numPartitions = 2, deviceAssignment = listOf(0)) }
        assertFailsWith<IllegalArgumentException> { CompileOptions(numPartitions = 2, deviceAssignment = listOf(3, 3)) }
        assertFailsWith<IllegalArgumentException> {
            CompileOptions(numReplicas = 2, numPartitions = 1, deviceAssignment = listOf(0))
        }
        assertFailsWith<IllegalArgumentException> { CompileOptions(numPartitions = 0) }
    }
}
