package io.tlaloc.runtime.pjrt.ffm

import java.io.ByteArrayOutputStream

/**
 * The `xla.CompileOptionsProto` handed to `PJRT_Client_Compile`, encoded by hand.
 *
 * Field numbers are from `xla/pjrt/proto/compile_options.proto` and
 * `xla/xla_data.proto` (openxla/xla):
 *
 *   CompileOptionsProto.executable_build_options = 3
 *   CompileOptionsProto.env_option_overrides     = 7 (map<string, OptionOverrideProto>)
 *   ExecutableBuildOptionsProto.num_replicas           = 4
 *   ExecutableBuildOptionsProto.num_partitions         = 5
 *   ExecutableBuildOptionsProto.use_spmd_partitioning  = 6
 *   ExecutableBuildOptionsProto.device_assignment      = 9
 *   ExecutableBuildOptionsProto.use_shardy_partitioner = 19
 *   ExecutableBuildOptionsProto.optimization_level     = 24 (ExecutionOptions.EffortLevel)
 *   DeviceAssignmentProto.replica_count = 1, computation_count = 2,
 *     computation_devices = 3 { replica_device_ids = 1 }
 *   OptionOverrideProto string_field = 1, bool_field = 2, int_field = 3, double_field = 4
 *
 * [DEFAULT] encodes to the six bytes every Tlaloc compile has used
 * (`1A 04 20 01 28 01`: one replica, one partition).
 *
 * A program over several devices (TorchTPU's recipe for collectives) sets
 * [numPartitions] to the device count, lists the global device ids in
 * [deviceAssignment] (one per partition, in partition order) and turns on
 * [useSpmdPartitioning] and [useShardyPartitioner]. [overrides] reach XLA's
 * DebugOptions by name (`xla_allow_excess_precision`, `xla_gpu_*`) and libtpu's
 * own options (`xla_tpu_*`); PJRT applies them when it reads the proto.
 */
data class CompileOptions(
    val numReplicas: Int = 1,
    val numPartitions: Int = 1,
    /** Global device ids, one per partition; null leaves the assignment to the plugin. */
    val deviceAssignment: List<Long>? = null,
    val useSpmdPartitioning: Boolean = false,
    val useShardyPartitioner: Boolean = false,
    val optimizationLevel: EffortLevel? = null,
    val overrides: Map<String, OptionValue> = emptyMap(),
) {
    init {
        require(numReplicas >= 1) { "CompileOptions: numReplicas must be at least 1, got $numReplicas" }
        require(numPartitions >= 1) { "CompileOptions: numPartitions must be at least 1, got $numPartitions" }
        if (deviceAssignment != null) {
            require(numReplicas == 1) {
                "CompileOptions: a device assignment is written for one replica and numPartitions " +
                    "computations; numReplicas is $numReplicas"
            }
            require(deviceAssignment.size == numPartitions) {
                "CompileOptions: deviceAssignment lists ${deviceAssignment.size} devices for " +
                    "$numPartitions partitions"
            }
            require(deviceAssignment.toSet().size == deviceAssignment.size) {
                "CompileOptions: deviceAssignment repeats a device: $deviceAssignment"
            }
        }
        require(overrides.keys.all { it.isNotEmpty() }) { "CompileOptions: an override has an empty name" }
    }

    /** `ExecutionOptions.EffortLevel`, by proto value. */
    enum class EffortLevel(val protoValue: Int) { O0(9), O1(19), O2(29), O3(39) }

    /** One `OptionOverrideProto` value. */
    sealed interface OptionValue {
        data class Text(val value: String) : OptionValue
        data class Flag(val value: Boolean) : OptionValue
        data class Int64(val value: Long) : OptionValue
        data class Real(val value: Double) : OptionValue
    }

    /** The serialized proto. Overrides are written in name order, so equal options give equal bytes. */
    fun toProtoBytes(): ByteArray {
        val build = ProtoWriter().apply {
            varintField(4, numReplicas.toLong())
            varintField(5, numPartitions.toLong())
            if (useSpmdPartitioning) varintField(6, 1)
            if (deviceAssignment != null) {
                messageField(9, ProtoWriter().apply {
                    varintField(1, 1)
                    varintField(2, numPartitions.toLong())
                    for (id in deviceAssignment) messageField(3, ProtoWriter().apply { varintField(1, id) })
                })
            }
            if (useShardyPartitioner) varintField(19, 1)
            if (optimizationLevel != null) varintField(24, optimizationLevel.protoValue.toLong())
        }
        return ProtoWriter().apply {
            messageField(3, build)
            for ((name, value) in overrides.toSortedMap()) {
                messageField(7, ProtoWriter().apply {
                    bytesField(1, name.toByteArray(Charsets.UTF_8))
                    messageField(2, ProtoWriter().apply {
                        when (value) {
                            is OptionValue.Text -> bytesField(1, value.value.toByteArray(Charsets.UTF_8))
                            is OptionValue.Flag -> varintField(2, if (value.value) 1 else 0)
                            is OptionValue.Int64 -> varintField(3, value.value)
                            is OptionValue.Real -> fixed64Field(4, java.lang.Double.doubleToRawLongBits(value.value))
                        }
                    })
                })
            }
        }.bytes()
    }

    /** These options with [name] overridden to [value]. */
    fun withOverride(name: String, value: OptionValue): CompileOptions = copy(overrides = overrides + (name to value))

    /**
     * These options for compiling [stablehlo] with XLA:GPU at the vendored commit (jax 0.12.0.1),
     * with two overrides that keep it from computing wrong values, each added unless already set:
     *
     * - `xla_gpu_experimental_gemm_fusion_v2 = false`. A v2 (nested) GEMM fusion that absorbs a
     *   layout-changing `copy` followed by a `bitcast` moves the bitcast above the copy, so the
     *   copy's transpose is lost and the dot reads its operand in the wrong order. Gradients through
     *   grouped-query attention hit it (the transpose after the sum over a key-value head's query
     *   heads). The v1 fusions still use Triton.
     * - `xla_gpu_enable_triton_gemm = false` when a `dot_general` has an FP8 operand. A Triton FP8
     *   GEMM tiled 16 deep along K returns twice the product on the GB10, and the autotuner picks
     *   that tile for some shapes; cuBLAS computes the product instead.
     */
    fun withXlaGpuFixes(stablehlo: String): CompileOptions {
        var fixed = this
        if (GEMM_FUSION_V2 !in overrides) fixed = fixed.withOverride(GEMM_FUSION_V2, OptionValue.Flag(false))
        if (TRITON_GEMM !in overrides && FP8_DOT.containsMatchIn(stablehlo)) fixed = fixed.withOverride(TRITON_GEMM, OptionValue.Flag(false))
        return fixed
    }

    companion object {
        /** One replica, one partition, nothing else: the bytes every compile used before this class. */
        val DEFAULT = CompileOptions()

        /**
         * [DEFAULT] with `xla_allow_excess_precision = false`: XLA keeps every narrowing the program
         * writes (no folded f32→bf16→f32 pairs, bf16 intermediates rounded where the program rounds
         * them), the interpreter's convention. TorchTPU compiles this way by default. For parity lanes.
         */
        val STRICT_PRECISION = DEFAULT.withOverride("xla_allow_excess_precision", OptionValue.Flag(false))

        /**
         * One program over [deviceIds] (global ids, in partition order), as TorchTPU compiles
         * collectives: `num_replicas = 1`, `num_partitions = N`, an explicit assignment, SPMD and
         * Shardy on even when the module carries no sharding.
         */
        fun spmd(deviceIds: List<Long>): CompileOptions = CompileOptions(
            numPartitions = deviceIds.size,
            deviceAssignment = deviceIds,
            useSpmdPartitioning = true,
            useShardyPartitioner = true,
        )

        private const val GEMM_FUSION_V2 = "xla_gpu_experimental_gemm_fusion_v2"
        private const val TRITON_GEMM = "xla_gpu_enable_triton_gemm"
        private val FP8_DOT = Regex("""dot_general[^\n]*xf8E""")
    }
}

/** The protobuf wire format, only as far as [CompileOptions] needs it. */
internal class ProtoWriter {
    private val out = ByteArrayOutputStream()

    fun bytes(): ByteArray = out.toByteArray()

    private fun varint(v: Long) {
        var x = v
        while (x and 0x7FL.inv() != 0L) {
            out.write(((x and 0x7F) or 0x80).toInt())
            x = x ushr 7
        }
        out.write(x.toInt())
    }

    private fun tag(field: Int, wireType: Int) = varint(((field shl 3) or wireType).toLong())

    fun varintField(field: Int, v: Long) { tag(field, 0); varint(v) }

    fun fixed64Field(field: Int, bits: Long) {
        tag(field, 1)
        for (i in 0 until 8) out.write(((bits ushr (8 * i)) and 0xFF).toInt())
    }

    fun bytesField(field: Int, b: ByteArray) { tag(field, 2); varint(b.size.toLong()); out.write(b) }

    fun messageField(field: Int, m: ProtoWriter) = bytesField(field, m.bytes())
}
