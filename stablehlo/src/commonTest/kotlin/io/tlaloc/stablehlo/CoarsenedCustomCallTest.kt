package io.tlaloc.stablehlo

import io.tlaloc.core.F32
import io.tlaloc.ir.DxirAxisRef
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirDimSharding
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirMesh
import io.tlaloc.ir.DxirMeshAxis
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirSharding
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.kernel.KernelDescriptor
import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.kernel.KernelTemplate
import io.tlaloc.ir.recognizer.kernel.lowerKernelChoice
import io.tlaloc.ir.recognizer.recognizeFlashAttention
import kotlin.test.Test
import kotlin.test.assertFails
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Layer 4.1 §0.4.261 — pin the StableHLO `custom_call` shape emitted for
 * a COARSENED op carrying a [KernelDescriptor]. Round-trip through
 * `stablehlo-translate` is deferred to L4 follow-up (see
 * `setup-dgx-spark-userspace.sh`); these tests pin emitted text only.
 */
class CoarsenedCustomCallTest {

    private val mDim = 8
    private val kDim = 4
    private val nDim = 8
    private val dvDim = 4

    private val qType = DxirType(F32, listOf(mDim, kDim))
    private val kType = DxirType(F32, listOf(kDim, nDim))
    private val vType = DxirType(F32, listOf(nDim, dvDim))
    private val sType = DxirType(F32, listOf(mDim, nDim))
    private val oType = DxirType(F32, listOf(mDim, dvDim))

    private fun buildCoarsenedFn(): DxirFunction {
        val raw = DxirBuilder.function("attn") {
            val q = param("Q", qType)
            val k = param("K", kType)
            val v = param("V", vType)
            val qk = op(OpKind.MATMUL, listOf(q, k), sType)
            val sm = op(OpKind.SOFTMAX, listOf(qk), sType)
            val out = op(OpKind.MATMUL, listOf(sm, v), oType)
            listOf(out)
        }
        return coarsenRecognizedPatterns(raw, recognizeFlashAttention(raw))
    }

    private fun emitFor(target: KernelTarget): String =
        lowerKernelChoice(buildCoarsenedFn(), target).toStablehlo()

    @Test
    fun h100EmitsFlashAttnV3CustomCall() {
        val mlir = emitFor(KernelTarget.NVIDIA_H100)
        assertTrue(mlir.contains("stablehlo.custom_call @flash_attn_v3"), mlir)
        assertTrue(
            mlir.contains("supported_kv_dtypes = [f32, bf16, fp8_e4m3, fp8_e5m2, int8]"),
            mlir,
        )
        assertTrue(mlir.contains("has_side_effect = false"), mlir)
    }

    @Test
    fun a100EmitsFlashAttnV2CustomCall() {
        val mlir = emitFor(KernelTarget.NVIDIA_A100)
        assertTrue(mlir.contains("stablehlo.custom_call @flash_attn_v2"), mlir)
        assertTrue(mlir.contains("supported_kv_dtypes = [f32, bf16, int8]"), mlir)
    }

    @Test
    fun mi300xEmitsFlashAttnAmdCustomCall() {
        val mlir = emitFor(KernelTarget.AMD_MI300X)
        assertTrue(mlir.contains("stablehlo.custom_call @flash_attn_amd"), mlir)
    }

    @Test
    fun tpuV5eEmitsTpuPallasFlashAttention() {
        val mlir = emitFor(KernelTarget.GOOGLE_TPU_V5E)
        assertTrue(mlir.contains("stablehlo.custom_call @tpu_pallas_flash_attention"), mlir)
        assertTrue(mlir.contains("supported_kv_dtypes = [f32, bf16, int8]"), mlir)
    }

    @Test
    fun tpuV6eIncludesFp8InSupportedKvDtypes() {
        val mlir = emitFor(KernelTarget.GOOGLE_TPU_V6E)
        assertTrue(mlir.contains("stablehlo.custom_call @tpu_pallas_flash_attention"), mlir)
        // v6e gets the FP8 kernel variant; older v4/v5 stay BF16/INT8.
        assertTrue(mlir.contains("supported_kv_dtypes = [f32, bf16, fp8_e4m3, int8]"), mlir)
    }

    @Test
    fun trainium2EmitsNkiFlashAttention() {
        val mlir = emitFor(KernelTarget.AWS_TRAINIUM2)
        assertTrue(mlir.contains("stablehlo.custom_call @nki_flash_attention"), mlir)
    }

    @Test
    fun gb10EmitsFlashAttnV3CustomCall() {
        // GB10 is the dev-host (DGX Spark, Grace-Blackwell sm_100) target —
        // it reuses the FA3 kernel name and the full fp8_e4m3 + fp8_e5m2
        // KV-quant matrix. Pinned here so a registry edit that accidentally
        // swaps it to cuDNN (or strips fp8_e5m2) is caught by the emitter
        // tests, not just the recognizer-side KernelLoweringTest.
        val mlir = emitFor(KernelTarget.NVIDIA_GB10)
        assertTrue(mlir.contains("stablehlo.custom_call @flash_attn_v3"), mlir)
        assertTrue(
            mlir.contains("supported_kv_dtypes = [f32, bf16, fp8_e4m3, fp8_e5m2, int8]"),
            mlir,
        )
    }

    @Test
    fun b100EmitsCudnnMultiHeadAttentionCustomCall() {
        val mlir = emitFor(KernelTarget.NVIDIA_B100)
        assertTrue(mlir.contains("stablehlo.custom_call @cudnn_multi_head_attention"), mlir)
        // Conservative KV-quant set on cuDNN MHA — no fp8_e5m2.
        assertTrue(mlir.contains("supported_kv_dtypes = [f32, bf16, fp8_e4m3, int8]"), mlir)
    }

    @Test
    fun customCallSyntaxIncludesOperandsAndTypeSignature() {
        val mlir = emitFor(KernelTarget.NVIDIA_H100)
        // Three operands (Q, K, V), single tensor result, type signature
        // with parens around the input types and bare result type.
        assertTrue(
            mlir.contains("(%0, %1, %2)"),
            "expected (Q, K, V) operand list: $mlir",
        )
        assertTrue(
            mlir.contains(
                "(tensor<8x4xf32>, tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<8x4xf32>",
            ),
            "expected (qType, kType, vType) -> oType signature: $mlir",
        )
    }

    @Test
    fun decomposeFallbackProducesPlainPrimitivesNoCustomCall() {
        // CPU_GENERIC forces decompose: the COARSENED is replaced by its
        // primal_body (MATMUL → SOFTMAX → MATMUL), so the emitter sees
        // no COARSENED at all and produces no custom_call.
        val mlir = emitFor(KernelTarget.CPU_GENERIC)
        assertFalse(mlir.contains("custom_call"), "decompose path emits no custom_call: $mlir")
        assertTrue(mlir.contains("stablehlo.dot_general"), mlir)
    }

    @Test
    fun coarsenedWithoutDescriptorRaisesEmitterError() {
        // The COARSENED produced by coarsenRecognizedPatterns has the
        // canonical primal_body / gradient_body / reads_primal_indices
        // attrs but no kernel_descriptor — that's the "compiler bug"
        // shape: the lowering chain neither annotated nor decomposed it.
        val fn = buildCoarsenedFn()
        val coarsened = fn.body.filterIsInstance<DxirOp>().single()
        // Sanity: confirm the test's premise — no descriptor on the bare
        // post-coarsen op.
        check(coarsened.attrs[KernelDescriptor.ATTR_KEY] == null) {
            "test invariant: bare coarsened op should not yet carry a descriptor"
        }
        assertFails { fn.toStablehlo() }
    }

    @Test
    fun emptyCustomCallAttrsProducesEmptyBackendConfig() {
        // Synthetic descriptor with no attrs: backend_config = "".
        val registry = mapOf<String, KernelTemplate>(
            "FlashAttention" to KernelTemplate { _, _ ->
                KernelDescriptor("synth_kernel", "tlaloc", "test", customCallAttrs = emptyMap())
            },
        )
        val mlir = lowerKernelChoice(buildCoarsenedFn(), KernelTarget.CPU_GENERIC, registry)
            .toStablehlo()
        assertTrue(mlir.contains("stablehlo.custom_call @synth_kernel"), mlir)
        assertTrue(mlir.contains("backend_config = \"\""), mlir)
    }

    @Test
    fun backendConfigSerializesAttrsInAlphabeticKeyOrder() {
        // Synthetic descriptor with multiple attr types in deliberately
        // non-alphabetic insertion order. The encoder must emit them
        // sorted alphabetically so the on-the-wire shape is deterministic.
        val registry = mapOf<String, KernelTemplate>(
            "FlashAttention" to KernelTemplate { _, _ ->
                KernelDescriptor(
                    "synth_kernel", "tlaloc", "test",
                    customCallAttrs = linkedMapOf(
                        "softmax_scale" to 0.125f,
                        "is_causal" to true,
                        "head_dim" to 64,
                    ),
                )
            },
        )
        val mlir = lowerKernelChoice(buildCoarsenedFn(), KernelTarget.CPU_GENERIC, registry)
            .toStablehlo()
        // Alphabetical: head_dim, is_causal, softmax_scale.
        assertTrue(
            mlir.contains(
                "backend_config = \"{head_dim = 64, is_causal = true, softmax_scale = 0.125}\"",
            ),
            mlir,
        )
    }

    @Test
    fun backendConfigEncodesScalarListAndBoolValues() {
        // Each supported value type (Number, Boolean, String, List)
        // round-trips into the expected literal form.
        val registry = mapOf<String, KernelTemplate>(
            "FlashAttention" to KernelTemplate { _, _ ->
                KernelDescriptor(
                    "synth_kernel", "tlaloc", "test",
                    customCallAttrs = mapOf(
                        "dtype_tag" to "bf16",
                        "is_causal" to false,
                        "head_dim" to 64,
                        "supported_kv_dtypes" to listOf("f32", "bf16", "int8"),
                    ),
                )
            },
        )
        val mlir = lowerKernelChoice(buildCoarsenedFn(), KernelTarget.CPU_GENERIC, registry)
            .toStablehlo()
        assertTrue(
            mlir.contains(
                "backend_config = \"{dtype_tag = bf16, head_dim = 64, is_causal = false, " +
                    "supported_kv_dtypes = [f32, bf16, int8]}\"",
            ),
            mlir,
        )
    }

    // ---------- Layer 4.2 §0.4.262 — sharding-aware emit ----------

    /**
     * Rebuild [annotated] (post-`lowerKernelChoice`) with [coarsenedSharding]
     * attached to its single COARSENED op. Used to construct sharded test
     * shapes; the L3.3 lowering pass preserves `node.sharding`, but L3.2's
     * coarsener doesn't *set* one, so tests inject it after annotation.
     */
    private val testMesh = DxirMesh("m", listOf(DxirMeshAxis("data", 8)))

    private fun withCoarsenedSharding(
        annotated: DxirFunction,
        coarsenedSharding: DxirSharding,
    ): DxirFunction = DxirBuilder.function(annotated.name) {
        declareMesh(testMesh)
        val nodeMap = HashMap<Int, DxirNode>()
        for (p in annotated.params) nodeMap[p.id] = param(p.name, p.type, p.sharding)
        for (n in annotated.body) {
            if (n !is DxirOp) continue
            val operands = n.operands.map { nodeMap[it.id]!! }
            val sharding = if (n.op == OpKind.COARSENED) coarsenedSharding else n.sharding
            nodeMap[n.id] = op(n.op, operands, n.type, n.attrs, sharding)
        }
        annotated.returns.map { nodeMap[it.id]!! }
    }

    private fun dataDimSharding(): DxirSharding = DxirSharding(
        meshName = "m",
        dimShardings = listOf(
            DxirDimSharding(axes = listOf(DxirAxisRef.Full("data"))),
            DxirDimSharding(axes = emptyList()), // dim 1 replicated
        ),
    )

    @Test
    fun shardedCoarsenedAttachesSdyShardingPerValue() {
        val annotated = lowerKernelChoice(buildCoarsenedFn(), KernelTarget.NVIDIA_H100)
        val sharded = withCoarsenedSharding(annotated, dataDimSharding())
        val mlir = sharded.toStablehlo()
        assertTrue(mlir.contains("stablehlo.custom_call @flash_attn_v3"), mlir)
        // Canonical SDY op-level sharding form: per_value with one entry for
        // the single result. The dim list mirrors toSdyAttr's `<@m, [...]>`.
        assertTrue(
            mlir.contains(
                "sdy.sharding = #sdy.sharding_per_value<[<@m, [{\"data\"}, {}]>]>",
            ),
            mlir,
        )
    }

    @Test
    fun unshardedCoarsenedHasNoSdyShardingAttribute() {
        // Regression guard for L4.1: when node.sharding is null, the
        // emit must not introduce an sdy.sharding attribute (otherwise
        // the L4.1 default emit would change shape on every kernel).
        val mlir = emitFor(KernelTarget.NVIDIA_H100)
        assertTrue(mlir.contains("stablehlo.custom_call"), mlir)
        assertFalse(
            mlir.contains("sdy.sharding"),
            "unsharded COARSENED must not emit sdy.sharding: $mlir",
        )
    }

    @Test
    fun meshAxesViaCustomCallAttrsSerializesIntoBackendConfig() {
        // The audit §16 calls out `mesh_axes` as the natural place to
        // plumb SDY axis names into the kernel-API contract. Because
        // customCallAttrs is `Map<String, Any>` and L4.1's encoder
        // already handles `List<String>`, this is a free convention —
        // the test pins it so future refactors don't accidentally hide
        // mesh_axes behind a different key or encoding.
        val registry = mapOf<String, KernelTemplate>(
            "FlashAttention" to KernelTemplate { _, _ ->
                KernelDescriptor(
                    "sharded_attention", "tlaloc", "test",
                    customCallAttrs = mapOf(
                        "mesh_axes" to listOf("data", "model"),
                    ),
                )
            },
        )
        val mlir = lowerKernelChoice(buildCoarsenedFn(), KernelTarget.CPU_GENERIC, registry)
            .toStablehlo()
        assertTrue(
            mlir.contains("backend_config = \"{mesh_axes = [data, model]}\""),
            mlir,
        )
    }

    @Test
    fun shardedCustomCallStillCarriesBackendConfigAndHasSideEffect() {
        // Composition: backend_config (from descriptor.customCallAttrs)
        // and sdy.sharding (from node.sharding) co-exist in the attribute
        // dict. has_side_effect = false stays present.
        val annotated = lowerKernelChoice(buildCoarsenedFn(), KernelTarget.NVIDIA_H100)
        val sharded = withCoarsenedSharding(annotated, dataDimSharding())
        val mlir = sharded.toStablehlo()
        assertTrue(mlir.contains("backend_config ="), mlir)
        assertTrue(mlir.contains("has_side_effect = false"), mlir)
        assertTrue(mlir.contains("sdy.sharding ="), mlir)
        // Single line — the three attrs are comma-separated inside one `{...}`.
        assertTrue(
            mlir.lines().any {
                it.contains("backend_config") && it.contains("has_side_effect") &&
                    it.contains("sdy.sharding")
            },
            "expected all three attrs on one custom_call line: $mlir",
        )
    }

    @Test
    fun replicatedOnlyShardingEmitsEmptyDimList() {
        // Boundary: a sharding that's fully replicated on every dim.
        // The SDY attr is still emitted (some passes treat the
        // explicit replicated annotation as a hint), and the per_value
        // wrapper still appears.
        val replicated = DxirSharding(
            meshName = "m",
            dimShardings = listOf(
                DxirDimSharding(axes = emptyList()),
                DxirDimSharding(axes = emptyList()),
            ),
        )
        val annotated = lowerKernelChoice(buildCoarsenedFn(), KernelTarget.NVIDIA_H100)
        val mlir = withCoarsenedSharding(annotated, replicated).toStablehlo()
        assertTrue(
            mlir.contains("sdy.sharding = #sdy.sharding_per_value<[<@m, [{}, {}]>]>"),
            mlir,
        )
    }
}
