package io.tlaloc.ir.recognizer.quant

import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirOpResult
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.kernel.KernelDescriptor

/**
 * KV-cache quantization application pass.
 *
 * # What
 *
 * Walks `fn` and, for each `OpKind.COARSENED` op that:
 *
 * 1. Has a [KernelDescriptor] annotated under [KernelDescriptor.ATTR_KEY]
 *    (i.e., [io.tlaloc.ir.recognizer.kernel.lowerKernelChoice] picked a vendor kernel for the
 * recognized pattern), and
 * 2. The descriptor's `customCallAttrs["supported_kv_dtypes"]` list
 *    contains the requested dtype's [KvQuantDtype.nameTag],
 *
 * stamps the COARSENED with [KvQuantConfig.ATTR_KEY] = `[config]`. The
 * downstream kernel custom-call (StableHLO emit) reads the attr to pass
 * the quantization directive through to the vendor runtime.
 *
 * # Why best-effort, not enforced
 *
 * The user's "I want FP8 KV cache" request is best-effort: if the
 * target's kernel doesn't support FP8 (e.g., A100, which has no native
 * FP8 path), we silently skip the annotation rather than error. This
 * makes the pass safe to apply across heterogeneous targets in a single
 * compile — the resulting function is correct on every target, just
 * with KV quantization where supported.
 *
 * Tracking unmet requests is via the second return value of
 * [applyKvQuantWithDiagnostics] — a list of `(coarsenedOpId, reason)`
 * pairs explaining why each declined COARSENED was skipped. Enables
 * IDE / debug tooling without forcing the pass to be fail-loud.
 *
 * # What's NOT in v1
 *
 * - **Per-head scale tensors aren't materialized as DxirNodes.** The
 *   [KvScaleStrategy.PER_HEAD] config is a directive; runtime computes
 *   the scales online (or reads them from a side-channel manifest). A
 *   future pass could insert explicit `compute_scales` ops if a target
 *   needs them in-graph.
 * - **No primal_body type rewrite.** The K and V tensors stay F32 in
 *   DXIR; only memory layout at the kernel-call boundary changes.
 *   Adding native `OpKind.QUANTIZE` / `OpKind.DEQUANTIZE` ops with I8 /
 *   FP8 dtypes is a separate type-system extension.
 * - **No automatic dtype selection.** Callers explicitly pick the
 *   target dtype; the request is honored verbatim.
 */
fun applyKvQuant(fn: DxirFunction, requested: KvQuantConfig): DxirFunction =
    applyKvQuantWithDiagnostics(fn, requested).first

/**
 * Two-return variant of [applyKvQuant]. The second list reports each
 * COARSENED that was skipped + the reason — useful for IDE tooling and
 * for examples that want to surface a "tried FP8, fell back to
 * BF16" line to the user.
 */
fun applyKvQuantWithDiagnostics(
    fn: DxirFunction,
    requested: KvQuantConfig,
): Pair<DxirFunction, List<KvQuantDiagnostic>> {
    val coarsenedOps = fn.body.filterIsInstance<DxirOp>().filter { it.op == OpKind.COARSENED }
    if (coarsenedOps.isEmpty()) return fn to emptyList()

    val diagnostics = mutableListOf<KvQuantDiagnostic>()
    val toAnnotate = HashSet<Int>()

    for (co in coarsenedOps) {
        val descriptor = co.attrs[KernelDescriptor.ATTR_KEY] as? KernelDescriptor
        if (descriptor == null) {
            diagnostics += KvQuantDiagnostic(co.id, "no kernel descriptor — pattern not lowered yet, or decomposed")
            continue
        }
        @Suppress("UNCHECKED_CAST")
        val supported = (descriptor.customCallAttrs["supported_kv_dtypes"] as? List<String>) ?: emptyList()
        if (supported.isEmpty()) {
            diagnostics += KvQuantDiagnostic(
                co.id,
                "kernel ${descriptor.kernelName} does not advertise supported_kv_dtypes",
            )
            continue
        }
        if (requested.dtype.nameTag !in supported) {
            diagnostics += KvQuantDiagnostic(
                co.id,
                "kernel ${descriptor.kernelName} on ${descriptor.targetArch} doesn't support " +
                    "${requested.dtype.nameTag} KV cache (supports: ${supported.joinToString(",")})",
            )
            continue
        }
        toAnnotate += co.id
    }

    if (toAnnotate.isEmpty()) return fn to diagnostics

    val rebuilt = DxirBuilder.function(fn.name) {
        val nodeMap = HashMap<Int, DxirNode>()
        for (p in fn.params) nodeMap[p.id] = param(p.name, p.type, p.sharding)

        for (node in fn.body) {
            when (node) {
                is DxirConst -> {
                    nodeMap[node.id] = const(node.value, node.type, node.sharding)
                }
                is DxirOp -> {
                    val operands = node.operands.map {
                        require(it !is DxirOpResult) {
                            "applyKvQuant: DxirOpResult operand out of v1 scope"
                        }
                        nodeMap[it.id] ?: error("applyKvQuant: operand id=${it.id} not in nodeMap")
                    }
                    require(node.regions.isEmpty()) {
                        "applyKvQuant: region-bearing op (id=${node.id}, ${node.op}) out of v1 scope"
                    }
                    require(!node.isMultiResult) {
                        "applyKvQuant: multi-result op (id=${node.id}, ${node.op}) out of v1 scope"
                    }
                    val annotated = if (node.id in toAnnotate) {
                        node.attrs + mapOf(KvQuantConfig.ATTR_KEY to requested)
                    } else {
                        node.attrs
                    }
                    nodeMap[node.id] = op(node.op, operands, node.type, annotated, node.sharding)
                }
                else -> error("applyKvQuant: unsupported body node $node")
            }
        }
        fn.returns.map {
            nodeMap[it.id] ?: error("applyKvQuant: return ref id=${it.id} not in nodeMap")
        }
    }
    return rebuilt to diagnostics
}

/**
 * Why a particular COARSENED was skipped during KV-quant application.
 *
 * @property coarsenedOpId id of the COARSENED op the pass declined to
 *   annotate.
 * @property reason short string describing why — surfaced via IDE
 *   tooling or example output.
 */
data class KvQuantDiagnostic(val coarsenedOpId: Int, val reason: String)
