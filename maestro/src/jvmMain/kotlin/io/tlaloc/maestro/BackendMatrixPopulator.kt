package io.tlaloc.maestro

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.recognizer.coarsener.coarsenRecognizedPatterns
import io.tlaloc.ir.recognizer.cost.DeviceDescriptors
import io.tlaloc.ir.recognizer.cost.estimateRooflineMicros
import io.tlaloc.ir.recognizer.kernel.KernelDescriptor
import io.tlaloc.ir.recognizer.kernel.KernelTarget
import io.tlaloc.ir.recognizer.kernel.lowerKernelChoice
import io.tlaloc.ir.recognizer.quant.KvQuantConfig
import io.tlaloc.ir.recognizer.quant.applyKvQuant
import io.tlaloc.ir.recognizer.recognizeAll

/**
 * Populate `ProgramManifest.backendMatrix` from the
 * kernel-selection pipeline.
 *
 * # Pipeline
 *
 * For each requested [KernelTarget], runs:
 *
 * ```
 * fn → recognizeAll → coarsenRecognizedPatterns
 *    → lowerKernelChoice(target)
 *    → [optional] applyKvQuant(target, requestedKvQuant)
 *    → estimateRooflineMicros(target)
 * ```
 *
 * and projects the post-pipeline state into a [BackendTarget] tuple.
 *
 * # When to call this
 *
 * `Tlaloc.program { }` does NOT call this automatically — running the
 * full pipeline per (vendor × arch × kv-quant) combo at trace time
 * would balloon trace cost. Instead, callers that *want* a populated
 * backend matrix invoke this explicitly:
 *
 * ```
 * val step = Tlaloc.program("encode", input, Mesh0) { ... }
 * val matrix = populateBackendMatrix(
 *     fn = step.capturedFn,  // not exposed by MaestroStep yet
 *     targets = listOf(KernelTarget.NVIDIA_H100, KernelTarget.AWS_TRAINIUM2),
 *     kvQuant = KvQuantConfig.FP8_PER_HEAD,
 * )
 * val richer = step.copy(manifest = step.manifest.copy(backendMatrix = matrix))
 * ```
 *
 * [MaestroStep] doesn't expose the captured `DxirFunction` (its public
 * surface is the StableHLO bytes + manifest), so this populator takes the
 * function directly.
 *
 * @param fn the captured `DxirFunction` from `Tlaloc.program { }`
 *   tracing — pre-coarsening, pre-lowering.
 * @param targets the device targets to populate decisions for.
 * @param kvQuant optional KV-quant request. `null` skips the quant
 *   pass entirely; the resulting [BackendTarget.kvQuantDtype] is
 *   `null`. When non-null, applied after kernel lowering — the
 *   per-target [BackendTarget.kvQuantDtype] reflects whether the
 *   target's kernel actually accepted the dtype (best-effort: a
 *   request that isn't supported on a given target produces a
 *   `kvQuantDtype = null` entry, matching the KV-quant pass's skip semantics).
 * @return one [BackendTarget] per element of [targets], in the same
 *   order. No diagnostics — for skip reasons, callers should run
 *   `applyKvQuantWithDiagnostics` separately.
 */
fun populateBackendMatrix(
    fn: DxirFunction,
    targets: List<KernelTarget>,
    kvQuant: KvQuantConfig? = null,
): List<BackendTarget> {
    if (targets.isEmpty()) return emptyList()

    // Pre-pipeline: recognize + coarsen once. The coarsened function is
    // target-independent; per-target we only run kernel lowering + KV
    // quant + cost.
    val matches = recognizeAll(fn)
    val coarsened = coarsenRecognizedPatterns(fn, matches)

    return targets.map { target ->
        val lowered = lowerKernelChoice(coarsened, target)
        val (afterQuant, kvDtype) = if (kvQuant != null) {
            val applied = applyKvQuant(lowered, kvQuant)
            // Determine whether any COARSENED actually got the quant
            // attr — if so, record the requested dtype's name tag.
            val accepted = applied.body.filterIsInstance<DxirOp>()
                .any { it.op == OpKind.COARSENED && it.attrs[KvQuantConfig.ATTR_KEY] != null }
            applied to (if (accepted) kvQuant.dtype.nameTag else null)
        } else {
            lowered to null
        }

        // Pick the kernel name: any COARSENED with a kernel_descriptor
        // attached records that descriptor's kernel name. If multiple
        // COARSENED ops exist (multiple recognized patterns), we record
        // the first — v1 ships with FlashAttention only, so the multi-
        // pattern case is hypothetical.
        val kernelName = afterQuant.body.filterIsInstance<DxirOp>()
            .firstOrNull { it.op == OpKind.COARSENED }
            ?.attrs?.get(KernelDescriptor.ATTR_KEY)
            ?.let { (it as KernelDescriptor).kernelName }

        val cost = DeviceDescriptors.byName(target.arch ?: "")
            ?.let { estimateRooflineMicros(afterQuant, it) }

        BackendTarget(
            vendor = target.vendor,
            arch = target.arch ?: "any",
            kernelName = kernelName,
            kvQuantDtype = kvDtype,
            costMicroseconds = cost,
        )
    }
}
