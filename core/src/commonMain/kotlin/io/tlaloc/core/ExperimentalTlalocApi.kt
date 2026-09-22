package io.tlaloc.core

/**
 * §0.4.505 (Tier 4) — the opt-in marker for the part of Tlaloc's public surface
 * that is genuinely provisional.
 *
 * `docs/COMPATIBILITY.md` says every `0.1.0-alpha01` API may change without a
 * deprecation cycle. That sentence is true and it is also useless as a signal,
 * because it grades a heavily-certified surface (`grad`, the op surface, `:nn`'s
 * layers and optimizers — thousands of tests against analytic and
 * cross-implementation oracles) exactly the same as a surface nothing has ever
 * executed. This annotation is the difference, in the type system rather than in
 * prose.
 *
 * ## What carries it, and why
 *
 * Three criteria, and a declaration needs one of them. They are deliberately
 * narrow: an annotation applied to everything teaches a reader to add
 * `-opt-in=...` once and never think about it again, which is worse than no
 * annotation at all.
 *
 * 1. **`docs/CAPABILITIES.md` marks the capability 🧪 (written, never run end to
 *    end) or 📐 (designed).** The collective / multi-device attribute convention
 *    (`io.tlaloc.ir.AllReduceAttrs`) is the case: "Distributed / multi-GPU
 *    training" is 📐 — the design and the marshalling exist and nothing has ever
 *    run on two hosts — and its own documentation says v1 supports `"sum"` and
 *    nothing else.
 * 2. **The declaration's own KDoc scopes itself to "v1" and names what it would
 *    have to become.** The four-worlds scope taxonomy ([KernelScope],
 *    [OrchestrationScope], [ProgramScope], [ClusterScope], [Tlaloc],
 *    [BufferHandle], [HandleRef]) records that "v1 keeps each op single-scope"
 *    and that a multi-scope op would send the whole design to Kotlin's context
 *    parameters — a rewrite of every signature in the taxonomy.
 * 3. **The API selects behaviour whose *result* is uncertified even though the
 *    selection itself is tested.** The kernel-choice and cost-model surface
 *    (`io.tlaloc.ir.recognizer.kernel`, `io.tlaloc.ir.recognizer.cost`) is
 *    unit-certified and the artifacts it emits are pinned — but what it is *for*
 *    is picking a kernel, and the only kernel measured against XLA lost at small
 *    shapes (`docs/KPTX_PAGED_PERF.md`). Six of its files scope themselves to
 *    "v1" in their own text, and `DeviceDescriptor`'s numbers are dated
 *    `v1 (2026-05)`.
 *
 * ## What deliberately does NOT carry it
 *
 * `grad` and the transformation family, the tensor and op surface, `:nn`'s
 * layers / optimizers / schedules, safetensors, StableHLO emission and the PJRT
 * and IREE runtimes. Every one of those is ✅ in `CAPABILITIES.md` with a named
 * gate. They are still alpha — `COMPATIBILITY.md` still governs them — but they
 * are not *provisional*, and saying otherwise would make the marker noise.
 *
 * `:kptx`'s PTX DSL is a fourth surface that meets criterion 3 and is **not**
 * marked, for a mechanical reason recorded in `docs/ALPHA_PLAN.md`: `:kptx`
 * declares no dependency on `:core`, so this annotation cannot reach it without
 * adding one, and a documentation tier is the wrong place to change a published
 * module's dependency graph.
 *
 * ## Using it
 *
 * ```
 * @OptIn(ExperimentalTlalocApi::class)
 * fun main() { /* … */ }
 * ```
 *
 * or, for a whole module, `-opt-in=io.tlaloc.core.ExperimentalTlalocApi`
 * (Gradle: `compilerOptions { optIn.add("io.tlaloc.core.ExperimentalTlalocApi") }`).
 * Tlaloc's own modules do the latter — see the root `build.gradle.kts` — so the
 * marker is a signal to *consumers*, which is who it is for.
 */
@RequiresOptIn(
    level = RequiresOptIn.Level.ERROR,
    message =
    "This Tlaloc API is provisional: either the capability behind it has never run end to end " +
        "(docs/CAPABILITIES.md marks it as written-not-run or designed-only), or its own " +
        "documentation scopes it to a v1 shape it says it would have to leave. It may change " +
        "shape, not just signature. Opt in with @OptIn(ExperimentalTlalocApi::class) to " +
        "acknowledge that.",
)
@Retention(AnnotationRetention.BINARY)
@Target(
    AnnotationTarget.CLASS,
    AnnotationTarget.ANNOTATION_CLASS,
    AnnotationTarget.PROPERTY,
    AnnotationTarget.FIELD,
    AnnotationTarget.LOCAL_VARIABLE,
    AnnotationTarget.VALUE_PARAMETER,
    AnnotationTarget.CONSTRUCTOR,
    AnnotationTarget.FUNCTION,
    AnnotationTarget.PROPERTY_GETTER,
    AnnotationTarget.PROPERTY_SETTER,
    AnnotationTarget.TYPEALIAS,
)
public annotation class ExperimentalTlalocApi
