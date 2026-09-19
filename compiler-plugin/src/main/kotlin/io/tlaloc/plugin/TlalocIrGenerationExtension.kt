package io.tlaloc.plugin

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.passes.CoarseningCache
import io.tlaloc.ir.passes.DiskCoarseningCache
import io.tlaloc.ir.passes.DxirForwardTransform
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.ir.passes.NoOpCoarseningCache
import io.tlaloc.ir.passes.PhiCalculus
import io.tlaloc.ir.passes.SymbolicEngine
import io.tlaloc.ir.passes.SymjaEngine
import io.tlaloc.ir.pretty
import java.nio.file.Path
import java.nio.file.Paths
import org.jetbrains.kotlin.backend.common.IrElementTransformerVoidWithContext
import org.jetbrains.kotlin.backend.common.extensions.IrGenerationExtension
import org.jetbrains.kotlin.backend.common.extensions.IrPluginContext
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import org.jetbrains.kotlin.cli.common.messages.MessageCollector
import org.jetbrains.kotlin.ir.declarations.IrModuleFragment
import org.jetbrains.kotlin.ir.expressions.IrCall
import org.jetbrains.kotlin.ir.expressions.IrExpression
import org.jetbrains.kotlin.ir.symbols.UnsafeDuringIrConstructionAPI
import org.jetbrains.kotlin.ir.util.callableId

/**
 * Session-4 IR phase. Walks every [IrCall] in the module fragment; for each call whose
 * callee FQN matches a Tlaloc `grad` / `valueAndGrad` intrinsic AND whose source range
 * has a matching [TlalocLoweringHandoff] entry (populated by the FIR checker), synthesise
 * a fresh lambda expression that evaluates the forward pass of the stored [DxirFunction]
 * and return it in place of the original call.
 *
 * Session-4 scope is forward-only: `grad` and friends, at the IR level, currently return
 * the same scalar value the user's lambda would produce (not the derivative). The `grad`
 * transform is the next item; this work is the machinery that lets `grad` lower to plain
 * Kotlin bytecode in the first place. If the synthesis fails (e.g. the DxirFunction
 * references a tensor type or an op outside the primitive-scalar surface), we fall back
 * to the original call so the runtime-tape path in `:autograd` still runs.
 *
 * For every match — even when synthesis is skipped — we emit a `WARNING` via the plugin's
 * message collector naming the DxirFunction. That diagnostic is also what the older
 * scaffolding-era test asserts against.
 */
class TlalocIrGenerationExtension : IrGenerationExtension {

    @OptIn(UnsafeDuringIrConstructionAPI::class)
    override fun generate(moduleFragment: IrModuleFragment, pluginContext: IrPluginContext) {
        // §0.4.94 — KT-78277 deprecation investigation. Both
        // `IrPluginContext.messageCollector` AND `createDiagnosticReporter(name)` are
        // deprecated as of Kotlin 2.2.x. The only non-deprecated path is
        // `pluginContext.diagnosticReporter: IrDiagnosticReporter`, whose API is
        // factory-based and requires anchoring each diagnostic to an `IrDeclaration`
        // / `IrElement` / `IrFile`. Migrating means:
        //   1. Declaring IR-phase `KtDiagnosticFactory` instances in `TlalocDiagnostics.kt`
        //      mirroring the existing FIR-phase ones (LAMBDA_LOWERED, LAMBDA_UNSUPPORTED).
        //   2. Adding an `IrDiagnosticRenderer` so the existing
        //      "Tlaloc IR extension saw handoff …" prefix the scaffolding test
        //      asserts on still appears in the rendered message.
        //   3. Either rewriting every `mc.report(severity, message, null)` call as
        //      `diagnosticReporter.at(currentFile).report(factory, args)`, OR adding
        //      a thin `MessageCollector`-style adapter wrapping the new reporter.
        //   4. Updating `TlalocPluginDiagnosticTest`'s assertion mechanism: the test
        //      currently filters compile-time warnings off a MessageCollector hook;
        //      the new path emits `KtDiagnostic` instances, which surface differently.
        // Each of those four pieces is a session-or-more of work. The deprecation is
        // a warning, not an error — keeping the @Suppress here lets the rest of §0.4.N
        // sessions ship without rewriting the IR-phase diagnostic surface, and the
        // plan above is the recipe for a future "diagnosticReporter migration" session
        // when it becomes a hard prerequisite (e.g. a Kotlin version bump removes the
        // deprecated paths entirely).
        @Suppress("DEPRECATION")
        val mc = pluginContext.messageCollector
        val synth = DxirToIrSynthesis(pluginContext)
        // §0.4.24 — SymjaEngine is lazy-initialised once per compilation invocation so
        // Symja's ~1-3s classload cost amortises across all `grad`/`valueAndGrad` call
        // sites in the module. Null if instantiation fails (classpath issue, Log4j init,
        // …); in that case PhiCalculus.apply runs engine-free, which is sufficient for
        // IF-only primals (F1/F2/F3/C3 fire; C5-C9 silently skip). Loop primals (Stage
        // B.4b) will want a hard error instead.
        val engineLazy: Lazy<SymbolicEngine?> = lazy(LazyThreadSafetyMode.NONE) {
            try { SymjaEngine() } catch (_: Throwable) { null }
        }
        // §0.4.26 — coarsening cache (plan §5.4). Opt-in via system property
        // `tlaloc.cache.dir=<path>` (default: disabled). When enabled, PhiCalculus.apply's
        // output is memoised against the input primal's canonical SHA-256 hash, keyed
        // additionally by the CAS version string so a Symja / plugin bump invalidates
        // stale entries. Disabled by default because introducing disk state to every
        // compilation unit is a deliberate opt-in; most existing test harnesses don't
        // need it. CI / dev environments set the property in Gradle to enable.
        val cache: CoarseningCache = buildCoarseningCache(mc)

        val transformer = object : IrElementTransformerVoidWithContext() {
            override fun visitCall(expression: IrCall): IrExpression {
                val transformed = super.visitCall(expression) as IrCall
                val ownerFn = transformed.symbol.owner
                // §0.4.201 — local functions (declared inside another function's
                // body) lack a `callableId` and Kotlin's IR raises
                // `IllegalStateException` from `getCallableIdImpl`. Skip them
                // defensively rather than crash compilation: a local function
                // can't be an io.tlaloc.autograd intrinsic anyway.
                val cid = try {
                    ownerFn.callableId
                } catch (_: IllegalStateException) {
                    return transformed
                }
                if (cid.className != null ||
                    cid.packageName.asString() != "io.tlaloc.autograd" ||
                    cid.callableName.asString() !in INTRINSIC_NAMES
                ) return transformed

                val fn: DxirFunction = TlalocLoweringHandoff.take(
                    transformed.startOffset, transformed.endOffset,
                ) ?: return transformed

                mc.report(
                    CompilerMessageSeverity.WARNING,
                    "Tlaloc IR extension saw handoff for '${fn.name}':\n${fn.pretty().trimEnd()}",
                    null,
                )

                // Stage A follow-up (§0.4.4): route all four intrinsics through the
                // reverse-mode transform. `grad` / `grad2` emit only the gradients;
                // `valueAndGrad` / `valueAndGrad2` prepend the primal return so the
                // synthesiser can box a Pair / Triple with the value alongside the
                // gradients. Gate violations (multi-return, non-scalar return, regions,
                // multi-result ops, unsupported OpKinds) fall back to the runtime tape.
                val callableName = cid.callableName.asString()

                // §0.4.372 — forward-mode intrinsics (Phase B1): route through
                // DxirForwardTransform instead of the reverse pipeline. The
                // transform rewrites f(x)->y into jvp_f(x, dx)->(y, dy). For
                // `valueAndJvp` we synthesise both returns (boxed Pair<y, dy>);
                // for `jvp` we drop the primal returns and synthesise dy alone.
                // v1 scope = straight-line bodies (the transform errors loudly on
                // regions), so we SKIP coarsening/lift entirely — a region-bearing
                // body simply falls back to the runtime tape here.
                // §0.4.387 — the two-argument forms (`jvp2`/`valueAndJvp2`) route
                // through the same branch: the transform emits all primals then all
                // tangents for any arity, and the tangent/value split below is
                // `returns.size / 2`, so nothing here is arity-specific.
                val forwardIntrinsic = callableName == "jvp" || callableName == "jvp2" ||
                    callableName == "valueAndJvp" || callableName == "valueAndJvp2"
                val tangentOnly = callableName == "jvp" || callableName == "jvp2"
                if (forwardIntrinsic) {
                    val jvpFn: DxirFunction = try {
                        DxirForwardTransform.apply(fn)
                    } catch (t: Throwable) {
                        mc.report(
                            CompilerMessageSeverity.WARNING,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "DxirForwardTransform failed (${t::class.simpleName}: ${t.message}); " +
                                "falling back to the runtime tape\n${fn.pretty().trimEnd()}",
                            null,
                        )
                        return transformed
                    }
                    // jvp(f): keep only the tangent returns (the second half —
                    // DxirForwardTransform emits values(m) ++ tangents(m)); the
                    // full body stays (tangents depend on the primal values).
                    val toSynthesise: DxirFunction = if (tangentOnly) {
                        val m = fn.returns.size
                        DxirFunction(
                            jvpFn.name,
                            jvpFn.params,
                            jvpFn.body,
                            jvpFn.returns.subList(m, jvpFn.returns.size),
                            jvpFn.meshes,
                        )
                    } else {
                        jvpFn
                    }
                    val replacement = synth.synthesise(toSynthesise, transformed, currentDeclarationParent!!)
                    if (replacement == null) {
                        val reason = synth.lastFailureReason ?: "(no specific gate stamped)"
                        mc.report(
                            CompilerMessageSeverity.WARNING,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "forward-transformed function falls outside the synthesis scope " +
                                "[$reason]\npost-forward jvp function:\n${toSynthesise.pretty().trimEnd()}",
                            null,
                        )
                        return transformed
                    }
                    if (replacement.type != transformed.type) {
                        mc.report(
                            CompilerMessageSeverity.WARNING,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "synthesised type ${replacement.type} doesn't match call type " +
                                "${transformed.type}",
                            null,
                        )
                        return transformed
                    }
                    mc.report(
                        CompilerMessageSeverity.WARNING,
                        "Tlaloc lowered '${callableName}' to forward-mode dxir:\n" +
                            toSynthesise.pretty().trimEnd(),
                        null,
                    )
                    return replacement
                }

                val includeForward = callableName == "valueAndGrad" || callableName == "valueAndGrad2"

                // §0.4.24 — Stage B.4a. Run PhiCalculus.apply before SCT so IF/WHILE
                // primals are coarsened ahead of the reverse transform. For IF-only
                // primals the engine-backed corollaries never fire, so F1/F2/F3/C3
                // suffice and null-engine is also valid. For loops (B.4b), the engine
                // will be required. If Symja blew up during lazy init, fall through
                // without coarsening — Stage A's post-§0.4.23 SCT handles bare IFs.
                //
                // §0.4.33 — Stage C.3b.3a. When the system property `tlaloc.soi.enabled`
                // is set to "true", replace the PhiCalculus.apply call with
                // PhiCalculus.coarsenFunction, which wraps the primal in an OpKind.
                // COARSENED op carrying the pre-computed gradient_body. C.3b.3a scope
                // is single-return + root-is-leaf only; multi-return + valueAndGrad
                // paths fall back to the existing pipeline. The SOI path skips the
                // coarsening cache because the cache is keyed by input hash (same key
                // as the non-SOI path) — a follow-up can add a mode suffix to the key.
                val useSoiCoarsening = System.getProperty(SOI_ENABLED_PROPERTY) == "true" &&
                    !includeForward &&
                    fn.returns.size == 1
                // §0.4.36 — runtime-tunable L via the `tlaloc.soi.size.limit` property.
                // Default 50 per plan §8.2 + paper §5 (paper's starting recommendation).
                // Smaller values (e.g., 10-20) trigger multi-SOI branch coarsening on
                // smaller IF primals; larger values keep whole-function coarsening for
                // bigger primals. The sweep in `LSweepTest` shows no measurable compile-
                // time sensitivity in the 5-200 range on typical scalar benchmarks.
                val soiSizeLimit = System.getProperty(SOI_SIZE_LIMIT_PROPERTY)?.toIntOrNull()?.takeIf { it > 0 }
                    ?: 50
                val coarsened: DxirFunction = try {
                    if (useSoiCoarsening) {
                        PhiCalculus.coarsenFunction(fn, engineLazy.value, sizeLimit = soiSizeLimit)
                    } else {
                        cache.getOrCompute(fn) { PhiCalculus.apply(fn, engineLazy.value) }
                    }
                } catch (t: Throwable) {
                    mc.report(
                        CompilerMessageSeverity.WARNING,
                        "Tlaloc IR extension: PhiCalculus.${if (useSoiCoarsening) "coarsenFunction" else "apply"} " +
                            "failed on '${fn.name}' (${t::class.simpleName}: ${t.message}); " +
                            "continuing with the raw primal",
                        null,
                    )
                    fn
                }
                // §0.4.174 — pre-SCT region-body lift. Coarsening's `distribute` rule
                // can produce IFs whose region body ops reference OUTER-scope IFs as
                // forward operands (`%59 = MUL(%58, %57-OUTER-IF)` inside a sibling
                // IF's branch). DxirReverseTransform.apply's clone loop maps
                // `nodeMap[primal-IF] = primal-IF` (skipping the deep clone), so
                // walkBranchReverse step 1 emits a top-level cloned MUL whose
                // `operand[1]` leaks the primal-IF id into the grad body's SSA.
                // Lifting body ops to top level (where DxirReverseTransform's clone
                // path rebuilds operands through nodeMap) closes the leak AND leaves
                // the post-lift IF with empty regions, satisfying irIfOp's "branches
                // yield outer-scope values only" gate. The pass bails out (no-op
                // returns) when the function contains shapes that aren't safe to lift
                // — see [PhiCalculus.liftIfRegionBodies].
                val lifted: DxirFunction = PhiCalculus.liftIfRegionBodies(coarsened)
                val toSynthesise: DxirFunction = tryReverseTransform(
                    lifted, includeForward, mc, fn.name,
                ) ?: run {
                    // §0.4.173 — augment the §0.4.169 warning: also dump the post-
                    // coarsening + post-lift dxir so the next firing has full visibility
                    // into the input that DxirReverseTransform rejected. §0.4.174 added
                    // the post-lift dump.
                    mc.report(
                        CompilerMessageSeverity.WARNING,
                        "Tlaloc IR extension post-coarsening dxir for '${fn.name}':\n" +
                            coarsened.pretty().trimEnd() + "\n" +
                            "post-lift dxir:\n${lifted.pretty().trimEnd()}",
                        null,
                    )
                    return transformed
                }

                // §0.4.105 — D.1i Phase 3. Optionally run PhiCalculus.simplifyReturns over
                // the gradient function before synthesis. Gated on `tlaloc.simplify.enabled`
                // (default off) and on the engine being available — null engine, property
                // unset, or anything other than "true" all leave the gradient body
                // unchanged. simplifyReturns has its own internal bail-out (returns the
                // input fn on any lift / lower failure), so the pipeline stays correct
                // for gradient bodies whose op set falls outside the lift surface.
                val simplifyEnabled =
                    System.getProperty(SIMPLIFY_ENABLED_PROPERTY) == "true"
                val simplified: DxirFunction = if (simplifyEnabled) {
                    val engine = engineLazy.value
                    if (engine == null) toSynthesise
                    else try {
                        PhiCalculus.simplifyReturns(toSynthesise, engine)
                    } catch (t: Throwable) {
                        mc.report(
                            CompilerMessageSeverity.WARNING,
                            "Tlaloc IR extension: PhiCalculus.simplifyReturns threw on " +
                                "'${fn.name}' (${t::class.simpleName}: ${t.message}); " +
                                "synthesising the un-simplified gradient",
                            null,
                        )
                        toSynthesise
                    }
                } else toSynthesise

                val replacement = synth.synthesise(simplified, transformed, currentDeclarationParent!!)
                if (replacement == null) {
                    val reason = synth.lastFailureReason ?: "(no specific gate stamped)"
                    mc.report(
                        CompilerMessageSeverity.WARNING,
                        "Tlaloc IR extension kept original call for '${fn.name}' — " +
                            "DxirFunction falls outside the scalar-primitive synthesis scope " +
                            "[$reason]\n" +
                            "post-coarsening primal:\n${coarsened.pretty().trimEnd()}\n" +
                            "post-lift dxir:\n${lifted.pretty().trimEnd()}\n" +
                            "post-SCT grad function:\n${simplified.pretty().trimEnd()}",
                        null,
                    )
                    return transformed
                }
                // With Pair/Triple boxing in synthesis (§0.4.4), the four intrinsics'
                // declared return types line up when the surface is scalar-primitive:
                // `grad` → `(P) -> R`, `grad2` → `(A, B) -> Pair<A, B>`, `valueAndGrad` →
                // `(P) -> Pair<R, P>`, `valueAndGrad2` → `(A, B) -> Triple<R, A, B>`. The
                // guard still fires for surfaces we can't synthesise (tensors, unsupported
                // op kinds, DScalar boxing, etc.) — in that case we keep the original call
                // so the runtime-tape path runs.
                if (replacement.type != transformed.type) {
                    mc.report(
                        CompilerMessageSeverity.WARNING,
                        "Tlaloc IR extension kept original call for '${fn.name}' — " +
                            "synthesised type ${replacement.type} doesn't match call type " +
                            "${transformed.type} (forward-only scope)",
                        null,
                    )
                    return transformed
                }
                return replacement
            }
        }

        for (file in moduleFragment.files) {
            file.transformChildren(transformer, null)
        }

        // Clear any unclaimed entries so a re-run of the in-process compiler harness
        // doesn't find stale handoffs from a previous invocation.
        TlalocLoweringHandoff.clear()
    }

    /**
     * Runs [DxirReverseTransform.apply] guarded against its hard gates. Returns `null` if
     * the primal violates a gate (e.g., non-scalar return, regions, unsupported op kind);
     * the caller falls back to the original runtime call.
     *
     * §0.4.169 — emits a per-failure WARNING that surfaces the specific exception
     * message (replacing the prior generic "(gate violation)" string). Two consecutive
     * port attempts (§0.4.163 HMC nested-loop, §0.4.168 CartPole Phase 1) hit the
     * downstream gate but couldn't pinpoint the failing op without exception text;
     * the warning text now names which dxir node / op kind / require-string failed,
     * enabling targeted fixes in subsequent firings.
     */
    private fun tryReverseTransform(
        primal: DxirFunction,
        includeForward: Boolean,
        mc: MessageCollector,
        fnName: String,
    ): DxirFunction? = try {
        DxirReverseTransform.apply(primal, includeForward)
    } catch (t: IllegalArgumentException) {
        mc.report(
            CompilerMessageSeverity.WARNING,
            "Tlaloc IR extension kept original call for '$fnName' — DxirReverseTransform " +
                "rejected the dxir (${t::class.simpleName}: ${t.message})",
            null,
        )
        null
    } catch (t: IllegalStateException) {
        mc.report(
            CompilerMessageSeverity.WARNING,
            "Tlaloc IR extension kept original call for '$fnName' — DxirReverseTransform " +
                "rejected the dxir (${t::class.simpleName}: ${t.message})",
            null,
        )
        null
    }

    /**
     * §0.4.26 — resolve the [CoarseningCache] impl for this compilation. Reads the
     * system property `tlaloc.cache.dir`:
     *
     *  - Unset or empty → [NoOpCoarseningCache] (caching disabled; default behaviour).
     *  - Set to a filesystem path → [DiskCoarseningCache] under that directory, keyed
     *    by `CAS_VERSION` (Symja version × plugin version; any bump invalidates cached
     *    entries).
     *  - Set to `":memory:"` → an anonymous in-memory cache for tests. Not useful in
     *    real builds but keeps the test harness from touching disk.
     *
     * Failures (permission errors, unwritable path) demote to NoOp + emit a WARNING;
     * compilation continues without caching rather than erroring.
     */
    private fun buildCoarseningCache(mc: org.jetbrains.kotlin.cli.common.messages.MessageCollector): CoarseningCache {
        val raw = System.getProperty(CACHE_DIR_PROPERTY).orEmpty().trim()
        if (raw.isEmpty()) return NoOpCoarseningCache
        if (raw == ":memory:") return io.tlaloc.ir.passes.InMemoryCoarseningCache()
        return try {
            val dir: Path = Paths.get(raw).toAbsolutePath()
            DiskCoarseningCache(dir, CAS_VERSION)
        } catch (t: Throwable) {
            mc.report(
                CompilerMessageSeverity.WARNING,
                "Tlaloc IR extension: coarsening cache disabled — failed to open '$raw' " +
                    "(${t::class.simpleName}: ${t.message})",
                null,
            )
            NoOpCoarseningCache
        }
    }

    companion object {
        private val INTRINSIC_NAMES: Set<String> = setOf(
            "grad", "grad2", "valueAndGrad", "valueAndGrad2",
            // §0.4.372 — forward-mode (Phase B1). §0.4.387 — its two-argument forms.
            "jvp", "valueAndJvp", "jvp2", "valueAndJvp2",
        )

        /**
         * System property name the plugin reads to decide where to write cache entries.
         * Unset = caching disabled. See [buildCoarseningCache] for accepted values.
         */
        const val CACHE_DIR_PROPERTY: String = "tlaloc.cache.dir"

        /**
         * §0.4.33 — system property toggling Stage C.3b.3a SOI-based coarsening.
         * When set to "true", `grad { ... }` calls route through
         * [PhiCalculus.coarsenFunction] (wraps the primal in an OpKind.COARSENED op
         * with pre-computed gradient_body). `valueAndGrad` + multi-return primals
         * continue to use the existing [PhiCalculus.apply] path because C.3b.3a
         * doesn't yet handle them.
         */
        const val SOI_ENABLED_PROPERTY: String = "tlaloc.soi.enabled"

        /**
         * §0.4.36 — system property overriding the SOI size-limit `L` when coarsening is
         * enabled. Must be a positive integer. Unset / invalid values fall back to the
         * paper-informed default of 50 (plan §8.2). `LSweepTest` in `:ir/jvmTest` sweeps
         * L ∈ {5, 25, 50, 100, 200} across a benchmark suite to inform tuning; see
         * §0.4.36 for empirical findings.
         */
        const val SOI_SIZE_LIMIT_PROPERTY: String = "tlaloc.soi.size.limit"

        /**
         * §0.4.105 — D.1i Phase 3. System property toggling [PhiCalculus.simplifyReturns]
         * over each gradient `DxirFunction` after [DxirReverseTransform.apply]. When set
         * to "true", the IR extension lifts each return expression to a Symja `SymExpr`,
         * runs `Simplify`, and lowers back. Unset / "false" / anything else: gradient
         * body passes through unchanged. The pass has internal bail-out semantics, so a
         * `true` setting is safe for gradient bodies outside the lift surface — they
         * synthesise un-simplified rather than failing.
         */
        const val SIMPLIFY_ENABLED_PROPERTY: String = "tlaloc.simplify.enabled"

        /**
         * CAS version string per plan §3.2.2 — bumps invalidate cached entries. Tied to
         * the plugin build (bump on plugin code changes that alter PhiCalculus output or
         * reverse-transform semantics) AND to Symja's resolved runtime version. Format:
         * `"tlaloc-<plugin>-symja-<symja>"`. The Symja version is read lazily (Symja's
         * package-info may not load until the first engine instantiation), so we bake a
         * build-time placeholder here and callers that need Symja-version-keyed cache
         * behaviour can override by recomputing.
         */
        const val CAS_VERSION: String = "tlaloc-0.4.26-symja-3.0.0"
    }
}
