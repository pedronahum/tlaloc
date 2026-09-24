package io.tlaloc.plugin

import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.passes.CoarseningCache
import io.tlaloc.ir.passes.DiskCoarseningCache
import io.tlaloc.ir.passes.DxirForwardTransform
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.ir.passes.NoOpCoarseningCache
import io.tlaloc.ir.passes.PhiCalculus
import io.tlaloc.ir.passes.SymbolicEngine
import io.tlaloc.ir.passes.SymbolicEngines
import io.tlaloc.ir.pretty
import io.tlaloc.ir.render.toKotlinSource
import java.io.File
import java.nio.file.Path
import java.nio.file.Paths
import org.jetbrains.kotlin.backend.common.IrElementTransformerVoidWithContext
import org.jetbrains.kotlin.backend.common.extensions.IrGenerationExtension
import org.jetbrains.kotlin.backend.common.extensions.IrPluginContext
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageLocation
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSourceLocation
import org.jetbrains.kotlin.cli.common.messages.MessageCollector
import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.declarations.IrFile
import org.jetbrains.kotlin.ir.declarations.IrModuleFragment
import org.jetbrains.kotlin.ir.declarations.IrValueDeclaration
import org.jetbrains.kotlin.ir.declarations.IrValueParameter
import org.jetbrains.kotlin.ir.declarations.IrVariable
import org.jetbrains.kotlin.ir.expressions.IrCall
import org.jetbrains.kotlin.ir.expressions.IrExpression
import org.jetbrains.kotlin.ir.expressions.impl.IrCallImpl
import org.jetbrains.kotlin.ir.expressions.impl.fromSymbolOwner
import org.jetbrains.kotlin.ir.symbols.UnsafeDuringIrConstructionAPI
import org.jetbrains.kotlin.ir.types.IrSimpleType
import org.jetbrains.kotlin.ir.types.typeOrNull
import org.jetbrains.kotlin.ir.types.typeWith
import org.jetbrains.kotlin.ir.util.callableId
import org.jetbrains.kotlin.ir.visitors.IrVisitorVoid
import org.jetbrains.kotlin.ir.visitors.acceptChildrenVoid
import org.jetbrains.kotlin.ir.visitors.acceptVoid
import org.jetbrains.kotlin.name.CallableId
import org.jetbrains.kotlin.name.ClassId
import org.jetbrains.kotlin.name.FqName
import org.jetbrains.kotlin.name.Name

/**
 * The IR phase. Walks every [IrCall] in the module fragment; for each call whose
 * callee FQN matches a Tlaloc differentiation intrinsic AND whose source range has a
 * matching [TlalocLoweringHandoff] entry (populated by the FIR checker), it transforms
 * the stored [DxirFunction] (reverse mode for the `grad` family, forward mode for the
 * `jvp` family), synthesises a fresh lambda expression from the result and returns it
 * in place of the original call, so the derivative compiles to plain Kotlin bytecode.
 * If a step fails, the original call is kept and reported (see
 * [TlalocPluginOptions.strictLowering]).
 *
 * For every match the extension emits an `INFO` naming the DxirFunction, only with
 * [TlalocPluginOptions.dumpLoweredIr] on: it is developer introspection and stays out of
 * a consumer's build log by default.
 *
 * The readable-reverse dump: with [dumpGradSource] on (the
 * `dumpGradSource` / `dumpGradSourceDir` plugin CLI options), every reverse-gradient
 * intrinsic (`grad` / `grad2` / `grad3` / `valueAndGrad{,2,3}`) whose lambda the plugin
 * SUCCESSFULLY synthesises also emits the reverse-transformed gradient
 * [DxirFunction] rendered as Kotlin source by [toKotlinSource] — an INFO message
 * headed by the lambda's source location, plus (dir form) a `.kt` file named after
 * that location. The dump happens at the dxir level BEFORE synthesis: the rendered
 * function is the IDENTICAL object handed to [DxirToIrSynthesis.synthesise], so the
 * gradient the user reads is the gradient the compiler compiles. Gradients the
 * renderer cannot print faithfully — above all tensor `grad {}` bodies, whose -1
 * SENTINEL dims forbid every ranked-literal rendering — dump a SKIPPED message naming
 * the refusal instead of wrong source.
 */
class TlalocIrGenerationExtension(
    private val dumpGradSource: Boolean = false,
    private val dumpGradSourceDir: String? = null,
    private val options: TlalocPluginOptions = TlalocPluginOptions(),
    private val handoff: TlalocLoweringHandoff = TlalocLoweringHandoff(),
) : IrGenerationExtension {

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
        //
        // §0.4.503 (Tier 3, item 3) — that "hard error" is now here, and the reason it
        // had to arrive is that Symja stopped being a mandatory dependency. `:ir`
        // declares it `compileOnly`, so ABSENT is the default state for a consumer who
        // did not ask for an 8.3 MB LGPL-3.0 jar. The old `catch (_: Throwable) { null }`
        // folded three different situations into one silent null — absent, present but
        // broken, and present but failing to initialise — and then degraded quietly.
        // `SymbolicEngines` separates them, and `reportMissingSymbolicEngine` below turns
        // the one case that MATTERS (the reverse transform rejected a primal that still
        // has a loop in it, on a classpath with no CAS) into a refusal that names the
        // missing dependency, its licence and the single line that adds it.
        val symjaAvailable = SymbolicEngines.symjaAvailable
        val engineLazy: Lazy<SymbolicEngine?> = lazy(LazyThreadSafetyMode.NONE) {
            SymbolicEngines.symjaOrNull()
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
            /** The file being transformed, for the dump's location header. */
            var irFileForDump: IrFile? = null

            /**
             * Every `IrVariable` / `IrValueParameter` in the file being
             * transformed, keyed by its source START OFFSET. This is the IR side of
             * the FIR→IR handoff for a captured runtime value: FIR recorded the
             * DECLARATION's source range, and the two phases share source offsets, so
             * this is how the synthesised gradient gets a handle on the very
             * declaration the user's lambda closed over. A list per offset (rather
             * than a single entry) so an ambiguous match can be REFUSED instead of
             * guessed at.
             */
            var valueDeclIndex: Map<Int, List<IrValueDeclaration>> = emptyMap()

            /** The source location of the call being rewritten, so every
             * IR-phase diagnostic points at the user's line instead of at nothing. */
            var callLocation: CompilerMessageSourceLocation? = null

            override fun visitCall(expression: IrCall): IrExpression {
                val transformed = super.visitCall(expression) as IrCall
                // §0.4.514 — the top-level guard, the IR twin of the FIR checker's. An
                // unexpected exception while rewriting ONE call (a synthesis bug, a
                // `!!` on an IR shape nobody anticipated) becomes a diagnostic at that
                // call naming the exception and where to report it, instead of an
                // internal compiler error. Compiler control-flow exceptions and JVM
                // errors are rethrown untouched.
                val location = locationOf(irFileForDump, transformed.startOffset)
                callLocation = location
                return try {
                    rewriteCall(transformed)
                } catch (t: Throwable) {
                    if (TlalocInternalErrors.mustRethrow(t)) throw t
                    mc.report(
                        if (options.strictLowering) CompilerMessageSeverity.ERROR else CompilerMessageSeverity.WARNING,
                        TlalocInternalErrors.describe(t, options.strictLowering),
                        location,
                    )
                    transformed
                }
            }

            private fun rewriteCall(transformed: IrCall): IrExpression {
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

                // §0.4.514 — keyed by this FILE's path as well as the range (see
                // TlalocLoweringHandoff): the same offsets in another file are another call.
                val filePath = irFileForDump?.fileEntry?.name ?: return transformed
                if (TlalocInternalErrors.faultInjected(TlalocInternalErrors.Phase.IR_HANDOFF_MISS)) return transformed
                val lowered: TlalocLoweringHandoff.LoweredLambda = handoff.take(
                    filePath, transformed.startOffset, transformed.endOffset,
                ) ?: return transformed
                TlalocInternalErrors.maybeInjectFault(TlalocInternalErrors.Phase.IR)
                val fn: DxirFunction = lowered.fn

                // §0.4.499 — developer introspection, OFF by default and INFO when on.
                // This was an unconditional WARNING: together with the FIR checker's
                // LAMBDA_LOWERED it put two dxir dumps into a consumer's build log for
                // every single `grad {}`, and broke any build using -Werror outright.
                if (options.dumpLoweredIr) {
                    mc.report(
                        CompilerMessageSeverity.INFO,
                        "Tlaloc IR extension saw handoff for '${fn.name}':\n${fn.pretty().trimEnd()}",
                        null,
                    )
                }

                // §0.4.501 — captured runtime values. The FIR side admits them for the
                // reverse-mode `grad` family only (see
                // TlalocIntrinsicCallChecker.captureCarryingIntrinsics), so this is a
                // belt-and-braces guard: were one ever to arrive at a branch that
                // rebuilds the parameter list, the extra param would change what the
                // returned function takes, and that must never happen silently.
                if (lowered.captures.isNotEmpty() &&
                    cid.callableName.asString() !in CAPTURE_CARRYING_INTRINSICS
                ) {
                    keptOriginal(
                        mc,
                        "Tlaloc IR extension kept original call for '${fn.name}' — it captures " +
                            "the runtime value(s) ${lowered.captures.joinToString { "'" + it.name + "'" }} " +
                            "and '${cid.callableName.asString()}' cannot carry a captured value " +
                            "as an input-only parameter",
                        callLocation,
                    )
                    return transformed
                }

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
                // §0.4.387 — the two-argument forms (`jvp2`/`valueAndJvp2`) route
                // through the same branch: the transform emits all primals then all
                // tangents for any arity, and the tangent/value split below is
                // `returns.size / 2`, so nothing here is arity-specific.
                // §0.4.403 — Phase B3: region-bearing bodies coarsen FIRST, exactly
                // as the reverse branch does below (PhiCalculus.apply + the §0.4.174
                // region-body lift), so a WHILE-bearing `jvp {}` body reaches
                // DxirForwardTransform as straight-line / COARSENED shapes instead
                // of falling back to the runtime tape. Straight-line bodies skip
                // the pipeline, keeping the §0.4.372 path byte-identical. §0.4.407 —
                // an IF that survives the coarsening (PhiCalculus's F-rules keep
                // genuine two-branch conditionals) no longer errors: the transform's
                // IF direct forward arm lowers it, so `jvp {}` over a Kotlin
                // if/else stops falling back; only shapes the transform still
                // refuses (e.g. a WHILE nested inside an IF branch that the
                // pipeline could not close) error and fall back below.
                val forwardIntrinsic = callableName == "jvp" || callableName == "jvp2" ||
                    callableName == "valueAndJvp" || callableName == "valueAndJvp2"
                val tangentOnly = callableName == "jvp" || callableName == "jvp2"
                if (forwardIntrinsic) {
                    val fwdPrimal: DxirFunction = if (fn.body.any { it is DxirOp && it.regions.isNotEmpty() }) {
                        val coarsenedFwd = try {
                            cache.getOrCompute(fn) { PhiCalculus.apply(fn, engineLazy.value) }
                        } catch (t: Throwable) {
                            mc.report(
                                CompilerMessageSeverity.WARNING,
                                "Tlaloc IR extension: PhiCalculus.apply failed on '${fn.name}' " +
                                    "(${t::class.simpleName}: ${t.message}); " +
                                    "continuing with the raw primal",
                                null,
                            )
                            fn
                        }
                        PhiCalculus.liftIfRegionBodies(coarsenedFwd)
                    } else fn
                    val jvpFn: DxirFunction = try {
                        DxirForwardTransform.apply(fwdPrimal)
                    } catch (t: Throwable) {
                        keptOriginal(
                            mc,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "DxirForwardTransform failed (${t::class.simpleName}: ${t.message}); " +
                                "falling back to the runtime tape\n${fn.pretty().trimEnd()}",
                            callLocation,
                        )
                        return transformed
                    }
                    // jvp(f): keep only the tangent returns (the second half —
                    // DxirForwardTransform emits values(m) ++ tangents(m)); the
                    // full body stays (tangents depend on the primal values).
                    val toSynthesiseRaw: DxirFunction = if (tangentOnly) {
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
                    // §0.4.416 — Phase B5 (customJvp): the forward transform's
                    // COARSENED arm clones the node as the VALUE stream (its
                    // tangent came from the tangent_body / primal_body splice),
                    // so a `jvp {}` body containing a custom-derivative
                    // call-form reaches synthesis with a COARSENED in it —
                    // which synthesis has no arm for. The tangent splice
                    // already happened, so what remains is pure value
                    // recomputation: decomposeCoarsened inlines `primal_body`
                    // in place, exactly the §0.4.415 grad-branch treatment.
                    // No-op when no COARSENED survived (every §0.4.372/403
                    // straight-line path, byte-identical).
                    val toSynthesise: DxirFunction = if (
                        toSynthesiseRaw.body.any { it is DxirOp && it.op == io.tlaloc.ir.OpKind.COARSENED }
                    ) {
                        try {
                            io.tlaloc.ir.recognizer.coarsener.decomposeCoarsened(toSynthesiseRaw)
                        } catch (t: Throwable) {
                            mc.report(
                                CompilerMessageSeverity.WARNING,
                                "Tlaloc IR extension: decomposeCoarsened failed on the jvp of " +
                                    "'${fn.name}' (${t::class.simpleName}: ${t.message}); " +
                                    "synthesising with the COARSENED intact (synthesis will " +
                                    "reject it loudly)",
                                null,
                            )
                            toSynthesiseRaw
                        }
                    } else {
                        toSynthesiseRaw
                    }
                    val replacement = synth.synthesise(toSynthesise, transformed, currentDeclarationParent!!)
                    if (replacement == null) {
                        val reason = synth.lastFailureReason ?: "(no specific gate stamped)"
                        keptOriginal(
                            mc,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "forward-transformed function falls outside the synthesis scope " +
                                "[$reason]\npost-forward jvp function:\n${toSynthesise.pretty().trimEnd()}",
                            callLocation,
                        )
                        return transformed
                    }
                    if (replacement.type != transformed.type) {
                        keptOriginal(
                            mc,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "synthesised type ${replacement.type} doesn't match call type " +
                                "${transformed.type}",
                            callLocation,
                        )
                        return transformed
                    }
                    // §0.4.514 — a SUCCESS dump: developer introspection, INFO and only with
                    // dumpLoweredIr (it was an unconditional WARNING, which failed -Werror builds).
                    if (options.dumpLoweredIr) {
                        mc.report(
                            CompilerMessageSeverity.INFO,
                            "Tlaloc lowered '${callableName}' to forward-mode dxir:\n" +
                            toSynthesise.pretty().trimEnd(),
                            null,
                        )
                    }
                    return replacement
                }

                // §0.4.394 — Phase B2: the assembly intrinsics. `jacobian` synthesises
                // the forward transform's seeded `jvp(x, dx) → dy`; `hessian` composes
                // forward-OVER-reverse into `hvp(x, v) → H·v` (the HVP composition
                // pinned at IR level since §0.4.361 — the runtime-extent adjoint ops
                // carry forward tangents but no VjpRules, which is exactly why the
                // Hessian is not reverse-over-reverse). The seeded 2-param lambda is
                // synthesised with an explicit callTypeOverride (the call site's own
                // type is the 1-param ASSEMBLED function) and handed to the matching
                // io.tlaloc.autograd.assemble*Forward helper, which loops over the
                // input's standard basis at RUNTIME — where the actual extents are
                // known — and stacks the [m, n] / [n, n] dense result. There is no
                // runtime-tape fallback for these (the `concat` precedent): a failed
                // synthesis keeps the original call, which throws pluginMissing loudly.
                // §0.4.406 — the two-argument forms `jacobian2` / `hessian2` ride the
                // same branch: the transforms are arity-agnostic (all primals, then all
                // tangents), so the generalisation is the gate (2 params), the harvested
                // types (A, B from the call type's first two args), the override type
                // (Function4 over primals ++ tangents; hessian2's seeded return is
                // Pair<A, B> — the two gradient tangents, boxed exactly as synthesis
                // boxes any 2-return function), and the assemble*2Forward helpers,
                // which loop basis vectors on EACH input with a zero tangent on the
                // other. hessian2 returns the FULL [(nx+nw), (nx+nw)] matrix over the
                // concatenated flat input (blocks are runtime-sized under sentinels).
                val assemblyIntrinsic = callableName == "jacobian" || callableName == "hessian" ||
                    callableName == "jacobian2" || callableName == "hessian2"
                if (assemblyIntrinsic) {
                    val arity = if (callableName.endsWith("2")) 2 else 1
                    val isJacobian = callableName.startsWith("jacobian")
                    if (fn.params.size != arity || fn.returns.size != 1) {
                        keptOriginal(
                            mc,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "$callableName v1 scope is $arity-param single-return " +
                                "(got ${fn.params.size} params, ${fn.returns.size} returns)",
                            callLocation,
                        )
                        return transformed
                    }
                    val callSiteType = transformed.type as? IrSimpleType
                    val primalTypes = (0 until arity).map {
                        callSiteType?.arguments?.getOrNull(it)?.typeOrNull
                    }
                    val fArgType = transformed.arguments.getOrNull(0)?.type as? IrSimpleType
                    val rType = fArgType?.arguments?.getOrNull(arity)?.typeOrNull
                    if (primalTypes.any { it == null } || rType == null) {
                        keptOriginal(
                            mc,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "could not harvest the input/output IrTypes from the " +
                                "$callableName call site (call type ${transformed.type})",
                            callLocation,
                        )
                        return transformed
                    }
                    val primals = primalTypes.map { it!! }
                    val seeded: DxirFunction = try {
                        if (isJacobian) {
                            DxirForwardTransform.apply(fn)
                        } else {
                            DxirForwardTransform.apply(DxirReverseTransform.apply(fn))
                        }
                    } catch (t: Throwable) {
                        keptOriginal(
                            mc,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "the seeded ${if (isJacobian) "forward" else "forward-over-reverse"} " +
                                "transform failed (${t::class.simpleName}: ${t.message})\n${fn.pretty().trimEnd()}",
                            callLocation,
                        )
                        return transformed
                    }
                    // Tangent-only: DxirForwardTransform emits values(m) ++ tangents(m);
                    // the assembly loop wants only the tangent half (the full body stays —
                    // tangents depend on the primal values).
                    val half = seeded.returns.size / 2
                    val tangentFnRaw = DxirFunction(
                        seeded.name,
                        seeded.params,
                        seeded.body,
                        seeded.returns.subList(half, seeded.returns.size),
                        seeded.meshes,
                    )
                    // §0.4.416 — Phase B5: `hessian` composes forward OVER
                    // reverse, and a customVjpJvp COARSENED can survive BOTH
                    // transforms (the reverse clone for value recomputation,
                    // then the forward value clone) into the seeded body —
                    // decompose it before synthesis, the same treatment as the
                    // grad and jvp branches. No-op on every pre-B5 path.
                    val tangentFn: DxirFunction = if (
                        tangentFnRaw.body.any { it is DxirOp && it.op == io.tlaloc.ir.OpKind.COARSENED }
                    ) {
                        try {
                            io.tlaloc.ir.recognizer.coarsener.decomposeCoarsened(tangentFnRaw)
                        } catch (t: Throwable) {
                            mc.report(
                                CompilerMessageSeverity.WARNING,
                                "Tlaloc IR extension: decomposeCoarsened failed on the seeded " +
                                    "body of '${fn.name}' (${t::class.simpleName}: ${t.message}); " +
                                    "synthesising with the COARSENED intact (synthesis will " +
                                    "reject it loudly)",
                                null,
                            )
                            tangentFnRaw
                        }
                    } else {
                        tangentFnRaw
                    }
                    // Seeded return type: jvp/jvp2's dy has f's return type; hvp's H·v
                    // has x's; hvp2's two gradient tangents box as Pair<A, B>.
                    val seedRet: org.jetbrains.kotlin.ir.types.IrType? = when {
                        isJacobian -> rType
                        arity == 1 -> primals[0]
                        else -> pluginContext.referenceClass(ClassId.fromString("kotlin/Pair"))
                            ?.typeWith(primals)
                    }
                    val overrideType: IrSimpleType? = seedRet?.let {
                        pluginContext.irBuiltIns.functionN(2 * arity).symbol
                            .typeWith(primals + primals + it)
                    }
                    if (overrideType == null) {
                        keptOriginal(
                            mc,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "could not build the seeded lambda's Function${2 * arity} type",
                            callLocation,
                        )
                        return transformed
                    }
                    val seededLambda = synth.synthesise(
                        tangentFn, transformed, currentDeclarationParent!!,
                        callTypeOverride = overrideType,
                    )
                    if (seededLambda == null) {
                        val reason = synth.lastFailureReason ?: "(no specific gate stamped)"
                        keptOriginal(
                            mc,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "seeded function falls outside the synthesis scope " +
                                "[$reason]\nseeded function:\n${tangentFn.pretty().trimEnd()}",
                            callLocation,
                        )
                        return transformed
                    }
                    val helperName = when (callableName) {
                        "jacobian" -> "assembleJacobianForward"
                        "hessian" -> "assembleHessianForward"
                        "jacobian2" -> "assembleJacobian2Forward"
                        else -> "assembleHessian2Forward"
                    }
                    val helperSym = pluginContext.referenceFunctions(
                        CallableId(FqName("io.tlaloc.autograd"), Name.identifier(helperName)),
                    ).singleOrNull()
                    if (helperSym == null) {
                        keptOriginal(
                            mc,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "io.tlaloc.autograd.$helperName not resolvable on the compile classpath",
                            callLocation,
                        )
                        return transformed
                    }
                    val assembled = IrCallImpl.fromSymbolOwner(
                        startOffset = transformed.startOffset,
                        endOffset = transformed.endOffset,
                        type = transformed.type,
                        symbol = helperSym,
                    )
                    // Helper type args mirror the declarations: assembleJacobian*Forward
                    // takes the primals + R; assembleHessian*Forward takes the primals only.
                    val helperTypeArgs = if (isJacobian) primals + rType else primals
                    helperTypeArgs.forEachIndexed { i, t ->
                        if (i < assembled.typeArguments.size) assembled.typeArguments[i] = t
                    }
                    assembled.arguments[0] = seededLambda
                    // §0.4.514 — a SUCCESS dump: developer introspection, INFO and only with
                    // dumpLoweredIr (it was an unconditional WARNING, which failed -Werror builds).
                    if (options.dumpLoweredIr) {
                        mc.report(
                            CompilerMessageSeverity.INFO,
                            "Tlaloc lowered '$callableName' to a seeded " +
                            "${if (isJacobian) "forward" else "forward-over-reverse"} " +
                            "pass + runtime basis assembly:\n${tangentFn.pretty().trimEnd()}",
                            null,
                        )
                    }
                    return assembled
                }

                // §0.4.412 — the REVERSE-assembled (tall) Jacobian, the m ≪ n tail
                // §0.4.394 recorded. Same shape as the assembly branch above but over
                // the OTHER seeded pass: `jacobianReverse` synthesises the §0.4.398
                // seeded reverse pullback `vjp_f(x, ȳ) → x̄` (one Jacobian ROW per
                // output-basis cotangent) and hands it to assembleJacobianReverse,
                // which loops the OUTPUT basis at runtime. The output extent m and
                // dims are unknowable before y exists — a basis cotangent needs y's
                // shape to be built at all — so the helper takes the ORIGINAL user
                // lambda too (passed through verbatim; its eager host execution IS
                // the primal) and runs it once: m + 1 passes versus jacobian's n.
                // Like the assembly branch, the call site's own type is the 1-param
                // ASSEMBLED function, so the seeded lambda synthesises under an
                // explicit callTypeOverride — Function2<A, R, A>, the pullback's
                // true type (the vjp branch below needs none because its call site
                // IS the seeded type). No runtime-tape fallback (the `concat`
                // precedent): a failed synthesis keeps the original call →
                // pluginMissing, loudly.
                // §0.4.424 — the two-argument form `jacobianReverse2` rides the same
                // branch: the seeded reverse transform and the §0.4.398 param rotation
                // are arity-agnostic, so the generalisation is the gate (2 params),
                // the harvested primal types (A, B from the call type's first two
                // args), the override type (Function3<A, B, R, Pair<A, B>> — the
                // 2-return pullback boxes as Pair exactly as vjp2's does), and the
                // assembleJacobianReverse2 helper, which writes each pullback pass's
                // (x̄, w̄) into row i of BOTH per-argument blocks.
                if (callableName == "jacobianReverse" || callableName == "jacobianReverse2") {
                    val jrArity = if (callableName.endsWith("2")) 2 else 1
                    if (fn.params.size != jrArity || fn.returns.size != 1) {
                        keptOriginal(
                            mc,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "$callableName v1 scope is $jrArity-param single-return " +
                                "(got ${fn.params.size} params, ${fn.returns.size} returns)",
                            callLocation,
                        )
                        return transformed
                    }
                    val callSiteType = transformed.type as? IrSimpleType
                    val jrPrimalTypes = (0 until jrArity).map {
                        callSiteType?.arguments?.getOrNull(it)?.typeOrNull
                    }
                    val fArg = transformed.arguments.getOrNull(0)
                    val rType = (fArg?.type as? IrSimpleType)?.arguments?.getOrNull(jrArity)?.typeOrNull
                    if (jrPrimalTypes.any { it == null } || rType == null || fArg == null) {
                        keptOriginal(
                            mc,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "could not harvest the input/output IrTypes from the " +
                                "$callableName call site (call type ${transformed.type})",
                            callLocation,
                        )
                        return transformed
                    }
                    val jrPrimals = jrPrimalTypes.map { it!! }
                    val seededGrad: DxirFunction = try {
                        DxirReverseTransform.apply(fn, seedAsParam = true)
                    } catch (t: Throwable) {
                        keptOriginal(
                            mc,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "the seeded reverse transform failed " +
                                "(${t::class.simpleName}: ${t.message})\n${fn.pretty().trimEnd()}",
                            callLocation,
                        )
                        return transformed
                    }
                    // (upstream, x) → (x, ȳ): rotate the upstream param to the back
                    // (the §0.4.398 metadata rotation — synthesis resolves body
                    // references by node id, params are positional metadata only).
                    val pullback = DxirFunction(
                        seededGrad.name,
                        seededGrad.params.drop(1) + seededGrad.params.first(),
                        seededGrad.body,
                        seededGrad.returns,
                        seededGrad.meshes,
                    )
                    // The pullback's return: x̄ at arity 1; the Pair-boxed (x̄, w̄)
                    // at arity 2 (synthesis boxes 2 returns as kotlin.Pair).
                    val pbReturn: org.jetbrains.kotlin.ir.types.IrType? = if (jrArity == 1) {
                        jrPrimals[0]
                    } else {
                        pluginContext.referenceClass(ClassId.fromString("kotlin/Pair"))
                            ?.typeWith(jrPrimals)
                    }
                    if (pbReturn == null) {
                        keptOriginal(
                            mc,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "could not build the seeded pullback's return type",
                            callLocation,
                        )
                        return transformed
                    }
                    val overrideType = pluginContext.irBuiltIns.functionN(jrArity + 1).symbol
                        .typeWith(jrPrimals + rType + pbReturn)
                    val seededLambda = synth.synthesise(
                        pullback, transformed, currentDeclarationParent!!,
                        callTypeOverride = overrideType,
                    )
                    if (seededLambda == null) {
                        val reason = synth.lastFailureReason ?: "(no specific gate stamped)"
                        keptOriginal(
                            mc,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "seeded pullback falls outside the synthesis scope " +
                                "[$reason]\npullback function:\n${pullback.pretty().trimEnd()}",
                            callLocation,
                        )
                        return transformed
                    }
                    val jrHelperName =
                        if (jrArity == 1) "assembleJacobianReverse" else "assembleJacobianReverse2"
                    val helperSym = pluginContext.referenceFunctions(
                        CallableId(FqName("io.tlaloc.autograd"), Name.identifier(jrHelperName)),
                    ).singleOrNull()
                    if (helperSym == null) {
                        keptOriginal(
                            mc,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "io.tlaloc.autograd.$jrHelperName not resolvable " +
                                "on the compile classpath",
                            callLocation,
                        )
                        return transformed
                    }
                    val assembled = IrCallImpl.fromSymbolOwner(
                        startOffset = transformed.startOffset,
                        endOffset = transformed.endOffset,
                        type = transformed.type,
                        symbol = helperSym,
                    )
                    (jrPrimals + rType).forEachIndexed { i, t ->
                        if (i < assembled.typeArguments.size) assembled.typeArguments[i] = t
                    }
                    assembled.arguments[0] = fArg
                    assembled.arguments[1] = seededLambda
                    // §0.4.514 — a SUCCESS dump: developer introspection, INFO and only with
                    // dumpLoweredIr (it was an unconditional WARNING, which failed -Werror builds).
                    if (options.dumpLoweredIr) {
                        mc.report(
                            CompilerMessageSeverity.INFO,
                            "Tlaloc lowered '$callableName' to a seeded reverse pullback + " +
                            "runtime output-basis assembly:\n${pullback.pretty().trimEnd()}",
                            null,
                        )
                    }
                    return assembled
                }

                // §0.4.398 — the seeded-cotangent intrinsics (audit item 10): `vjp` is
                // grad{} generalised to TENSOR-valued f — the pullback of a
                // user-supplied cotangent ȳ (of f's OUTPUT type) through f at x, in ONE
                // reverse pass. The reverse transform's seedAsParam mode (§0.4.33's
                // COARSENED gradient_body machinery) already produces exactly this
                // function — `(upstream, x) → x̄` — and §0.4.398 lifted its
                // scalar-return gate, so unlike `jacobian` there is NO runtime helper
                // and NO callTypeOverride: the call site's own type IS the 2-param
                // seeded function type, Function2<A, R, A> (Pair<R, A>-returning for
                // `valueAndVjp`, which rides the same transform's includeForward mode).
                // The only seam is parameter ORDER: the transform emits the upstream
                // first, the declared surface takes `(x, ȳ)` — and since synthesis
                // resolves body references by node id (params are positional metadata
                // only), reordering the params list is a pure metadata rotation.
                // No runtime-tape fallback (the `concat`/`jacobian` precedent): a
                // failed synthesis keeps the original call → pluginMissing, loudly.
                // §0.4.406 — the two-argument forms `vjp2` / `valueAndVjp2` ride the
                // same branch verbatim: `DxirReverseTransform(seedAsParam = true)` has
                // emitted `(upstream, *params) → (*grads)` for ANY arity since §0.4.33
                // (that IS the COARSENED gradient_body signature), the param rotation
                // below is already arity-agnostic (`drop(1) + first()`), and the call
                // site's own type is again the seeded function's type —
                // Function3<A, B, R, Pair<A, B>> (Triple-returning for valueAndVjp2).
                // Only the gate changes: 2 primal params for the "2" spellings.
                val vjpIntrinsic = callableName == "vjp" || callableName == "valueAndVjp" ||
                    callableName == "vjp2" || callableName == "valueAndVjp2"
                if (vjpIntrinsic) {
                    val vjpArity = if (callableName.endsWith("2")) 2 else 1
                    if (fn.params.size != vjpArity || fn.returns.size != 1) {
                        keptOriginal(
                            mc,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "$callableName v1 scope is $vjpArity-param single-return " +
                                "(got ${fn.params.size} params, ${fn.returns.size} returns)",
                            callLocation,
                        )
                        return transformed
                    }
                    val includeValue = callableName.startsWith("valueAnd")
                    val seededGrad: DxirFunction = try {
                        DxirReverseTransform.apply(
                            fn,
                            includeForward = includeValue,
                            seedAsParam = true,
                        )
                    } catch (t: Throwable) {
                        keptOriginal(
                            mc,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "the seeded reverse transform failed " +
                                "(${t::class.simpleName}: ${t.message})\n${fn.pretty().trimEnd()}",
                            callLocation,
                        )
                        return transformed
                    }
                    // (upstream, x) → (x, ȳ): rotate the upstream param to the back.
                    val pullback = DxirFunction(
                        seededGrad.name,
                        seededGrad.params.drop(1) + seededGrad.params.first(),
                        seededGrad.body,
                        seededGrad.returns,
                        seededGrad.meshes,
                    )
                    val replacement = synth.synthesise(pullback, transformed, currentDeclarationParent!!)
                    if (replacement == null) {
                        val reason = synth.lastFailureReason ?: "(no specific gate stamped)"
                        keptOriginal(
                            mc,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "seeded pullback falls outside the synthesis scope " +
                                "[$reason]\npullback function:\n${pullback.pretty().trimEnd()}",
                            callLocation,
                        )
                        return transformed
                    }
                    if (replacement.type != transformed.type) {
                        keptOriginal(
                            mc,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "synthesised type ${replacement.type} doesn't match call type " +
                                "${transformed.type}",
                            callLocation,
                        )
                        return transformed
                    }
                    // §0.4.514 — a SUCCESS dump: developer introspection, INFO and only with
                    // dumpLoweredIr (it was an unconditional WARNING, which failed -Werror builds).
                    if (options.dumpLoweredIr) {
                        mc.report(
                            CompilerMessageSeverity.INFO,
                            "Tlaloc lowered '$callableName' to a seeded reverse pullback:\n" +
                            pullback.pretty().trimEnd(),
                            null,
                        )
                    }
                    return replacement
                }

                val includeForward = callableName == "valueAndGrad" || callableName == "valueAndGrad2" ||
                    callableName == "valueAndGrad3"

                // §0.4.501 — bind each captured runtime value to the IR declaration the
                // user's own lambda closed over, BEFORE anything is transformed: a
                // capture we cannot bind must keep the original call, not synthesise a
                // gradient that reads the wrong value. The FIR side matched them by the
                // DECLARATION's source range; `valueDeclIndex` is that range's IR side.
                // Everything downstream of here is POSITIONAL — the reverse transform
                // drops the last N gradients, synthesis binds the last N params — so
                // the FIR side's claim that the captures ARE the last N params of the
                // lowered function is checked rather than trusted.
                if (lowered.captures.isNotEmpty() &&
                    fn.params.takeLast(lowered.captures.size).map { it.id } !=
                    lowered.captures.map { it.paramId }
                ) {
                    keptOriginal(
                        mc,
                        "Tlaloc IR extension kept original call for '${fn.name}' — the captured " +
                            "runtime values are not the trailing params of the lowered function " +
                            "(params=${fn.params.map { it.id }}, " +
                            "captures=${lowered.captures.map { it.paramId }})",
                        callLocation,
                    )
                    return transformed
                }
                val capturedBindings = ArrayList<IrValueDeclaration>(lowered.captures.size)
                for (c in lowered.captures) {
                    val candidates = valueDeclIndex[c.declStartOffset].orEmpty()
                        .filter { it.name.asString() == c.name }
                        // §0.4.501 — the IR phase's OWN scope check, independent of the
                        // FIR side's (which refuses a declaration inside the lambda's
                        // source range). A declaration the gradient can close over is
                        // written BEFORE the intrinsic call — Kotlin has no forward
                        // reference to a local — and the call's range covers the lambda,
                        // so this also excludes anything declared inside it. Two halves
                        // that must both agree before an `irGet` is emitted.
                        .filter { it.startOffset < transformed.startOffset }
                    val decl = candidates.singleOrNull()
                    if (decl == null) {
                        keptOriginal(
                            mc,
                            "Tlaloc IR extension kept original call for '${fn.name}' — the " +
                                "captured runtime value '${c.name}' was lowered as an input-only " +
                                "gradient parameter, but its declaration at source offset " +
                                "${c.declStartOffset} " +
                                (
                                    if (candidates.isEmpty()) "is not an IR local or parameter " +
                                        "of this file declared before the call"
                                    else "is ambiguous (${candidates.size} declarations share " +
                                        "that name and offset)"
                                    ) +
                                " — the gradient cannot be bound to it, so the call is left as " +
                                "written",
                            callLocation,
                        )
                        return transformed
                    }
                    capturedBindings += decl
                }

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
                    lifted, includeForward, mc, fn.name, callLocation,
                    // §0.4.501 — the captured params are INPUTS. No gradient is emitted
                    // for them, so `grad` still returns one gradient, `grad2` a Pair and
                    // `grad3` a Triple, whatever the lambda captured.
                    inputOnlyTrailingParams = lowered.captures.size,
                ) ?: run {
                    // §0.4.503 (Tier 3, item 3) — FIRST, the one refusal a user can act
                    // on. If the primal that was rejected still contains a loop and
                    // there is no symbolic engine on the classpath, the most likely
                    // cause of this failure is the missing optional dependency, and the
                    // dxir dump below is of no use to anybody who does not work on
                    // Tlaloc. Say so by name.
                    reportMissingSymbolicEngine(mc, fn.name, coarsened, symjaAvailable)
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

                // §0.4.415 — Phase B5 (customVjp): a COARSENED node can now SURVIVE
                // into the gradient function. Pre-B5 it never did — a machine
                // coarsening wraps the WHOLE body, so nothing downstream consumes
                // its result and no adjoint dereferences it. A customVjp node sits
                // MID-BODY (`f(x).sum()`), and under `grad {}`'s sentinels the
                // reduction adjoints carry their operand as a runtime shape
                // template, so the reverse transform clones the COARSENED into the
                // gradient body — where synthesis has no arm for it. The gradient
                // splice already happened (handleCoarsenedAdjoint consumed
                // `gradient_body` during the transform), so what remains is purely
                // the VALUE recomputation: decomposeCoarsened inlines `primal_body`
                // in place, exactly its Layer-4 CPU-baseline job. No-op when no
                // COARSENED survived (every pre-B5 path, byte-identical).
                val decomposed: DxirFunction = if (
                    toSynthesise.body.any { it is DxirOp && it.op == io.tlaloc.ir.OpKind.COARSENED }
                ) {
                    try {
                        io.tlaloc.ir.recognizer.coarsener.decomposeCoarsened(toSynthesise)
                    } catch (t: Throwable) {
                        mc.report(
                            CompilerMessageSeverity.WARNING,
                            "Tlaloc IR extension: decomposeCoarsened failed on '${fn.name}' " +
                                "(${t::class.simpleName}: ${t.message}); synthesising with the " +
                                "COARSENED intact (synthesis will reject it loudly)",
                            null,
                        )
                        toSynthesise
                    }
                } else {
                    toSynthesise
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
                    if (engine == null) decomposed
                    else try {
                        PhiCalculus.simplifyReturns(decomposed, engine)
                    } catch (t: Throwable) {
                        mc.report(
                            CompilerMessageSeverity.WARNING,
                            "Tlaloc IR extension: PhiCalculus.simplifyReturns threw on " +
                                "'${fn.name}' (${t::class.simpleName}: ${t.message}); " +
                                "synthesising the un-simplified gradient",
                            null,
                        )
                        decomposed
                    }
                } else decomposed

                val replacement = synth.synthesise(
                    simplified, transformed, currentDeclarationParent!!,
                    capturedBindings = capturedBindings,
                )
                if (replacement == null) {
                    val reason = synth.lastFailureReason ?: "(no specific gate stamped)"
                    keptOriginal(
                        mc,
                        "Tlaloc IR extension kept original call for '${fn.name}' — " +
                            "DxirFunction falls outside the scalar-primitive synthesis scope " +
                            "[$reason]\n" +
                            "post-coarsening primal:\n${coarsened.pretty().trimEnd()}\n" +
                            "post-lift dxir:\n${lifted.pretty().trimEnd()}\n" +
                            "post-SCT grad function:\n${simplified.pretty().trimEnd()}",
                        callLocation,
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
                    keptOriginal(
                        mc,
                        "Tlaloc IR extension kept original call for '${fn.name}' — " +
                            "synthesised type ${replacement.type} doesn't match call type " +
                            "${transformed.type} (forward-only scope)",
                        callLocation,
                    )
                    return transformed
                }
                // §0.4.450 — the readable-reverse dump: synthesis SUCCEEDED, so
                // `simplified` (the reverse-transformed gradient dxir, post-decompose,
                // post-optional-simplify) is exactly what the bytecode now computes.
                // Render THAT object as Kotlin source next to the lambda it
                // differentiates. See [dumpGradKotlinSource].
                if (dumpGradSource) {
                    dumpGradKotlinSource(
                        simplified, callableName, irFileForDump, transformed.startOffset, mc,
                    )
                }
                return replacement
            }
        }

        val filesWithIndexFailure = HashSet<String>()
        for (file in moduleFragment.files) {
            transformer.irFileForDump = file
            transformer.callLocation = null
            // §0.4.514 — the per-call guard lives in visitCall; this one covers the
            // per-file work around it (the declaration index), with the file as the
            // only location there is.
            try {
                transformer.valueDeclIndex = indexValueDeclarations(file)
            } catch (t: Throwable) {
                if (TlalocInternalErrors.mustRethrow(t)) throw t
                mc.report(
                    if (options.strictLowering) CompilerMessageSeverity.ERROR else CompilerMessageSeverity.WARNING,
                    TlalocInternalErrors.describe(t, options.strictLowering),
                    CompilerMessageLocation.create(file.fileEntry.name),
                )
                filesWithIndexFailure += TlalocLoweringHandoff.normalisePath(file.fileEntry.name)
                continue
            }
            file.transformChildren(transformer, null)
        }

        // §0.4.514a — every entry the FIR checker recorded is a call it lowered and
        // expects the IR phase to rewrite (Tracer-overload calls are never recorded).
        // One still here means the IR phase never matched that call — the two phases
        // disagreed on its file path or offsets — and the call is left as written, so
        // its fallback body throws at the first call. That is refused here by name
        // instead of surfacing at run time. Files whose declaration index already
        // failed are skipped: they carry their own internal error.
        val filesByPath = moduleFragment.files.associateBy {
            TlalocLoweringHandoff.normalisePath(it.fileEntry.name)
        }
        for (u in handoff.drainUnclaimed()) {
            if (u.file in filesWithIndexFailure) continue
            mc.report(
                if (options.strictLowering) CompilerMessageSeverity.ERROR else CompilerMessageSeverity.WARNING,
                TlalocInternalErrors.describeUnclaimed(u.fnName.removeSuffix("_body"), options.strictLowering),
                locationOf(filesByPath[u.file], u.startOffset) ?: CompilerMessageLocation.create(u.file),
            )
        }
        // §0.4.514 — this compilation's table only (it is a per-compilation instance),
        // so nothing another compilation in the same daemon recorded is touched.
        handoff.clear()
    }

    /**
     * The one reporting path for "kept original call": the call is left
     * unrewritten, so the `io.tlaloc.autograd` fallback body would throw at the first
     * call. That is a compile-time ERROR under [TlalocPluginOptions.strictLowering]
     * (the default), exactly like the FIR phase's LAMBDA_NOT_LOWERABLE, and a WARNING
     * only when the build opted out with `strictLowering=false`.
     */
    private fun keptOriginal(
        mc: MessageCollector,
        text: String,
        location: CompilerMessageSourceLocation?,
    ) {
        val versionHint = if ("not resolvable" in text || "no IR symbol" in text || "symbol not found" in text) {
            "\n" + VERSION_MISMATCH_HINT
        } else {
            ""
        }
        if (options.strictLowering) {
            mc.report(CompilerMessageSeverity.ERROR, text + versionHint + "\n" + KEPT_ORIGINAL_STRICT_HINT, location)
        } else {
            mc.report(CompilerMessageSeverity.WARNING, text + versionHint + "\n" + KEPT_ORIGINAL_LENIENT_NOTE, location)
        }
    }

    private fun locationOf(file: IrFile?, startOffset: Int): CompilerMessageSourceLocation? {
        val entry = file?.fileEntry ?: return null
        if (startOffset < 0) return CompilerMessageLocation.create(entry.name)
        return CompilerMessageLocation.create(
            entry.name,
            entry.getLineNumber(startOffset) + 1,
            entry.getColumnNumber(startOffset) + 1,
            null,
        )
    }

    /**
     * Index every value DECLARATION in [file] by its source start offset.
     *
     * Only `IrVariable` (a local `val` / `var`) and `IrValueParameter` (a function's or
     * lambda's parameter) are collected, because those are the two shapes a captured
     * runtime value is admitted in on the FIR side: both are read with a plain
     * `irGet`, which is exactly what the user's own lambda compiles to, and the JVM
     * backend's closure conversion then captures them for the synthesised lambda the
     * same way it would have captured them for the original one. A top-level or member
     * property is a getter CALL, not a value declaration, and the FIR side refuses it
     * by name.
     *
     * Several declarations can share a start offset (a destructuring declaration's
     * components, a synthesised parameter over the same source range), so the value is
     * a LIST and the caller refuses an ambiguous match rather than picking one.
     */
    private fun indexValueDeclarations(file: IrFile): Map<Int, List<IrValueDeclaration>> {
        val byOffset = HashMap<Int, MutableList<IrValueDeclaration>>()
        file.acceptVoid(object : IrVisitorVoid() {
            override fun visitElement(element: IrElement) {
                element.acceptChildrenVoid(this)
            }

            override fun visitVariable(declaration: IrVariable) {
                record(declaration)
                super.visitVariable(declaration)
            }

            override fun visitValueParameter(declaration: IrValueParameter) {
                record(declaration)
                super.visitValueParameter(declaration)
            }

            private fun record(declaration: IrValueDeclaration) {
                if (declaration.startOffset < 0) return
                byOffset.getOrPut(declaration.startOffset) { ArrayList(1) } += declaration
            }
        })
        return byOffset
    }

    /**
     * Runs [DxirReverseTransform.apply] guarded against its hard gates. Returns `null` if
     * the primal violates a gate (e.g., non-scalar return, regions, unsupported op kind);
     * the caller falls back to the original runtime call.
     *
     * Each failure is reported with the transform's own exception message, which names
     * the dxir node, op kind or requirement that failed.
     */
    /**
     * Refuse BY NAME when a body needed the computer
     * algebra system and Symja was not on the classpath.
     *
     * The predicate is deliberately narrow, and its two halves are both necessary.
     * *Symja absent* alone is not a problem: most programs never need a CAS, and
     * nobody who does not need it has to know Symja exists.
     * *A loop surviving coarsening* alone is not a problem either: the
     * LoopInvariant rewrite legitimately leaves nested WHILEs that the reverse
     * transform handles. It is the CONJUNCTION — a loop survived, the reverse
     * transform then rejected the primal, and there was no engine to close the loop
     * with — that makes the missing dependency the actionable cause. Anything
     * narrower would be silent; anything wider would blame Symja for other bugs.
     *
     * Severity follows [TlalocPluginOptions.strictLowering]: with
     * `strictLowering=false` this is a warning and the call fails at runtime instead,
     * as every other lowering failure does.
     */
    private fun reportMissingSymbolicEngine(
        mc: MessageCollector,
        fnName: String,
        coarsened: DxirFunction,
        symjaAvailable: Boolean,
    ) {
        val text = missingSymbolicEngineMessage(
            fnName = fnName,
            symjaAvailable = symjaAvailable,
            loopSurvivedCoarsening = PhiCalculus.containsLoop(coarsened),
        ) ?: return
        mc.report(
            if (options.strictLowering) CompilerMessageSeverity.ERROR else CompilerMessageSeverity.WARNING,
            text,
            null,
        )
    }

    private fun tryReverseTransform(
        primal: DxirFunction,
        includeForward: Boolean,
        mc: MessageCollector,
        fnName: String,
        location: CompilerMessageSourceLocation?,
        inputOnlyTrailingParams: Int = 0,
    ): DxirFunction? = try {
        DxirReverseTransform.apply(
            primal,
            includeForward,
            inputOnlyTrailingParams = inputOnlyTrailingParams,
        )
    } catch (t: IllegalArgumentException) {
        keptOriginal(
            mc,
            "Tlaloc IR extension kept original call for '$fnName' — DxirReverseTransform " +
                "rejected the dxir (${t::class.simpleName}: ${t.message})",
            location,
        )
        null
    } catch (t: IllegalStateException) {
        keptOriginal(
            mc,
            "Tlaloc IR extension kept original call for '$fnName' — DxirReverseTransform " +
                "rejected the dxir (${t::class.simpleName}: ${t.message})",
            location,
        )
        null
    }

    /**
     * Render the successfully synthesised gradient [gradFn] as Kotlin
     * source (the `toKotlinSource` host-twin renderer) and emit it as a
     * compiler INFO message headed by the intrinsic call's source location; when
     * [dumpGradSourceDir] is set, ALSO write it as a `.kt` file named after that
     * location. The dump renders the dxir handed to synthesis — the gradient the
     * user reads is the same function the synthesis compiles, by construction.
     *
     * The renderer's refusals stay LOUD here rather than fatal: a gradient it
     * cannot print faithfully — a tensor `grad {}` body whose types carry the -1
     * SENTINEL dims (a ranked-literal rendering of those would bake
     * sentinel-derived garbage), an op kind with no host twin, control
     * flow — dumps a SKIPPED message that repeats the refusal's named reason.
     * Compilation is never affected: the dump is a window, not a gate.
     */
    private fun dumpGradKotlinSource(
        gradFn: DxirFunction,
        callableName: String,
        irFile: IrFile?,
        startOffset: Int,
        mc: MessageCollector,
    ) {
        val entry = irFile?.fileEntry
        val line = if (entry != null && startOffset >= 0) entry.getLineNumber(startOffset) + 1 else 0
        val col = if (entry != null && startOffset >= 0) entry.getColumnNumber(startOffset) + 1 else 0
        val loc = if (entry != null) "${entry.name}:$line:$col" else "(unknown location)"
        val source = try {
            gradFn.toKotlinSource()
        } catch (t: Throwable) {
            mc.report(
                CompilerMessageSeverity.INFO,
                "Tlaloc grad source dump SKIPPED for '$callableName' at $loc — the gradient " +
                    "compiled, but it cannot be printed as Kotlin source: ${t.message}",
                null,
            )
            return
        }
        mc.report(
            CompilerMessageSeverity.INFO,
            "Tlaloc grad source for '$callableName' at $loc — the reverse-transformed " +
                "gradient rendered as Kotlin, the SAME dxir function the synthesis compiles:\n" +
                source.trimEnd(),
            null,
        )
        val dirRaw = dumpGradSourceDir ?: return
        try {
            val dir = File(dirRaw).absoluteFile
            dir.mkdirs()
            val fileBase = entry?.name?.substringAfterLast('/') ?: "unknown"
            val base = "${fileBase}_${line}_${col}_$callableName"
                .map { if (it.isLetterOrDigit() || it == '_') it else '_' }
                .joinToString("")
            File(dir, "$base.kt").writeText(source)
        } catch (t: Throwable) {
            mc.report(
                CompilerMessageSeverity.WARNING,
                "Tlaloc grad source dump: failed to write the .kt file for '$callableName' " +
                    "at $loc under '$dirRaw' (${t::class.simpleName}: ${t.message})",
                null,
            )
        }
    }

    /**
     * Resolve the [CoarseningCache] impl for this compilation. Reads the
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

        /**
         * The refusal text, as a pure function of the three
         * facts that decide it, so the wording and the predicate are both certified by
         * `SymjaOptionalDependencyTest` without needing a Symja-free compiler run.
         *
         * Returns null when there is nothing to say: Symja present, or no loop left in
         * the primal. See [reportMissingSymbolicEngine] for why both halves matter.
         */
        internal fun missingSymbolicEngineMessage(
            fnName: String,
            symjaAvailable: Boolean,
            loopSurvivedCoarsening: Boolean,
        ): String? {
            if (symjaAvailable || !loopSurvivedCoarsening) return null
            return SymbolicEngines.absenceMessage(
                "Tlaloc cannot build a gradient for '$fnName': the phi-calculus coarsener " +
                    "left a LOOP in the primal (a dxir WHILE op) and the reverse transform " +
                    "then rejected it. Closing a loop into a closed form is what the " +
                    "engine-backed corollaries C6-C9 do",
            )
        }
        /** Appended to every strict "kept original call" error. */
        internal const val KEPT_ORIGINAL_STRICT_HINT: String =
            "The call cannot be compiled to a gradient, so the build stops here instead of " +
                "throwing at the first call. Rewrite the body within the supported surface " +
                "(docs/GETTING_STARTED.md), or use the Tracer-capture API " +
                "(io.tlaloc.autograd.gradWithScalars) for this call. " +
                "`tlaloc { strictLowering.set(false) }` (or -P plugin:io.tlaloc.plugin:strictLowering=false) " +
                "turns this into a warning, " +
                "and the call then throws IllegalStateException when it runs."

        /** Appended to every "kept original call" warning under
         * `strictLowering=false`. */
        internal const val KEPT_ORIGINAL_LENIENT_NOTE: String =
            "The call is left as written (strictLowering=false), so it throws " +
                "IllegalStateException when it runs."

        /** Appended when the failure is a library symbol the plugin could not
         * find: the usual cause is a library/plugin version mismatch. */
        internal const val VERSION_MISMATCH_HINT: String =
            "If the io.tlaloc libraries on the compile classpath are a different version from " +
                "the Tlaloc compiler plugin, use the same version for both."

        private val INTRINSIC_NAMES: Set<String> = setOf(
            "grad", "grad2", "valueAndGrad", "valueAndGrad2",
            // §0.4.424 — the three-argument reverse spellings (the transform and
            // synthesis were arity-agnostic all along; the gate is the surface).
            "grad3", "valueAndGrad3",
            // §0.4.372 — forward-mode (Phase B1). §0.4.387 — its two-argument forms.
            "jvp", "valueAndJvp", "jvp2", "valueAndJvp2",
            // §0.4.394 — Phase B2: the assembly intrinsics. §0.4.406 — their
            // two-argument forms.
            "jacobian", "hessian", "jacobian2", "hessian2",
            // §0.4.412 — the reverse-assembled (tall) Jacobian. §0.4.424 — its
            // two-argument form.
            "jacobianReverse", "jacobianReverse2",
            // §0.4.398 — the seeded-cotangent user surface. §0.4.406 — its
            // two-argument forms.
            "vjp", "valueAndVjp", "vjp2", "valueAndVjp2",
        )

        /**
         * The intrinsics whose synthesised function can carry a captured
         * runtime value as a trailing input-only parameter. Mirrors
         * `TlalocIntrinsicCallChecker.captureCarryingIntrinsics`, which is the gate
         * that decides whether the FIR lowering promotes a capture at all; this set
         * is the IR phase's independent check that nothing else ever sees one.
         */
        private val CAPTURE_CARRYING_INTRINSICS: Set<String> = setOf(
            "grad", "grad2", "grad3", "valueAndGrad", "valueAndGrad2", "valueAndGrad3",
        )

        /**
         * System property name the plugin reads to decide where to write cache entries.
         * Unset = caching disabled. See [buildCoarseningCache] for accepted values.
         */
        const val CACHE_DIR_PROPERTY: String = "tlaloc.cache.dir"

        /**
         * System property toggling SOI-based coarsening.
         * When set to "true", `grad { ... }` calls route through
         * [PhiCalculus.coarsenFunction] (wraps the primal in an OpKind.COARSENED op
         * with pre-computed gradient_body). `valueAndGrad` + multi-return primals
         * continue to use the existing [PhiCalculus.apply] path because SOI
         * coarsening does not handle them.
         */
        const val SOI_ENABLED_PROPERTY: String = "tlaloc.soi.enabled"

        /**
         * System property overriding the SOI size-limit `L` when coarsening is
         * enabled. Must be a positive integer. Unset / invalid values fall back to the
         * paper-informed default of 50. `LSweepTest` in `:ir/jvmTest` sweeps
         * L ∈ {5, 25, 50, 100, 200} across a benchmark suite to inform tuning.
         */
        const val SOI_SIZE_LIMIT_PROPERTY: String = "tlaloc.soi.size.limit"

        /**
         * System property toggling [PhiCalculus.simplifyReturns]
         * over each gradient `DxirFunction` after [DxirReverseTransform.apply]. When set
         * to "true", the IR extension lifts each return expression to a Symja `SymExpr`,
         * runs `Simplify`, and lowers back. Unset / "false" / anything else: gradient
         * body passes through unchanged. The pass has internal bail-out semantics, so a
         * `true` setting is safe for gradient bodies outside the lift surface — they
         * synthesise un-simplified rather than failing.
         */
        const val SIMPLIFY_ENABLED_PROPERTY: String = "tlaloc.simplify.enabled"

        /**
         * CAS version string — bumps invalidate cached entries. Tied to
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
