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
import org.jetbrains.kotlin.ir.expressions.impl.IrCallImpl
import org.jetbrains.kotlin.ir.expressions.impl.fromSymbolOwner
import org.jetbrains.kotlin.ir.symbols.UnsafeDuringIrConstructionAPI
import org.jetbrains.kotlin.ir.types.IrSimpleType
import org.jetbrains.kotlin.ir.types.typeOrNull
import org.jetbrains.kotlin.ir.types.typeWith
import org.jetbrains.kotlin.ir.util.callableId
import org.jetbrains.kotlin.name.CallableId
import org.jetbrains.kotlin.name.ClassId
import org.jetbrains.kotlin.name.FqName
import org.jetbrains.kotlin.name.Name

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
                        mc.report(
                            CompilerMessageSeverity.WARNING,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "$callableName v1 scope is $arity-param single-return " +
                                "(got ${fn.params.size} params, ${fn.returns.size} returns)",
                            null,
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
                        mc.report(
                            CompilerMessageSeverity.WARNING,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "could not harvest the input/output IrTypes from the " +
                                "$callableName call site (call type ${transformed.type})",
                            null,
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
                        mc.report(
                            CompilerMessageSeverity.WARNING,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "the seeded ${if (isJacobian) "forward" else "forward-over-reverse"} " +
                                "transform failed (${t::class.simpleName}: ${t.message})\n${fn.pretty().trimEnd()}",
                            null,
                        )
                        return transformed
                    }
                    // Tangent-only: DxirForwardTransform emits values(m) ++ tangents(m);
                    // the assembly loop wants only the tangent half (the full body stays —
                    // tangents depend on the primal values).
                    val half = seeded.returns.size / 2
                    val tangentFn = DxirFunction(
                        seeded.name,
                        seeded.params,
                        seeded.body,
                        seeded.returns.subList(half, seeded.returns.size),
                        seeded.meshes,
                    )
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
                        mc.report(
                            CompilerMessageSeverity.WARNING,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "could not build the seeded lambda's Function${2 * arity} type",
                            null,
                        )
                        return transformed
                    }
                    val seededLambda = synth.synthesise(
                        tangentFn, transformed, currentDeclarationParent!!,
                        callTypeOverride = overrideType,
                    )
                    if (seededLambda == null) {
                        val reason = synth.lastFailureReason ?: "(no specific gate stamped)"
                        mc.report(
                            CompilerMessageSeverity.WARNING,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "seeded function falls outside the synthesis scope " +
                                "[$reason]\nseeded function:\n${tangentFn.pretty().trimEnd()}",
                            null,
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
                        mc.report(
                            CompilerMessageSeverity.WARNING,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "io.tlaloc.autograd.$helperName not resolvable on the compile classpath",
                            null,
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
                    mc.report(
                        CompilerMessageSeverity.WARNING,
                        "Tlaloc lowered '$callableName' to a seeded " +
                            "${if (isJacobian) "forward" else "forward-over-reverse"} " +
                            "pass + runtime basis assembly:\n${tangentFn.pretty().trimEnd()}",
                        null,
                    )
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
                if (callableName == "jacobianReverse") {
                    if (fn.params.size != 1 || fn.returns.size != 1) {
                        mc.report(
                            CompilerMessageSeverity.WARNING,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "jacobianReverse v1 scope is 1-param single-return " +
                                "(got ${fn.params.size} params, ${fn.returns.size} returns)",
                            null,
                        )
                        return transformed
                    }
                    val callSiteType = transformed.type as? IrSimpleType
                    val aType = callSiteType?.arguments?.getOrNull(0)?.typeOrNull
                    val fArg = transformed.arguments.getOrNull(0)
                    val rType = (fArg?.type as? IrSimpleType)?.arguments?.getOrNull(1)?.typeOrNull
                    if (aType == null || rType == null || fArg == null) {
                        mc.report(
                            CompilerMessageSeverity.WARNING,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "could not harvest the input/output IrTypes from the " +
                                "jacobianReverse call site (call type ${transformed.type})",
                            null,
                        )
                        return transformed
                    }
                    val seededGrad: DxirFunction = try {
                        DxirReverseTransform.apply(fn, seedAsParam = true)
                    } catch (t: Throwable) {
                        mc.report(
                            CompilerMessageSeverity.WARNING,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "the seeded reverse transform failed " +
                                "(${t::class.simpleName}: ${t.message})\n${fn.pretty().trimEnd()}",
                            null,
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
                    val overrideType = pluginContext.irBuiltIns.functionN(2).symbol
                        .typeWith(listOf(aType, rType, aType))
                    val seededLambda = synth.synthesise(
                        pullback, transformed, currentDeclarationParent!!,
                        callTypeOverride = overrideType,
                    )
                    if (seededLambda == null) {
                        val reason = synth.lastFailureReason ?: "(no specific gate stamped)"
                        mc.report(
                            CompilerMessageSeverity.WARNING,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "seeded pullback falls outside the synthesis scope " +
                                "[$reason]\npullback function:\n${pullback.pretty().trimEnd()}",
                            null,
                        )
                        return transformed
                    }
                    val helperSym = pluginContext.referenceFunctions(
                        CallableId(FqName("io.tlaloc.autograd"), Name.identifier("assembleJacobianReverse")),
                    ).singleOrNull()
                    if (helperSym == null) {
                        mc.report(
                            CompilerMessageSeverity.WARNING,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "io.tlaloc.autograd.assembleJacobianReverse not resolvable " +
                                "on the compile classpath",
                            null,
                        )
                        return transformed
                    }
                    val assembled = IrCallImpl.fromSymbolOwner(
                        startOffset = transformed.startOffset,
                        endOffset = transformed.endOffset,
                        type = transformed.type,
                        symbol = helperSym,
                    )
                    listOf(aType, rType).forEachIndexed { i, t ->
                        if (i < assembled.typeArguments.size) assembled.typeArguments[i] = t
                    }
                    assembled.arguments[0] = fArg
                    assembled.arguments[1] = seededLambda
                    mc.report(
                        CompilerMessageSeverity.WARNING,
                        "Tlaloc lowered 'jacobianReverse' to a seeded reverse pullback + " +
                            "runtime output-basis assembly:\n${pullback.pretty().trimEnd()}",
                        null,
                    )
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
                        mc.report(
                            CompilerMessageSeverity.WARNING,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "$callableName v1 scope is $vjpArity-param single-return " +
                                "(got ${fn.params.size} params, ${fn.returns.size} returns)",
                            null,
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
                        mc.report(
                            CompilerMessageSeverity.WARNING,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "the seeded reverse transform failed " +
                                "(${t::class.simpleName}: ${t.message})\n${fn.pretty().trimEnd()}",
                            null,
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
                        mc.report(
                            CompilerMessageSeverity.WARNING,
                            "Tlaloc IR extension kept original call for '${fn.name}' — " +
                                "seeded pullback falls outside the synthesis scope " +
                                "[$reason]\npullback function:\n${pullback.pretty().trimEnd()}",
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
                        "Tlaloc lowered '$callableName' to a seeded reverse pullback:\n" +
                            pullback.pretty().trimEnd(),
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
            // §0.4.394 — Phase B2: the assembly intrinsics. §0.4.406 — their
            // two-argument forms.
            "jacobian", "hessian", "jacobian2", "hessian2",
            // §0.4.412 — the reverse-assembled (tall) Jacobian.
            "jacobianReverse",
            // §0.4.398 — the seeded-cotangent user surface. §0.4.406 — its
            // two-argument forms.
            "vjp", "valueAndVjp", "vjp2", "valueAndVjp2",
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
