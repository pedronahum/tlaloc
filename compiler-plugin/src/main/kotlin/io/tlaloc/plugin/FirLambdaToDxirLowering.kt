package io.tlaloc.plugin

import io.tlaloc.core.Bool
import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.core.I32
import io.tlaloc.core.I64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirEmitter
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirRegion
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import org.jetbrains.kotlin.KtFakeSourceElementKind
import org.jetbrains.kotlin.fir.declarations.FirAnonymousFunction
import org.jetbrains.kotlin.fir.declarations.FirProperty
import org.jetbrains.kotlin.fir.expressions.FirBlock
import org.jetbrains.kotlin.fir.expressions.FirBreakExpression
import org.jetbrains.kotlin.fir.expressions.FirComparisonExpression
import org.jetbrains.kotlin.fir.expressions.FirExpression
import org.jetbrains.kotlin.fir.expressions.FirFunctionCall
import org.jetbrains.kotlin.fir.expressions.FirLiteralExpression
import org.jetbrains.kotlin.types.ConstantValueKind
import org.jetbrains.kotlin.fir.expressions.FirOperation
import org.jetbrains.kotlin.fir.expressions.FirPropertyAccessExpression
import org.jetbrains.kotlin.fir.expressions.FirReturnExpression
import org.jetbrains.kotlin.fir.expressions.FirVariableAssignment
import org.jetbrains.kotlin.fir.expressions.FirWhenBranch
import org.jetbrains.kotlin.fir.expressions.FirWhenExpression
import org.jetbrains.kotlin.fir.expressions.FirWhileLoop
import org.jetbrains.kotlin.fir.expressions.impl.FirElseIfTrueCondition
import org.jetbrains.kotlin.fir.references.toResolvedCallableSymbol
import org.jetbrains.kotlin.fir.symbols.impl.FirPropertySymbol
import org.jetbrains.kotlin.fir.symbols.impl.FirValueParameterSymbol
import org.jetbrains.kotlin.fir.types.ConeKotlinType
import org.jetbrains.kotlin.fir.types.classId
import org.jetbrains.kotlin.fir.types.coneType
import org.jetbrains.kotlin.fir.types.type

/**
 * Walks a resolved [FirAnonymousFunction] and produces a [DxirFunction], or a [Result.Failure]
 * with a human-readable reason. Supported surface:
 *
 * - `Float`/`Double`/`Int`/`Long` + `DScalar` + `DTensor<ScalarShape|Rank1<_>, …>` params.
 * - Binary `+ - * /` and unary `-`; numeric literals; parameter / local-val references.
 * - `:core` scalar + `:core/ops` tensor unary/reduction helpers (relu, sigmoid, tanh, exp,
 *   log, sqrt, neg, sum).
 * - §0.4.24 (B.4a): top-level `if (cond) … else …` whose condition is `a > b` / `a < b`
 *   between two lowerable scalar numeric operands. Each branch body is itself a lowerable
 *   block (no nested when/if, no tensor ops inside branches for this slice).
 * - §0.4.25 (B.4b): `for (i in 0 until N)` loops over a concrete integer literal range,
 *   where the body mutates exactly one outer-scope `var` via scalar arithmetic that does
 *   NOT read the loop index `i`. Emits [OpKind.WHILE] in the canonical C5-pattern shape
 *   (`operands = [carried, counter=0i32]`, counter cond `STEP(SUB(n, counter))`, counter
 *   step `ADD(counter, 1i32)`) so `PhiCalculus.apply` can close it via C5/C6 before SCT.
 *   `var d = x` inside the lambda body + simple `d = <expr>` assignments are supported.
 *
 * No raw `while`/`do-while`, no multi-branch `when`, no user-defined functions, no `break`
 * / `continue`, no nested loops, no body references to the loop index. Anything outside
 * that surface throws [LoweringException], which the caller surfaces as
 * `TLALOC_LAMBDA_UNSUPPORTED`.
 */
object FirLambdaToDxirLowering {

    sealed class Result {
        data class Success(val fn: DxirFunction) : Result()
        data class Failure(val reason: String) : Result()
    }

    fun lower(name: String, anonFn: FirAnonymousFunction): Result {
        val env = HashMap<Any, DxirNode>()
        return try {
            val fn = DxirBuilder.function(name) {
                for (firParam in anonFn.valueParameters) {
                    val ct = firParam.returnTypeRef.coneType
                    val paramType = resolveParamType(ct)
                        ?: throw LoweringException(
                            "lambda param '${firParam.name}' has unsupported type ${ct.renderForError()}",
                        )
                    val dp = param(firParam.name.asString(), paramType)
                    env[firParam.symbol] = dp
                }
                val body = anonFn.body ?: throw LoweringException("lambda has no body")
                val returnNode = lowerBlock(body, env, this)
                listOf(returnNode)
            }
            Result.Success(fn)
        } catch (e: LoweringException) {
            Result.Failure(e.message ?: "unknown")
        }
    }

    private class LoweringException(message: String) : RuntimeException(message)

    private fun lowerBlock(
        block: FirBlock,
        env: MutableMap<Any, DxirNode>,
        emitter: DxirEmitter,
    ): DxirNode {
        val statements = block.statements
        if (statements.isEmpty()) throw LoweringException("empty lambda body")
        var last: DxirNode? = null
        for ((i, stmt) in statements.withIndex()) {
            last = lowerStatement(stmt, env, emitter, isLast = i == statements.lastIndex) ?: last
        }
        return last ?: throw LoweringException("lambda body has no return expression")
    }

    /**
     * Single statement dispatch shared by [lowerBlock] and B.4b's for-loop body processor.
     * Returns the statement's yielded SSA value when the statement is a trailing expression
     * (per Kotlin's block-as-expression rules), or null for "side-effect-only" forms like
     * [FirVariableAssignment] / desugared for-loops. Callers track the final non-null
     * return as the block's result.
     */
    private fun lowerStatement(
        stmt: Any,
        env: MutableMap<Any, DxirNode>,
        emitter: DxirEmitter,
        isLast: Boolean,
    ): DxirNode? = when (stmt) {
        is FirProperty -> {
            val init = stmt.initializer
                ?: throw LoweringException("${if (stmt.isVar) "var" else "val"} '${stmt.name}' has no initializer")
            val node = lowerExpr(init, env, emitter)
            env[stmt.symbol] = node
            if (isLast) node else null
        }
        is FirVariableAssignment -> {
            val targetSym = resolveAssignmentTarget(stmt)
                ?: throw LoweringException("assignment target is not a local property")
            if (targetSym !in env) {
                throw LoweringException(
                    "assignment to variable '${targetSym.name}' declared outside the lambda",
                )
            }
            val newValue = lowerExpr(stmt.rValue, env, emitter)
            env[targetSym] = newValue
            // Variable assignment is a unit-typed statement; don't surface it as a block return.
            null
        }
        is FirReturnExpression -> lowerExpr(stmt.result, env, emitter)
        is FirWhileLoop -> {
            lowerRawWhileLoop(stmt, env, emitter)
            null
        }
        is FirBlock -> {
            if (stmt.source?.kind == KtFakeSourceElementKind.DesugaredForLoop) {
                lowerDesugaredForLoop(stmt, env, emitter)
                null
            } else {
                // Plain [FirBlock]s appear as `FirSingleExpressionBlock` around a braceless
                // for-loop body (`for (i in ...) d = d * 2` → block wrapping the assignment)
                // and around explicit `{ }` blocks. Flatten transparently — the lambda
                // scope is already Kotlin's block scope, and var bindings handle shadowing
                // via SSA id freshness.
                var last: DxirNode? = null
                val inner = stmt.statements
                for ((i, s) in inner.withIndex()) {
                    last = lowerStatement(s, env, emitter, isLast && i == inner.lastIndex) ?: last
                }
                last
            }
        }
        is FirExpression -> {
            val node = lowerExpr(stmt, env, emitter)
            if (isLast) node else null
        }
        else -> throw LoweringException(
            "unsupported statement kind ${stmt::class.simpleName}",
        )
    }

    private fun resolveAssignmentTarget(stmt: FirVariableAssignment): FirPropertySymbol? {
        val lvalue = stmt.lValue
        if (lvalue !is FirPropertyAccessExpression) return null
        return lvalue.calleeReference.toResolvedCallableSymbol() as? FirPropertySymbol
    }

    private fun lowerExpr(
        expr: FirExpression,
        env: MutableMap<Any, DxirNode>,
        emitter: DxirEmitter,
    ): DxirNode = when (expr) {
        is FirLiteralExpression -> lowerLiteral(expr, emitter)
        is FirPropertyAccessExpression -> lookupReference(expr, env)
        is FirFunctionCall -> lowerCall(expr, env, emitter)
        is FirReturnExpression -> lowerExpr(expr.result, env, emitter)
        is FirWhenExpression -> lowerWhen(expr, env, emitter)
        else -> throw LoweringException("unsupported expression ${expr::class.simpleName}")
    }

    private fun lowerLiteral(expr: FirLiteralExpression, emitter: DxirEmitter): DxirNode {
        val value = expr.value ?: throw LoweringException("null literal is not a numeric scalar")
        // §0.4.51 — FIR stores all integer literals' value as kotlin.Long regardless of
        // the Kotlin source type. Look at `expr.kind` to distinguish `0` (Int) from `0L`
        // (Long). Without this, `var k = 0` lowered to `const 0 : i64`, which then
        // cascaded through the raw-while counter into the wrong STEP/SUB dtype and broke
        // gradient correctness for kernels that combine while-loops with nested for-loops.
        return when (expr.kind) {
            ConstantValueKind.Int, ConstantValueKind.IntegerLiteral -> {
                val intValue = (value as Number).toInt()
                emitter.const(intValue, DxirType(I32, emptyList()))
            }
            else -> {
                val dtype = literalDType(value)
                    ?: throw LoweringException("unsupported literal type ${value::class.simpleName}")
                emitter.const(value, DxirType(dtype, emptyList()))
            }
        }
    }

    private fun lookupReference(
        expr: FirPropertyAccessExpression,
        env: Map<Any, DxirNode>,
    ): DxirNode {
        val sym = expr.calleeReference.toResolvedCallableSymbol()
            ?: throw LoweringException("unresolved property access")
        return when (sym) {
            is FirValueParameterSymbol -> env[sym]
            is FirPropertySymbol -> env[sym]
            else -> null
        } ?: throw LoweringException(
            "reference to symbol outside the lowering scope: ${sym.callableId}",
        )
    }

    /**
     * §0.4.24 — `if (cond) a else b` (a [FirWhenExpression] with an `else`-branch synthetic
     * true-condition) lowers to [OpKind.IF] with a Bool-scalar predicate + two regions.
     * The condition shape that B.4a accepts is a [FirComparisonExpression] whose operation
     * is [FirOperation.GT] or [FirOperation.LT]; the operands must be lowerable as scalars.
     * Other when shapes (subject form, multi-branch, `>=` / `<=` / `==`) throw.
     *
     * The branch bodies are [FirBlock]s lowered via [lowerBlock]; they share the same
     * `env` (val bindings introduced outside flow into branches) but ops emitted inside
     * each branch land in the respective [DxirRegion] body via a [DxirRegionBuilder]. The
     * branch's yielded value becomes the region's single terminator — resolved through
     * [lowerBlock]'s trailing expression.
     *
     * §0.4.162 — nested when-expressions (if/when inside another region body, e.g.
     * inside a WHILE body or inside another IF branch) are now supported. The
     * dispatch on `emitter` (DxirBuilder vs DxirRegionBuilder) mirrors the pattern
     * `PhiCalculus.cloneRegion` uses: both subclasses expose `region { … }` and
     * `ifOp(…)` with the same signatures, but those aren't on the `DxirEmitter`
     * interface, so we type-switch at call sites. Required by HMC Phase 3's
     * numerical-stability mask `if (-Xβ_i > 80) -Xβ_i else log(1 + exp(-Xβ_i))`,
     * which lives inside a `for`-loop body (lowered to a WHILE body region).
     */
    private fun lowerWhen(
        expr: FirWhenExpression,
        env: MutableMap<Any, DxirNode>,
        emitter: DxirEmitter,
    ): DxirNode {
        if (expr.subjectVariable != null) {
            throw LoweringException("when-expression with subject not supported (B.4a scope)")
        }
        val branches = expr.branches
        if (branches.size != 2) {
            throw LoweringException(
                "when-expression must have exactly 2 branches (then + else) for B.4a; got ${branches.size}",
            )
        }
        val thenBranch = branches[0]
        val elseBranch = branches[1]
        if (elseBranch.condition !is FirElseIfTrueCondition) {
            throw LoweringException(
                "when-expression second branch must be the else arm (FirElseIfTrueCondition)",
            )
        }
        val pred = lowerPredicate(thenBranch.condition, env, emitter)
        // Both DxirBuilder and DxirRegionBuilder expose `region { … }` with the same
        // signature, but the method isn't on DxirEmitter. Build via a helper closure
        // that dispatches on the runtime type.
        fun emitterRegion(block: io.tlaloc.ir.DxirRegionBuilder.() -> Unit): DxirRegion = when (emitter) {
            is DxirBuilder -> emitter.region(block)
            is io.tlaloc.ir.DxirRegionBuilder -> emitter.region(block)
            else -> throw LoweringException(
                "lowerWhen: unsupported emitter type ${emitter::class.simpleName}",
            )
        }
        val thenRegion: DxirRegion = emitterRegion {
            val node = lowerBlock(thenBranch.result, env, this)
            yields(node)
        }
        val elseRegion: DxirRegion = emitterRegion {
            val node = lowerBlock(elseBranch.result, env, this)
            yields(node)
        }
        val resultType = thenRegion.blocks.single().terminator.single().type
        return when (emitter) {
            is DxirBuilder -> emitter.ifOp(
                cond = pred,
                types = listOf(resultType),
                thenRegion = thenRegion,
                elseRegion = elseRegion,
            )
            is io.tlaloc.ir.DxirRegionBuilder -> emitter.ifOp(
                cond = pred,
                types = listOf(resultType),
                thenRegion = thenRegion,
                elseRegion = elseRegion,
            )
            else -> throw LoweringException(
                "lowerWhen: unsupported emitter type ${emitter::class.simpleName}",
            )
        }
    }

    /**
     * §0.4.50 — lower a raw `while (cond) { body }` to [OpKind.WHILE]. Unlike the
     * desugared-for-loop path, there is no synthetic counter: carried vars are exactly
     * the user-scope mutations in the body, and the condition is a user expression
     * (currently a [FirComparisonExpression] accepted by [lowerPredicate]).
     *
     * Shape:
     *   inits   = [env[v_k] for each mutated outer-scope var v_k, in first-mutation order]
     *   cond    = region with one arg per carried var; body rebinds env[v_k] = args[k]
     *             and yields the lowered predicate (Bool)
     *   body    = region with one arg per carried var; body rebinds env[v_k] = args[k],
     *             lowers user stmts, yields new env[v_k] for each carried var
     * Post-loop: env[v_k] = w.result(k).
     *
     * Constraints (first cut): comparison predicate (GT/LT), no break, no nested when
     * reading carried vars across the cond region, only scalar-typed carried vars.
     */
    private fun lowerRawWhileLoop(
        loop: FirWhileLoop,
        env: MutableMap<Any, DxirNode>,
        emitter: DxirEmitter,
    ) {
        val allStatements = loop.block.statements
        // §0.4.50 Gap 3 — recognise a trailing `if (cond) break` and hoist `cond` into
        // the WHILE's cond region. Supported shape: FirWhenExpression as the last body
        // stmt with a 2-branch structure where the then-arm's block contains exactly
        // one statement, a FirBreakExpression targeting this loop. Anything else
        // (break elsewhere, continue, labeled break) is out of scope.
        val breakPattern = detectTrailingBreak(allStatements, loop)
        val bodyStatements = if (breakPattern != null) allStatements.dropLast(1) else allStatements
        val breakCond = breakPattern?.breakCond

        val mutatedTargets = collectMutatedTargets(bodyStatements).toList()
        if (mutatedTargets.isEmpty()) {
            throw LoweringException(
                "while-loop must mutate at least one outer `var` (got 0)",
            )
        }
        val carriedSyms = mutatedTargets
        val carriedInits = carriedSyms.map { sym ->
            env[sym] ?: throw LoweringException(
                "while-loop mutates '${sym.name}' which isn't in scope at loop entry",
            )
        }

        val boolS = DxirType(Bool, emptyList())
        val w = emitter.whileOp(
            inits = carriedInits,
            cond = { args ->
                for (k in carriedSyms.indices) {
                    env[carriedSyms[k]] = args[k]
                }
                val primaryPred = lowerPredicate(loop.condition, env, this)
                val pred = if (breakCond != null) {
                    // §0.4.50 — LAND-hoist for break cond. Works when break_cond
                    // references only carried vars (cond region has them via args[k]).
                    // §0.4.56 D.3ii-tape — body-local-dep breaks (e.g., `if (d < eps) break`
                    // where d is a body-local val) surface here as a `reference to symbol
                    // outside the lowering scope` from lookupReference. Rewrap so the
                    // runtime-tape fallback diagnostic names the actual trigger rather
                    // than the low-level env miss.
                    val breakPred = try {
                        lowerPredicate(breakCond, env, this)
                    } catch (e: LoweringException) {
                        throw LoweringException(
                            "break condition references a value not carried across the " +
                                "loop iteration — only break conditions over carried `var`s " +
                                "are supported at compile time (${e.message})",
                        )
                    }
                    val notBreak = op(OpKind.NOT, listOf(breakPred), boolS)
                    op(OpKind.LAND, listOf(primaryPred, notBreak), boolS)
                } else {
                    primaryPred
                }
                yields(pred)
            },
            body = { args ->
                for (k in carriedSyms.indices) {
                    env[carriedSyms[k]] = args[k]
                }
                for (stmt in bodyStatements) {
                    lowerStatement(stmt, env, this, isLast = false)
                }
                val carriedYields: Array<DxirNode> = Array(carriedSyms.size) { k -> env[carriedSyms[k]]!! }
                yields(*carriedYields)
            },
        )
        for (k in carriedSyms.indices) {
            env[carriedSyms[k]] = w.result(k)
        }
    }

    private data class TrailingBreak(val breakCond: FirExpression)

    /**
     * §0.4.50 Gap 3 — match `... ; if (break_cond) break` as the tail of a while body.
     * Returns the extracted `break_cond` or `null` if the pattern doesn't match.
     * Also returns null if the body contains any OTHER `FirBreakExpression` (i.e., a
     * break not in the trailing-if position) — we reject rather than silently drop.
     */
    private fun detectTrailingBreak(
        bodyStatements: List<Any>,
        loop: FirWhileLoop,
    ): TrailingBreak? {
        val last = bodyStatements.lastOrNull() ?: return null
        // FIR often wraps a unit-typed `if (cond) break` in a FirBlock (FirSingleExpressionBlock).
        // Peel plain-block wrappers to find the FirWhenExpression underneath.
        var unwrapped: Any = last
        while (unwrapped is FirBlock && unwrapped.source?.kind != KtFakeSourceElementKind.DesugaredForLoop
            && unwrapped.statements.size == 1) {
            unwrapped = unwrapped.statements[0]
        }
        val trailing = unwrapped as? FirWhenExpression ?: run {
            assertNoStrayBreak(bodyStatements, loop)
            return null
        }
        val branches = trailing.branches
        if (branches.isEmpty() || branches.size > 2) {
            assertNoStrayBreak(bodyStatements, loop)
            return null
        }
        val thenBranch = branches[0]
        // Bare `if (cond) break` may normalize to 1 branch or 2 branches with an empty
        // else arm. Reject any else-arm that does actual work.
        if (branches.size == 2) {
            val elseBranch = branches[1]
            if (elseBranch.condition !is FirElseIfTrueCondition) {
                assertNoStrayBreak(bodyStatements, loop)
                return null
            }
            if (!isEffectivelyEmpty(elseBranch.result)) {
                assertNoStrayBreak(bodyStatements, loop)
                return null
            }
            if (containsBreakTargeting(elseBranch.result, loop)) {
                throw LoweringException("break inside the else branch of trailing if-break not supported (Gap 3 scope)")
            }
        }
        var thenBody: Any = thenBranch.result
        while (thenBody is FirBlock && thenBody.statements.size == 1) {
            thenBody = thenBody.statements[0]
        }
        val breakExpr = thenBody as? FirBreakExpression ?: run {
            assertNoStrayBreak(bodyStatements, loop)
            return null
        }
        if (breakExpr.target.labeledElement !== loop) {
            throw LoweringException("labeled break targeting an outer loop not supported (Gap 3 scope)")
        }
        // No other break in earlier body positions.
        assertNoStrayBreak(bodyStatements.dropLast(1), loop)
        return TrailingBreak(thenBranch.condition)
    }

    private fun isEffectivelyEmpty(block: FirBlock): Boolean {
        return block.statements.isEmpty()
    }


    private fun assertNoStrayBreak(statements: List<Any>, loop: FirWhileLoop) {
        for (stmt in statements) {
            if (containsBreakTargeting(stmt, loop)) {
                throw LoweringException(
                    "break outside a trailing `if (cond) break` in loop body not supported (Gap 3 scope)",
                )
            }
        }
    }

    private fun containsBreakTargeting(node: Any?, loop: FirWhileLoop): Boolean {
        return when (node) {
            null -> false
            is FirBreakExpression -> node.target.labeledElement === loop
            is FirBlock -> node.statements.any { containsBreakTargeting(it, loop) }
            is FirWhenExpression -> node.branches.any { containsBreakTargeting(it.result, loop) }
            else -> false
        }
    }

    /**
     * §0.4.25 — lower a desugared `for (i in 0 until N)` block to [OpKind.WHILE] in the
     * canonical C5-pattern shape. Kotlin's raw-FIR builder rewrites `for` loops into a
     * `FirBlock { val <iterator> = range.iterator(); FirWhileLoop(hasNext) { val i =
     * iter.next(); <body> } }` (marker: `source.kind == KtFakeSourceElementKind
     * .DesugaredForLoop`). We re-recognise that shape here and emit a counter-based WHILE
     * so `PhiCalculus.apply`'s C5/C6 corollaries can detect + close it into straight-line
     * dxir before SCT.
     *
     * Constraints (B.4b first cut):
     *  - Range is `0 until N` where `N` is a concrete `Int` literal.
     *  - Body mutates exactly one outer-scope `var` via [FirVariableAssignment].
     *  - Body does not read the loop index `i` (any reference throws — falls back to tape).
     *  - No nested loops, no `break`/`continue`, no nested region emission from the body.
     *
     * Emission shape matches the `iterateConcreteN` test primal in `PhiCalculusTest.kt`:
     * `inits = [carried_init, const(0, i32)]`, cond `STEP(SUB(const(n, i32), args[1]))`,
     * body yields `[new_carried, ADD(args[1], const(1, i32))]`. C5's `detectSimpleLoop`
     * pattern matches this directly.
     */
    private fun lowerDesugaredForLoop(
        block: FirBlock,
        env: MutableMap<Any, DxirNode>,
        emitter: DxirEmitter,
    ): Unit {
        val statements = block.statements
        if (statements.size != 2) {
            throw LoweringException(
                "desugared-for-loop block must have [iterator, while]; got ${statements.size} statements",
            )
        }
        val iterProp = statements[0] as? FirProperty
            ?: throw LoweringException("desugared-for-loop first statement is not a FirProperty")
        val whileLoop = statements[1] as? FirWhileLoop
            ?: throw LoweringException("desugared-for-loop second statement is not a FirWhileLoop")

        val tripBound = extractForLoopTripCount(iterProp)
            ?: throw LoweringException(
                "for-loop range must be `0 until <int-literal-or-expr>` (B.4b scope)",
            )

        val innerBlock = whileLoop.block
        val bodyStatements = innerBlock.statements
        if (bodyStatements.isEmpty()) {
            throw LoweringException("for-loop body is empty (unexpected FIR shape)")
        }
        // §0.4.40 — first statement is the synthetic `val <i> = iter.next()` where
        // `<i>` is the user-visible loop-index name. Bind it to the counter block arg
        // so body references to `i` resolve to the i32 counter. Prior sessions
        // (§0.4.25) deliberately left this unbound; kernel-faithful brachistochrone
        // needs `i` (or `i.toFloat()`) in the body for per-segment computations.
        val loopParamSym = (bodyStatements[0] as? FirProperty)?.symbol
            ?: throw LoweringException("desugared-for-loop body's first statement is not `val i = iter.next()`")
        val userBodyStatements = bodyStatements.drop(1)

        val mutatedTargets = collectMutatedTargets(userBodyStatements).toList()
        if (mutatedTargets.isEmpty()) {
            throw LoweringException(
                "for-loop must mutate at least one outer `var` (got 0)",
            )
        }
        // §0.4.39 — multi-var for-loop bodies. `mutatedTargets` preserves first-mutation
        // order (LinkedHashSet in collectMutatedTargets). The whileOp's operand list is
        // user_vars ++ [counter]; counter sits at the END so user indices stay
        // compact [0, N-1] for the C5 unroll's single-referenced-index lookup.
        val carriedSyms = mutatedTargets  // List<FirPropertySymbol> in declaration order
        val carriedInits = carriedSyms.map { sym ->
            env[sym] ?: throw LoweringException(
                "for-loop mutates '${sym.name}' which isn't in scope at loop entry",
            )
        }

        val i32s = DxirType(I32, emptyList())
        val boolS = DxirType(Bool, emptyList())
        // §0.4.53 — counter dtype matches the bound. Concrete Int bound keeps I32 counter
        // (preserving §0.4.39's established C5 shape); expression bound uses whatever
        // dtype the lowered expression yields (typically F32 for a `Float` lambda param).
        val (nBound, counterDtype) = when (tripBound) {
            is ForLoopBound.Concrete -> emitter.const(tripBound.value, i32s) to i32s
            is ForLoopBound.Expression -> {
                val lowered = lowerExpr(tripBound.expr, env, emitter)
                lowered to lowered.type
            }
        }
        val counterInit = emitter.const(zeroOfDtype(counterDtype), counterDtype)
        val counterIncr = emitter.const(oneOfDtype(counterDtype), counterDtype)
        val counterArgIdx = carriedSyms.size  // last block-arg is the counter

        val w = emitter.whileOp(
            inits = carriedInits + counterInit,
            cond = { args ->
                val diff = op(OpKind.SUB, listOf(nBound, args[counterArgIdx]), counterDtype)
                val pred = op(OpKind.STEP, listOf(diff), boolS)
                yields(pred)
            },
            body = { args ->
                // Rebind each mutated var to its carried block-arg so body reads resolve
                // to args[k]. Saving outer env entries is unnecessary — the post-loop
                // rebind to `w.result(k)` overwrites them, and lambda scope ends at the
                // lambda boundary anyway.
                for (k in carriedSyms.indices) {
                    env[carriedSyms[k]] = args[k]
                }
                // §0.4.40 — bind the loop index `i` to the counter block arg (i32).
                // Body references to `i` now resolve to `args[counterArgIdx]` instead
                // of throwing "reference to symbol outside the lowering scope".
                env[loopParamSym] = args[counterArgIdx]
                for (stmt in userBodyStatements) {
                    lowerStatement(stmt, env, this, isLast = false)
                }
                val newCounter = op(OpKind.ADD, listOf(args[counterArgIdx], counterIncr), counterDtype)
                val carriedYields: Array<DxirNode> = Array(carriedSyms.size) { k -> env[carriedSyms[k]]!! }
                yields(*carriedYields, newCounter)
            },
        )
        for (k in carriedSyms.indices) {
            env[carriedSyms[k]] = w.result(k)
        }
    }

    /**
     * §0.4.53 — for-loop trip-count is either a concrete Int literal or an arbitrary
     * FIR expression (typically a lambda-param reference, e.g., `for (i in 0 until T)`
     * where T is `Int`/`Float`). The expression path lowers inside
     * [lowerDesugaredForLoop] via [lowerExpr] so PhiCalculus's C6 `TripCount.Symbolic`
     * path can close loops whose trip count is a runtime parameter.
     */
    private sealed interface ForLoopBound {
        data class Concrete(val value: Int) : ForLoopBound
        data class Expression(val expr: FirExpression) : ForLoopBound
    }

    /**
     * Extract the trip-count `N` from the synthetic `val <iterator> = (0 until N).iterator()`
     * property. Returns null on shape mismatch (not a `0 until …` range).
     */
    private fun extractForLoopTripCount(iterProp: FirProperty): ForLoopBound? {
        val iterInit = iterProp.initializer as? FirFunctionCall ?: return null
        val iterSym = iterInit.calleeReference.toResolvedCallableSymbol() ?: return null
        if (iterSym.callableId?.callableName?.asString() != "iterator") return null
        val rangeExpr = receiver(iterInit) as? FirFunctionCall ?: return null
        val rangeSym = rangeExpr.calleeReference.toResolvedCallableSymbol() ?: return null
        val rangeCid = rangeSym.callableId ?: return null
        if (rangeCid.packageName.asString() != "kotlin.ranges") return null
        if (rangeCid.callableName.asString() != "until") return null
        val startExpr = receiver(rangeExpr) as? FirLiteralExpression ?: return null
        if ((startExpr.value as? Number)?.toInt() != 0) return null
        val endExpr = rangeExpr.argumentList.arguments.firstOrNull() ?: return null
        if (endExpr is FirLiteralExpression) {
            val endValue = (endExpr.value as? Number)?.toInt() ?: return null
            if (endValue < 0) return null
            return ForLoopBound.Concrete(endValue)
        }
        return ForLoopBound.Expression(endExpr)
    }

    /**
     * Walk [statements] and collect every [FirVariableAssignment] target symbol,
     * descending through plain [FirBlock]s (e.g., the `FirSingleExpressionBlock` wrapping
     * a braceless `for (...) body` statement) AND through nested desugared-for-loop
     * blocks (§0.4.50 — the outer loop's carried-var set must include vars mutated by
     * the inner loop, otherwise the outer WHILE body wouldn't yield the updated value).
     * Other nested control flow (raw while, when) is still out of scope.
     */
    private fun collectMutatedTargets(statements: List<Any>): Set<FirPropertySymbol> {
        val out = LinkedHashSet<FirPropertySymbol>()
        // §0.4.163 — track properties declared INSIDE the body so a mutation of one
        // doesn't bubble up as a "carried var of the enclosing loop". Without this,
        // `for (i …) { var xb = 0; for (j …) { xb = xb + … } }` reports `xb` as a
        // mutation of the OUTER loop, but `xb` is locally-scoped to the outer body
        // and reset each outer iteration — it is NOT a carried var of the outer.
        val localDecls = HashSet<FirPropertySymbol>()
        fun visit(list: List<Any>) {
            for (stmt in list) {
                when (stmt) {
                    is FirProperty ->
                        localDecls += stmt.symbol
                    is FirVariableAssignment ->
                        resolveAssignmentTarget(stmt)?.let { sym ->
                            if (sym !in localDecls) out += sym
                        }
                    is FirBlock -> {
                        if (stmt.source?.kind == KtFakeSourceElementKind.DesugaredForLoop) {
                            // statements[1] is the FirWhileLoop; descend into its body,
                            // skipping the first stmt (synthetic `val <i> = iter.next()`).
                            val whileLoop = stmt.statements.getOrNull(1) as? FirWhileLoop
                            if (whileLoop != null) {
                                visit(whileLoop.block.statements.drop(1))
                            }
                        } else {
                            visit(stmt.statements)
                        }
                    }
                    else -> {}
                }
            }
        }
        visit(statements)
        return out
    }

    /**
     * Lower a boolean expression suitable for the IF-predicate slot. B.4a's shape is a
     * [FirComparisonExpression] with GT or LT. `a > b` → `STEP(SUB(a, b))` with Bool result
     * type; `a < b` → `STEP(SUB(b, a))`. `>=`, `<=`, and equality are rejected — they
     * would need distinct ops (STEP(0)=0 per §0.4.7 gives strict inequality only).
     */
    private fun lowerPredicate(
        expr: FirExpression,
        env: MutableMap<Any, DxirNode>,
        emitter: DxirEmitter,
    ): DxirNode {
        if (expr !is FirComparisonExpression) {
            throw LoweringException(
                "if-condition must be a comparison `a > b` or `a < b` (B.4a scope); " +
                    "got ${expr::class.simpleName}",
            )
        }
        val call = expr.compareToCall
        val lhsExpr = call.dispatchReceiver ?: call.extensionReceiver
            ?: throw LoweringException("comparison compareTo has no receiver")
        val rhsExpr = call.argumentList.arguments.firstOrNull()
            ?: throw LoweringException("comparison compareTo missing rhs")
        val lhs = lowerExpr(lhsExpr, env, emitter)
        val rhs = lowerExpr(rhsExpr, env, emitter)
        val (a, b) = when (expr.operation) {
            FirOperation.GT -> lhs to rhs
            FirOperation.LT -> rhs to lhs
            else -> throw LoweringException(
                "if-condition operation ${expr.operation} not supported (B.4a accepts GT/LT only)",
            )
        }
        val diff = emitter.op(OpKind.SUB, listOf(a, b), a.type)
        return emitter.op(OpKind.STEP, listOf(diff), DxirType(Bool, emptyList()))
    }

    private fun lowerCall(
        call: FirFunctionCall,
        env: MutableMap<Any, DxirNode>,
        emitter: DxirEmitter,
    ): DxirNode {
        val sym = call.calleeReference.toResolvedCallableSymbol()
            ?: throw LoweringException("unresolved call")
        val callableId = sym.callableId ?: throw LoweringException("call has no callableId")
        val classId = callableId.classId
        val fqn = if (classId != null) {
            "${classId.asFqNameString()}.${callableId.callableName.asString()}"
        } else {
            "${callableId.packageName.asString()}.${callableId.callableName.asString()}"
        }

        // §0.4.40 — dtype-changing receiver-only conversions (`Int.toFloat()`,
        // `Long.toDouble()`, etc.). Dispatched before UNARY_OP_MAP because CAST's
        // result dtype differs from its operand's, whereas every other UNARY_OP_MAP
        // entry is shape-and-dtype-preserving (or a reduction that stays in operand
        // dtype). Needed at minimum for `i.toFloat()` in body expressions after the
        // loop-index-binding that landed in §0.4.40.
        CAST_OP_MAP[fqn]?.let { targetDtype ->
            val operandExpr = receiver(call)
                ?: throw LoweringException("cast '$fqn' has no receiver")
            val operand = lowerExpr(operandExpr, env, emitter)
            return emitter.op(
                kind = OpKind.CAST,
                operands = listOf(operand),
                type = DxirType(targetDtype, operand.type.dims),
            )
        }

        // §0.4.42 — scalar-index-into-rank-1 gather. The `io.tlaloc.core.ops.get`
        // operator (declared in :core/ops/HostOps.kt) lets users write `arr[i]` in a
        // grad lambda. FIR lowers it to `OpKind.GATHER(arr, idx)` with a scalar
        // result whose dtype matches the operand array. Narrow shape validation
        // (rank-1 operand + scalar-Int index) happens here so a rank-2 `mat[0]`
        // fails compile rather than surfacing as a silent runtime bug downstream.
        if (fqn == "io.tlaloc.core.ops.get") {
            val arrExpr = receiver(call)
                ?: throw LoweringException("gather call 'get' has no receiver")
            val idxExpr = call.argumentList.arguments.firstOrNull()
                ?: throw LoweringException("gather call 'get' missing index argument")
            val arr = lowerExpr(arrExpr, env, emitter)
            val rawIdx = lowerExpr(idxExpr, env, emitter)
            if (arr.type.rank != 1) {
                throw LoweringException(
                    "gather array operand must be rank-1 (got rank=${arr.type.rank}); " +
                        "multi-dim indexing is out of scope (post-Stage-D)",
                )
            }
            if (!rawIdx.type.isScalar) {
                throw LoweringException(
                    "gather index must be a scalar (got ${rawIdx.type})",
                )
            }
            // §0.4.42 — FIR literals for integer constants land as I64 (FIR stores
            // `IntegerLiteral` kind's value as a Long until typed-resolution narrows
            // it; at IR-gen time we see Long even when the source literal was `0`
            // with Int context). GATHER's interpreter arm requires I32, so auto-cast
            // anything else via OpKind.CAST. Same coercion covers a user-written
            // `arr[someLong.toInt()]` that lowered to I32 already (no-op cast).
            val idx = if (rawIdx.type.dtype == I32) {
                rawIdx
            } else {
                emitter.op(
                    kind = OpKind.CAST,
                    operands = listOf(rawIdx),
                    type = DxirType(I32, emptyList()),
                )
            }
            return emitter.op(
                kind = OpKind.GATHER,
                operands = listOf(arr, idx),
                type = DxirType(arr.type.dtype, emptyList()),
            )
        }

        // §0.4.188 — DTensor → Float bridge. `DTensor<ScalarShape, F32>.toFloat()`
        // (in :core/ops/HostOps.kt) is a no-op at the dxir level: a scalar-shape
        // DTensor and a primitive Float share `DxirType(F32, [])`. The plugin's
        // dispatch returns the receiver's already-lowered value directly. Without
        // this bridge, lambda bodies can't terminate in a Float computed from
        // tensor intermediates (e.g., `a.sum().toFloat()` for active SUM
        // gradient on rank-2 inputs, or future MATMUL-based primals). Special-
        // cased here rather than added to UNARY_OP_MAP because no op kind is
        // emitted — the dispatch returns the receiver expression as-is.
        if (fqn == "io.tlaloc.core.ops.toFloat") {
            val operandExpr = receiver(call)
                ?: throw LoweringException("toFloat call has no receiver")
            val operand = lowerExpr(operandExpr, env, emitter)
            if (!operand.type.isScalar) {
                throw LoweringException(
                    "toFloat receiver must be scalar (got ${operand.type})",
                )
            }
            if (operand.type.dtype != F32) {
                throw LoweringException(
                    "toFloat receiver dtype must be F32 (got ${operand.type.dtype})",
                )
            }
            return operand
        }

        BINARY_OP_MAP[fqn]?.let { kind ->
            val lhsExpr = receiver(call)
                ?: throw LoweringException("binary op '$fqn' has no receiver")
            val lhs = lowerExpr(lhsExpr, env, emitter)
            val rhsExpr = call.argumentList.arguments.firstOrNull()
                ?: throw LoweringException("binary op '$fqn' missing rhs argument")
            val rhs = lowerExpr(rhsExpr, env, emitter)
            return emitter.op(
                kind = kind,
                operands = listOf(lhs, rhs),
                type = lhs.type,
            )
        }

        UNARY_OP_MAP[fqn]?.let { kind ->
            val operandExpr = receiver(call)
                ?: throw LoweringException("unary op '$fqn' has no receiver")
            val operand = lowerExpr(operandExpr, env, emitter)
            // Reductions collapse all operand dims to scalar — every other unary op is
            // shape-preserving. Keeping this as a per-kind branch here (rather than a
            // second OP_MAP) matches the BINARY_OP_MAP convention: the receiver side
            // inspects only the `kind`, never the FQN itself.
            val resultType = when (kind) {
                OpKind.SUM, OpKind.MEAN -> DxirType(operand.type.dtype, emptyList())
                else -> operand.type
            }
            return emitter.op(
                kind = kind,
                operands = listOf(operand),
                type = resultType,
            )
        }

        throw LoweringException("unsupported call '$fqn'")
    }

    private fun receiver(call: FirFunctionCall): FirExpression? =
        call.dispatchReceiver ?: call.extensionReceiver

    private fun resolveParamType(type: ConeKotlinType): DxirType? {
        val cid = type.classId ?: return null
        val fqn = cid.asString()
        PRIMITIVE_DTYPE_MAP[fqn]?.let { return DxirType(it, emptyList()) }
        if (fqn == "io/tlaloc/core/DTensor") {
            val args = type.typeArguments
            if (args.size != 2) return null
            val shapeType = args[0].type ?: return null
            val dtypeType = args[1].type ?: return null
            val shapeFqn = shapeType.classId?.asString() ?: return null
            val dtypeFqn = dtypeType.classId?.asString() ?: return null
            val dtype = DTENSOR_DTYPE_MAP[dtypeFqn] ?: return null
            // Shape-to-dims mapping. ScalarShape (rank 0) is the legacy path and still
            // types as scalar. Rank1 gets a single sentinel `-1` — no rule in
            // [io.tlaloc.ir.passes.VjpRegistry] dereferences dim *values* on the
            // synthesis-bound path (§0.4.10 — SumRule reads `.dims` only for rank /
            // identity), so a sentinel is enough here.
            //
            // §0.4.185 — Rank2 + Rank3 entries: substrate addition for Phase 0c
            // (plugin MATMUL recognition). FIR-side recognition unblocks lowering of
            // `grad { A: DTensor<Rank2<R, K>, F32> -> ... }` style primals to dxir.
            // Note that DxirToIrSynthesis today supports only scalar + rank-1 F32, so
            // ANY rank-2/3 grad-body op (e.g., the BROADCAST emitted by SumRule on a
            // rank-2 reduction) still trips synthesis-scope rejection — the runtime
            // tape stub fires for end-to-end gradient calls. The FIR-side
            // recognition lands here as the first slice; rank-2/3 synthesis-side
            // widening is the next slice in the Phase 0c arc.
            val dims: List<Int> = when (shapeFqn) {
                "io/tlaloc/core/ScalarShape" -> emptyList()
                "io/tlaloc/core/Rank1" -> listOf(-1)
                "io/tlaloc/core/Rank2" -> listOf(-1, -1)
                "io/tlaloc/core/Rank3" -> listOf(-1, -1, -1)
                else -> return null
            }
            return DxirType(dtype, dims)
        }
        return null
    }

    private fun literalDType(value: Any): DType? = when (value) {
        is Float -> F32
        is Double -> F64
        is Int -> I32
        is Long -> I64
        else -> null
    }

    /** §0.4.53 — dtype-matched zero/one constants for synthesizing for-loop counters. */
    private fun zeroOfDtype(type: DxirType): Any = when (type.dtype) {
        F32 -> 0.0f
        F64 -> 0.0
        I32 -> 0
        I64 -> 0L
        else -> throw LoweringException("unsupported counter dtype ${type.dtype}")
    }

    private fun oneOfDtype(type: DxirType): Any = when (type.dtype) {
        F32 -> 1.0f
        F64 -> 1.0
        I32 -> 1
        I64 -> 1L
        else -> throw LoweringException("unsupported counter dtype ${type.dtype}")
    }

    private fun ConeKotlinType.renderForError(): String =
        classId?.asFqNameString() ?: this::class.simpleName ?: "unknown"

    private val BINARY_OP_MAP: Map<String, OpKind> = buildMap {
        for (t in listOf("Float", "Double", "Int", "Long")) {
            put("kotlin.$t.plus", OpKind.ADD)
            put("kotlin.$t.minus", OpKind.SUB)
            put("kotlin.$t.times", OpKind.MUL)
            put("kotlin.$t.div", OpKind.DIV)
        }
        // :core DScalar / FloatScalar / DoubleScalar operators are top-level extensions
        // in package io.tlaloc.core, so their callableId has no classId.
        put("io.tlaloc.core.plus", OpKind.ADD)
        put("io.tlaloc.core.minus", OpKind.SUB)
        put("io.tlaloc.core.times", OpKind.MUL)
        put("io.tlaloc.core.div", OpKind.DIV)
        // §0.4.187 — Phase 0c slice (c): plugin recognition for `infix fun matmul`
        // (declared in :core/ops/HostOps.kt). The IR-side MATMUL was shipped at
        // §0.4.135 / §0.4.137 / §0.4.138 (rank-2 → rank-2-or-3 batched → any rank ≥ 2).
        // §0.4.185 + §0.4.186 closed Phase 0c slices (a) + (b) (Rank2/3 param
        // recognition + synthesis-side rank-1/2/3 acceptance via broadcastLike). With
        // both substrate pieces in place, MATMUL-bearing primal bodies now lower
        // through the K2 plugin's BINARY_OP_MAP dispatch the same way ADD/SUB/MUL/DIV
        // do — `lhs.type` for matmul is rank-2 with sentinel dims, the result type is
        // also rank-2 with sentinel dims (same shape structurally), so the existing
        // `type = lhs.type` dispatch produces the correct DxirType.
        put("io.tlaloc.core.ops.matmul", OpKind.MATMUL)
    }

    // §0.4.40 — dtype-conversion calls. Maps receiver-only `Int.toFloat()` /
    // `Long.toDouble()` / etc. to the target DType. These emit `OpKind.CAST` with
    // a result dtype different from the operand's. Int → Int, Float → Float, and
    // other identity conversions are intentionally omitted (zero-benefit noise).
    private val CAST_OP_MAP: Map<String, DType> = buildMap {
        put("kotlin.Int.toFloat", F32)
        put("kotlin.Int.toDouble", F64)
        put("kotlin.Int.toLong", I64)
        put("kotlin.Long.toFloat", F32)
        put("kotlin.Long.toDouble", F64)
        put("kotlin.Long.toInt", I32)
        put("kotlin.Float.toDouble", F64)
        put("kotlin.Float.toInt", I32)
        put("kotlin.Float.toLong", I64)
        put("kotlin.Double.toFloat", F32)
        put("kotlin.Double.toInt", I32)
        put("kotlin.Double.toLong", I64)
    }

    private val UNARY_OP_MAP: Map<String, OpKind> = buildMap {
        for (t in listOf("Float", "Double", "Int", "Long")) {
            put("kotlin.$t.unaryMinus", OpKind.NEG)
        }
        put("io.tlaloc.core.unaryMinus", OpKind.NEG)
        // :core scalar relu entries on Float/Double/DScalar (see DScalar.kt).
        // Top-level extension functions, so they land in package io.tlaloc.core with
        // no classId — the FQN is "io.tlaloc.core.relu".
        put("io.tlaloc.core.relu", OpKind.RELU)
        // :core scalar sqrt entries on Float/Double/DScalar (see DScalar.kt). Same
        // top-level-extension pattern as `relu`. Enables scalar `y.sqrt()` inside
        // `grad { y: Float -> ... }` bodies to route through OpKind.SQRT + SqrtRule.
        put("io.tlaloc.core.sqrt", OpKind.SQRT)
        // §0.4.158 — :core scalar exp / log entries. Mirrors the sqrt pattern. Enables
        // scalar `y.exp()` / `y.log()` inside `grad { y: Float -> ... }` bodies to route
        // through OpKind.EXP / OpKind.LOG + ExpRule / LogRule. Needed by HMC's logistic-
        // regression port for the `log(1 + exp(-Xβ))` per-record term.
        put("io.tlaloc.core.exp", OpKind.EXP)
        put("io.tlaloc.core.log", OpKind.LOG)
        // §0.4.188 — `:core/ops/HostOps.kt`'s tensor SUM extension. The receiver is a
        // `DTensor<S, F32>` of any rank; the result is `DTensor<ScalarShape, F32>`.
        // The result-type dispatch above this map's lookup site special-cases SUM /
        // MEAN to produce a scalar `DxirType` regardless of the operand's rank. With
        // the §0.4.185 + §0.4.186 + §0.4.187 substrate (rank-2/3 param + synthesis +
        // matmul), wiring SUM completes the active-gradient path for tensor-bearing
        // primal lambdas — `grad { a -> a.sum().toFloat() }` on a rank-2 input now
        // produces a rank-2 ones-tensor gradient (the BROADCAST(1.0, a's shape) from
        // SumRule).
        put("io.tlaloc.core.ops.sum", OpKind.SUM)
        // §0.4.166 — :core scalar sin / cos entries. Mirrors the §0.4.158 pattern.
        // Needed by CartPole's pole-angle physics step (per docs/CARTPOLE_PORT_PLAN.md
        // Phase 0a).
        put("io.tlaloc.core.sin", OpKind.SIN)
        put("io.tlaloc.core.cos", OpKind.COS)
        // §0.4.167 — :core scalar abs entry. AbsRule and the synthesis arm `irAbs`
        // ship in the same firing. Needed by CartPole's loss-clipping
        // `(2.4 - |xt+1,0|) · (0.21 - |xt+1,2|)`.
        put("io.tlaloc.core.abs", OpKind.ABS)
        // :core DTensor shape-preserving unary ops (io.tlaloc.core.ops package).
        put("io.tlaloc.core.ops.relu", OpKind.RELU)
        put("io.tlaloc.core.ops.neg", OpKind.NEG)
        put("io.tlaloc.core.ops.sigmoid", OpKind.SIGMOID)
        put("io.tlaloc.core.ops.tanh", OpKind.TANH)
        put("io.tlaloc.core.ops.sign", OpKind.SIGN)
        put("io.tlaloc.core.ops.exp", OpKind.EXP)
        put("io.tlaloc.core.ops.log", OpKind.LOG)
        put("io.tlaloc.core.ops.sqrt", OpKind.SQRT)
        // :core DTensor reductions (io.tlaloc.core.ops package). These collapse all
        // operand dims to scalar — special-cased in the unary-lowering arm above.
        put("io.tlaloc.core.ops.sum", OpKind.SUM)
    }

    private val PRIMITIVE_DTYPE_MAP: Map<String, DType> = mapOf(
        "kotlin/Float" to F32,
        "kotlin/Double" to F64,
        "kotlin/Int" to I32,
        "kotlin/Long" to I64,
        // DScalar default-precisions: FloatScalar → F32, DoubleScalar → F64. DScalar (the
        // sealed super-interface) has no static precision; we default to F32 since Float is
        // the dominant ML precision and matches the runtime-tape implementation.
        "io/tlaloc/core/DScalar" to F32,
        "io/tlaloc/core/FloatScalar" to F32,
        "io/tlaloc/core/DoubleScalar" to F64,
    )

    private val DTENSOR_DTYPE_MAP: Map<String, DType> = mapOf(
        "io/tlaloc/core/F32" to F32,
        "io/tlaloc/core/F64" to F64,
        "io/tlaloc/core/I32" to I32,
        "io/tlaloc/core/I64" to I64,
    )
}
