package io.tlaloc.plugin

import io.tlaloc.core.Bool
import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.core.I32
import io.tlaloc.core.I64
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirConst
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
import org.jetbrains.kotlin.fir.expressions.FirNamedArgumentExpression
import org.jetbrains.kotlin.fir.expressions.FirVarargArgumentsExpression
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
import org.jetbrains.kotlin.fir.types.resolvedType
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
        /** §0.4.353 — [namedIndex] classifies contract/named-axis violations
         * so the checker can report them as error-severity
         * [TlalocErrors.NAMED_INDEX_MISMATCH] instead of the generic
         * (warning-severity, tape-fallback) [TlalocErrors.LAMBDA_UNSUPPORTED]. */
        data class Failure(val reason: String, val namedIndex: Boolean = false) : Result()
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
            Result.Failure(e.message ?: "unknown", namedIndex = e is NamedIndexException)
        }
    }

    private open class LoweringException(message: String) : RuntimeException(message)

    /** §0.4.353 — named-axis misuse (the user's type-level contract is
     * inconsistent): reported as a compile ERROR, not a lowering fallback. */
    private class NamedIndexException(message: String) : LoweringException(message)

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

        // Layer 1 §0.4.241+ — named-index contraction. `contract`'s result
        // type structurally differs from its operands (the contracted axis is
        // dropped), so it can't go through BINARY_OP_MAP's `type = lhs.type`
        // dispatch. Special-cased here to inspect both operands' axisNames,
        // derive the shared name + positions, and emit MATMUL/DOT with the
        // right contracted_names + lhs_contracting_dims attrs. Kotlin's
        // native overload resolution has already validated that the operands
        // share a name in the type system — this lowering relies on that
        // pre-condition (a contract call only reaches this site when the
        // overload resolution succeeded, i.e. when there *is* a shared name).
        if (fqn == "io.tlaloc.core.ops.contract") {
            val lhsExpr = receiver(call)
                ?: throw LoweringException("contract call has no receiver")
            val lhs = lowerExpr(lhsExpr, env, emitter)
            val rhsExpr = call.argumentList.arguments.firstOrNull()
                ?: throw LoweringException("contract call missing rhs argument")
            val rhs = lowerExpr(rhsExpr, env, emitter)
            return emitContract(lhs, rhs, emitter)
        }

        // §0.4.368 — NN ops (DiffKT parity, Phase A3): `softmax(axis)` and
        // `logSoftmax(axis)` over a single axis (default last). Both are
        // shape-preserving (unlike the reductions), so the result DxirType is
        // the operand's; the axis folds into an `"axis"` Int attr (the
        // emitter's readAxis requires Int, and the value is normalised
        // non-negative here so the interpreter/VJP/emitter all agree).
        // `logSoftmax` lowers to `LOG(SOFTMAX(x, axis))` — both ops carry full
        // VJP + JVP rules, so the gradient flows through LogRule ∘ SoftmaxRule
        // with no new AD math (LOGSUMEXP stays emitter-only).
        if (fqn == "io.tlaloc.core.ops.softmax" || fqn == "io.tlaloc.core.ops.logSoftmax") {
            val operandExpr = receiver(call)
                ?: throw LoweringException("$fqn has no receiver")
            val operand = lowerExpr(operandExpr, env, emitter)
            val rank = operand.type.rank
            if (rank == 0) throw LoweringException("$fqn requires a tensor operand (got scalar)")
            val rawAxis = call.argumentList.arguments.firstOrNull()?.let {
                intLiteralArg(it) ?: throw LoweringException("$fqn axis must be an integer literal")
            } ?: -1
            val axis = if (rawAxis < 0) rawAxis + rank else rawAxis
            if (axis !in 0 until rank) {
                throw LoweringException("$fqn axis $rawAxis out of range for rank $rank")
            }
            val sm = emitter.op(
                kind = OpKind.SOFTMAX,
                operands = listOf(operand),
                type = operand.type,
                attrs = mapOf("axis" to axis),
            )
            return if (fqn == "io.tlaloc.core.ops.softmax") sm
            else emitter.op(kind = OpKind.LOG, operands = listOf(sm), type = operand.type)
        }

        // §0.4.370 — Phase A3b (DiffKT parity): the softmax cross-entropy and
        // NLL losses, composed onto existing fully-ruled ops (no new VjpRule).
        // `crossEntropyLoss(logits, oneHot)` = NEG(SUM(MUL(oneHot, LOG(SOFTMAX(
        // logits, -1))))) — logSoftmax over the last (class) axis, one-hot
        // weighted, negated, summed to a scalar (the sum-reduction convention).
        // `nllLoss(logProbs, oneHot)` skips the softmax (logProbs is already
        // log-normalised): NEG(SUM(MUL(oneHot, logProbs))). Both return a scalar
        // so the lambda body terminates in `.toFloat()`.
        if (fqn == "io.tlaloc.core.ops.crossEntropyLoss" || fqn == "io.tlaloc.core.ops.nllLoss") {
            val args = call.argumentList.arguments
            if (args.size != 2) {
                throw LoweringException("$fqn requires 2 arguments (scores, oneHot); got ${args.size}")
            }
            val scores = lowerExpr(args[0], env, emitter)
            val oneHot = lowerExpr(args[1], env, emitter)
            val rank = scores.type.rank
            if (rank == 0) throw LoweringException("$fqn requires a tensor operand (got scalar)")
            // crossEntropyLoss runs logSoftmax over the class axis first; nllLoss
            // takes the caller's already-log-normalised scores as-is.
            val logProbs = if (fqn == "io.tlaloc.core.ops.crossEntropyLoss") {
                val sm = emitter.op(
                    kind = OpKind.SOFTMAX,
                    operands = listOf(scores),
                    type = scores.type,
                    attrs = mapOf("axis" to rank - 1),
                )
                emitter.op(kind = OpKind.LOG, operands = listOf(sm), type = scores.type)
            } else {
                scores
            }
            val prod = emitter.op(kind = OpKind.MUL, operands = listOf(oneHot, logProbs), type = scores.type)
            val summed = emitter.op(
                kind = OpKind.SUM,
                operands = listOf(prod),
                type = DxirType(scores.type.dtype, emptyList()),
            )
            return emitter.op(
                kind = OpKind.NEG,
                operands = listOf(summed),
                type = DxirType(scores.type.dtype, emptyList()),
            )
        }

        // §0.4.384 — Phase A3b: the conv user surface (NCHW, the layout the
        // interpreter/emitter fix). `x.conv2d(w, …)` takes an OIHW
        // `[Co, Ci, kh, kw]` kernel; `x.convTranspose2d(w, …)` takes IOHW
        // `[Ci, Co, kh, kw]` — the reverse channel order, because the transposed
        // op contracts over the kernel's first axis and emits its second.
        //
        // Every attr is a compile-time Int literal folded onto the op here, the
        // way the reductions fold their axes: `window_strides`, `padding` as
        // [[top, bottom], [left, right]], and — for the transposed spelling only
        // — `lhs_dilation`. The user-facing `stride` of `convTranspose2d` is the
        // UPSAMPLING factor, which in the IR is `lhs_dilation` with
        // `window_strides` left at [1, 1]: the mapping Conv2dRule's `dX` and
        // `stablehlo.convolution` both use.
        //
        // The result's spatial extents are a floor-division over the input's, so
        // any symbolic operand dim (-1, which is what every `grad {}` /
        // `jvp {}` param carries) makes the matching output dim symbolic too —
        // `flatten`'s convention. Reverse-mode conv is NOT reachable from here
        // yet: Conv2dRule solves its adjoint padding from concrete extents and
        // rejects sentinels loudly (see its §0.4.384 guard). Forward mode works,
        // because the tangent rule replays these same literal attrs.
        if (fqn == "io.tlaloc.core.ops.conv2d" || fqn == "io.tlaloc.core.ops.convTranspose2d") {
            val transposed = fqn == "io.tlaloc.core.ops.convTranspose2d"
            val operandExpr = receiver(call)
                ?: throw LoweringException("$fqn has no receiver")
            val x = lowerExpr(operandExpr, env, emitter)
            if (x.type.rank != 4) {
                throw LoweringException("$fqn requires a rank-4 NCHW receiver; got ${x.type}")
            }
            if (x.type.dtype != F32) {
                throw LoweringException("$fqn is F32-only in v1; got ${x.type.dtype}")
            }
            // Two arities, matching the host surface: `conv2d(w)` (valid conv) and
            // the 7-argument positional form. K2 unwraps a named argument to its
            // bare expression BEFORE this lowering runs and does not reorder it
            // into its parameter's position, so reading attrs by name would
            // silently mis-fold (`padTop = 1` would land on `strideH`); arity is
            // the only trustworthy signal here, and the host signature has no
            // default values precisely so that no other spelling compiles.
            val args = call.argumentList.arguments
            val wExpr: FirExpression
            val attrExprs: List<FirExpression>
            when (args.size) {
                1 -> {
                    wExpr = args[0]
                    attrExprs = emptyList()
                }
                7 -> {
                    wExpr = args[0]
                    attrExprs = args.drop(1)
                }
                else -> throw LoweringException(
                    "$fqn takes (w) or (w, strideH, strideW, padTop, padBottom, padLeft, padRight); " +
                        "got ${args.size} arguments",
                )
            }
            val w = lowerExpr(
                (wExpr as? FirNamedArgumentExpression)?.expression ?: wExpr, env, emitter,
            )
            if (w.type.rank != 4) {
                throw LoweringException("$fqn requires a rank-4 kernel; got ${w.type}")
            }
            val attrLits = attrExprs.map { e ->
                intLiteralArg((e as? FirNamedArgumentExpression)?.expression ?: e)
                    ?: throw LoweringException("$fqn stride/padding arguments must be Int literals")
            }
            val strideH = attrLits.getOrNull(0) ?: 1
            val strideW = attrLits.getOrNull(1) ?: 1
            val padTop = attrLits.getOrNull(2) ?: 0
            // The 1-argument form is the valid conv; the 7-argument form supplies
            // all four sides. Mirrors the host defaults exactly.
            val padBottom = attrLits.getOrNull(3) ?: padTop
            val padLeft = attrLits.getOrNull(4) ?: padTop
            val padRight = attrLits.getOrNull(5) ?: padTop
            if (strideH <= 0 || strideW <= 0) {
                throw LoweringException("$fqn strides must be positive; got [$strideH, $strideW]")
            }
            val xd = x.type.dims
            val wd = w.type.dims
            // Kernel input channels must be the receiver's channels — checkable
            // only when both are concrete.
            val kInAxis = if (transposed) 0 else 1
            if (wd[kInAxis] > 0 && xd[1] > 0 && wd[kInAxis] != xd[1]) {
                throw LoweringException(
                    "$fqn kernel input channels ${wd[kInAxis]} ≠ receiver channels ${xd[1]} " +
                        "(x=${xd.toList()}, w=${wd.toList()})",
                )
            }
            // The user surface exposes no dilation, so `rhs_dilation` is [1,1] and
            // the stride lands on `window_strides` (conv2d) or `lhs_dilation`
            // (convTranspose2d) — the other stays [1,1].
            val winStride = if (transposed) listOf(1, 1) else listOf(strideH, strideW)
            val lhsDil = if (transposed) listOf(strideH, strideW) else listOf(1, 1)
            fun outExtent(inDim: Int, kDim: Int, axis: Int): Int {
                if (inDim <= 0 || kDim <= 0) return -1
                val inDil = (inDim - 1) * lhsDil[axis] + 1
                val pad = if (axis == 0) padTop + padBottom else padLeft + padRight
                return (inDil + pad - kDim) / winStride[axis] + 1
            }
            val cOut = if (transposed) wd[1] else wd[0]
            val attrs = buildMap<String, Any> {
                put("window_strides", winStride)
                put("padding", listOf(listOf(padTop, padBottom), listOf(padLeft, padRight)))
                if (transposed) put("lhs_dilation", lhsDil)
            }
            return emitter.op(
                kind = if (transposed) OpKind.CONV_TRANSPOSE2D else OpKind.CONV2D,
                operands = listOf(x, w),
                type = DxirType(
                    x.type.dtype,
                    listOf(xd[0], cOut, outExtent(xd[2], wd[2], 0), outExtent(xd[3], wd[3], 1)),
                ),
                attrs = attrs,
            )
        }

        // §0.4.386 — Phase A3b: the pooling user surfaces (NCHW). Two arities and
        // no default parameter values, for the same reason as conv (K2 unwraps a
        // named argument before this lowering runs and does not reorder it, so
        // attrs must be positional to be unambiguous): `avgPool2d(kh, kw)` /
        // `maxPool2d(kh, kw)` is the non-overlapping pool (strides default to the
        // window, the interpreter's own convention) and the 8-argument form adds
        // strides and all four padding sides. §0.4.389 generalised this arm from
        // avg-only to both kinds — they differ only in the OpKind they produce.
        // `window`/`window_strides`/`padding` fold onto the op as literal attrs;
        // the result's spatial extents are a floor-division over the input's, so a
        // symbolic input dim gives a symbolic output dim.
        if (fqn == "io.tlaloc.core.ops.avgPool2d" || fqn == "io.tlaloc.core.ops.maxPool2d") {
            val isMax = fqn == "io.tlaloc.core.ops.maxPool2d"
            val operandExpr = receiver(call)
                ?: throw LoweringException("$fqn has no receiver")
            val x = lowerExpr(operandExpr, env, emitter)
            if (x.type.rank != 4) {
                throw LoweringException("$fqn requires a rank-4 NCHW receiver; got ${x.type}")
            }
            if (x.type.dtype != F32) {
                throw LoweringException("$fqn is F32-only in v1; got ${x.type.dtype}")
            }
            val args = call.argumentList.arguments
            if (args.size != 2 && args.size != 8) {
                throw LoweringException(
                    "$fqn takes (windowH, windowW) or (windowH, windowW, strideH, strideW, " +
                        "padTop, padBottom, padLeft, padRight); got ${args.size} arguments",
                )
            }
            val lits = args.map { e ->
                intLiteralArg((e as? FirNamedArgumentExpression)?.expression ?: e)
                    ?: throw LoweringException("$fqn arguments must be Int literals")
            }
            val windowH = lits[0]
            val windowW = lits[1]
            if (windowH <= 0 || windowW <= 0) {
                throw LoweringException("$fqn window must be positive; got [$windowH, $windowW]")
            }
            // Strides default to the window (non-overlapping), matching both the
            // interpreter's attr default and PyTorch's `AvgPool2d(k)`.
            val strideH = lits.getOrNull(2) ?: windowH
            val strideW = lits.getOrNull(3) ?: windowW
            val padTop = lits.getOrNull(4) ?: 0
            val padBottom = lits.getOrNull(5) ?: 0
            val padLeft = lits.getOrNull(6) ?: 0
            val padRight = lits.getOrNull(7) ?: 0
            if (strideH <= 0 || strideW <= 0) {
                throw LoweringException("$fqn strides must be positive; got [$strideH, $strideW]")
            }
            val xd = x.type.dims
            fun outExtent(inDim: Int, k: Int, stride: Int, padLo: Int, padHi: Int): Int =
                if (inDim <= 0) -1 else (inDim + padLo + padHi - k) / stride + 1
            return emitter.op(
                kind = if (isMax) OpKind.MAXPOOL2D else OpKind.AVGPOOL2D,
                operands = listOf(x),
                type = DxirType(
                    x.type.dtype,
                    listOf(
                        xd[0], xd[1],
                        outExtent(xd[2], windowH, strideH, padTop, padBottom),
                        outExtent(xd[3], windowW, strideW, padLeft, padRight),
                    ),
                ),
                attrs = mapOf(
                    "window" to listOf(windowH, windowW),
                    "window_strides" to listOf(strideH, strideW),
                    "padding" to listOf(listOf(padTop, padBottom), listOf(padLeft, padRight)),
                ),
            )
        }

        // §0.4.390 — Phase A3b: TRAINING-mode batch normalisation, DESUGARED into
        // primitives here rather than given a first-class op — the `maximum`/`clip`
        // pattern. `OpKind.BATCHNORM` already exists but it is the INFERENCE form
        // (five operands: input, scale, offset, mean, variance) that the Layer-3
        // recognizer emits for fused kernels; training mode computes the statistics
        // from the argument, so its gradient needs the mean/variance dependencies
        // that an inference-form rule could never produce.
        //
        // Desugaring is what makes `grad {}` work with NO new VjpRule, interpreter
        // arm, host delegate or synthesis arm: every node below already has a
        // sentinel-safe adjoint (MEAN via SUM_TO's runtime template, the elementwise
        // binaries via A5c broadcasting, the unit-axis RESHAPEs since §0.4.375).
        //
        //   μ = mean(x, [0,2,3], keepdims)          ν = mean((x−μ)², [0,2,3], keepdims)
        //   y = (x−μ)/√(ν+eps) · γ + β
        //
        // Biased variance (divide by N·H·W), matching PyTorch's training-mode
        // normalisation and the host twin in `:core/ops`.
        if (fqn == "io.tlaloc.core.ops.batchNorm") {
            val args = call.argumentList.arguments
            if (args.size != 2 && args.size != 3) {
                throw LoweringException(
                    "$fqn takes (scale, offset) or (scale, offset, eps); got ${args.size} arguments",
                )
            }
            val operandExpr = receiver(call) ?: throw LoweringException("$fqn has no receiver")
            val x = lowerExpr(operandExpr, env, emitter)
            if (x.type.rank != 4) {
                throw LoweringException("$fqn requires a rank-4 NCHW receiver; got ${x.type}")
            }
            if (x.type.dtype != F32) {
                throw LoweringException("$fqn is F32-only in v1; got ${x.type.dtype}")
            }
            fun unwrap(e: FirExpression): FirExpression =
                (e as? FirNamedArgumentExpression)?.expression ?: e
            val scale = lowerExpr(unwrap(args[0]), env, emitter)
            val offset = lowerExpr(unwrap(args[1]), env, emitter)
            val eps = if (args.size == 3) {
                floatLiteralArg(unwrap(args[2]))
                    ?: throw LoweringException("$fqn eps must be a Float literal")
            } else {
                1e-5f
            }
            for ((paramName, t) in listOf("scale" to scale, "offset" to offset)) {
                if (t.type.rank != 1) {
                    throw LoweringException("$fqn $paramName must be rank-1 [C]; got ${t.type}")
                }
                if (t.type.dims[0] > 0 && x.type.dims[1] > 0 && t.type.dims[0] != x.type.dims[1]) {
                    throw LoweringException(
                        "$fqn $paramName has ${t.type.dims[0]} channels but the input has " +
                            "${x.type.dims[1]}",
                    )
                }
            }

            // [C] → [1,C,1,1], a unit-axis RESHAPE, so the broadcast aligns on the
            // FEATURE axis: NumPy right-alignment of a bare [C] against [N,C,H,W]
            // would match C up with W.
            val perChannel = DxirType(x.type.dtype, listOf(1, x.type.dims[1], 1, 1))
            val scaleR = emitter.op(OpKind.RESHAPE, listOf(scale), perChannel)
            val offsetR = emitter.op(OpKind.RESHAPE, listOf(offset), perChannel)
            val reduceAxes = mapOf("reduction_dims" to listOf(0, 2, 3))
            val mean = emitter.op(OpKind.MEAN, listOf(x), perChannel, attrs = reduceAxes)
            val centred = emitter.op(OpKind.SUB, listOf(x, mean), x.type)
            val squared = emitter.op(OpKind.MUL, listOf(centred, centred), x.type)
            val variance = emitter.op(OpKind.MEAN, listOf(squared), perChannel, attrs = reduceAxes)
            // eps splats to the variance's shape (not x's), so the ADD is same-shape;
            // `splatLiteral` routes through a template BROADCAST under sentinels.
            val varEps = emitter.op(
                OpKind.ADD, listOf(variance, splatLiteral(eps, variance, emitter)), perChannel,
            )
            val std = emitter.op(OpKind.SQRT, listOf(varEps), perChannel)
            val normalised = emitter.op(OpKind.DIV, listOf(centred, std), x.type)
            val scaled = emitter.op(OpKind.MUL, listOf(normalised, scaleR), x.type)
            return emitter.op(OpKind.ADD, listOf(scaled, offsetR), x.type)
        }

        // §0.4.369 — Phase A4 (DiffKT parity): elementwise `maximum(a, b)` /
        // `minimum(a, b)` as sugar over the §0.4.364 where/compare surface.
        // `maximum` = WHERE(COMPARE(a, b, GE), a, b); `minimum` uses LE. The
        // COMPARE result is Bool (WHERE's pred contract), so unlike the
        // user-facing `where` (which re-derives Bool via `pred ≠ 0`) we feed
        // the compare directly. Same-shape (elementwise); result type = a.type.
        if (fqn == "io.tlaloc.core.ops.maximum" || fqn == "io.tlaloc.core.ops.minimum") {
            val args = call.argumentList.arguments
            if (args.size != 2) {
                throw LoweringException("$fqn requires 2 arguments; got ${args.size}")
            }
            val a = lowerExpr(args[0], env, emitter)
            val b = lowerExpr(args[1], env, emitter)
            val direction = if (fqn == "io.tlaloc.core.ops.maximum") "GE" else "LE"
            val cmp = emitter.op(
                kind = OpKind.COMPARE,
                operands = listOf(a, b),
                type = DxirType(Bool, a.type.dims),
                attrs = mapOf("direction" to direction),
            )
            return emitter.op(kind = OpKind.WHERE, operands = listOf(cmp, a, b), type = a.type)
        }

        // §0.4.369 — `clip(x, lo, hi)` = minimum(maximum(x, lo), hi). lo/hi are
        // compile-time Float literals lowered to splat consts of x's shape
        // (the const's sentinel-dim splat materialises in the gradient body via
        // irConstFor's axis-matching against x — the recomputed COMPARE(x, lo)
        // mask reads lo). Two COMPARE+WHERE pairs; the gradient is 1 inside
        // [lo, hi] and 0 outside.
        if (fqn == "io.tlaloc.core.ops.clip") {
            val args = call.argumentList.arguments
            if (args.size != 3) {
                throw LoweringException("clip requires 3 arguments (x, lo, hi); got ${args.size}")
            }
            val x = lowerExpr(args[0], env, emitter)
            val lo = floatLiteralArg(args[1])
                ?: throw LoweringException("clip lo bound must be a Float literal")
            val hi = floatLiteralArg(args[2])
                ?: throw LoweringException("clip hi bound must be a Float literal")
            if (lo > hi) throw LoweringException("clip: lo ($lo) must be ≤ hi ($hi)")
            val loConst = splatLiteral(lo, x, emitter)
            val hiConst = splatLiteral(hi, x, emitter)
            val geCmp = emitter.op(
                kind = OpKind.COMPARE,
                operands = listOf(x, loConst),
                type = DxirType(Bool, x.type.dims),
                attrs = mapOf("direction" to "GE"),
            )
            val maxed = emitter.op(kind = OpKind.WHERE, operands = listOf(geCmp, x, loConst), type = x.type)
            val leCmp = emitter.op(
                kind = OpKind.COMPARE,
                operands = listOf(maxed, hiConst),
                type = DxirType(Bool, x.type.dims),
                attrs = mapOf("direction" to "LE"),
            )
            return emitter.op(kind = OpKind.WHERE, operands = listOf(leCmp, maxed, hiConst), type = x.type)
        }

        // §0.4.369 — `outerProduct(a, b)` for rank-1 operands: the outer product
        // a[n]⊗b[m] IS the matmul reshape(a,[n,1]) × reshape(b,[1,m]) → [n,m], so
        // the gradient flows through MatmulRule ∘ ReshapeRule with no new AD
        // math. Dims may be sentinels — the RESHAPE result types carry them and
        // the synthesis reshape arm reads the axis-matched param dim. v1: rank-1
        // ⊗ rank-1 only (higher ranks would need a batched/general contraction).
        if (fqn == "io.tlaloc.core.ops.outerProduct") {
            val args = call.argumentList.arguments
            if (args.size != 2) {
                throw LoweringException("outerProduct requires 2 arguments; got ${args.size}")
            }
            val a = lowerExpr(args[0], env, emitter)
            val b = lowerExpr(args[1], env, emitter)
            if (a.type.rank != 1 || b.type.rank != 1) {
                throw LoweringException(
                    "outerProduct v1 supports rank-1 ⊗ rank-1 only; got rank ${a.type.rank} ⊗ ${b.type.rank}",
                )
            }
            val n = a.type.dims[0]
            val m = b.type.dims[0]
            val aU = emitter.op(
                kind = OpKind.RESHAPE,
                operands = listOf(a),
                type = DxirType(F32, listOf(n, 1)),
            )
            val bU = emitter.op(
                kind = OpKind.RESHAPE,
                operands = listOf(b),
                type = DxirType(F32, listOf(1, m)),
            )
            return emitter.op(
                kind = OpKind.MATMUL,
                operands = listOf(aU, bU),
                type = DxirType(F32, listOf(n, m)),
            )
        }

        // §0.4.364 — elementwise comparisons (the DiffKT-gap user surface).
        // `a gt b` lowers to COMPARE(direction):Bool + CAST back to the
        // operand dtype: the user-visible value is a 0/1 mask matching the
        // :core host semantics, while XLA still sees a genuine tensor<xi1>
        // through the intermediate. Can't go through BINARY_OP_MAP — the
        // direction attr and the Bool intermediate don't fit its
        // `type = lhs.type` single-op dispatch.
        COMPARE_DIRECTION_MAP[fqn]?.let { direction ->
            val lhsExpr = receiver(call)
                ?: throw LoweringException("comparison '$fqn' has no receiver")
            val lhs = lowerExpr(lhsExpr, env, emitter)
            val rhsExpr = call.argumentList.arguments.firstOrNull()
                ?: throw LoweringException("comparison '$fqn' missing rhs argument")
            val rhs = lowerExpr(rhsExpr, env, emitter)
            val cmp = emitter.op(
                kind = OpKind.COMPARE,
                operands = listOf(lhs, rhs),
                type = DxirType(Bool, lhs.type.dims),
                attrs = mapOf("direction" to direction),
            )
            return emitter.op(
                kind = OpKind.CAST,
                operands = listOf(cmp),
                type = DxirType(lhs.type.dtype, lhs.type.dims),
            )
        }

        // §0.4.364 — `where(pred, a, b)`: the differentiable routing
        // primitive. The user-level pred is a 0/1 F32 mask; WHERE's IR
        // contract wants Bool, so re-derive it via COMPARE(pred ≠ 0) —
        // XLA folds the round-trip, and the interpreter's Bool encoding
        // is the same 0f/1f floats either way.
        if (fqn == "io.tlaloc.core.ops.where") {
            val args = call.argumentList.arguments
            if (args.size != 3) {
                throw LoweringException("where requires 3 arguments (pred, a, b); got ${args.size}")
            }
            val pred = lowerExpr(args[0], env, emitter)
            val a = lowerExpr(args[1], env, emitter)
            val b = lowerExpr(args[2], env, emitter)
            val zero = splatLiteral(0.0f, pred, emitter)
            val boolPred = emitter.op(
                kind = OpKind.COMPARE,
                operands = listOf(pred, zero),
                type = DxirType(Bool, pred.type.dims),
                attrs = mapOf("direction" to "NE"),
            )
            return emitter.op(
                kind = OpKind.WHERE,
                operands = listOf(boolPred, a, b),
                type = a.type,
            )
        }

        BINARY_OP_MAP[fqn]?.let { kind ->
            val lhsExpr = receiver(call)
                ?: throw LoweringException("binary op '$fqn' has no receiver")
            val rhsExpr = call.argumentList.arguments.firstOrNull()
                ?: throw LoweringException("binary op '$fqn' missing rhs argument")
            // Phase A5 (DiffKT parity) — scalar × tensor mixing. `a * 2.0f` and
            // `3.0f - a` share these FQNs with the tensor⊙tensor operators (the
            // scalar overloads live in the same `io.tlaloc.core.ops` package), so
            // the mixed case arrives here as one rank-0 and one rank-N operand.
            // Lowering it verbatim leaves an ill-typed binary op — every consumer
            // downstream requires matching operands (the interpreter's size check,
            // the emitter's `broadcastIfNeeded`, and MulRule/DivRule, which type
            // their adjoint products as `upstream.type`) — and synthesis then
            // builds an IrCall passing a Float where a DTensor is expected.
            // Splatting the scalar side to the tensor side's type makes the op
            // well-typed with NO new op kind and NO new VjpRule: a literal splats
            // to a shaped const (the §0.4.369 `clip` bound pattern), a computed
            // rank-0 value splats through BROADCAST with an empty
            // `broadcast_dimensions` (the §0.4.359 scalar-seed polymorphism), and
            // the adjoint of a DIFFERENTIABLE scalar side falls out of
            // BroadcastRule as the runtime-extent SUM_TO full reduce (§0.4.373).
            // Operand order is preserved — `3.0f - a` is not `a - 3.0f`.
            // MATMUL is excluded: its operands are rank ≥ 2 by contract.
            if (kind in ELEMENTWISE_BINARY_KINDS) {
                lowerScalarMixedBinary(kind, lhsExpr, rhsExpr, env, emitter)?.let { return it }
            }
            val lhs = lowerExpr(lhsExpr, env, emitter)
            val rhs = lowerExpr(rhsExpr, env, emitter)
            // Phase A5c-2 — with implicit broadcasting the result of an elementwise
            // binary is the NumPy broadcast of its two operand shapes, not `lhs`'s.
            // Ranks are known even under `grad {}`'s -1 sentinels, so the result
            // rank is exact; an extent is exact only when both aligned operand
            // extents are concrete, and stays a sentinel otherwise.
            val resultType = if (kind in ELEMENTWISE_BINARY_KINDS) {
                broadcastResultType(lhs.type, rhs.type, fqn)
            } else {
                lhs.type
            }
            return emitter.op(
                kind = kind,
                operands = listOf(lhs, rhs),
                type = resultType,
            )
        }

        // §0.4.374 — single-axis `slice(start, end, axis)` (DiffKT parity,
        // Phase A2b). start/end/axis are compile-time literals: the result shape
        // is the operand's with `axis`'s extent replaced by `end − start`; the
        // non-sliced axes stay full (their `limit_indices` ride as the operand's
        // -1 SENTINEL dims, but only the sliced axis's start/limit are ever read
        // downstream — irSlice recovers (start, end, axis) from the explicit
        // `slice_axis` attr, and SliceRule's PAD_TO adjoint reads the operand's
        // full extent at runtime, never from these attrs).
        if (fqn == "io.tlaloc.core.ops.slice") {
            val operandExpr = receiver(call)
                ?: throw LoweringException("slice has no receiver")
            val operand = lowerExpr(operandExpr, env, emitter)
            val rank = operand.type.rank
            val intArgs = call.argumentList.arguments.map { arg ->
                intLiteralArg(arg) ?: throw LoweringException("slice args must be integer literals")
            }
            if (intArgs.size != 3) throw LoweringException("slice takes (start, end, axis)")
            val (start, end, rawAxis) = intArgs
            val axis = if (rawAxis < 0) rawAxis + rank else rawAxis
            if (axis !in 0 until rank) throw LoweringException("slice axis $rawAxis out of range for rank $rank")
            if (start < 0 || end < start) throw LoweringException("slice: invalid range [$start, $end)")
            val od = operand.type.dims[axis]
            if (od > 0 && end > od) throw LoweringException("slice: end $end exceeds axis $axis extent $od")
            val resultDims = operand.type.dims.mapIndexed { i, d -> if (i == axis) end - start else d }
            val starts = (0 until rank).map { if (it == axis) start else 0 }
            val limits = (0 until rank).map { if (it == axis) end else operand.type.dims[it] }
            return emitter.op(
                kind = OpKind.SLICE,
                operands = listOf(operand),
                type = DxirType(operand.type.dtype, resultDims),
                attrs = mapOf(
                    "start_indices" to starts,
                    "limit_indices" to limits,
                    "strides" to List(rank) { 1 },
                    "slice_axis" to axis,
                    "slice_start" to start,
                    "slice_end" to end,
                ),
            )
        }

        // Phase A2b — `concat(axis, vararg tensors)` and `stack(axis, vararg
        // tensors)` (DiffKT parity). Two things make these unlike every op lowered
        // so far:
        //
        //  * the operand count is the USER's, so the call arrives with a
        //    `FirVarargArgumentsExpression` to flatten (the `SHAPE_OP_SET` and
        //    `REDUCE_OP_MAP` arms already do this for `vararg Int` axes);
        //  * synthesis cannot build an `IrVararg` — the documented reason the whole
        //    `…RankN` host-shim family exists — so an n-ary CONCAT node would be
        //    unsynthesizable whenever it survived into a gradient body, which it
        //    does for the ordinary `concat(…).sum()` loss tail. Concat is
        //    associative along the axis, so the lowering folds n operands into a
        //    right-fold of BINARY CONCATs: unbounded in n, and each node matches the
        //    fixed-arity `concatPair` host op.
        //
        // `stack` is sugar and lowers as such: unsqueeze every operand at `axis`
        // (a unit-axis RESHAPE, which synthesis has handled since §0.4.375) and then
        // concat along it. Axis extents sum, so the result axis is a -1 sentinel
        // whenever any operand's is; every other extent must agree.
        if (fqn == "io.tlaloc.core.ops.concat" || fqn == "io.tlaloc.core.ops.stack") {
            val isStack = fqn == "io.tlaloc.core.ops.stack"
            val args = call.argumentList.arguments
            if (args.isEmpty()) throw LoweringException("'$fqn' takes (axis, vararg tensors)")
            val rawAxis = intLiteralArg(args[0])
                ?: throw LoweringException("'$fqn' axis must be an Int literal")
            val tensorExprs = mutableListOf<FirExpression>()
            for (arg in args.drop(1)) {
                if (arg is FirVarargArgumentsExpression) tensorExprs += arg.arguments else tensorExprs += arg
            }
            if (tensorExprs.size < 2) {
                throw LoweringException("'$fqn' needs at least 2 tensors, got ${tensorExprs.size}")
            }
            var nodes = tensorExprs.map { lowerExpr(it, env, emitter) }
            val operandRank = nodes[0].type.rank
            val axis = if (rawAxis < 0) rawAxis + operandRank + (if (isStack) 1 else 0) else rawAxis
            if (isStack) {
                if (axis !in 0..operandRank) {
                    throw LoweringException("stack axis $rawAxis out of range for result rank ${operandRank + 1}")
                }
                nodes = nodes.map { n ->
                    val dims = (0..n.type.rank).map { i ->
                        when {
                            i == axis -> 1
                            i < axis -> n.type.dims[i]
                            else -> n.type.dims[i - 1]
                        }
                    }
                    emitter.op(kind = OpKind.RESHAPE, operands = listOf(n), type = DxirType(n.type.dtype, dims))
                }
            }
            val rank = nodes[0].type.rank
            if (axis !in 0 until rank) throw LoweringException("'$fqn' axis $rawAxis out of range for rank $rank")
            for (n in nodes) {
                if (n.type.rank != rank) {
                    throw LoweringException("'$fqn' operands must share a rank (${n.type.rank} vs $rank)")
                }
                if (n.type.dtype != nodes[0].type.dtype) {
                    throw LoweringException("'$fqn' operands must share a dtype (${n.type.dtype} vs ${nodes[0].type.dtype})")
                }
            }
            var acc = nodes[0]
            for (i in 1 until nodes.size) {
                acc = emitter.op(
                    kind = OpKind.CONCAT,
                    operands = listOf(acc, nodes[i]),
                    type = concatResultType(acc.type, nodes[i].type, axis, fqn),
                    attrs = mapOf("dimension" to axis),
                )
            }
            return acc
        }

        // §0.4.367 — shape ops (DiffKT parity, Phase A2a): the RESHAPE family
        // (`squeeze(axis)` / `unsqueeze(axis)` / `flatten()` / `reshape(dims)`)
        // and permutation `transpose(perm)` (no-arg = reverse all axes; the
        // rank-2 `.transpose()` receiver spelling included). Axis positions
        // and target dims are compile-time literals folded into the result
        // DxirType (and TRANSPOSE's `permutation` attr) here. Flatten's
        // result dim is the operand's element count — a product of possibly
        // -1 sentinels, so any symbolic operand flattens to the rank-1
        // sentinel [-1].
        if (fqn in SHAPE_OP_SET) {
            val operandExpr = receiver(call)
                ?: throw LoweringException("shape op '$fqn' has no receiver")
            val operand = lowerExpr(operandExpr, env, emitter)
            val rank = operand.type.rank
            val args = call.argumentList.arguments
            val intArgs = mutableListOf<Int>()
            for (arg in args) {
                when (arg) {
                    is FirVarargArgumentsExpression ->
                        for (e in arg.arguments) {
                            intArgs += intLiteralArg(e) ?: throw LoweringException(
                                "shape op '$fqn' arguments must be integer literals",
                            )
                        }
                    else -> {
                        intArgs += intLiteralArg(arg) ?: throw LoweringException(
                            "shape op '$fqn' arguments must be integer literals",
                        )
                    }
                }
            }
            when (fqn) {
                "io.tlaloc.core.ops.squeeze" -> {
                    if (intArgs.size != 1) throw LoweringException("squeeze takes exactly one axis")
                    val a = intArgs[0].let { if (it < 0) it + rank else it }
                    if (a !in 0 until rank) {
                        throw LoweringException("squeeze axis ${intArgs[0]} out of range for rank $rank")
                    }
                    val d = operand.type.dims[a]
                    if (d > 0 && d != 1) {
                        throw LoweringException("squeeze axis $a has size $d (must be 1)")
                    }
                    val resultDims = operand.type.dims.filterIndexed { i, _ -> i != a }
                    return emitter.op(
                        kind = OpKind.RESHAPE,
                        operands = listOf(operand),
                        type = DxirType(operand.type.dtype, resultDims),
                    )
                }
                "io.tlaloc.core.ops.unsqueeze" -> {
                    if (intArgs.size != 1) throw LoweringException("unsqueeze takes exactly one axis")
                    val outRank = rank + 1
                    val a = intArgs[0].let { if (it < 0) it + outRank else it }
                    if (a !in 0 until outRank) {
                        throw LoweringException("unsqueeze axis ${intArgs[0]} out of range for result rank $outRank")
                    }
                    val resultDims = buildList {
                        addAll(operand.type.dims)
                        add(a, 1)
                    }
                    return emitter.op(
                        kind = OpKind.RESHAPE,
                        operands = listOf(operand),
                        type = DxirType(operand.type.dtype, resultDims),
                    )
                }
                "io.tlaloc.core.ops.flatten" -> {
                    if (intArgs.isNotEmpty()) throw LoweringException("flatten takes no arguments")
                    val n = if (operand.type.dims.any { it < 0 }) -1
                    else operand.type.dims.fold(1) { acc, d -> acc * d }
                    return emitter.op(
                        kind = OpKind.RESHAPE,
                        operands = listOf(operand),
                        type = DxirType(operand.type.dtype, listOf(n)),
                    )
                }
                "io.tlaloc.core.ops.reshape" -> {
                    if (intArgs.isEmpty()) throw LoweringException("reshape requires target dims")
                    if (intArgs.any { it <= 0 }) {
                        throw LoweringException("reshape dims must be positive literals, got $intArgs")
                    }
                    if (operand.type.dims.all { it > 0 }) {
                        val have = operand.type.dims.fold(1) { acc, d -> acc * d }
                        val want = intArgs.fold(1) { acc, d -> acc * d }
                        if (have != want) {
                            throw LoweringException(
                                "reshape element count mismatch: ${operand.type.dims} vs $intArgs",
                            )
                        }
                    }
                    return emitter.op(
                        kind = OpKind.RESHAPE,
                        operands = listOf(operand),
                        type = DxirType(operand.type.dtype, intArgs.toList()),
                    )
                }
                "io.tlaloc.core.ops.transpose" -> {
                    val perm = if (intArgs.isEmpty()) (rank - 1 downTo 0).toList()
                    else intArgs.map { p ->
                        val a = if (p < 0) p + rank else p
                        if (a !in 0 until rank) {
                            throw LoweringException("transpose axis $p out of range for rank $rank")
                        }
                        a
                    }
                    if (perm.size != rank || perm.toSet().size != rank) {
                        throw LoweringException("transpose perm $intArgs is not a rank-$rank permutation")
                    }
                    val resultDims = perm.map { operand.type.dims[it] }
                    return emitter.op(
                        kind = OpKind.TRANSPOSE,
                        operands = listOf(operand),
                        type = DxirType(operand.type.dtype, resultDims),
                        attrs = mapOf("permutation" to perm),
                    )
                }
                "io.tlaloc.core.ops.broadcastTo" -> {
                    // §0.4.371 — Phase A2b: rank-increasing broadcast (DiffKT
                    // `broadcastTo`/`expand`, NumPy right-alignment). The operand's
                    // `rank` axes map to the TRAILING axes of the target; the new
                    // leading axes are replicated. `broadcast_dimensions` is the
                    // right-aligned suffix [outRank-rank .. outRank-1] — pure
                    // compile-time POSITIONS, so BroadcastRule's adjoint (a SUM over
                    // the complement = the new leading axes) is sentinel-safe: it
                    // never reads an operand extent. In-place size-1 stretch (e.g.
                    // [1,C]→[N,C]) is deliberately NOT supported here — its adjoint
                    // would need to know which operand axis was size-1, unknowable
                    // under the -1 sentinel dims of `grad {}` (deferred, see plan).
                    if (intArgs.isEmpty()) throw LoweringException("broadcastTo requires target dims")
                    if (intArgs.any { it <= 0 }) {
                        throw LoweringException("broadcastTo dims must be positive literals, got $intArgs")
                    }
                    val outRank = intArgs.size
                    if (outRank < rank) {
                        throw LoweringException(
                            "broadcastTo target rank $outRank < operand rank $rank (only new leading axes)",
                        )
                    }
                    val offset = outRank - rank
                    // §0.4.373 — trailing target dims must either MATCH the
                    // operand (only enforceable when the operand dim is concrete)
                    // or be an in-place size-1 stretch (operand dim == 1 → the
                    // BroadcastRule adjoint sums the stretched axis via SUM_TO,
                    // reading the extent at runtime). Under sentinels the runtime
                    // host op / interpreter fail loudly on a genuine mismatch.
                    // The MIXED case — a simultaneous rank-increase (offset > 0)
                    // AND an aligned size-1 stretch — stays deferred: its adjoint
                    // (the non-empty-reduceDims SUM path) can't ALSO sum a
                    // stretched aligned axis, so guard it when concretely
                    // detectable (see the DIFFKT_PARITY_PLAN A2b deferral note).
                    for (j in 0 until rank) {
                        val od = operand.type.dims[j]
                        val target = intArgs[offset + j]
                        if (od > 1 && od != target) {
                            throw LoweringException(
                                "broadcastTo: operand dim $j = $od does not match target " +
                                    "$target (not an equal dim nor a size-1 stretch)",
                            )
                        }
                        if (offset > 0 && od == 1 && target > 1) {
                            throw LoweringException(
                                "broadcastTo: simultaneous rank-increase and in-place size-1 " +
                                    "stretch (axis $j: 1 -> $target) is unsupported in grad {} " +
                                    "(deferred); split into reshape + broadcastTo",
                            )
                        }
                    }
                    val bcastDims = (0 until rank).map { offset + it }
                    return emitter.op(
                        kind = OpKind.BROADCAST,
                        operands = listOf(operand),
                        type = DxirType(operand.type.dtype, intArgs.toList()),
                        attrs = mapOf("broadcast_dimensions" to bcastDims),
                    )
                }
            }
        }

        // §0.4.366 — axis-wise reductions (DiffKT parity, Phase A1):
        // `x.sum(1)`, `x.mean(0, keepDims = true)`, `x.max(0, 1)`. The vararg
        // axis arguments and the `keepDims` flag must be compile-time literals;
        // they fold into the op's `reduction_dims` attr and its result
        // [DxirType] here — the IR carries the exact reduced shape even though
        // the host-side Kotlin return type erases to `Shape`. No-arg calls
        // fall through to the UNARY_OP_MAP full-reduce arm below. There is
        // deliberately no `keep_dims` attr: keepdims-vs-squeeze is expressed
        // structurally by the result type (the emitter's `resolveReduceShape`
        // and the interpreter both disambiguate by rank — the established
        // §0.4.359 convention).
        REDUCE_OP_MAP[fqn]?.let { kind ->
            val args = call.argumentList.arguments
            if (args.isNotEmpty()) {
                val operandExpr = receiver(call)
                    ?: throw LoweringException("reduction '$fqn' has no receiver")
                val operand = lowerExpr(operandExpr, env, emitter)
                val rank = operand.type.rank
                val dims = mutableListOf<Int>()
                var keepDims = false
                for (arg in args) {
                    when {
                        arg is FirVarargArgumentsExpression ->
                            for (e in arg.arguments) {
                                dims += intLiteralArg(e) ?: throw LoweringException(
                                    "reduction '$fqn' axis arguments must be integer literals",
                                )
                            }
                        arg is FirNamedArgumentExpression -> {
                            val inner = arg.expression
                            if (inner is FirLiteralExpression && inner.kind == ConstantValueKind.Boolean) {
                                keepDims = inner.value as Boolean
                            } else {
                                throw LoweringException(
                                    "reduction '$fqn' keepDims must be a Boolean literal",
                                )
                            }
                        }
                        arg is FirLiteralExpression && arg.kind == ConstantValueKind.Boolean ->
                            keepDims = arg.value as Boolean
                        else -> {
                            dims += intLiteralArg(arg) ?: throw LoweringException(
                                "reduction '$fqn' axis arguments must be integer literals",
                            )
                        }
                    }
                }
                if (dims.isEmpty()) {
                    throw LoweringException("reduction '$fqn' requires at least one axis argument")
                }
                val normalized = dims.map { d ->
                    val a = if (d < 0) d + rank else d
                    if (a !in 0 until rank) {
                        throw LoweringException("reduction '$fqn' axis $d out of range for rank $rank")
                    }
                    a
                }.sorted()
                if (normalized.toSet().size != normalized.size) {
                    throw LoweringException("reduction '$fqn' has duplicate axes $dims")
                }
                val resultDims = if (keepDims) {
                    operand.type.dims.mapIndexed { i, d -> if (i in normalized) 1 else d }
                } else {
                    operand.type.dims.filterIndexed { i, _ -> i !in normalized }
                }
                return emitter.op(
                    kind = kind,
                    operands = listOf(operand),
                    type = DxirType(operand.type.dtype, resultDims),
                    attrs = mapOf("reduction_dims" to normalized),
                )
            }
        }

        UNARY_OP_MAP[fqn]?.let { kind ->
            // §0.4.395 — the operand is the receiver for the extension spellings
            // (`x.tan()`), or the sole positional argument for the top-level
            // `kotlin.math` spellings (`tan(x)` — no receiver at all). Every
            // UNARY_OP_MAP entry is arity-1, so the fallback is unambiguous.
            val operandExpr = receiver(call)
                ?: call.argumentList.arguments.singleOrNull()
                ?: throw LoweringException("unary op '$fqn' has no receiver")
            val operand = lowerExpr(operandExpr, env, emitter)
            // Reductions collapse all operand dims to scalar — every other unary op is
            // shape-preserving. Keeping this as a per-kind branch here (rather than a
            // second OP_MAP) matches the BINARY_OP_MAP convention: the receiver side
            // inspects only the `kind`, never the FQN itself.
            val resultType = when (kind) {
                OpKind.SUM, OpKind.MEAN, OpKind.MAX, OpKind.MIN ->
                    DxirType(operand.type.dtype, emptyList())
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

    /**
     * §0.4.366 — constant-fold an integer axis argument. Accepts a plain
     * integer literal or the FIR spelling of a negative literal (`-1` resolves
     * to `1.unaryMinus()` — a [FirFunctionCall] over a literal receiver).
     * Returns null for anything non-constant; callers turn that into a
     * [LoweringException] naming the reduction.
     */
    private fun intLiteralArg(expr: FirExpression): Int? {
        if (expr is FirLiteralExpression && expr.value is Number) {
            return (expr.value as Number).toInt()
        }
        if (expr is FirFunctionCall) {
            val id = expr.calleeReference.toResolvedCallableSymbol()?.callableId
            if (id?.callableName?.asString() == "unaryMinus") {
                val rec = expr.dispatchReceiver ?: expr.extensionReceiver
                if (rec is FirLiteralExpression && rec.value is Number) {
                    return -(rec.value as Number).toInt()
                }
            }
        }
        return null
    }

    /**
     * §0.4.369 — constant-fold a Float scalar bound (clip's `lo`/`hi`).
     * Accepts any numeric literal (Float/Double/Int spelling) and the negative
     * literal form (`-1.0f` = `1.0f.unaryMinus()`). Returns null for anything
     * non-constant; callers turn that into a [LoweringException].
     */
    private fun floatLiteralArg(expr: FirExpression): Float? {
        if (expr is FirLiteralExpression && expr.value is Number) {
            return (expr.value as Number).toFloat()
        }
        if (expr is FirFunctionCall) {
            val id = expr.calleeReference.toResolvedCallableSymbol()?.callableId
            if (id?.callableName?.asString() == "unaryMinus") {
                val rec = expr.dispatchReceiver ?: expr.extensionReceiver
                if (rec is FirLiteralExpression && rec.value is Number) {
                    return -(rec.value as Number).toFloat()
                }
            }
        }
        return null
    }

    /**
     * Phase A5 — the mixed-rank arm of the [BINARY_OP_MAP] dispatch: one side is
     * an F32 `DTensor`, the other a scalar. Returns the lowered op, or null when
     * the call is not a scalar × tensor mix (so the caller's uniform path runs).
     *
     * Which side is the tensor is read off the FIR types rather than off lowered
     * nodes, so the literal case can splat straight to a shaped const without
     * first materialising — and then orphaning — a rank-0 const: the same IR shape
     * the §0.4.369 `clip` arm produces for its Float bounds. A COMPUTED scalar
     * side (a rank-0 value: a reduction result, a scalar param, a `val`) has no
     * literal to fold, so it splats through [splatScalarTo]'s BROADCAST form.
     *
     * Sentinel-safe either way: the splat's target dims are the tensor operand's
     * (possibly -1), and synthesis materialises them at runtime by axis-matching
     * the params, so no dim value is ever baked.
     */
    private fun lowerScalarMixedBinary(
        kind: OpKind,
        lhsExpr: FirExpression,
        rhsExpr: FirExpression,
        env: MutableMap<Any, DxirNode>,
        emitter: DxirEmitter,
    ): DxirNode? {
        val lhsIsTensor = isDTensorExpr(lhsExpr)
        if (lhsIsTensor == isDTensorExpr(rhsExpr)) return null
        val tensor = lowerExpr(if (lhsIsTensor) lhsExpr else rhsExpr, env, emitter)
        val scalarExpr = if (lhsIsTensor) rhsExpr else lhsExpr
        // A rank-0 DTensor operand (both sides scalar) or a non-F32 tensor needs no
        // splat. Finish the uniform lowering here rather than returning null — the
        // tensor side is already lowered, and letting the caller lower it again
        // would emit it twice.
        if (tensor.type.isScalar || tensor.type.dtype != F32) {
            val scalarSide = lowerExpr(scalarExpr, env, emitter)
            val operands = if (lhsIsTensor) listOf(tensor, scalarSide) else listOf(scalarSide, tensor)
            return emitter.op(kind = kind, operands = operands, type = operands[0].type)
        }
        val literal = floatLiteralArg(scalarExpr)
        val splat = if (literal != null) {
            splatLiteral(literal, tensor, emitter)
        } else {
            splatScalarTo(lowerExpr(scalarExpr, env, emitter), tensor, emitter)
        }
        val operands = if (lhsIsTensor) listOf(tensor, splat) else listOf(splat, tensor)
        return emitter.op(kind = kind, operands = operands, type = tensor.type)
    }

    /**
     * Phase A5c-3 — a literal splatted over [tensor]'s shape.
     *
     * Concrete dims bake a shaped const: synthesis materialises every extent as a
     * const and there is nothing to guess. SYMBOLIC dims carry [tensor] as a
     * shape-only BROADCAST template instead, because a shaped const has no runtime
     * shape source of its own — synthesis would have to pick its extents by
     * axis-matching static atoms against the params, and once operands broadcast that
     * guess can land on the wrong param (`[3,3]` where the tensor is `[2,3]`).
     */
    private fun splatLiteral(value: Any, tensor: DxirNode, emitter: DxirEmitter): DxirNode =
        if (tensor.type.dims.any { it <= 0 }) {
            emitter.op(
                kind = OpKind.BROADCAST,
                operands = listOf(emitter.const(value, DxirType(tensor.type.dtype, emptyList())), tensor),
                type = tensor.type,
                attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
            )
        } else {
            emitter.const(value, tensor.type)
        }

    /** True when [expr]'s resolved FIR type is `io.tlaloc.core.DTensor` (any shape / dtype args). */
    private fun isDTensorExpr(expr: FirExpression): Boolean =
        expr.resolvedType.classId?.asString() == "io/tlaloc/core/DTensor"

    /**
     * Phase A5c-2 — the NumPy broadcast of two operand shapes: right-aligned, each
     * aligned pair equal or 1 on one side, the result taking the max of each pair, a
     * rank-deficient operand gaining replicated leading axes.
     *
     * Sentinel-safe by construction. Under `grad {}` the operand dims are -1
     * placeholders, but the RANKS come from the call-site type and are exact, so the
     * result rank always is too. An extent is exact only when both aligned extents
     * are concrete (or one is a literal 1 — keepdims axes and `Lit<Int>` atoms the
     * user pinned); anything touching a sentinel stays a sentinel and is resolved at
     * runtime by the host broadcasting op. Incompatible concrete extents are a
     * [LoweringException] at the call site rather than a runtime shape error.
     *
     * Axis names follow the wider operand (the one whose rank equals the result's);
     * a broadcast that would have to invent names for the new leading axes leaves
     * them null, and an unnamed pair stays `emptyList()` so the type still compares
     * equal to the unnamed types the rest of the pipeline builds.
     */
    private fun broadcastResultType(a: DxirType, b: DxirType, fqn: String): DxirType {
        if (a.dims == b.dims) return a
        if (a.dtype != b.dtype) {
            throw LoweringException(
                "'$fqn' operands have different dtypes (${a.dtype} vs ${b.dtype}); broadcasting is shape-only",
            )
        }
        val r = maxOf(a.dims.size, b.dims.size)
        val dims = (0 until r).map { k ->
            val x = alignedDim(a.dims, k, r)
            val y = alignedDim(b.dims, k, r)
            when {
                x == y -> x
                x == 1 -> y
                y == 1 -> x
                x < 0 || y < 0 -> -1
                else -> throw LoweringException(
                    "'$fqn' operand shapes ${a.dims} and ${b.dims} are not broadcast-compatible " +
                        "at result axis $k ($x vs $y, neither is 1)",
                )
            }
        }
        val wider = if (a.dims.size >= b.dims.size) a else b
        val axisNames = if (wider.hasNamedAxes) {
            val offset = r - wider.axisNames.size
            List(r) { k -> if (k >= offset) wider.axisNames[k - offset] else null }
        } else {
            emptyList()
        }
        return DxirType(a.dtype, dims, axisNames)
    }

    /** Right-aligned axis [k] of [dims] within a rank-[rank] result: 1 for axes the operand lacks. */
    private fun alignedDim(dims: List<Int>, k: Int, rank: Int): Int {
        val i = k - (rank - dims.size)
        return if (i < 0) 1 else dims[i]
    }

    /**
     * Phase A2b — the result type of a binary CONCAT along [axis]: the axis extents
     * SUM and every other extent must agree. Sentinel-propagating like
     * [broadcastResultType] — under `grad {}` the operands' extents are -1, so the
     * sum is too, and the concat axis's runtime extent is recovered by the host
     * `concatPair` (and, in the adjoint, by `SLICE_LIKE`'s templates). A concrete
     * disagreement on a non-axis extent is a call-site [LoweringException] rather
     * than a runtime shape error.
     */
    private fun concatResultType(a: DxirType, b: DxirType, axis: Int, fqn: String): DxirType {
        val dims = a.dims.indices.map { i ->
            val x = a.dims[i]
            val y = b.dims[i]
            if (i == axis) {
                if (x <= 0 || y <= 0) -1 else x + y
            } else {
                if (x > 0 && y > 0 && x != y) {
                    throw LoweringException("'$fqn' operands disagree on non-axis $i extent ($x vs $y)")
                }
                if (x <= 0 || y <= 0) -1 else x
            }
        }
        return DxirType(a.dtype, dims)
    }

    /**
     * Layer 1 §0.4.241+ + Layer 1.5 §0.4.242+ — emit a [OpKind.MATMUL] (or
     * [OpKind.DOT] for the rank-1 × rank-1 case) for a named-index
     * `contract` call.
     *
     * Pre-condition: the call site reached us only because Kotlin's
     * overload resolution succeeded, which means both operands carry
     * named axes with at least one shared [io.tlaloc.core.IndexName].
     *
     * Layer 1.5 generalises the v1 "exactly one shared name" rule to
     * support batched contractions:
     *
     * - **Batching axes**: shared names that appear at the *same dim
     *   position* in both operands. These are preserved in the output one-
     *   to-one (StableHLO `dot_general` batching dims).
     * - **Contracting axes**: shared names that appear at *different dim
     *   positions* in lhs vs rhs. Summed over and removed from the output
     *   (StableHLO `dot_general` contracting dims). v1.5 still requires
     *   exactly one contracting axis (the position-mismatch case).
     *
     * The position-based partition is the natural reading of the user's
     * type-level overload: when the overload signature shares a type
     * variable (`NameB` at lhs[0] = rhs[0]), the user is asserting "this
     * is the same index"; when it shares it at different positions, "this
     * gets summed."
     *
     * Result-axis ordering follows StableHLO convention: batching dims
     * first (in lhs-position order), then lhs preserved, then rhs preserved
     * — matches the type-level overloads' declared output shape.
     *
     * Throws [LoweringException] on any post-condition violation; that
     * path surfaces as a `NAMED_INDEX_MISMATCH` diagnostic via the
     * existing failure handler in [lower].
     */
    private fun emitContract(lhs: DxirNode, rhs: DxirNode, emitter: DxirEmitter): DxirNode {
        val lhsType = lhs.type
        val rhsType = rhs.type
        val lhsNames = lhsType.axisNames
        val rhsNames = rhsType.axisNames
        if (lhsNames.isEmpty() || rhsNames.isEmpty()) {
            throw NamedIndexException(
                "contract requires both operands to carry named axes; got lhs=$lhsType rhs=$rhsType",
            )
        }
        val lhsByName = lhsNames.withIndex().mapNotNull { (i, n) -> n?.let { it to i } }.toMap()
        val rhsByName = rhsNames.withIndex().mapNotNull { (i, n) -> n?.let { it to i } }.toMap()
        val sharedNames = lhsByName.keys.intersect(rhsByName.keys)
        if (sharedNames.isEmpty()) {
            throw NamedIndexException(
                "contract operands share no named axis: lhs=$lhsNames rhs=$rhsNames",
            )
        }

        // Partition shared names by position alignment:
        //   - same position  → batching axis (preserved in output)
        //   - different pos  → contracting axis (summed over)
        //
        // Special case for rank-1 × rank-1: the unique shared axis MUST be
        // contracting (no point in calling `contract` if there's nothing to
        // sum). The position-based heuristic alone would misfire here —
        // both operands have the axis at position 0, which would otherwise
        // categorize as batching. This matches the user-facing semantics of
        // the rank-1 `contract` overload (`Rank1<Named<K, _>> contract Rank1<Named<K, _>>: ScalarShape`).
        val isRank1Pair = lhsType.rank == 1 && rhsType.rank == 1
        val batchingNames = if (isRank1Pair) {
            emptySet<String>()
        } else {
            sharedNames.filter { lhsByName[it] == rhsByName[it] }.toSet()
        }
        val contractingNames = sharedNames - batchingNames
        if (contractingNames.size != 1) {
            throw NamedIndexException(
                "contract supports exactly one contracting axis in v1.5 " +
                    "(got contracting=$contractingNames batching=$batchingNames). " +
                    "A shared name at the *same* dim position is batching; at *different* positions, contracting.",
            )
        }
        val contractingName = contractingNames.single()
        val lhsContractDim = lhsByName.getValue(contractingName)
        val rhsContractDim = rhsByName.getValue(contractingName)

        // Sort batching dims by lhs position to give a stable + emitter-
        // friendly ordering. Rhs ordering follows the same name ordering.
        val sortedBatching = batchingNames.sortedBy { lhsByName.getValue(it) }
        val lhsBatchingDims = sortedBatching.map { lhsByName.getValue(it) }
        val rhsBatchingDims = sortedBatching.map { rhsByName.getValue(it) }
        val batchingPositionsLhs = lhsBatchingDims.toSet()
        val batchingPositionsRhs = rhsBatchingDims.toSet()

        // Build result dims + axisNames + preserved-name list.
        // Output ordering: [batching dims (lhs order), lhs preserved, rhs preserved].
        val resultDims = ArrayList<Int>(lhsType.rank + rhsType.rank - sharedNames.size - 1)
        val resultNames = ArrayList<String?>(lhsType.rank + rhsType.rank - sharedNames.size - 1)
        val preservedNames = ArrayList<String>()
        for (i in lhsBatchingDims) {
            resultDims += lhsType.dims[i]
            resultNames += lhsNames[i]
            lhsNames[i]?.let { preservedNames += it }
        }
        for (i in lhsNames.indices) {
            if (i == lhsContractDim) continue
            if (i in batchingPositionsLhs) continue
            resultDims += lhsType.dims[i]
            resultNames += lhsNames[i]
            lhsNames[i]?.let { preservedNames += it }
        }
        for (i in rhsNames.indices) {
            if (i == rhsContractDim) continue
            if (i in batchingPositionsRhs) continue
            resultDims += rhsType.dims[i]
            resultNames += rhsNames[i]
            rhsNames[i]?.let { preservedNames += it }
        }
        val resultAxisNames = if (resultNames.all { it == null }) emptyList() else resultNames.toList()

        // Op kind: rank-1 × rank-1 with no batching produces a scalar (DOT);
        // everything else is matmul-shaped (MATMUL). The emitter's MATMUL
        // explicit path reads the four list attrs, so we always populate them.
        val isDot = lhsType.rank == 1 && rhsType.rank == 1 && batchingNames.isEmpty()
        val kind = if (isDot) OpKind.DOT else OpKind.MATMUL
        val attrs: Map<String, Any> = mapOf(
            "lhs_contracting_dims" to listOf(lhsContractDim),
            "rhs_contracting_dims" to listOf(rhsContractDim),
            "lhs_batching_dims" to lhsBatchingDims,
            "rhs_batching_dims" to rhsBatchingDims,
            "contracted_names" to setOf(contractingName),
            "preserved_names" to preservedNames.toList(),
            "batching_names" to sortedBatching.toList(),
        )
        val resultType = DxirType(lhsType.dtype, resultDims, resultAxisNames)
        return emitter.op(
            kind = kind,
            operands = listOf(lhs, rhs),
            type = resultType,
            attrs = attrs,
        )
    }

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
            // §0.4.242 — Rank4 / Rank5 / Rank6 added for Layer 1.5 (closes
            // OQ-5: rank-3+ batched named contraction). The axis-name walker
            // below already iterates `0 until rank` so it lifts named axes
            // for any rank uniformly; the only required change here is the
            // FQN → rank lookup. Synthesis-side acceptance for rank ≥ 4 is
            // a separate concern (the rank-2/3 BROADCAST gate from §0.4.186
            // remains the synthesis-side ceiling); FIR-stage recognition is
            // sufficient for `LAMBDA_LOWERED` to fire on rank-4 grad bodies.
            val rank: Int = when (shapeFqn) {
                "io/tlaloc/core/ScalarShape" -> 0
                "io/tlaloc/core/Rank1" -> 1
                "io/tlaloc/core/Rank2" -> 2
                "io/tlaloc/core/Rank3" -> 3
                "io/tlaloc/core/Rank4" -> 4
                "io/tlaloc/core/Rank5" -> 5
                "io/tlaloc/core/Rank6" -> 6
                else -> return null
            }
            val dims: List<Int> = List(rank) { -1 }
            // §0.4.241 — Layer 1 named-index lift. Walk the shape's type arguments
            // and, where an atom is `Named<N, A>` (FQN `io/tlaloc/core/Named`),
            // extract the IndexName subtype's simple class name as the axis name.
            // For positional atoms (Sym, Lit, Mul, Add, ...) we record null. The
            // resulting axisNames list is empty when no named atoms appeared, so
            // pre-Layer-1 callers and tests stay byte-for-byte equivalent.
            //
            // v1 lifts the IndexName *class simple name* (e.g. `Batch` -> "Batch")
            // rather than the singleton's `override val name` property. Reading the
            // const initializer would require a FirSession plumbed through here,
            // which is disproportionate plumbing for v1; downstream consumers that
            // want the friendly name can recover it from the FQN. v2 will switch
            // to reading the property — tracked as an open question in §18.
            val axisNames: List<String?> = if (rank == 0) {
                emptyList()
            } else {
                val shapeArgs = shapeType.typeArguments
                val names = ArrayList<String?>(rank)
                var anyNamed = false
                for (i in 0 until rank) {
                    val atomType = shapeArgs.getOrNull(i)?.type
                    val atomFqn = atomType?.classId?.asString()
                    if (atomFqn == "io/tlaloc/core/Named") {
                        val nameArg = atomType.typeArguments.firstOrNull()?.type
                        val nameClassId = nameArg?.classId
                        val nameStr = nameClassId?.shortClassName?.asString()
                        names += nameStr
                        if (nameStr != null) anyNamed = true
                    } else {
                        names += null
                    }
                }
                if (anyNamed) names else emptyList()
            }
            return DxirType(dtype, dims, axisNames)
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

    /**
     * §0.4.364 — `:core/ops` comparison FQN → COMPARE `direction` attr
     * (stablehlo.compare's spelling). Consumed by the dedicated dispatch
     * arm (COMPARE + CAST pair), not BINARY_OP_MAP.
     */
    private val COMPARE_DIRECTION_MAP: Map<String, String> = mapOf(
        "io.tlaloc.core.ops.gt" to "GT",
        "io.tlaloc.core.ops.ge" to "GE",
        "io.tlaloc.core.ops.lt" to "LT",
        "io.tlaloc.core.ops.le" to "LE",
        "io.tlaloc.core.ops.eq" to "EQ",
        "io.tlaloc.core.ops.ne" to "NE",
    )

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
        // §0.4.364 — the :core/ops DTensor elementwise operators (HostOps.kt).
        // Until now only matmul lowered from the tensor operator surface —
        // `a * b` on same-shape DTensors hit "unsupported call". Elementwise
        // `type = lhs.type` dispatch is exactly right for these. (The
        // scalar-mixing overloads — `a * 2.0f`, `3.0f - a` — share these FQNs
        // and arrive as one rank-0 and one rank-N operand; the mixed-rank arm
        // above splats the scalar side, per Phase A5.)
        put("io.tlaloc.core.ops.plus", OpKind.ADD)
        put("io.tlaloc.core.ops.minus", OpKind.SUB)
        put("io.tlaloc.core.ops.times", OpKind.MUL)
        put("io.tlaloc.core.ops.div", OpKind.DIV)
        // Phase A5b (DiffKT parity) — `pow`. POW has been fully ruled below the
        // surface since Stage B.3 (PowRule incl. the integer-exponent CAST, the
        // interpreter arm, `stablehlo.power` emission, the forward-mode tangent and
        // synthesis's scalar `kotlin.math.pow` arm) but had no lowering entry, so
        // it was an orphan. Both spellings map here: the `:core/ops` tensor
        // extension (`a.pow(2.0f)` / `a.pow(b)`) and `kotlin.math.pow` on scalars
        // (`x.pow(n)` inside a `grad { x: Float -> … }` body, which the scalar
        // synthesis arm already handles). A literal exponent arrives as the
        // mixed-rank case and is splatted by the Phase A5a arm, so the IR always
        // sees the uniform two-operand POW PowRule expects.
        put("io.tlaloc.core.ops.pow", OpKind.POW)
        put("kotlin.math.pow", OpKind.POW)
    }

    /**
     * Phase A5 — the [BINARY_OP_MAP] kinds whose operands must agree in shape,
     * i.e. the ones the mixed-rank scalar-splat arm applies to. MATMUL is in the
     * same map but contracts rank ≥ 2 operands, where a scalar side is a type
     * error rather than a broadcast.
     */
    private val ELEMENTWISE_BINARY_KINDS: Set<OpKind> =
        setOf(OpKind.ADD, OpKind.SUB, OpKind.MUL, OpKind.DIV, OpKind.POW)

    /**
     * Phase A5 — splat a rank-0 [scalar] over [tensor]'s shape.
     *
     * A compile-time constant goes through [splatLiteral]. A computed rank-0 value
     * splats through BROADCAST with an EMPTY `broadcast_dimensions`, the §0.4.359
     * scalar-seed polymorphism the §0.4.371 generalisation preserved: the interpreter
     * fills the output with the single input element, the emitter emits a
     * scalar→shape `broadcast_in_dim`, and BroadcastRule's adjoint is the full reduce
     * — emitted since §0.4.373 as the runtime-extent `SUM_TO`, which reads the target
     * shape from its template operand at execution. A differentiable scalar side
     * therefore gets a correct adjoint for free.
     *
     * Phase A5c-3 — [tensor] rides along as the BROADCAST's shape-only template in
     * both forms, so synthesis reads the target extents off a real runtime value
     * instead of axis-matching static atoms against the params. It is already a body
     * node (the binary op's other operand), so the reference costs nothing.
     */
    private fun splatScalarTo(scalar: DxirNode, tensor: DxirNode, emitter: DxirEmitter): DxirNode =
        if (scalar is DxirConst) {
            splatLiteral(scalar.value, tensor, emitter)
        } else {
            emitter.op(
                kind = OpKind.BROADCAST,
                operands = listOf(scalar, tensor),
                type = tensor.type,
                attrs = mapOf("broadcast_dimensions" to emptyList<Int>()),
            )
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
        // Phase A5b (DiffKT parity) — the SCALAR tanh / sigmoid surface. Both were
        // fully ruled at the IR level (TanhRule `1 − tanh²`, SigmoidRule `σ(1−σ)`,
        // interpreter + emitter arms, forward tangents) and the tensor spellings
        // `io.tlaloc.core.ops.tanh` / `.sigmoid` have mapped since §0.4.200, but
        // `:core` had no scalar host fns, so `grad { x: Float -> x.tanh() }` was an
        // orphan. Synthesis: scalar TANH already routes to `kotlin.math.tanh`;
        // scalar SIGMOID resolves the new `io.tlaloc.core.sigmoid` (no stdlib
        // equivalent exists).
        put("io.tlaloc.core.tanh", OpKind.TANH)
        put("io.tlaloc.core.sigmoid", OpKind.SIGMOID)
        // §0.4.395 — Phase C2 trig tails: the SCALAR tan / atan surface, in both
        // the `:core` receiver spelling (`x.tan()`, the §0.4.166 sin/cos pattern)
        // and the bare `kotlin.math` spelling (`tan(x)` — a top-level one-arg
        // call, which the UNARY arm's argument fallback below now accepts; the
        // §0.4.377 `kotlin.math.pow` precedent for mapping stdlib FQNs directly).
        put("io.tlaloc.core.tan", OpKind.TAN)
        put("io.tlaloc.core.atan", OpKind.ATAN)
        put("kotlin.math.tan", OpKind.TAN)
        put("kotlin.math.atan", OpKind.ATAN)
        // :core DTensor shape-preserving unary ops (io.tlaloc.core.ops package).
        put("io.tlaloc.core.ops.relu", OpKind.RELU)
        put("io.tlaloc.core.ops.neg", OpKind.NEG)
        put("io.tlaloc.core.ops.sigmoid", OpKind.SIGMOID)
        put("io.tlaloc.core.ops.tanh", OpKind.TANH)
        put("io.tlaloc.core.ops.sign", OpKind.SIGN)
        // §0.4.395 — the TENSOR tan / atan spellings (:core/ops/HostOps.kt).
        put("io.tlaloc.core.ops.tan", OpKind.TAN)
        put("io.tlaloc.core.ops.atan", OpKind.ATAN)
        put("io.tlaloc.core.ops.exp", OpKind.EXP)
        put("io.tlaloc.core.ops.log", OpKind.LOG)
        put("io.tlaloc.core.ops.sqrt", OpKind.SQRT)
        // :core DTensor reductions (io.tlaloc.core.ops package). These collapse all
        // operand dims to scalar — special-cased in the unary-lowering arm above.
        // §0.4.366 — mean/max/min join sum (mean's absence was an audit-flagged
        // orphan: the dispatch arm handled MEAN but no FQN mapped to it).
        put("io.tlaloc.core.ops.sum", OpKind.SUM)
        put("io.tlaloc.core.ops.mean", OpKind.MEAN)
        put("io.tlaloc.core.ops.max", OpKind.MAX)
        put("io.tlaloc.core.ops.min", OpKind.MIN)
    }

    /**
     * §0.4.366 — the axis-reduction user surface (Phase A1). Calls WITH axis
     * arguments dispatch through the REDUCE_OP_MAP arm (literal axes →
     * `reduction_dims` attr + exact result type); no-arg calls fall through
     * to UNARY_OP_MAP's full-reduce-to-scalar arm.
     */
    private val REDUCE_OP_MAP: Map<String, OpKind> = mapOf(
        "io.tlaloc.core.ops.sum" to OpKind.SUM,
        "io.tlaloc.core.ops.mean" to OpKind.MEAN,
        "io.tlaloc.core.ops.max" to OpKind.MAX,
        "io.tlaloc.core.ops.min" to OpKind.MIN,
    )

    /** §0.4.367 — the RESHAPE-family + transpose user surface (Phase A2a). */
    private val SHAPE_OP_SET: Set<String> = setOf(
        "io.tlaloc.core.ops.squeeze",
        "io.tlaloc.core.ops.unsqueeze",
        "io.tlaloc.core.ops.flatten",
        "io.tlaloc.core.ops.reshape",
        "io.tlaloc.core.ops.transpose",
        "io.tlaloc.core.ops.broadcastTo",
    )

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
