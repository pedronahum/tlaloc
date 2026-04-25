package io.tlaloc.ir.passes

import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirType
import org.matheclipse.core.eval.ExprEvaluator
import org.matheclipse.core.expression.F
import org.matheclipse.core.interfaces.IExpr
import org.matheclipse.core.interfaces.ISymbol

/**
 * Stage B coarsening — Symja-backed [SymbolicEngine]. JVM-only. LGPL-3.0 (the dependency
 * carries that license; Tlaloc itself stays Apache-compatible because LGPL allows linking
 * from non-LGPL code — see DIFFKTX_SPEC.md §0.4.13 for the licensing analysis).
 *
 * Wraps Symja's [ExprEvaluator] with the Tlaloc-side [SymbolicEngine] interface. All
 * `evaluator` calls run inside `synchronized(evaluator)` per docs/STAGE_B_PLAN.md §13
 * risk #13 — Symja's evaluator is not documented as thread-safe and the K2 compiler
 * plugin may invoke `PhiCalculus.apply` from concurrent compilation tasks. If contention
 * becomes a measurable bottleneck (Stage B.3 timing data), switch to thread-local
 * evaluators; deferred until measured.
 *
 * **Status (B.0b):** real bodies for arithmetic, composition, summation, differentiation,
 * simplification + first-cut [liftNode] / [liftFunction] / [lowerToDxir]. The first
 * caller is the Symja adequacy bake-off (`SymjaBakeoffTest`); `PhiCalculus.apply` lands
 * in Stage B.1 on top of this engine.
 *
 * **Treat as runtime-only:** never fork-and-patch matheclipse-core (any modifications
 * would inherit LGPL on our side). The interface boundary in `SymbolicEngine.kt`
 * insulates us — if Symja's adequacy fails for our use cases, we swap in a custom
 * Kotlin CAS implementation per spec §11.7 / plan §5.3.
 */
class SymjaEngine : SymbolicEngine {

    private val evaluator: ExprEvaluator = ExprEvaluator(false, 100)

    override val version: String = run {
        val pkgVersion = ExprEvaluator::class.java.`package`?.implementationVersion ?: "unknown"
        "symja-$pkgVersion-tlaloc-b0b"
    }

    private fun eval(expr: IExpr): IExpr = synchronized(evaluator) { evaluator.eval(expr) }

    private fun evalString(form: String): IExpr = synchronized(evaluator) { evaluator.eval(form) }

    private fun unwrap(e: SymExpr): IExpr = (e as SymjaExpr).inner
    private fun unwrap(f: SymFn): IExpr = (f as SymjaFn).inner
    private fun wrap(e: IExpr): SymExpr = SymjaExpr(e)

    // ------------------------------------------------------------------------
    // Constants + variables
    // ------------------------------------------------------------------------

    override fun rational(numerator: Long, denominator: Long): SymExpr =
        wrap(if (denominator == 1L) F.ZZ(numerator) else F.QQ(numerator, denominator))

    /**
     * Construct a [SymExpr] from a concrete `Double`. Promoted to the [SymbolicEngine]
     * interface in Stage B.3 — C6's affine-coefficient lifting from concrete Float
     * consts uses this (the rational form would lose precision on arbitrary floats).
     */
    override fun realLiteral(value: Double): SymExpr = wrap(F.num(value))

    /**
     * Symja-specific: extract a `Double` from a fully-numeric [SymExpr]. Errors if the
     * expression has any unresolved free variables — substitute first to ground them.
     */
    @Suppress("DEPRECATION") // see comment on [lowerImpl]
    fun toDouble(expr: SymExpr): Double = unwrap(expr).evalDouble()

    override fun variable(name: String): SymExpr =
        wrap(F.symbol(name) as ISymbol)

    override fun opaqueFunction(name: String, arity: Int): SymFn {
        require(arity == 1) {
            "SymjaEngine.opaqueFunction: only unary opaque functions supported in B.0b; got arity=$arity"
        }
        // Represent as a fresh symbol that Symja will treat as an unknown function head.
        return SymjaFn(F.symbol(name) as ISymbol)
    }

    override fun closure(varName: String, body: SymExpr): SymFn {
        // Build `Function[{varName}, body]`. Symja represents this as an unevaluated
        // Function expression that, when applied to an arg via `unaryAST1`, substitutes
        // the bound variable in the body. Used by C6's geometric-series construction:
        // `closure("i", a^i)` becomes `Function[{i}, a^i]`, then `engine.sum(it, 0, n-1)`
        // lifts it to `Sum[a^i, {i, 0, n-1}]` which Symja closes to `(a^n - 1) / (a - 1)`.
        val fnExpr = F.Function(F.List(F.symbol(varName)), unwrap(body))
        return SymjaFn(eval(fnExpr))
    }

    // ------------------------------------------------------------------------
    // Arithmetic
    // ------------------------------------------------------------------------

    override fun add(a: SymExpr, b: SymExpr): SymExpr = wrap(eval(F.Plus(unwrap(a), unwrap(b))))
    override fun sub(a: SymExpr, b: SymExpr): SymExpr = wrap(eval(F.Subtract(unwrap(a), unwrap(b))))
    override fun mul(a: SymExpr, b: SymExpr): SymExpr = wrap(eval(F.Times(unwrap(a), unwrap(b))))
    override fun div(a: SymExpr, b: SymExpr): SymExpr = wrap(eval(F.Divide(unwrap(a), unwrap(b))))
    override fun neg(a: SymExpr): SymExpr = wrap(eval(F.Negate(unwrap(a))))
    override fun pow(a: SymExpr, b: SymExpr): SymExpr = wrap(eval(F.Power(unwrap(a), unwrap(b))))

    // ------------------------------------------------------------------------
    // Composition + iteration
    // ------------------------------------------------------------------------

    override fun apply(f: SymFn, x: SymExpr): SymExpr {
        // For an opaque function symbol `g`, Symja's `g[x]` constructs the unevaluated
        // application. For a Function[{v}, body], `Function[{v}, body][x]` evaluates by
        // substituting v -> x in the body. Both are correct via the same path.
        return wrap(eval(F.unaryAST1(unwrap(f), unwrap(x))))
    }

    override fun nest(f: SymFn, x: SymExpr, n: SymExpr): SymExpr =
        wrap(eval(F.Nest(unwrap(f), unwrap(x), unwrap(n))))

    override fun sum(f: SymFn, lo: SymExpr, hi: SymExpr): SymExpr {
        // Symja's `Sum[expr, {i, lo, hi}]` requires a bound variable and an expr in that
        // variable. Convert the [SymFn] to an applied form by introducing a fresh dummy
        // symbol. If `f` is `Function[{k}, ...]`, applying it to the dummy substitutes;
        // if `f` is an opaque symbol `g`, the result is `g[i]` which Symja can sum if the
        // closed form exists.
        val dummy = F.Dummy("k")
        val applied = eval(F.unaryAST1(unwrap(f), dummy))
        return wrap(eval(F.Sum(applied, F.List(dummy, unwrap(lo), unwrap(hi)))))
    }

    override fun product(f: SymFn, lo: SymExpr, hi: SymExpr): SymExpr {
        val dummy = F.Dummy("k")
        val applied = eval(F.unaryAST1(unwrap(f), dummy))
        return wrap(eval(F.Product(applied, F.List(dummy, unwrap(lo), unwrap(hi)))))
    }

    // ------------------------------------------------------------------------
    // Differentiation
    // ------------------------------------------------------------------------

    override fun diff(expr: SymExpr, v: SymExpr): SymExpr =
        wrap(eval(F.D(unwrap(expr), unwrap(v))))

    // ------------------------------------------------------------------------
    // Simplification
    // ------------------------------------------------------------------------

    override fun simplify(expr: SymExpr): SymExpr = wrap(eval(F.Simplify(unwrap(expr))))
    override fun expand(expr: SymExpr): SymExpr = wrap(eval(F.Expand(unwrap(expr))))
    override fun factor(expr: SymExpr): SymExpr = wrap(eval(F.Factor(unwrap(expr))))
    override fun collect(expr: SymExpr, v: SymExpr): SymExpr =
        wrap(eval(F.Collect(unwrap(expr), unwrap(v))))

    override fun containsVariable(expr: SymExpr, v: SymExpr): Boolean {
        val freeQ = eval(F.FreeQ(unwrap(expr), unwrap(v)))
        // FreeQ returns True when `v` does NOT occur; containsVariable is its negation.
        return !freeQ.isTrue
    }

    override fun substitute(expr: SymExpr, subst: Map<SymExpr, SymExpr>): SymExpr {
        if (subst.isEmpty()) return expr
        // Build a Symja `{var1 -> val1, var2 -> val2, ...}` rule list.
        val rules = subst.entries.map { (k, v) -> F.Rule(unwrap(k), unwrap(v)) }
        val ruleList = F.List(*rules.toTypedArray())
        return wrap(eval(F.ReplaceAll(unwrap(expr), ruleList)))
    }

    // ------------------------------------------------------------------------
    // Lifting + lowering — first-cut for B.0b. Stage B.1+ tightens these.
    // ------------------------------------------------------------------------

    /**
     * Promote a [Number] DxirConst payload to its closest Symja form. Integer-valued
     * payloads (Long/Int and Float/Double whose value round-trips through Long) lift to
     * `rational` so Symja's integer-domain rules fire (`Times[1, x] → x`). Fractional
     * Float/Double payloads lift to `realLiteral` to preserve precision (D.1i Phase 2 —
     * §0.4.104).
     */
    private fun liftNumber(n: Number): SymExpr {
        val d = n.toDouble()
        if (d.isFinite()) {
            val asLong = d.toLong()
            if (asLong.toDouble() == d) return rational(asLong)
        }
        return realLiteral(d)
    }

    /**
     * First-cut [liftNode]. Handles the subtree shapes Stage B.0b's bake-off and Stage B.1's
     * F1/F2/F3/C1/C3 tests construct: scalar `DxirParam` / `DxirConst` / arithmetic
     * elementwise ops (ADD/SUB/MUL/DIV/NEG/POW). Every other op kind throws — a future
     * Stage B step that needs a wider lift surface (e.g., embedded WHILE inside a lifted
     * subtree) explicitly grows this routine and the matching [lowerToDxir] half.
     */
    override fun liftNode(node: DxirNode): SymExpr {
        return when (node) {
            is io.tlaloc.ir.DxirParam -> variable(node.name)
            is io.tlaloc.ir.DxirConst -> {
                val n = node.value as? Number
                    ?: error("SymjaEngine.liftNode: non-numeric const value ${node.value}")
                liftNumber(n)
            }
            is io.tlaloc.ir.DxirOp -> when (node.op) {
                io.tlaloc.ir.OpKind.ADD -> add(liftNode(node.operands[0]), liftNode(node.operands[1]))
                io.tlaloc.ir.OpKind.SUB -> sub(liftNode(node.operands[0]), liftNode(node.operands[1]))
                io.tlaloc.ir.OpKind.MUL -> mul(liftNode(node.operands[0]), liftNode(node.operands[1]))
                io.tlaloc.ir.OpKind.DIV -> div(liftNode(node.operands[0]), liftNode(node.operands[1]))
                io.tlaloc.ir.OpKind.NEG -> neg(liftNode(node.operands[0]))
                io.tlaloc.ir.OpKind.POW -> pow(liftNode(node.operands[0]), liftNode(node.operands[1]))
                else -> error(
                    "SymjaEngine.liftNode: op ${node.op} not yet supported (B.0b first-cut). " +
                        "Add support when a Stage B.1+ rewrite needs it.",
                )
            }
            else -> error(
                "SymjaEngine.liftNode: unsupported node kind ${node::class.simpleName} " +
                    "(id=${node.id}); B.0b first-cut handles only Param/Const/scalar arithmetic ops.",
            )
        }
    }

    /**
     * First-cut [liftFunction]. Treats a single-param + single-return [DxirFunction] as the
     * closure `λparam. liftNode(returnExpr)`. Multi-param / multi-return / region-bearing
     * functions error — Stage B.1+ widens this when needed.
     */
    override fun liftFunction(fn: DxirFunction): SymFn {
        require(fn.params.size == 1) {
            "SymjaEngine.liftFunction: B.0b first-cut requires exactly 1 param; got ${fn.params.size}"
        }
        require(fn.returns.size == 1) {
            "SymjaEngine.liftFunction: B.0b first-cut requires exactly 1 return; got ${fn.returns.size}"
        }
        val paramName = fn.params[0].name
        val body = liftNode(fn.returns[0])
        // Construct `Function[{paramName}, body]`.
        val fnExpr = F.Function(F.List(F.symbol(paramName)), unwrap(body))
        return SymjaFn(eval(fnExpr))
    }

    /**
     * First-cut [lowerToDxir]. Walks Symja [IExpr] trees emitting [DxirOp] / [DxirConst]
     * via [builder]. Handles: integer / rational literals, `Plus[a, b, ...]` (variadic →
     * left-fold ADD), `Times[a, b, ...]` (left-fold MUL), `Subtract`, `Divide`, `Negate`,
     * `Power`, and named symbols (resolved against [symbolMap]).
     *
     * Variadic Plus/Times lower as a left-fold of binary ops, which is the only shape the
     * dxir surface supports today (ADD / MUL are strictly binary). Symja's evaluator may
     * normalise `a + b + c` into `Plus[a, b, c]`; the lowering folds it into `ADD(ADD(a, b), c)`.
     *
     * `Power[base, exp]` lowers to `OpKind.POW` regardless of whether the exponent is
     * symbolic, integer, or rational. The dxir interpreter doesn't yet evaluate POW
     * numerically (that's a future addition); the bake-off doesn't exercise the lowered
     * form numerically — it only checks that lowering produces a structurally well-formed
     * dxir subtree.
     *
     * Throws [SymbolicEngine.LoweringException] if the IExpr contains a head Symja
     * resolved to (e.g., `Sum`, `Nest`, `Sqrt`, `Exp`, `Log`) that has no current dxir
     * counterpart. Callers should catch + fall back.
     */
    override fun lowerToDxir(
        expr: SymExpr,
        outputType: DxirType,
        builder: DxirBuilder,
    ): DxirNode = lowerImpl(unwrap(expr), outputType, builder, emptyMap())

    /**
     * Variant of [lowerToDxir] that accepts a `symbol-name → DxirNode` map for resolving
     * free variables. Promoted to the [SymbolicEngine] interface in Stage B.3 — C6's
     * lowering needs to map closed-form free variables (e.g., `p`, `n`) back to dxir
     * params in the rewritten function.
     */
    override fun lowerToDxir(
        expr: SymExpr,
        outputType: DxirType,
        builder: DxirBuilder,
        symbolMap: Map<String, DxirNode>,
    ): DxirNode = lowerImpl(unwrap(expr), outputType, builder, symbolMap)

    @Suppress("DEPRECATION") // IExpr.evalDouble() is the simplest cross-version path; Symja's
    // newer `evalf()`-based API churns by minor version. Re-evaluate on a Symja bump.
    private fun lowerImpl(
        expr: IExpr,
        outputType: DxirType,
        builder: DxirBuilder,
        symbolMap: Map<String, DxirNode>,
    ): DxirNode {
        // All numeric IExpr kinds (integer, rational, real) → single `DxirConst` via
        // Symja's `evalDouble()` first-cut. This collapses `Rational[2, 3]` into the
        // concrete float 0.666… rather than emitting `DIV(const(2), const(3))`. Stage B.1+
        // can split rationals back into a structural DIV if downstream passes (e.g., a
        // CSE that wants to share rational-coefficient subexpressions) need it; for B.0b
        // the bake-off only checks numerical agreement.
        if (expr.isNumber) {
            val d = expr.evalDouble()
            return builder.const(d.toFloatLiteral(outputType.dtype), outputType)
        }
        if (expr.isSymbol) {
            val name = expr.toString()
            val mapped = symbolMap[name]
                ?: throw SymbolicEngine.LoweringException(
                    "SymjaEngine.lowerToDxir: free symbol '$name' has no entry in symbolMap",
                )
            return mapped
        }
        if (expr.isAST) {
            val ast = expr as org.matheclipse.core.interfaces.IAST
            val head = ast.head().toString()
            val args = (1..ast.argSize()).map { ast[it] }
            return when (head) {
                "Plus" -> foldBinary(args, io.tlaloc.ir.OpKind.ADD, outputType, builder, symbolMap)
                "Times" -> foldBinary(args, io.tlaloc.ir.OpKind.MUL, outputType, builder, symbolMap)
                "Subtract" -> {
                    require(args.size == 2) { "Subtract expects 2 args; got ${args.size}" }
                    val a = lowerImpl(args[0], outputType, builder, symbolMap)
                    val b = lowerImpl(args[1], outputType, builder, symbolMap)
                    builder.op(io.tlaloc.ir.OpKind.SUB, listOf(a, b), outputType)
                }
                "Divide" -> {
                    require(args.size == 2) { "Divide expects 2 args; got ${args.size}" }
                    val a = lowerImpl(args[0], outputType, builder, symbolMap)
                    val b = lowerImpl(args[1], outputType, builder, symbolMap)
                    builder.op(io.tlaloc.ir.OpKind.DIV, listOf(a, b), outputType)
                }
                "Negate" -> {
                    require(args.size == 1) { "Negate expects 1 arg; got ${args.size}" }
                    val a = lowerImpl(args[0], outputType, builder, symbolMap)
                    builder.op(io.tlaloc.ir.OpKind.NEG, listOf(a), outputType)
                }
                "Power" -> {
                    require(args.size == 2) { "Power expects 2 args; got ${args.size}" }
                    val base = lowerImpl(args[0], outputType, builder, symbolMap)
                    val expArg = lowerImpl(args[1], outputType, builder, symbolMap)
                    builder.op(io.tlaloc.ir.OpKind.POW, listOf(base, expArg), outputType)
                }
                else -> throw SymbolicEngine.LoweringException(
                    "SymjaEngine.lowerToDxir: AST head '$head' has no dxir counterpart " +
                        "(B.0b first-cut). Caller should fall back to leaving the original " +
                        "subtree in place.",
                )
            }
        }
        throw SymbolicEngine.LoweringException(
            "SymjaEngine.lowerToDxir: cannot lower IExpr of kind ${expr::class.simpleName}: $expr",
        )
    }

    private fun foldBinary(
        args: List<IExpr>,
        kind: io.tlaloc.ir.OpKind,
        outputType: DxirType,
        builder: DxirBuilder,
        symbolMap: Map<String, DxirNode>,
    ): DxirNode {
        require(args.isNotEmpty()) { "foldBinary: variadic op needs at least one arg" }
        var acc = lowerImpl(args[0], outputType, builder, symbolMap)
        for (i in 1 until args.size) {
            val rhs = lowerImpl(args[i], outputType, builder, symbolMap)
            acc = builder.op(kind, listOf(acc, rhs), outputType)
        }
        return acc
    }

    private fun Double.toFloatLiteral(dtype: io.tlaloc.core.DType): Any = when (dtype) {
        io.tlaloc.core.F32 -> this.toFloat()
        io.tlaloc.core.F64 -> this
        io.tlaloc.core.I32 -> this.toInt()
        io.tlaloc.core.I64 -> this.toLong()
        else -> error("SymjaEngine.lowerToDxir: unsupported dtype $dtype for numeric const")
    }
}

// ----- Sealed-impl wrappers for the SymExpr / SymFn opaque handles. ---------
//
// Sealed interfaces in commonMain may be extended within the same module (the :ir KMP
// module's commonMain + jvmMain count as one module); these data classes are the only
// concrete impls until / unless a custom CAS lands as the v0.5 fallback per spec §11.7.

internal data class SymjaExpr(val inner: IExpr) : SymExpr {
    override fun toString(): String = inner.toString()
}

internal data class SymjaFn(val inner: IExpr) : SymFn {
    override fun toString(): String = inner.toString()
}
