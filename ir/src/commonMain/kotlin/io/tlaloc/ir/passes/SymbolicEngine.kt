package io.tlaloc.ir.passes

import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirType

/**
 * The single swap boundary between the φ-calculus coarsening pass and the underlying
 * computer algebra system. All φ-calculus rewrites go through [SymbolicEngine]; rule
 * code never depends on Symja (or any specific CAS) types directly.
 *
 * The shipped backend is Symja (`org.matheclipse:matheclipse-core`, **LGPL-3.0**,
 * JVM-native), implemented by [SymjaEngine] in `jvmMain`. A minimal custom Kotlin CAS
 * with a focused surface (rational arithmetic, symbolic differentiation, [nest], and
 * polynomial-only [simplify]) could replace it behind this interface.
 *
 * LICENSE: Symja is LGPL-3.0. LGPL permits what Tlaloc does (link, never fork or
 * patch), which is why this interface is the only thing `commonMain` sees. Tlaloc
 * itself is Apache-2.0 (see `LICENSE`).
 *
 * MODULE PLACEMENT: the interface lives in `commonMain` because it has no
 * JVM-specific dependencies — [SymExpr] and [SymFn] are opaque interfaces.
 * Symja-backed implementations must live in `jvmMain` because Symja itself is
 * JVM-only. Targets other than JVM never see a running [SymbolicEngine]; coarsening
 * happens at compile time.
 *
 * @see io.tlaloc.ir.passes.DxirReverseTransform — reverse-mode AD; coarsening runs
 *   before it and rewrites control-flow-bearing subtrees.
 */
interface SymbolicEngine {

    /**
     * Version string identifying the backend + plugin version. Used as a cache-key
     * component.
     *
     * Example values: `"symja-2.0.0-tlaloc-b1"`, `"kotlin-cas-v0.5-tlaloc-b3"`.
     */
    val version: String

    // ------------------------------------------------------------------------
    // Lifting dxir to symbolic
    // ------------------------------------------------------------------------

    /**
     * Lift a straight-line [DxirNode] subtree to a symbolic expression. Rejects
     * nodes carrying regions (IF / WHILE) — callers must close those via
     * φ-calculus first, then lift the resulting closed-form subtree.
     */
    fun liftNode(node: DxirNode): SymExpr

    /**
     * Lift a [DxirFunction] to a symbolic function. Each parameter becomes a free
     * symbolic variable; the function's (single) return becomes the expression body.
     *
     * Only single-return functions are lifted as one [SymFn]. Multi-return (tuple) functions
     * remain lifted as separate [SymFn]s, one per return position.
     */
    fun liftFunction(fn: DxirFunction): SymFn

    // ------------------------------------------------------------------------
    // Constants + variables
    // ------------------------------------------------------------------------

    /** Construct a symbolic exact rational. Defaults to integer when [denominator] == 1. */
    fun rational(numerator: Long, denominator: Long = 1L): SymExpr

    /**
     * Construct a symbolic real (`Double`) literal. Implementations may store this as
     * a finite-precision real (Symja's `Num`), as an exact rational approximation, or
     * any equivalent. Needed for affine-coefficient lifting from concrete
     * `Float` consts in the dxir into symbolic expressions; the [rational] form would
     * lose precision on arbitrary float values.
     */
    fun realLiteral(value: Double): SymExpr

    /** Construct a symbolic variable with the given name. Two calls with the same name produce references to the same variable. */
    fun variable(name: String): SymExpr

    /**
     * Construct an opaque symbolic function of the given arity. Used when lifting a
     * dxir subtree whose body the engine chooses NOT to expand (e.g., the body
     * contains an unclosed WHILE). The resulting [SymFn] can be composed with
     * [apply] / [nest] but cannot be simplified further until lowered.
     */
    fun opaqueFunction(name: String, arity: Int): SymFn

    /**
     * Construct a unary closure `Function[{varName}, body]` from an existing [SymExpr]
     * body in the named variable. Used for C6-style closed-form
     * construction — `Σ_{i=0}^{n-1} a^i` requires a `λi. a^i` closure that the
     * existing [opaqueFunction] (which is just a free symbol) doesn't supply.
     */
    fun closure(varName: String, body: SymExpr): SymFn

    // ------------------------------------------------------------------------
    // Arithmetic
    // ------------------------------------------------------------------------

    fun add(a: SymExpr, b: SymExpr): SymExpr
    fun sub(a: SymExpr, b: SymExpr): SymExpr
    fun mul(a: SymExpr, b: SymExpr): SymExpr
    fun div(a: SymExpr, b: SymExpr): SymExpr
    fun neg(a: SymExpr): SymExpr
    fun pow(a: SymExpr, b: SymExpr): SymExpr

    // ------------------------------------------------------------------------
    // Composition + iteration (load-bearing for C5, C6, C7)
    // ------------------------------------------------------------------------

    /** Apply a unary [SymFn] `f` to an argument `x`. Returns `f(x)`. */
    fun apply(f: SymFn, x: SymExpr): SymExpr

    /**
     * N-fold composition: `Nest(f, x, n) = f(f(f(...f(x)...)))` applied `n` times.
     * [n] may be a symbolic variable or a concrete [rational]; Symja handles both
     * via `Nest[f, x, n]` with symbolic n closing via `Sum` / `Product` when the
     * recurrence is simple.
     *
     * Load-bearing for rule C5.
     */
    fun nest(f: SymFn, x: SymExpr, n: SymExpr): SymExpr

    /**
     * Closed-form summation: `Sum(f(i), i, lo, hi)`. [f] is a unary function of
     * the bound variable. Load-bearing for rule C6.
     */
    fun sum(f: SymFn, lo: SymExpr, hi: SymExpr): SymExpr

    /**
     * Closed-form product: `Product(f(i), i, lo, hi)`. Load-bearing for rule C7.
     */
    fun product(f: SymFn, lo: SymExpr, hi: SymExpr): SymExpr

    // ------------------------------------------------------------------------
    // Differentiation
    // ------------------------------------------------------------------------

    /**
     * Partial derivative of [expr] with respect to variable [v]. The result is
     * another [SymExpr] in the same free-variable environment as [expr].
     */
    fun diff(expr: SymExpr, v: SymExpr): SymExpr

    // ------------------------------------------------------------------------
    // Simplification
    // ------------------------------------------------------------------------

    /**
     * Algebraic simplification. Idempotent: `simplify(simplify(x)) == simplify(x)`.
     * The main compile-time cost of coarsening — [SymjaEngine.simplify] should be
     * cached.
     */
    fun simplify(expr: SymExpr): SymExpr

    /** Expand polynomial forms. E.g., `(x + 1)^3` → `x^3 + 3x^2 + 3x + 1`. */
    fun expand(expr: SymExpr): SymExpr

    /** Factor a polynomial. E.g., `x^2 - 1` → `(x - 1) * (x + 1)`. */
    fun factor(expr: SymExpr): SymExpr

    /** Collect terms in [v]. E.g., `Collect[a*x + b*x + c, x]` → `(a + b) * x + c`. */
    fun collect(expr: SymExpr, v: SymExpr): SymExpr

    /** Substitute each `subst[k]` for variable `k` in [expr]. */
    fun substitute(expr: SymExpr, subst: Map<SymExpr, SymExpr>): SymExpr

    /**
     * Returns true if [expr] references [v] as a free variable. Used by C6's
     * algebraic pre-normalization: after computing `aSym = simplify(diff(backEdge, w))`,
     * checking `containsVariable(aSym, w) == false` is the affine-in-w test — if the
     * linear coefficient itself depends on w, the expression has a quadratic-or-higher
     * term and isn't a valid C6 recurrence. Backed by Symja's `FreeQ` for [SymjaEngine];
     * a pure-Kotlin CAS alternative would implement this via a tree walk.
     */
    fun containsVariable(expr: SymExpr, v: SymExpr): Boolean

    // ------------------------------------------------------------------------
    // Lowering to dxir
    // ------------------------------------------------------------------------

    /**
     * Emit a dxir subtree that evaluates [expr]. [outputType] pins the expected
     * dxir result type; [builder] receives the emitted [io.tlaloc.ir.DxirOp] /
     * [io.tlaloc.ir.DxirConst] nodes; returns the root node of the emitted subtree.
     *
     * Callers should catch [LoweringException] and fall back to leaving the
     * original dxir subtree in place — the expression contained a symbolic
     * construct (e.g., `Gamma[x]`, unclosed `Sum`, irreducible `Nest`) that has
     * no finite dxir representation.
     */
    fun lowerToDxir(
        expr: SymExpr,
        outputType: DxirType,
        builder: DxirBuilder,
    ): DxirNode

    /**
     * Variant of [lowerToDxir] that accepts a `symbol-name → DxirNode` map for resolving
     * free variables. Used when re-emitting a lifted closed-form whose
     * free variables correspond to existing dxir nodes (typically the original
     * function's params: e.g., `p` and `n` in C6's `a^n * p + b * Σa^i`).
     */
    fun lowerToDxir(
        expr: SymExpr,
        outputType: DxirType,
        builder: DxirBuilder,
        symbolMap: Map<String, DxirNode>,
    ): DxirNode

    /** Thrown by [lowerToDxir] when an expression cannot be lowered. */
    class LoweringException(message: String) : RuntimeException(message)
}

/**
 * Opaque handle for a symbolic expression. Implementations wrap Symja's `IExpr`,
 * a custom Kotlin AST, or any other CAS representation.
 *
 * **Contract:** implementations MUST treat [SymExpr] instances as immutable. Do
 * not mutate fields via reflection. Sharing a [SymExpr] across multiple
 * [SymbolicEngine] calls is safe; sharing across threads requires the backend's
 * own synchronisation (Symja's `ExprEvaluator` is not documented as thread-safe).
 *
 * NOTE on sealing: this was originally declared `sealed` to prevent random external
 * impls. KMP rejects extending a sealed type across source-set boundaries (commonMain
 * and jvmMain are treated as different "modules" for sealing purposes), so we drop
 * `sealed` to let `SymjaEngine` provide its concrete impl in jvmMain. The intended
 * impl set is still small + controlled (Symja; possibly a custom Kotlin CAS);
 * external implementations outside the `:ir` module are not part of the public API.
 */
interface SymExpr

/**
 * Opaque handle for a symbolic function (a unary closure). Implementations wrap
 * Symja's `Function[{x}, ...]` or a custom closure representation. Same
 * immutability + thread-safety contract as [SymExpr]; same KMP-compat reason for
 * not being `sealed` (see [SymExpr]).
 */
interface SymFn
