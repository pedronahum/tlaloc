package io.tlaloc.ir.passes

/**
 * §0.4.503 (Tier 3, item 3) — **Symja is an optional dependency now, and this object
 * is the seam.**
 *
 * The defect: `ir/build.gradle.kts` declared `implementation(libs.symja.core)` in
 * `jvmMain`, so `org.matheclipse:matheclipse-core` — an 8.3 MB **LGPL-3.0** jar plus
 * its own transitive tree — was a mandatory runtime dependency of every consumer of
 * `:ir`, and therefore of `:autograd`, `:nn` and `:stablehlo` too. Most programs
 * never differentiate a loop-bearing body and never touch the computer algebra
 * system at all; they paid for it anyway, and — this is the part that matters to a
 * corporate policy reviewer — they inherited an LGPL artifact into their dependency
 * graph without being asked.
 *
 * It is now `compileOnly`. `:ir` still compiles against it (so [SymjaEngine] is
 * built and shipped exactly as before) and the coarsening test suite still puts it
 * on the test classpath, so nothing about the certified behaviour changes. What
 * changes is that a consumer who wants the CAS asks for it:
 *
 *     implementation("org.matheclipse:matheclipse-core:3.1.1")   // LGPL-3.0
 *
 * ## Licensing, stated plainly because this is where a reader will look for it
 *
 * matheclipse-core is LGPL-3.0. Tlaloc is Apache-2.0 and only ever LINKS Symja
 * across the [SymbolicEngine] interface — it does not fork it, patch it, or
 * redistribute it. LGPL permits that, which is why the arrangement was legal
 * before and is legal now. Making the dependency optional does not change the
 * licence analysis; it changes who has to accept it. A consumer whose policy
 * forbids LGPL in the graph can now simply not add the line, and will get a named
 * refusal from the compiler plugin on the day a body actually needs the CAS
 * instead of an audit finding.
 *
 * ## Why a separate object and not a check inside SymjaEngine
 *
 * [SymjaEngine]'s fields and method signatures mention `org.matheclipse` types, so
 * merely LOADING that class can raise `NoClassDefFoundError` during verification —
 * before any `init` block of its own could run. This object names no Symja type at
 * all: the probe is a `Class.forName` on a string, and the only reference to
 * [SymjaEngine] is a constructor call that is not reached when the probe fails.
 */
object SymbolicEngines {

    /**
     * The class the probe looks for: Symja's evaluator, which is the one type
     * [SymjaEngine] cannot work without.
     */
    const val SYMJA_PROBE_CLASS: String = "org.matheclipse.core.eval.ExprEvaluator"

    /** The Maven coordinate a user adds, version included. */
    const val SYMJA_COORDINATE: String = "org.matheclipse:matheclipse-core:3.1.1"

    /** Symja's licence. Stated in the refusal text on purpose — see the class KDoc. */
    const val SYMJA_LICENSE: String = "LGPL-3.0"

    /**
     * Whether Symja is on the runtime classpath. `false` is a normal, supported
     * state — it is the DEFAULT state for a consumer who took no action.
     */
    val symjaAvailable: Boolean by lazy { probe(SymbolicEngines::class.java.classLoader) }

    /**
     * The presence probe, with the class loader as a parameter so it can be tested
     * against a loader that genuinely cannot see Symja. `Class.forName(…, false, …)`
     * — `initialize = false` — because the question is whether the class is
     * resolvable, not whether Symja's static initialisers succeed; running Log4j
     * initialisation as a side effect of asking a yes/no question would be a
     * surprise, and [symja] runs it moments later anyway.
     */
    fun probe(loader: ClassLoader?): Boolean =
        try {
            Class.forName(SYMJA_PROBE_CLASS, false, loader)
            true
        } catch (_: ClassNotFoundException) {
            false
        } catch (_: LinkageError) {
            // A half-present Symja (wrong version, missing transitive dep) is ABSENT
            // for our purposes, and saying so here is what keeps the refusal message
            // accurate rather than optimistic.
            false
        }

    /**
     * The message a caller shows when a body needs the CAS and Symja is not there.
     * A function of [need] — the caller's own description of what wanted it — so
     * the refusal names the situation and not just the missing jar.
     */
    fun absenceMessage(need: String): String =
        "$need, which requires a symbolic engine, and none is available: Symja " +
            "($SYMJA_COORDINATE, $SYMJA_LICENSE) is not on the classpath of the process that " +
            "needs it. Since 0.1.0-alpha01 Symja is an OPTIONAL dependency of io.github.pedronahum:ir — it " +
            "is an 8.3 MB $SYMJA_LICENSE jar that most programs never need, so it is no longer " +
            "forced on every consumer. Note WHERE it goes: the CAS runs inside the KOTLIN " +
            "COMPILER, not inside your program — coarsening happens in this plugin's IR phase — " +
            "so the jar must be on the COMPILER PLUGIN's classpath, and an `implementation` " +
            "dependency is invisible to it. Add exactly one line to get it back:\n" +
            "    kotlinCompilerPluginClasspath(\"$SYMJA_COORDINATE\")\n" +
            "Tlaloc only links Symja across the io.tlaloc.ir.passes.SymbolicEngine interface; " +
            "it does not modify or redistribute it, which is what makes an $SYMJA_LICENSE " +
            "dependency compatible with Tlaloc's own Apache-2.0 licence. If your policy " +
            "forbids $SYMJA_LICENSE in the dependency graph, this body cannot be coarsened " +
            "here — the alternative is a Symja-free SymbolicEngine implementation, which does " +
            "not exist yet (docs/ALPHA_PLAN.md, Tier 3)."

    /**
     * A [SymjaEngine], or a named refusal. Never a `NoClassDefFoundError`.
     *
     * @param need what wanted the engine, for the message.
     */
    fun symja(need: String): SymbolicEngine {
        if (!symjaAvailable) throw IllegalStateException(absenceMessage(need))
        return SymjaEngine()
    }

    /**
     * A [SymjaEngine] when one can be had, else null. For callers whose behaviour
     * without an engine is defined and correct — the coarsener's engine-free
     * F1/F2/F3/C3/C5 rules are exactly that. A caller using this MUST be able to
     * say what it did instead; the compiler plugin refuses by name at the point
     * where the difference becomes visible.
     */
    fun symjaOrNull(): SymbolicEngine? =
        if (!symjaAvailable) {
            null
        } else {
            // Symja's own construction can still fail (Log4j initialisation has done
            // so historically — §0.4.24). That is a DIFFERENT condition from absence
            // and is deliberately not swallowed into it.
            runCatching { SymjaEngine() }.getOrNull()
        }
}
