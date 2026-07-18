package io.tlaloc.kptx

/**
 * KPTX v2.4 (§0.4.341) — the KernelScope builder DSL (plan task 12).
 *
 * Kernels written in pure Kotlin, one call = one instruction:
 *
 * ```kotlin
 * val module = ptxKernel("add_one") {
 *     val inPtr = param(".u64", "in_ptr")
 *     val outPtr = param(".u64", "out_ptr")
 *     val n = param(".u32", "n")
 *     val p1 = pred(); val r1 = r32(); val rd1 = r64(); val f1 = f32()
 *
 *     inst("ld.param.u64", rd1, mem(inPtr))
 *     inst("mov.u32", r1, ctaidX)
 *     val done = label("DONE")
 *     inst("bra", done, guard = p1)      // @%p1 bra DONE;
 *     inst("bra", done, guard = !p1)     // @!%p1 bra DONE;
 *     place(done)
 *     inst("ret")
 * }
 * ```
 *
 * Design (the pyptx lessons, adapted):
 * - **`inst()` is the validated escape hatch** (task 11's DoD): every
 *   call constructs a [PtxInst] and runs [validateInst] against the ISA
 *   table immediately — a typo'd modifier, wrong operand count, or a
 *   class-mismatched register throws at the Kotlin call site (with the
 *   construction stack trace), not at driver-JIT time. Typed per-opcode
 *   wrappers (task 14) are sugar over this same call.
 * - **Registers are typed handles, auto-numbered per class** following
 *   the v1 naming convention (`%p`/`%r`/`%f`/`%rd`, indices from 1);
 *   the `.reg` bank declarations are synthesized from the allocation
 *   high-water mark (`<count+1>`, matching the hand-written style) and
 *   prepended in the canonical order pred, b32, f32, b64.
 * - **Guards read naturally**: a predicate register *is* a positive
 *   [KGuard]; `!p` negates it.
 * - **Everything lowers to the §0.4.338 value-type IR** — the DSL is a
 *   builder over [PtxModule], so DSL-written kernels inherit the
 *   emitter's canonical format and the parser's byte-identical
 *   round-trip guarantees for free.
 */
@DslMarker
annotation class KptxDsl

/** An operand the DSL can pass to [KernelScope.inst]. */
sealed interface KOp {
    fun toOperand(): PtxOperand
}

/** A predicate guard: a bare predicate [KReg] is the positive form;
 * `!reg` is the negated form. */
sealed interface KGuard {
    val reg: KReg
    val negated: Boolean
}

/** A typed register handle. [name] includes the `%`. */
class KReg internal constructor(val name: String, val cls: IsaRegClass) : KOp, KGuard {
    override val reg: KReg get() = this
    override val negated: Boolean get() = false
    override fun toOperand(): PtxOperand = PtxReg(name)

    /** Negated guard: `inst("bra", l, guard = !p)` → `@!%p …`. */
    operator fun not(): KGuard = Negated(this)

    private class Negated(override val reg: KReg) : KGuard {
        override val negated: Boolean get() = true
    }

    override fun toString(): String = name
}

/** An immediate operand, exact spelling preserved. */
class KImm internal constructor(private val text: String) : KOp {
    override fun toOperand(): PtxOperand = PtxImm(text)
}

/** A memory operand `[base]` / `[base+offset]`. */
class KMem internal constructor(private val base: String, private val offset: Int) : KOp {
    override fun toOperand(): PtxOperand = PtxMem(base, offset)
}

/** A bare symbol operand (shared-array names). */
class KSym internal constructor(val name: String) : KOp {
    override fun toOperand(): PtxOperand = PtxSym(name)
}

/** A branch-target label handle; usable as a `bra` operand before or
 * after being [KernelScope.place]d. */
class KLabel internal constructor(val name: String) : KOp {
    override fun toOperand(): PtxOperand = PtxSym(name)
}

/** A `.param` handle; address it with [KernelScope.mem]. */
class KParam internal constructor(val name: String)

/** Integer immediate. */
fun imm(value: Int): KImm = KImm(value.toString())

/** Immediate with an exact spelling — hex-float constants
 * (`imm("0f3F800000")`) are bit-exact kernel identity, never derived
 * from a Kotlin Float. */
fun imm(text: String): KImm = KImm(text)

/** Build a single-kernel [PtxModule] with the DSL. */
fun ptxKernel(
    name: String,
    version: String = "7.0",
    target: String = "sm_75",
    build: KernelScope.() -> Unit,
): PtxModule = ptxModule(version, target) { kernel(name, build) }

/** Build a multi-kernel [PtxModule] with the DSL. */
fun ptxModule(
    version: String = "7.0",
    target: String = "sm_75",
    build: ModuleScope.() -> Unit,
): PtxModule {
    val scope = ModuleScope()
    scope.build()
    require(scope.kernels.isNotEmpty()) { "ptxModule: no kernels declared" }
    return PtxModule(scope.kernels.toList(), version, target)
}

@KptxDsl
class ModuleScope internal constructor() {
    internal val kernels = ArrayList<PtxKernel>()

    fun kernel(name: String, build: KernelScope.() -> Unit) {
        val scope = KernelScope(name)
        scope.build()
        kernels.add(scope.assemble())
    }
}

@KptxDsl
class KernelScope internal constructor(private val name: String) {

    private val params = ArrayList<PtxParam>()
    private val sharedDecls = ArrayList<PtxSharedDecl>()
    private val stmts = ArrayList<PtxStmt>()
    private val nextIndex = HashMap<IsaRegClass, Int>()
    private val labels = HashSet<String>()
    private val placed = HashSet<String>()

    // Special registers (32-bit per the ISA table's convention).
    val tidX: KReg = KReg("%tid.x", IsaRegClass.R32)
    val ntidX: KReg = KReg("%ntid.x", IsaRegClass.R32)
    val ctaidX: KReg = KReg("%ctaid.x", IsaRegClass.R32)
    val nctaidX: KReg = KReg("%nctaid.x", IsaRegClass.R32)

    /** Declare a `.param`; declaration order is the launch ABI. */
    fun param(type: String, name: String): KParam {
        require(type.startsWith(".")) { "param type must include the leading dot, got `$type`" }
        params.add(PtxParam(type, name))
        return KParam(name)
    }

    private fun alloc(cls: IsaRegClass): KReg {
        val i = nextIndex.getOrPut(cls) { 1 }
        nextIndex[cls] = i + 1
        return KReg("${cls.prefix}$i", cls)
    }

    /** Allocate the next predicate register (`%p1`, `%p2`, …). */
    fun pred(): KReg = alloc(IsaRegClass.PRED)

    /** Allocate the next 32-bit register (`%r1`, …). */
    fun r32(): KReg = alloc(IsaRegClass.R32)

    /** Allocate the next f32 register (`%f1`, …). */
    fun f32(): KReg = alloc(IsaRegClass.F32)

    /** Allocate the next 64-bit register (`%rd1`, …). */
    fun r64(): KReg = alloc(IsaRegClass.R64)

    /** Declare a static shared-memory byte array. */
    fun shared(name: String, sizeBytes: Int, align: Int = 4): KSym {
        sharedDecls.add(PtxSharedDecl(align, name, sizeBytes))
        return KSym(name)
    }

    /** Create a label handle (place it with [place]). */
    fun label(name: String): KLabel {
        require(labels.add(name)) { "label `$name` declared twice" }
        return KLabel(name)
    }

    /** Place [label] at the current position. */
    fun place(label: KLabel) {
        require(placed.add(label.name)) { "label `${label.name}` placed twice" }
        stmts.add(PtxLabel(label.name))
    }

    /** `[reg]` / `[reg+offset]` memory operand. */
    fun mem(base: KReg, offset: Int = 0): KMem = KMem(base.name, offset)

    /** `[param]` memory operand (for `ld.param.*`). */
    fun mem(base: KParam): KMem = KMem(base.name, 0)

    /** Full-line comment. */
    fun comment(text: String) {
        stmts.add(PtxComment(text))
    }

    /** Blank separator line. */
    fun blank() {
        stmts.add(PtxBlank)
    }

    /**
     * The validated escape hatch: append one instruction. [opcode] is
     * the full dotted spelling (`"fma.rn.f32"`); the ISA table
     * validates modifiers, operand count/kinds, and register classes
     * **now** — errors throw [IllegalArgumentException] at this call
     * site with the table's diagnostics.
     */
    fun inst(opcode: String, vararg ops: KOp, guard: KGuard? = null, comment: String? = null) {
        val ptxInst = PtxInst(
            opcode = opcode,
            operands = ops.map { it.toOperand() },
            guard = guard?.let { PtxGuard(it.reg.name, it.negated) },
            comment = comment,
        )
        val errors = validateInst(ptxInst)
        require(errors.isEmpty()) {
            "inst(\"$opcode\") in kernel `$name` is invalid:\n  " + errors.joinToString("\n  ")
        }
        stmts.add(ptxInst)
    }

    internal fun assemble(): PtxKernel {
        val unplaced = labels - placed
        require(unplaced.isEmpty()) { "kernel `$name`: labels declared but never placed: $unplaced" }

        val decls = ArrayList<PtxStmt>()
        // Canonical bank order: pred, b32, f32, b64 (the v1 style).
        val declOrder = listOf(
            IsaRegClass.PRED to ".pred",
            IsaRegClass.R32 to ".b32",
            IsaRegClass.F32 to ".f32",
            IsaRegClass.R64 to ".b64",
        )
        for ((cls, type) in declOrder) {
            val next = nextIndex[cls] ?: continue
            decls.add(PtxRegDecl(type, cls.prefix, next))
        }
        decls.addAll(sharedDecls)

        val body = if (decls.isEmpty()) stmts.toList() else decls + PtxBlank + stmts
        return PtxKernel(name, params.toList(), body)
    }
}
