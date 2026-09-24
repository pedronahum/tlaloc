package io.tlaloc.kptx

/**
 * The declarative ISA spec table.
 *
 * The parser ([parsePtx]) is opcode-agnostic on purpose; this table is
 * where opcode knowledge lives, as **data**: per instruction family
 * (keyed by base opcode) an [IsaInstructionSpec] declares the ordered
 * modifier slots, the type-suffix positions, and the operand signature
 * (kinds + register classes). [validateIsa] checks a [PtxModule]
 * against the table; the [KernelScope] `inst("opcode.mods", …)` escape hatch
 * calls the same validator at the Kotlin call site, so a typo'd
 * modifier or a class-mismatched register fails at kernel-construction
 * time, not at driver-JIT time with an inscrutable `CUDA_ERROR_INVALID_PTX`.
 *
 * Register classes follow the KPTX naming convention
 * (`%p` pred, `%r` b32, `%f` f32, `%rd` b64; special
 * registers `%tid/%ntid/%ctaid/%nctaid` are 32-bit) — the table checks
 * that a `ld.global.f32` destination is an `%f` register, that `setp`
 * writes a predicate, and that `.wide` widens the destination to
 * `%rd`.
 *
 * Bit-typed operands are **width**-checked (see [widthOfBitType]), so
 * `mov.b32 %r1, %f1`
 * is accepted and `mov.b32 %rd1, %f1` is not — which is what `ptxas`
 * itself does, and what every warp-reduced float kernel needs.
 *
 * Coverage: the instruction families the KPTX kernel corpus exercises.
 * Adding a family is one [IsaInstructionSpec] literal — the validator
 * never grows per-opcode code. Instructions whose base is absent from
 * the table are **rejected**: the escape hatch's contract is "validated
 * or refused", never "passed through unchecked". (Widening the table
 * is the intended response, not bypassing it.)
 */

/** KPTX register classes, by naming convention. [widthBits] is the
 * class's storage width — the bit-typed rule checks *that*
 * rather than the class (see [widthOfBitType]). */
enum class IsaRegClass(val prefix: String, val widthBits: Int) {
    PRED("%p", 1), R32("%r", 32), F32("%f", 32), R64("%rd", 64);
}

/** Infer the class of a register name; null for unknown spellings. */
fun regClassOf(name: String): IsaRegClass? {
    val special = listOf("%tid", "%ntid", "%ctaid", "%nctaid", "%laneid", "%warpid")
    if (special.any { name == it || name.startsWith("$it.") }) return IsaRegClass.R32
    return when {
        name.startsWith("%rd") -> IsaRegClass.R64
        name.startsWith("%p") -> IsaRegClass.PRED
        name.startsWith("%r") -> IsaRegClass.R32
        name.startsWith("%f") -> IsaRegClass.F32
        else -> null
    }
}

/** Map a type suffix to the register class that holds values of it;
 * null means "no class check" — f64 (no KPTX register bank) and the
 * b-types: PTX `bN` is untyped bit storage, and b-typed instructions
 * (`mov.b32 %f2, %r8` — pyptx's bit-preserving cross-class move) are
 * class-agnostic by ISA semantics. u/s/f types stay class-checked. */
internal fun classOfType(type: String): IsaRegClass? = when (type) {
    "f32" -> IsaRegClass.F32
    "u32", "s32" -> IsaRegClass.R32
    "u64", "s64" -> IsaRegClass.R64
    "pred" -> IsaRegClass.PRED
    else -> null
}

/**
 * The **width** a bit-typed suffix constrains its register
 * operands to, or null for "no width check".
 *
 * [classOfType] skips the class check for `bN`; this function supplies
 * the other half of the ISA's rule. PTX `bN` is untyped bit
 * *storage of a stated size*: the class
 * is free, the **size is not**. Verified against `ptxas 13.0`
 * (`-arch=sm_75`) rather than assumed, since this is the arm every
 * warp-reduced float kernel leans on:
 *
 * | spelling | ptxas |
 * |---|---|
 * | `mov.b32 %r1, %f1` / `mov.b32 %f2, %r1` | **legal** — bit reinterpretation, the intended use |
 * | `shfl.sync.down.b32 %f2, %f1, 16, 0x1f, …` | **legal** — the shuffle moves `%f` registers directly |
 * | `mov.b32 %rd3, %f1` | rejected: `Arguments mismatch for instruction 'mov'` |
 * | `shfl.sync.down.b32 %rd3, %rd1, …` | rejected |
 *
 * So `b32` accepts any 32-bit class (`%r`, `%f`) and refuses `%rd`
 * and `%p`; `b64` accepts `%rd` alone. `b16`/`b128` map to null — KPTX
 * has no register class of either width, so there is nothing to check
 * and inventing a rejection would be guessing.
 *
 * Two boundaries this deliberately does **not** move. `ptxas` is
 * looser than this table on u/s/f-typed `mov` (`mov.f32 %r1, %f1`
 * assembles); KPTX keeps the stricter class check, because a `%r`
 * holding an f32 value is a bug in every kernel this repo has written.
 * And `ptxas` *is* class-strict on arithmetic (`add.u32 %f2, %f1, %f1`
 * is rejected), which is exactly what [classOfType] already enforces.
 */
internal fun widthOfBitType(type: String): Int? = when (type) {
    "b32" -> 32
    "b64" -> 64
    else -> null
}

/** Operand kind at the IR surface. */
enum class IsaOperandKind { REG, IMM, MEM, SYM, VEC }

/** How to derive a REG operand's expected class. */
sealed interface IsaClassRule {
    /** From the instruction's type suffix at [typeIndex] (cvt has two). */
    data class FromType(val typeIndex: Int = 0) : IsaClassRule

    /** A fixed class regardless of type (e.g. `setp`'s pred destination). */
    data class Fixed(val cls: IsaRegClass) : IsaClassRule

    /** No class check. */
    data object Any : IsaClassRule
}

/** One operand slot: accepted [kinds]; [classRule] applies when the
 * actual operand is a register. */
data class IsaOperand(
    val kinds: Set<IsaOperandKind>,
    val classRule: IsaClassRule = IsaClassRule.FromType(0),
)

/** One ordered modifier slot between base and type suffixes. */
data class IsaModifierSlot(val alternatives: Set<String>, val required: Boolean = true)

/**
 * One instruction family.
 *
 * @property base the first dotted token (`"fma"`).
 * @property mods ordered modifier slots (matched greedily; leftovers error).
 * @property types allowed type-suffix sets, one per trailing type token
 *   (`cvt` has two: dst then src). Empty = untyped (`bra`, `ret`).
 * @property operands the operand signature.
 * @property wideDest when true and the `wide` modifier is present, the
 *   destination's class is the 64-bit widening of the 32-bit type
 *   (`mul.wide.u32` writes `%rd`).
 */
data class IsaInstructionSpec(
    val base: String,
    val mods: List<IsaModifierSlot> = emptyList(),
    val types: List<Set<String>> = emptyList(),
    val operands: List<IsaOperand> = emptyList(),
    val wideDest: Boolean = false,
)

private val INT_TYPES = setOf("u32", "s32", "u64", "s64", "b32", "b64")
private val ALL_TYPES = INT_TYPES + setOf("f32", "f64")
private val FLOAT_TYPES = setOf("f32", "f64")
private val ROUND = setOf("rn", "rz", "rm", "rp")

private fun reg(rule: IsaClassRule = IsaClassRule.FromType(0)) =
    IsaOperand(setOf(IsaOperandKind.REG), rule)
private fun vec(rule: IsaClassRule = IsaClassRule.FromType(0)) =
    IsaOperand(setOf(IsaOperandKind.VEC), rule)
private fun regOrImm(rule: IsaClassRule = IsaClassRule.FromType(0)) =
    IsaOperand(setOf(IsaOperandKind.REG, IsaOperandKind.IMM), rule)
private fun mem() = IsaOperand(setOf(IsaOperandKind.MEM), IsaClassRule.Any)
private fun sym() = IsaOperand(setOf(IsaOperandKind.SYM), IsaClassRule.Any)
private fun immOnly() = IsaOperand(setOf(IsaOperandKind.IMM), IsaClassRule.Any)

/** The ISA table, keyed by base opcode. */
val PTX_ISA: Map<String, IsaInstructionSpec> = listOf(
    IsaInstructionSpec("ret"),
    IsaInstructionSpec("bra", operands = listOf(sym())),
    IsaInstructionSpec(
        "bar",
        mods = listOf(IsaModifierSlot(setOf("sync"))),
        operands = listOf(immOnly()),
    ),
    // §0.4.357 — the optional v2/v4 slot covers vectorized ld/st
    // (pyptx-style `ld.global.v4.f32 {%f0..%f3}, [addr]`); the data
    // operand accepts a fragment vector alongside a scalar register.
    IsaInstructionSpec(
        "ld",
        mods = listOf(
            IsaModifierSlot(setOf("param", "global", "shared")),
            IsaModifierSlot(setOf("v2", "v4"), required = false),
        ),
        types = listOf(ALL_TYPES),
        operands = listOf(
            IsaOperand(setOf(IsaOperandKind.REG, IsaOperandKind.VEC), IsaClassRule.FromType(0)),
            mem(),
        ),
    ),
    IsaInstructionSpec(
        "st",
        mods = listOf(
            IsaModifierSlot(setOf("global", "shared")),
            IsaModifierSlot(setOf("v2", "v4"), required = false),
        ),
        types = listOf(ALL_TYPES),
        operands = listOf(
            mem(),
            IsaOperand(setOf(IsaOperandKind.REG, IsaOperandKind.VEC), IsaClassRule.FromType(0)),
        ),
    ),
    IsaInstructionSpec(
        "cvta",
        mods = listOf(IsaModifierSlot(setOf("to")), IsaModifierSlot(setOf("global", "shared", "local"))),
        types = listOf(setOf("u64")),
        operands = listOf(reg(IsaClassRule.Fixed(IsaRegClass.R64)), reg(IsaClassRule.Fixed(IsaRegClass.R64))),
    ),
    IsaInstructionSpec(
        "mov",
        types = listOf(ALL_TYPES),
        operands = listOf(
            reg(),
            IsaOperand(setOf(IsaOperandKind.REG, IsaOperandKind.IMM, IsaOperandKind.SYM), IsaClassRule.Any),
        ),
    ),
    IsaInstructionSpec(
        "add",
        mods = listOf(IsaModifierSlot(ROUND, required = false), IsaModifierSlot(setOf("sat"), required = false)),
        types = listOf(ALL_TYPES),
        operands = listOf(reg(), reg(), regOrImm()),
    ),
    IsaInstructionSpec(
        "sub",
        mods = listOf(IsaModifierSlot(ROUND, required = false), IsaModifierSlot(setOf("sat"), required = false)),
        types = listOf(ALL_TYPES),
        operands = listOf(reg(), reg(), regOrImm()),
    ),
    IsaInstructionSpec(
        "mul",
        mods = listOf(IsaModifierSlot(setOf("lo", "hi", "wide"), required = false), IsaModifierSlot(ROUND, required = false)),
        types = listOf(ALL_TYPES),
        operands = listOf(reg(), reg(), regOrImm()),
        wideDest = true,
    ),
    IsaInstructionSpec(
        "mad",
        mods = listOf(IsaModifierSlot(setOf("lo", "hi", "wide"), required = false)),
        types = listOf(ALL_TYPES),
        operands = listOf(reg(), reg(), reg(), regOrImm()),
        wideDest = true,
    ),
    IsaInstructionSpec(
        "div",
        mods = listOf(IsaModifierSlot(ROUND + setOf("full", "approx"), required = false)),
        types = listOf(ALL_TYPES),
        operands = listOf(reg(), reg(), regOrImm()),
    ),
    IsaInstructionSpec(
        "fma",
        mods = listOf(IsaModifierSlot(ROUND)),
        types = listOf(FLOAT_TYPES),
        operands = listOf(reg(), reg(), reg(), regOrImm()),
    ),
    IsaInstructionSpec(
        "sqrt",
        mods = listOf(IsaModifierSlot(ROUND + setOf("approx"))),
        types = listOf(FLOAT_TYPES),
        operands = listOf(reg(), reg()),
    ),
    IsaInstructionSpec(
        "rcp",
        mods = listOf(IsaModifierSlot(ROUND + setOf("approx"))),
        types = listOf(FLOAT_TYPES),
        operands = listOf(reg(), reg()),
    ),
    // §0.4.349 — PTX ships only the approx transcendentals (no .rn sin/cos
    // instruction exists); the `approx` slot is required so the fast-math
    // grade is visible at every call site.
    IsaInstructionSpec(
        "sin",
        mods = listOf(IsaModifierSlot(setOf("approx")), IsaModifierSlot(setOf("ftz"), required = false)),
        types = listOf(setOf("f32")),
        operands = listOf(reg(), reg()),
    ),
    IsaInstructionSpec(
        "cos",
        mods = listOf(IsaModifierSlot(setOf("approx")), IsaModifierSlot(setOf("ftz"), required = false)),
        types = listOf(setOf("f32")),
        operands = listOf(reg(), reg()),
    ),
    IsaInstructionSpec(
        "neg",
        types = listOf(ALL_TYPES),
        operands = listOf(reg(), reg()),
    ),
    // §0.4.350 — min/max plus the approx-only exp2/log2 transcendentals
    // (like sin/cos, PTX offers no .rn variants; natural exp/log are
    // ex2(x·log2e) / lg2(x)·ln2 with the constants spelled in hex).
    IsaInstructionSpec(
        "max",
        types = listOf(ALL_TYPES),
        operands = listOf(reg(), reg(), regOrImm()),
    ),
    IsaInstructionSpec(
        "min",
        types = listOf(ALL_TYPES),
        operands = listOf(reg(), reg(), regOrImm()),
    ),
    IsaInstructionSpec(
        "ex2",
        mods = listOf(IsaModifierSlot(setOf("approx")), IsaModifierSlot(setOf("ftz"), required = false)),
        types = listOf(setOf("f32")),
        operands = listOf(reg(), reg()),
    ),
    IsaInstructionSpec(
        "lg2",
        mods = listOf(IsaModifierSlot(setOf("approx")), IsaModifierSlot(setOf("ftz"), required = false)),
        types = listOf(setOf("f32")),
        operands = listOf(reg(), reg()),
    ),
    IsaInstructionSpec(
        "shr",
        types = listOf(INT_TYPES),
        operands = listOf(reg(), reg(), regOrImm(IsaClassRule.Any)),
    ),
    // §0.4.357 — shifts + bitwise logicals (index math in pyptx kernels).
    IsaInstructionSpec(
        "shl",
        types = listOf(INT_TYPES),
        operands = listOf(reg(), reg(), regOrImm(IsaClassRule.Any)),
    ),
    IsaInstructionSpec(
        "and",
        types = listOf(setOf("b16", "b32", "b64", "pred")),
        operands = listOf(reg(), reg(), regOrImm(IsaClassRule.Any)),
    ),
    IsaInstructionSpec(
        "or",
        types = listOf(setOf("b16", "b32", "b64", "pred")),
        operands = listOf(reg(), reg(), regOrImm(IsaClassRule.Any)),
    ),
    IsaInstructionSpec(
        "xor",
        types = listOf(setOf("b16", "b32", "b64", "pred")),
        operands = listOf(reg(), reg(), regOrImm(IsaClassRule.Any)),
    ),
    IsaInstructionSpec(
        "rsqrt",
        mods = listOf(IsaModifierSlot(setOf("approx"), required = true), IsaModifierSlot(setOf("ftz"), required = false)),
        types = listOf(FLOAT_TYPES),
        operands = listOf(reg(), reg()),
    ),
    IsaInstructionSpec(
        "setp",
        mods = listOf(IsaModifierSlot(setOf("eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hi", "hs"))),
        types = listOf(ALL_TYPES),
        operands = listOf(reg(IsaClassRule.Fixed(IsaRegClass.PRED)), reg(), regOrImm()),
    ),
    IsaInstructionSpec(
        "cvt",
        mods = listOf(
            IsaModifierSlot(ROUND + setOf("rni", "rzi", "rmi", "rpi"), required = false),
            IsaModifierSlot(setOf("sat"), required = false),
        ),
        types = listOf(ALL_TYPES, ALL_TYPES),
        operands = listOf(reg(IsaClassRule.FromType(0)), reg(IsaClassRule.FromType(1))),
    ),
    // §0.4.343 — tensor-core matmul-accumulate. Type suffixes are
    // d.a.b.c; a/b fragments hold packed halves in %r registers
    // (f16/bf16 map to no class — the typed wrapper enforces %r), d/c
    // fragments class-check per element via FromType.
    IsaInstructionSpec(
        "mma",
        mods = listOf(
            IsaModifierSlot(setOf("sync")),
            IsaModifierSlot(setOf("aligned")),
            IsaModifierSlot(setOf("m16n8k16", "m16n8k8", "m8n8k4")),
            IsaModifierSlot(setOf("row", "col")),
            IsaModifierSlot(setOf("row", "col")),
        ),
        types = listOf(
            setOf("f32", "f16"),
            setOf("f16", "bf16"),
            setOf("f16", "bf16"),
            setOf("f32", "f16"),
        ),
        operands = listOf(
            vec(IsaClassRule.FromType(0)),
            vec(IsaClassRule.FromType(1)),
            vec(IsaClassRule.FromType(2)),
            vec(IsaClassRule.FromType(3)),
        ),
    ),
    // §0.4.343 — warp shuffle: shfl.sync.<mode>.b32 d, a, b, c, membermask.
    IsaInstructionSpec(
        "shfl",
        mods = listOf(
            IsaModifierSlot(setOf("sync")),
            IsaModifierSlot(setOf("up", "down", "bfly", "idx")),
        ),
        types = listOf(setOf("b32")),
        operands = listOf(reg(), reg(), regOrImm(IsaClassRule.Any), regOrImm(IsaClassRule.Any), regOrImm(IsaClassRule.Any)),
    ),
    // §0.4.343 — warp vote: vote.sync.<mode>.{pred|b32} d, p, membermask.
    IsaInstructionSpec(
        "vote",
        mods = listOf(
            IsaModifierSlot(setOf("sync")),
            IsaModifierSlot(setOf("all", "any", "uni", "ballot")),
        ),
        types = listOf(setOf("pred", "b32")),
        operands = listOf(
            reg(),
            reg(IsaClassRule.Fixed(IsaRegClass.PRED)),
            regOrImm(IsaClassRule.Any),
        ),
    ),
).associateBy { it.base }

/** Thrown by [validateIsa] when a module fails validation. */
class IsaValidationException(val errors: List<String>) :
    RuntimeException("PTX ISA validation failed:\n" + errors.joinToString("\n"))

/** Validate one instruction against [PTX_ISA]; empty list = valid. */
fun validateInst(inst: PtxInst): List<String> {
    val errors = ArrayList<String>()
    val at = "`${inst.opcode}`"

    inst.guard?.let { g ->
        if (regClassOf(g.reg) != IsaRegClass.PRED) {
            errors += "$at: guard register ${g.reg} is not a predicate (%p)"
        }
    }

    val tokens = inst.opcode.split(".")
    val spec = PTX_ISA[tokens[0]]
        ?: return errors + "$at: unknown opcode base `${tokens[0]}` (not in the ISA table; widen PTX_ISA, don't bypass it)"

    // Split trailing type tokens from middle modifier tokens.
    val nTypes = spec.types.size
    if (tokens.size - 1 < nTypes) {
        return errors + "$at: expected $nTypes type suffix(es)"
    }
    val types = tokens.takeLast(nTypes)
    val mods = tokens.subList(1, tokens.size - nTypes)

    for ((i, t) in types.withIndex()) {
        if (t !in spec.types[i]) errors += "$at: type suffix `$t` not allowed at position $i"
    }

    // Ordered greedy slot match.
    var m = 0
    for (slot in spec.mods) {
        if (m < mods.size && mods[m] in slot.alternatives) { m++; continue }
        if (slot.required) errors += "$at: missing required modifier (one of ${slot.alternatives.sorted()})"
    }
    if (m < mods.size) errors += "$at: unknown modifier `${mods[m]}`"

    if (inst.operands.size != spec.operands.size) {
        errors += "$at: expected ${spec.operands.size} operand(s), got ${inst.operands.size}"
        return errors
    }

    val wide = spec.wideDest && "wide" in mods
    for ((i, op) in inst.operands.withIndex()) {
        val opSpec = spec.operands[i]
        val kind = when (op) {
            is PtxReg -> IsaOperandKind.REG
            is PtxImm -> IsaOperandKind.IMM
            is PtxMem -> IsaOperandKind.MEM
            is PtxSym -> IsaOperandKind.SYM
            is PtxVec -> IsaOperandKind.VEC
        }
        if (kind !in opSpec.kinds) {
            errors += "$at: operand $i has kind $kind; expected one of ${opSpec.kinds.sorted()}"
            continue
        }
        val expected = when (val rule = opSpec.classRule) {
            is IsaClassRule.Any -> null
            is IsaClassRule.Fixed -> rule.cls
            is IsaClassRule.FromType -> {
                val base = classOfType(types.getOrNull(rule.typeIndex) ?: "")
                if (wide && i == 0 && (base == IsaRegClass.R32)) IsaRegClass.R64 else base
            }
        }
        // §0.4.493 — bit-typed operands are width-checked instead.
        val expectedWidth = when (val rule = opSpec.classRule) {
            is IsaClassRule.FromType -> {
                val w = widthOfBitType(types.getOrNull(rule.typeIndex) ?: "")
                if (wide && i == 0 && w == 32) 64 else w
            }
            else -> null
        }
        fun checkReg(name: String, what: String) {
            val actual = regClassOf(name)
            if (actual == null) {
                errors += "$at: $what `$name` has no recognizable class"
            } else if (expected != null && actual != expected) {
                errors += "$at: $what `$name` is ${actual.prefix}-class; expected ${expected.prefix}"
            } else if (expected == null && expectedWidth != null && actual.widthBits != expectedWidth) {
                errors += "$at: $what `$name` is ${actual.widthBits}-bit (${actual.prefix}-class); " +
                    "a bit-typed `.b$expectedWidth` operand must be $expectedWidth-bit"
            }
        }
        when (op) {
            is PtxReg -> checkReg(op.name, "operand $i register")
            // §0.4.343 — class rules apply per fragment element.
            is PtxVec -> op.regs.forEach { checkReg(it, "operand $i fragment element") }
            else -> {}
        }
    }
    return errors
}

/** Validate every instruction in [this]; empty list = valid. Errors are
 * prefixed with `kernel <name>, stmt <index>:` for location. */
fun PtxModule.validateIsaErrors(): List<String> = buildList {
    for (kernel in kernels) {
        for ((idx, stmt) in kernel.body.withIndex()) {
            if (stmt !is PtxInst) continue
            for (e in validateInst(stmt)) add("kernel ${kernel.name}, stmt $idx: $e")
        }
    }
}

/** Validate and throw [IsaValidationException] on any error. */
fun PtxModule.validateIsa() {
    val errors = validateIsaErrors()
    if (errors.isNotEmpty()) throw IsaValidationException(errors)
}
