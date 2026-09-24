package io.tlaloc.kptx

/**
 * The value-type PTX IR underlying the Kotlin PTX DSL. Design lifted from pyptx:
 * **immutable value types, byte-identical round-trip
 * discipline** — the IR models the *text*, not an abstraction over it,
 * so `emit(parse(text)) == text` can hold byte-for-byte on the
 * canonical corpus and every hand-written kernel is expressible without
 * loss (comments and blank lines are statements, not trivia).
 *
 * Scope: the surface the hand-written kernels exercise —
 * `.visible .entry` kernels, `.param` lists, `.reg`/`.shared`
 * directives, dotted-opcode instructions with register / immediate /
 * memory / symbol operands, `@pred` guards, labels. The opcode is an
 * **uninterpreted dotted string** (`"fma.rn.f32"`) — the IR stays
 * opcode-agnostic by construction; per-opcode validation is the ISA
 * spec table's job (see PtxIsa.kt), not the type system's.
 *
 * Everything here is a `data class`/`data object`: structural equality
 * is the identity the round-trip corpus and the specialization cache
 * ([PtxKernelTemplate.specialize]) key on.
 */

/**
 * A PTX translation unit: the `.version` / `.target` / `.address_size`
 * header followed by one or more kernel entries.
 *
 * @property version PTX ISA version line (`.version 7.0`). 7.0 is the
 *   default floor — the baseline KPTX kernels need ISA 6.x or later, and
 *   the driver JIT accepts anything ≤ its own ISA.
 * @property target virtual architecture (`.target sm_75`). The driver
 *   JITs to the real SASS (GB10 = sm_121); sm_75 is the default baseline.
 * @property addressSize `.address_size 64` — the only value KPTX
 *   supports (the FFM launch path is 64-bit throughout).
 */
data class PtxModule(
    val kernels: List<PtxKernel>,
    val version: String = "7.0",
    val target: String = "sm_75",
    val addressSize: Int = 64,
)

/**
 * A `.visible .entry` kernel: name, `.param` declarations (order is the
 * launch ABI — the registry marshals `void**` slots positionally,
 * inputs then outputs then trailing scalars), and the body
 * statement list (directives, instructions, labels, comments, blanks —
 * in source order).
 */
data class PtxKernel(
    val name: String,
    val params: List<PtxParam>,
    val body: List<PtxStmt>,
)

/**
 * One `.param` declaration. [type] is the bare PTX type token including
 * the leading dot (`".u64"`, `".u32"`) — kept textual for the same
 * reason opcodes are: the emitter/parser must not lose spellings.
 */
data class PtxParam(val type: String, val name: String)

/** A body statement. Order-preserving: the body list is the source. */
sealed interface PtxStmt

/**
 * `.reg .pred %p<4>;` — a virtual-register bank declaration.
 * @property type bare type token with dot (`".pred"`, `".b32"`, `".f32"`, `".b64"`).
 * @property prefix register name prefix including `%` (`"%p"`, `"%rd"`).
 * @property count bank size (the `<N>` upper bound; names go `%p0..%p{N-1}`).
 */
data class PtxRegDecl(val type: String, val prefix: String, val count: Int) : PtxStmt

/**
 * `.shared .align 4 .b8 sdata[1024];` — a static shared-memory array.
 * Only `.b8`-typed byte arrays are modelled (the kernels' reduction
 * scratch); typed shared arrays are not supported.
 */
data class PtxSharedDecl(val align: Int, val name: String, val sizeBytes: Int) : PtxStmt

/**
 * One instruction: uninterpreted dotted [opcode], positional [operands],
 * optional [guard] predicate, optional trailing [comment]
 * (`add.s64 %rd12, %rd6, %rd11;    // x row` — canonical separator is
 * `;` + four spaces + `// `). `ret;` is `PtxInst("ret")`.
 */
data class PtxInst(
    val opcode: String,
    val operands: List<PtxOperand> = emptyList(),
    val guard: PtxGuard? = null,
    val comment: String? = null,
) : PtxStmt

/** `@%p1` / `@!%p1` instruction guard. [reg] includes the `%`. */
data class PtxGuard(val reg: String, val negated: Boolean = false)

/** `LOOP_SUM:` — a branch target. Emitted at column 0. */
data class PtxLabel(val name: String) : PtxStmt

/** `// text` full-line comment, emitted at instruction indent. Comments
 * are statements so the round-trip corpus keeps them byte-identically. */
data class PtxComment(val text: String) : PtxStmt

/** A blank separator line inside a body. */
data object PtxBlank : PtxStmt

/** An instruction operand. */
sealed interface PtxOperand

/** A register reference, including special registers: `%f1`, `%rd3`,
 * `%ctaid.x`, `%tid.x`. [name] includes the `%`. */
data class PtxReg(val name: String) : PtxOperand

/** An immediate, kept as its exact source spelling: `0f3F800000`
 * (hex-float), `4`, `0`. Never reformatted — bit-exact constants are
 * part of kernel identity. */
data class PtxImm(val text: String) : PtxOperand

/** A memory operand `[base]` / `[base+offset]`. [base] is a register
 * (with `%`), a param name, or a shared-array symbol. */
data class PtxMem(val base: String, val offset: Int = 0) : PtxOperand

/** A bare symbol operand: branch-target labels (`bra LOOP_SUM;`) and
 * address-of-shared (`mov.u64 %rd13, sdata;`). */
data class PtxSym(val name: String) : PtxOperand

/**
 * A vector/fragment operand `{%f1, %f2, %f3, %f4}` (mma
 * fragments, vector ld/st). Elements are register names including `%`;
 * canonical spelling separates with `", "` like top-level operands.
 */
data class PtxVec(val regs: List<String>) : PtxOperand
