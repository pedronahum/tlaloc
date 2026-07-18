package io.tlaloc.kptx

/**
 * KPTX v2.6 (§0.4.343) — typed wrappers for tensor-core mma and warp
 * intrinsics (plan task 14). Sugar over [KernelScope.inst] — the ISA
 * table still validates every emitted instruction — but the wrappers
 * add what the table can't express generically: **fragment arity and
 * register class per operand position**, checked in Kotlin before the
 * instruction is even built, with mma-shape-specific messages.
 *
 * Scope: `mma.sync` (m16n8k16, f32.f16.f16.f32 — the workhorse
 * Ampere+ shape), `shfl.sync` (all four modes), `vote.sync.ballot`.
 * WGMMA / TMA / tcgen05 are deferred until a kernel needs them
 * (docs/KPTX_PLAN.md task 14 note).
 */

/** A fragment operand for [KernelScope.inst]. */
class KVec internal constructor(internal val regs: List<KReg>) : KOp {
    override fun toOperand(): PtxOperand = PtxVec(regs.map { it.name })
}

/** Build a `{…}` fragment operand from registers. */
fun vec(vararg regs: KReg): KVec = KVec(regs.toList())

private fun requireFragment(what: String, regs: List<KReg>, size: Int, cls: IsaRegClass) {
    require(regs.size == size) { "$what fragment must have $size registers, got ${regs.size}" }
    val wrong = regs.filter { it.cls != cls }
    require(wrong.isEmpty()) { "$what fragment must be ${cls.prefix}-class; got $wrong" }
}

/**
 * `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` — the Ampere+
 * f16-in/f32-accumulate tile: `d[4×%f] = a[4×%r] · b[2×%r] + c[4×%f]`,
 * a/b holding packed f16 pairs in 32-bit registers per the PTX
 * fragment layout. Fragment arities and classes are enforced here;
 * layouts are fixed row·col (the canonical layout for row-major A ·
 * column-major B — variants can widen the wrapper when needed).
 */
fun KernelScope.mmaSyncM16N8K16F32F16(
    d: List<KReg>,
    a: List<KReg>,
    b: List<KReg>,
    c: List<KReg>,
) {
    requireFragment("mma.m16n8k16 d", d, 4, IsaRegClass.F32)
    requireFragment("mma.m16n8k16 a", a, 4, IsaRegClass.R32)
    requireFragment("mma.m16n8k16 b", b, 2, IsaRegClass.R32)
    requireFragment("mma.m16n8k16 c", c, 4, IsaRegClass.F32)
    inst(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
        KVec(d), KVec(a), KVec(b), KVec(c),
    )
}

/** `shfl.sync` lane-exchange modes. */
enum class KShflMode(internal val token: String) {
    UP("up"), DOWN("down"), BFLY("bfly"), IDX("idx");
}

/**
 * `shfl.sync.<mode>.b32 d, a, b, c, membermask` — warp lane exchange.
 * [b] is the lane offset/index, [c] the clamp/segment word (`0x1f` for
 * a full warp in `down` mode), [mask] the member mask (defaults to all
 * 32 lanes). [d]/[a] must be 32-bit registers.
 */
fun KernelScope.shflSync(
    mode: KShflMode,
    d: KReg,
    a: KReg,
    b: KOp,
    c: KOp,
    mask: KOp = imm("0xffffffff"),
) {
    require(d.cls == IsaRegClass.R32 && a.cls == IsaRegClass.R32) {
        "shfl.sync moves 32-bit values: d/a must be %r-class, got d=$d a=$a"
    }
    inst("shfl.sync.${mode.token}.b32", d, a, b, c, mask)
}

/**
 * `vote.sync.ballot.b32 d, p, membermask` — each lane contributes
 * [p]'s value to the ballot word written to [d].
 */
fun KernelScope.voteBallotSync(d: KReg, p: KReg, mask: KOp = imm("0xffffffff")) {
    require(d.cls == IsaRegClass.R32) { "ballot destination must be %r-class, got $d" }
    require(p.cls == IsaRegClass.PRED) { "ballot source must be a predicate, got $p" }
    inst("vote.sync.ballot.b32", d, p, mask)
}
