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
 *
 * §0.4.493 adds the **float warp reduction** this surface was missing:
 * [warpReduceSumF32], plus the bit-reinterpreting [movB32], plus a
 * relaxation of [shflSync] from `%r`-only to any 32-bit class.
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
    // §0.4.493 — the constraint is 32 bits, not %r. `shfl.sync` is
    // `.b32`-typed, and `bN` is untyped bit storage of a stated size:
    // `shfl.sync.down.b32 %f2, %f1, 16, 0x1f, …` assembles (verified
    // against ptxas 13.0). §0.4.343 wrote `%r`-only here, and
    // docs/KPTX_PAGED_PERF.md §7.4 then recorded that as the blocker
    // standing between this DSL and a float warp reduction. It was a
    // wrapper's `require`, not the ISA.
    require(d.cls.widthBits == 32 && a.cls.widthBits == 32) {
        "shfl.sync moves 32 bits: d/a must be a 32-bit class (%r or %f), " +
            "got d=$d (${d.cls.prefix}) a=$a (${a.cls.prefix})"
    }
    inst("shfl.sync.${mode.token}.b32", d, a, b, c, mask)
}

/**
 * §0.4.493 — `mov.b32 d, a`: **bit reinterpretation** between 32-bit
 * register classes, with no conversion. `movB32(r, f)` is the PTX
 * spelling of `Float.toRawBits()`, and `movB32(f, r)` of
 * `Float.fromBits()` — this is *not* [KernelScope.inst]`("cvt…")`,
 * which changes the bits to preserve the value.
 *
 * Needed where a bit-typed instruction genuinely refuses a class (the
 * `%r`-only lane-index operands, `vote.sync.ballot`'s word). It is
 * **not** needed to warp-reduce floats — see [shflSync] — which is why
 * [warpReduceSumF32] does not call it.
 */
fun KernelScope.movB32(d: KReg, a: KReg) {
    require(d.cls.widthBits == 32 && a.cls.widthBits == 32) {
        "mov.b32 moves 32 bits: both operands must be a 32-bit class (%r or %f), " +
            "got d=$d (${d.cls.prefix}) a=$a (${a.cls.prefix})"
    }
    inst("mov.b32", d, a)
}

/**
 * §0.4.493 — the natural spelling of a **full-warp f32 sum**: after
 * this, lane 0 of every warp holds the sum of all 32 lanes' [acc]
 * (the other lanes hold partial sums and are conventionally discarded).
 *
 * Five `shfl.sync.down` steps at offsets 16, 8, 4, 2, 1, each followed
 * by `add.rn.f32` — no barrier and no shared memory, because a warp is
 * already synchronous. The scratch register is allocated from the
 * enclosing [KernelScope], so the call site writes one line.
 *
 * Deferred by name: **sub-warp widths.** The clamp/segment word is
 * pinned to `0x1f` (a full 32-lane segment); a `width < 32` reduction
 * needs `0x1f or ((32 - width) shl 8)` and a test that a partial warp
 * actually segments, which no kernel here needs yet.
 *
 * Summation order is the tree above, so the result is
 * order-deterministic but **not** bit-identical to a sequential sum —
 * the usual f32 non-associativity, stated because paged attention's
 * oracle tolerances are quoted to 1.2e-7.
 */
fun KernelScope.warpReduceSumF32(acc: KReg) {
    require(acc.cls == IsaRegClass.F32) {
        "warpReduceSumF32 accumulates an f32: acc must be %f-class, got $acc (${acc.cls.prefix})"
    }
    val tmp = f32()
    for (off in listOf(16, 8, 4, 2, 1)) {
        shflSync(KShflMode.DOWN, d = tmp, a = acc, b = imm(off), c = imm("0x1f"))
        inst("add.rn.f32", acc, acc, tmp)
    }
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
