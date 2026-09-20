package io.tlaloc.core

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * §0.4.408 — Phase D1: the stateless PRNG foundation (DiffKT `RandomKey`
 * parity, first slice).
 *
 * DiffKT's `random/` package is a counter-based (stateless) PRNG: a
 * `RandomKey` value deterministically names an infinite random stream, keys
 * `split` into statistically independent children, and every draw is a pure
 * function of (key, flat index) — no mutable generator state, so the same key
 * always reproduces the same tensor and parallel draws need no locking.
 * DiffKT realises this over SHA-512; Tlaloc uses the JAX design instead:
 * **Threefry-2x32** (Salmon et al., "Parallel Random Numbers: As Easy as
 * 1, 2, 3", SC'11 — the Random123 suite), the same choice JAX made for the
 * same reasons (cheap, crypto-strength-adjacent mixing, trivially
 * counter-parallel, exact reproducibility across hosts).
 *
 * Everything here is pinned bit-for-bit against **JAX 0.10's classic
 * (non-partitionable) threefry2x32 implementation** (`jax.random` with
 * `jax_threefry_partitionable=False`, the layout JAX shipped from day one):
 * the block function against the Random123 known-answer vectors, and the
 * counter layout of [threefryBits] / [split] / [foldIn] / [uniformFloats]
 * against values generated from the GB10's JAX install (see RandomTest).
 * JAX's newer "partitionable" stream (default-on since 0.4.30) is a
 * DIFFERENT derivation with the same block function — deliberately not
 * modelled here.
 *
 * [normalFloats] is the one deliberate deviation: JAX draws normals via
 * inverse-erf; we use the Box-Muller transform over two uniform draws
 * (documented at the function) because it validates against elementary
 * identities instead of an erfinv approximation. It is still a pure function
 * of (key, index) with the same determinism guarantees.
 *
 * All draws are NON-differentiable in this slice: a `grad {}` / jvp body
 * containing an RNG op refuses loudly by name (the output is
 * piecewise-constant in the key, and DiffKT's reparameterized-gradient story
 * — gradients through the loc/scale of sampled normals — is Phase D2).
 */

/**
 * A stateless PRNG key: two 32-bit words (the JAX convention — `PRNGKey(seed)`
 * is `[seed >>> 32, seed & 0xffffffff]`). The words are unsigned bit patterns
 * carried in Kotlin `Int`s; all arithmetic below is mod-2³² where sign never
 * matters.
 */
data class RandomKey(val k0: Int, val k1: Int) {
    companion object {
        /** JAX `PRNGKey(seed)`: high word, low word. */
        fun fromSeed(seed: Long): RandomKey =
            RandomKey((seed ushr 32).toInt(), seed.toInt())
    }

    override fun toString(): String =
        "RandomKey(${k0.toUInt().toString(16)}, ${k1.toUInt().toString(16)})"
}

/** Threefry-2x32 rotation schedule (R_2x32 from the Random123 paper). */
private val ROT_EVEN = intArrayOf(13, 15, 26, 6)
private val ROT_ODD = intArrayOf(17, 29, 16, 24)

/** The Threefish key-schedule parity constant (SKEIN_KS_PARITY32). */
private const val THREEFRY_PARITY: Int = 0x1BD11BDA

/**
 * The Threefry-2x32 block function at the reference 20 rounds: mixes the
 * 64-bit counter (`c0`, `c1`) under the 64-bit key (`k0`, `k1`) into two
 * output words, packed as `(y0 << 32) | (y1 & 0xffffffff)`.
 *
 * Reference schedule: ks = [k0, k1, k0 ^ k1 ^ 0x1BD11BDA]; x = c + ks[0..1];
 * five groups of four ARX rounds (rotations alternating [13,15,26,6] /
 * [17,29,16,24]), each group followed by the key injection
 * `x0 += ks[(i+1)%3]; x1 += ks[(i+2)%3] + (i+1)`.
 *
 * Known-answer pins (Random123 KAT vectors, reproduced by JAX's
 * `threefry_2x32` tests — asserted in RandomTest):
 *   key (0,0),   ctr (0,0)   → (0x6b200159, 0x99ba4efe)
 *   key (-1,-1), ctr (-1,-1) → (0x1cb996fc, 0xbb002be7)
 *   key (0x13198a2e, 0x03707344), ctr (0x243f6a88, 0x85a308d3)
 *                             → (0xc4923a9c, 0x483df7a0)
 */
fun threefry2x32(k0: Int, k1: Int, c0: Int, c1: Int): Long {
    val ks2 = k0 xor k1 xor THREEFRY_PARITY
    var x0 = c0 + k0
    var x1 = c1 + k1
    for (i in 0 until 5) {
        val rots = if (i % 2 == 0) ROT_EVEN else ROT_ODD
        for (r in rots) {
            x0 += x1
            x1 = (x1 shl r) or (x1 ushr (32 - r))
            x1 = x1 xor x0
        }
        when (i % 3) {
            0 -> { x0 += k1; x1 += ks2 + (i + 1) }
            1 -> { x0 += ks2; x1 += k0 + (i + 1) }
            else -> { x0 += k0; x1 += k1 + (i + 1) }
        }
    }
    return (x0.toLong() shl 32) or (x1.toLong() and 0xFFFFFFFFL)
}

/**
 * [n] random 32-bit words for [key] over the flat counter space `0 until n` —
 * JAX's classic `threefry_2x32(key, iota(n))` layout, bit-for-bit:
 * the (zero-padded-to-even) counter vector splits into two halves, lane `j`
 * runs the block function on counter pair `(j, half + j)`, the two output
 * half-vectors concatenate `[y0…, y1…]`, and an odd [n] pads one zero counter
 * at the END and drops the last output word.
 */
fun threefryBits(key: RandomKey, n: Int): IntArray {
    require(n >= 0) { "threefryBits: n must be non-negative, got $n" }
    if (n == 0) return IntArray(0)
    val odd = n % 2
    val padded = n + odd
    val half = padded / 2
    val out = IntArray(padded)
    for (j in 0 until half) {
        val c0 = j
        val c1 = half + j
        val packed = threefry2x32(key.k0, key.k1, c0, if (c1 < n) c1 else 0)
        out[j] = (packed ushr 32).toInt()
        out[half + j] = packed.toInt()
    }
    return if (odd == 0) out else out.copyOf(n)
}

/**
 * Derives [n] statistically independent child keys — JAX `random.split`:
 * `threefryBits(key, 2n)` reshaped to n pairs. Deterministic; children never
 * collide with each other or the parent in any realisable stream.
 */
fun RandomKey.split(n: Int): List<RandomKey> {
    require(n >= 1) { "split: n must be >= 1, got $n" }
    val bits = threefryBits(this, 2 * n)
    return List(n) { RandomKey(bits[2 * it], bits[2 * it + 1]) }
}

/**
 * Folds 32-bit [data] into the key, deriving a new independent key — JAX
 * `random.fold_in`: the block function applied at counter `(0, data)` (JAX
 * widens integer data to a u64 counter whose high word is 0; for a negative
 * Kotlin [data] we keep the raw 32-bit pattern with a zero high word — our
 * convention, pinned in RandomTest).
 */
fun RandomKey.foldIn(data: Int): RandomKey {
    val packed = threefry2x32(k0, k1, 0, data)
    return RandomKey((packed ushr 32).toInt(), packed.toInt())
}

/**
 * [n] uniform f32 draws in [0, 1) — the JAX recipe, bit-for-bit: take each
 * word's top 23 bits as a mantissa (`bits >>> 9`), OR in the exponent of 1.0f
 * (`0x3f800000`) to bitcast into [1, 2), subtract 1. Every value is an exact
 * multiple of 2⁻²³ in [0, 1).
 */
fun uniformFloats(key: RandomKey, n: Int): FloatArray {
    val bits = threefryBits(key, n)
    return FloatArray(n) {
        Float.fromBits((bits[it] ushr 9) or 0x3F800000) - 1f
    }
}

/**
 * [n] standard-normal f32 draws via the Box-Muller transform:
 * `z[i] = sqrt(-2 ln(1 - u[i])) · cos(2π u[n + i])` over a single
 * `uniformFloats(key, 2n)` stream (first half radial, second half angular;
 * `1 - u ∈ (0, 1]` keeps the log finite since u < 1 exactly). The sine
 * partner of each pair is deliberately discarded — counter-based bits are
 * cheap and the stream stays a pure function of (key, index).
 *
 * NOTE: this deviates from JAX, which draws normals as `√2 · erfinv(2u − 1)`;
 * same distribution, different bits. Our convention, pinned in RandomTest.
 */
fun normalFloats(key: RandomKey, n: Int): FloatArray {
    val u = uniformFloats(key, 2 * n)
    return FloatArray(n) {
        val r = sqrt(-2.0 * ln(1.0 - u[it].toDouble()))
        val theta = 2.0 * PI * u[n + it].toDouble()
        (r * cos(theta)).toFloat()
    }
}

// ---------------------------------------------------------------------------
// §0.4.431 — Phase D distributions (DiffKT `random/` parity: cauchy /
// chiSquare, plus the exponential the chi-square family factors through).
//
// Every distribution here is a PURE ELEMENTWISE TRANSFORM of the D1 streams —
// no new randomness primitives, no new IR OpKinds. That is the recorded
// compositional design: a draw-then-transform graph differentiates as a
// constant automatically (the D2 zero-gradient RNG arms absorb any upstream),
// and emits to GPU through §0.4.422's explicit threefry for free. The
// arithmetic below deliberately MIRRORS the IR arms the composition lowers
// to (f32 SUB/MUL, `kotlin.math.tan(double).toFloat()` — the §0.4.395 TAN
// arm), so a lowered draw and a host draw are bit-identical on the same JVM.
// ---------------------------------------------------------------------------

/**
 * [n] standard-Cauchy f32 draws via the quantile (inverse-CDF) transform:
 * `x = tan(π · (u − ½))` over one [uniformFloats] stream. `u ∈ [0, 1)` maps
 * to angle `[−π/2, π/2)`; `u = ½` (an exact f32 multiple of 2⁻²³, so it does
 * occur) lands exactly on tan(0) = 0, and the pole at −π/2 is never hit
 * exactly but nearby mantissas produce the honest heavy tail (finite-but-huge
 * in Double, never ±∞ — the TAN arm's IEEE note). Centring and scaling are
 * f32, the tangent goes through Double — bit-for-bit the composition
 * RNG_UNIFORM → SUB ½ → MUL π → TAN that the `grad {}` lowering emits.
 */
fun cauchyFloats(key: RandomKey, n: Int): FloatArray {
    val u = uniformFloats(key, n)
    return FloatArray(n) {
        val centred = u[it] - 0.5f
        val scaled = centred * PI.toFloat()
        kotlin.math.tan(scaled.toDouble()).toFloat()
    }
}

/**
 * [n] unit-rate exponential f32 draws via the quantile transform:
 * `x = −ln(1 − u)`. `1 − u ∈ (0, 1]` keeps the log finite (u < 1 exactly,
 * the same guard [normalFloats] leans on), and `u = 0` gives exactly 0 —
 * the distribution's support edge, honestly included. The log runs in
 * Double over the f32 complement, mirroring the LOG arm convention.
 */
fun exponentialFloats(key: RandomKey, n: Int): FloatArray {
    val u = uniformFloats(key, n)
    return FloatArray(n) {
        val complement = 1.0f - u[it]
        (-ln(complement.toDouble())).toFloat()
    }
}

/**
 * [n] chi-square f32 draws with [dof] degrees of freedom, by the DEFINITION:
 * the sum of [dof] squared standard normals. One [normalFloats] stream of
 * `n · dof` draws, laid out row-major — output element `i` consumes the
 * contiguous block `z[i·dof … i·dof+dof−1]` (the layout a future
 * reshape-[n, dof]-then-axis-sum `grad {}` lowering would read), squared and
 * accumulated in f32 left to right (the pinned accumulation order).
 * Definitional rather than quantile-based because the χ² inverse CDF has no
 * elementary form — the sum-of-squares IS the hand-checkable oracle, exact
 * over any pinned normal stream for any positive integer [dof].
 */
fun chiSquareFloats(key: RandomKey, n: Int, dof: Int): FloatArray {
    require(dof >= 1) { "chiSquareFloats: dof must be >= 1, got $dof" }
    val z = normalFloats(key, n * dof)
    return FloatArray(n) { i ->
        var acc = 0f
        for (j in 0 until dof) {
            val v = z[i * dof + j]
            acc += v * v
        }
        acc
    }
}

// ---------------------------------------------------------------------------
// Host tensor surface (this slice's whole user surface: draws materialise on
// host; RNG inside grad {} bodies is Phase D2).
// ---------------------------------------------------------------------------

/** Uniform [0, 1) rank-1 host tensor of [n] draws. */
fun <A : ShapeAtom> RandomKey.uniformVector(n: Int): DTensor<Rank1<A>, F32> =
    DTensor(HostF32Storage(uniformFloats(this, n)), intArrayOf(n), F32)

/** Uniform [0, 1) rank-2 host tensor, drawn over the flat index space. */
fun <R : ShapeAtom, C : ShapeAtom> RandomKey.uniformMatrix(
    rows: Int,
    cols: Int,
): DTensor<Rank2<R, C>, F32> =
    DTensor(HostF32Storage(uniformFloats(this, rows * cols)), intArrayOf(rows, cols), F32)

/** Standard-normal rank-1 host tensor of [n] draws. */
fun <A : ShapeAtom> RandomKey.normalVector(n: Int): DTensor<Rank1<A>, F32> =
    DTensor(HostF32Storage(normalFloats(this, n)), intArrayOf(n), F32)

/** Standard-normal rank-2 host tensor, drawn over the flat index space. */
fun <R : ShapeAtom, C : ShapeAtom> RandomKey.normalMatrix(
    rows: Int,
    cols: Int,
): DTensor<Rank2<R, C>, F32> =
    DTensor(HostF32Storage(normalFloats(this, rows * cols)), intArrayOf(rows, cols), F32)

/** Standard-Cauchy rank-1 host tensor of [n] draws (§0.4.431). */
fun <A : ShapeAtom> RandomKey.cauchyVector(n: Int): DTensor<Rank1<A>, F32> =
    DTensor(HostF32Storage(cauchyFloats(this, n)), intArrayOf(n), F32)

/** Standard-Cauchy rank-2 host tensor, drawn over the flat index space. */
fun <R : ShapeAtom, C : ShapeAtom> RandomKey.cauchyMatrix(
    rows: Int,
    cols: Int,
): DTensor<Rank2<R, C>, F32> =
    DTensor(HostF32Storage(cauchyFloats(this, rows * cols)), intArrayOf(rows, cols), F32)

/** Unit-rate exponential rank-1 host tensor of [n] draws (§0.4.431). */
fun <A : ShapeAtom> RandomKey.exponentialVector(n: Int): DTensor<Rank1<A>, F32> =
    DTensor(HostF32Storage(exponentialFloats(this, n)), intArrayOf(n), F32)

/** Unit-rate exponential rank-2 host tensor, drawn over the flat index space. */
fun <R : ShapeAtom, C : ShapeAtom> RandomKey.exponentialMatrix(
    rows: Int,
    cols: Int,
): DTensor<Rank2<R, C>, F32> =
    DTensor(HostF32Storage(exponentialFloats(this, rows * cols)), intArrayOf(rows, cols), F32)

/** Chi-square([dof]) rank-1 host tensor of [n] draws (§0.4.431). */
fun <A : ShapeAtom> RandomKey.chiSquareVector(n: Int, dof: Int): DTensor<Rank1<A>, F32> =
    DTensor(HostF32Storage(chiSquareFloats(this, n, dof)), intArrayOf(n), F32)

/** Chi-square([dof]) rank-2 host tensor, drawn over the flat index space. */
fun <R : ShapeAtom, C : ShapeAtom> RandomKey.chiSquareMatrix(
    rows: Int,
    cols: Int,
    dof: Int,
): DTensor<Rank2<R, C>, F32> =
    DTensor(HostF32Storage(chiSquareFloats(this, rows * cols, dof)), intArrayOf(rows, cols), F32)
