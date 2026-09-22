package io.tlaloc.ir.render

import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.449 — unit pins for the DxirFunction → Kotlin pretty-printer's
 * CONTRACT (the golden compile-and-run oracle lives in
 * `:compiler-plugin`'s PrintedGradientGoldenTest, where the K2 harness is):
 * readable host-twin spellings on the happy path, and LOUD, KIND-NAMING
 * refusals everywhere else — never a silent skip.
 */
class KotlinSourceRendererTest {

    private val f32s = DxirType(F32, emptyList())
    private val mat22 = DxirType(F32, listOf(2, 2))

    @Test
    fun rendersHostTwinSpellingsWithRankedTypes() {
        val fn = DxirBuilder.function("toy") {
            val a = param("a", mat22)
            val b = param("b", mat22)
            val m = op(OpKind.MUL, listOf(a, b), mat22)
            val r = op(OpKind.RELU, listOf(m), mat22)
            val s = op(OpKind.SUM, listOf(r), f32s)
            listOf(s)
        }
        val src = fn.toKotlinSource()
        assertTrue("fun toy(a: DTensor<Rank2<Sym, Sym>, F32>, b: DTensor<Rank2<Sym, Sym>, F32>): DTensor<ScalarShape, F32>" in src, src)
        assertTrue("(a * b)" in src, src)
        assertTrue(".relu()" in src, src)
        assertTrue(".sum()" in src, src)
        assertTrue("import io.tlaloc.core.ops.*" in src, src)
        // every body op is annotated with the SSA node it renders
        assertTrue("// %2 = MUL(%0, %1)" in src, src)
    }

    @Test
    fun sinCosPrintTheirHostTwins() {
        // §0.4.496 — SIN/COS used to refuse with the twin-gap message, which made
        // every gradient through trig unprintable (a `grad { }` over a launch
        // angle, for instance). `:core` grew the DTensor twins, so they render.
        val fn = DxirBuilder.function("trig") {
            val a = param("a", mat22)
            val s = op(OpKind.SIN, listOf(a), mat22)
            val c = op(OpKind.COS, listOf(a), mat22)
            val m = op(OpKind.MUL, listOf(s, c), mat22)
            listOf(m)
        }
        val src = fn.toKotlinSource()
        assertTrue("a.sin()" in src, src)
        assertTrue("a.cos()" in src, src)
        assertTrue("// %1 = SIN(%0)" in src, src)
        assertTrue("// %2 = COS(%0)" in src, src)
    }

    @Test
    fun multiReturnRendersAsPair() {
        val fn = DxirBuilder.function("two") {
            val a = param("a", mat22)
            val n = op(OpKind.NEG, listOf(a), mat22)
            val e = op(OpKind.EXP, listOf(a), mat22)
            listOf(n, e)
        }
        val src = fn.toKotlinSource()
        assertTrue("Pair<DTensor<Rank2<Sym, Sym>, F32>, DTensor<Rank2<Sym, Sym>, F32>>" in src, src)
        assertTrue("return Pair(v1, v2)" in src, src)
    }

    @Test
    fun demotedKindRefusesByNameWithTheSanctionedAlternative() {
        val fn = DxirBuilder.function("ln") {
            val a = param("a", mat22)
            val y = op(OpKind.LAYERNORM, listOf(a), mat22)
            listOf(y)
        }
        val ex = assertFailsWith<KotlinRenderRefusal> { fn.toKotlinSource() }
        val msg = ex.message ?: ""
        assertTrue("LAYERNORM" in msg, msg)
        assertTrue("coarsener" in msg, msg)
    }

    @Test
    fun twinGapKindRefusesByName() {
        val fn = DxirBuilder.function("absy") {
            val a = param("a", mat22)
            val y = op(OpKind.ABS, listOf(a), mat22)
            listOf(y)
        }
        val ex = assertFailsWith<KotlinRenderRefusal> { fn.toKotlinSource() }
        assertTrue("ABS" in (ex.message ?: ""), ex.message ?: "")
    }

    @Test
    fun sentinelDimsRefuseByName() {
        val fn = DxirBuilder.function("sentinel") {
            val a = param("a", DxirType(F32, listOf(-1, 2)))
            val y = op(OpKind.NEG, listOf(a), DxirType(F32, listOf(-1, 2)))
            listOf(y)
        }
        val ex = assertFailsWith<KotlinRenderRefusal> { fn.toKotlinSource() }
        assertTrue("sentinel" in (ex.message ?: ""), ex.message ?: "")
    }

    @Test
    fun i32ZerosLikePrintsTheIntTwin() {
        val idxT = DxirType(I32, listOf(4))
        val fn = DxirBuilder.function("zeros") {
            val i = param("i", idxT)
            val z = op(OpKind.ZEROS_LIKE, listOf(i), idxT)
            listOf(z)
        }
        val src = fn.toKotlinSource()
        assertTrue("intZerosLike(i)" in src, src)
        assertTrue("DTensor<Rank1<Sym>, I32>" in src, src)
    }
}
