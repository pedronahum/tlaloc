package io.tlaloc.ir.passes

import io.tlaloc.core.F32
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * [OpKind.MOE_EXPERTS] in both interpreters against transformers'
 * `Qwen3_5MoeTopKRouter` and `Qwen3_5MoeExperts`, from the committed fixture
 * `moe_ops_fixture.json` (`harness/python/moe_ops_fixture.py`).
 */
class MoeExpertsFixtureTest {

    private val fx = parseJson(javaClass.getResourceAsStream("moe_ops_fixture.json")!!.readBytes().toString(Charsets.UTF_8)) as JsonObject
    private fun floats(k: String) = (fx[k] as JsonArray).elements.map { (it as JsonNumber).value.toFloat() }.toFloatArray()
    private val d = fx["dims"] as JsonObject
    private fun dim(k: String) = (d[k] as JsonNumber).value.toInt()

    private val fn = DxirBuilder.function("moe") {
        val r = dim("R"); val h = dim("H"); val e = dim("E"); val i = dim("I")
        val x = param("x", DxirType(F32, listOf(r, h)))
        val l = param("logits", DxirType(F32, listOf(r, e)))
        val gu = param("gateUp", DxirType(F32, listOf(e, 2 * i, h)))
        val dn = param("down", DxirType(F32, listOf(e, h, i)))
        listOf(op(OpKind.MOE_EXPERTS, listOf(x, l, gu, dn), DxirType(F32, listOf(r, h)), mapOf("top_k" to dim("K"))))
    }

    private val inputs = listOf(floats("x"), floats("logits"), floats("gateUp"), floats("down"))

    private fun check(got: FloatArray) {
        val want = floats("y")
        var worst = 0f
        for (j in want.indices) worst = maxOf(worst, abs(got[j] - want[j]))
        assertTrue(worst <= 2e-5f, "worst |interpreter - transformers| = $worst")
    }

    @Test
    fun theF32InterpreterMatchesTransformers() = check(DxirInterpreter.evalFunction(fn, inputs)[0])

    @Test
    fun theF64InterpreterMatchesTransformers() = check(
        DxirInterpreterF64.evalFunction(fn, inputs.map { a -> DoubleArray(a.size) { a[it].toDouble() } })[0]
            .let { a -> FloatArray(a.size) { a[it].toFloat() } },
    )
}
