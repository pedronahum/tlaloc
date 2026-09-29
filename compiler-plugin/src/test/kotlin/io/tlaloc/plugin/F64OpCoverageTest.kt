package io.tlaloc.plugin

import io.tlaloc.plugin.F64TestHarness.lit
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The `grad {}` surface op by op at F64. Each case is a loss `L(x)` written once as a plain
 * Kotlin function (the F64 host ops, forward only) and once inside `grad {}`; the compiled
 * gradient is compared with fourth-order central differences (`h = 1e-3`) of the plain
 * function, computed in the same program. The same loss is also evaluated at F32, and the F32
 * and F64 values must agree to 1e-5 relative, which pins the F64 forward ops to the F32 ones.
 *
 * Tolerance 1e-8 of the largest gradient entry: the differences are good to about 1e-12 on
 * these smooth O(1) losses (the special functions' kernels to about 1e-13), the compiled
 * gradients to about 1e-14. Every input stays at least 0.05 from any kink of its loss.
 */
class F64OpCoverageTest {

    private data class Case(
        val name: String,
        val imports: List<String>,
        val body: String,
        val rank1: Boolean = false,
        val rank4: Boolean = false,
        val named: Boolean = false,
    )

    private val x2 = doubleArrayOf(0.31, -0.72, 1.13, 0.25, -0.44, 0.87, -1.29, 0.66, 0.92, 0.18, -0.35, -0.81)
    private val x1 = doubleArrayOf(0.31, -0.72, 1.13, 0.25, -0.44)

    /** `[2, 2, 3, 3]`: an NCHW input, and also a valid OIHW conv kernel and IOHW transposed-conv weight for itself. */
    private val x4 = DoubleArray(36) { (((it * 17) % 23) - 11) / 9.0 + 0.013 * it }

    private val cases = listOf(
        Case("sigmoid", listOf("sigmoid"), "(x.sigmoid() * x).sum()"),
        Case("sqrt_log", listOf("sqrt", "log"), "((x * x + 1.0).sqrt() + (x * x + 1.0).log() * x).sum()"),
        Case("neg_exp", listOf("neg", "exp"), "(x.neg() * x.exp()).sum()"),
        Case("pow", listOf("pow"), "(x.pow(3.0) + (x * x + 1.0).pow(x)).sum()"),
        Case("tan_atan", listOf("tan", "atan"), "((x * 0.5).tan() * x + x.atan()).sum()"),
        Case("special", listOf("lgamma", "digamma", "polygamma"), "((x * x + 1.0).lgamma() + (x * x + 1.0).digamma() * x + (x * x + 1.5).polygamma(1)).sum()"),
        Case("maximum_minimum", listOf("maximum", "minimum", "tanh"), "(maximum(x, x * 0.5) + minimum(x, x.tanh() * 2.0)).sum()"),
        Case("transpose_perm", listOf("transpose", "exp"), "(x.transpose(1, 0) * x.transpose(1, 0).exp()).sum()"),
        Case("reshape_flatten", listOf("reshape", "flatten", "exp"), "(x.reshape(4, 3) * x.reshape(4, 3).exp()).sum() + (x.flatten() * x.flatten()).sum()"),
        Case("squeeze_unsqueeze", listOf("squeeze", "unsqueeze", "exp"), "(x.unsqueeze(0).squeeze(0) * x.exp()).sum()"),
        Case("flip", listOf("flip"), "(x.flip(1) * x * x).sum()"),
        Case("broadcastTo", listOf("broadcastTo"), "(x.sum(0, keepDims = true).broadcastTo(3, 4) * x).sum()"),
        Case("axis_reductions", listOf("exp"), "(x.exp().mean(1) * x.min(1)).sum().toDouble() + x.max().toDouble() * x.mean().toDouble() + x.min().toDouble()"),
        Case("concat_stack", listOf("concat", "stack", "exp"), "(concat(0, x, x * 2.0) * concat(0, x.exp(), x)).sum() + (stack(0, x, x) * stack(0, x, x.exp())).sum()"),
        Case("slice_view", listOf("slice", "view", "exp"), "(x.slice(1, 3, 1) * x.slice(0, 2, 1).exp()).sum() + (x.view(0..1, 0) * x.view(1..2, 0)).sum()"),
        Case("comparisons", listOf("where", "gt", "ge", "lt", "le", "exp"), "(where(x gt 0.0, x * x, x.exp()) + where(x le -0.5, x, x * 3.0) + where(x ge x.exp(), x, x * x * x) + where(x lt 0.2, x * 0.5, x)).sum()"),
        Case("losses", listOf("crossEntropyLoss", "nllLoss", "logSoftmax", "softmax"), "crossEntropyLoss(x, x.softmax(1)).toDouble() + nllLoss(x.logSoftmax(1), (x * x).softmax(1)).toDouble()"),
        Case("relu_sign", listOf("relu", "sign"), "(x.relu() * x + x.sign() * x * x).sum()"),
        Case("outerProduct", listOf("outerProduct", "tanh"), "(outerProduct(x, x.tanh()) * outerProduct(x, x)).sum()", rank1 = true),
        Case("get", listOf("get"), "x[0] * x[1] * x[2] + x[3] * x[4]", rank1 = true),
        Case("conv2d", listOf("conv2d", "tanh"), "(x.conv2d(x) * x.conv2d(x).tanh()).sum()", rank4 = true),
        Case("convTranspose2d", listOf("convTranspose2d"), "(x.convTranspose2d(x) * x.convTranspose2d(x)).sum()", rank4 = true),
        Case("pooling", listOf("avgPool2d", "maxPool2d", "exp"), "(x.avgPool2d(2, 2) * x.maxPool2d(2, 2).exp()).sum() + (x.maxPool2d(2, 1) * x.avgPool2d(2, 1)).sum()", rank4 = true),
        Case("batchNorm", listOf("batchNorm", "exp"), "(x.batchNorm(x.mean(0, 2, 3).exp(), x.max(0, 2, 3), 0.001) * x).sum()", rank4 = true),
    )

    private fun program(c: Case, f64: Boolean): String {
        val d = if (f64) "F64" else "F32"
        val shape = if (c.named) "Rank1<Named<I, Sym>>" else if (c.rank1) "Rank1<Sym>" else if (c.rank4) "Rank4<Sym, Sym, Sym, Sym>" else "Rank2<Sym, Sym>"
        val data = if (c.rank1) x1 else if (c.rank4) x4 else x2
        val make = if (c.named) {
            if (f64) "Tensors.f64Vector<Named<I, Sym>>(doubleArrayOf(${lit(x1)}))"
            else "Tensors.f32Vector<Named<I, Sym>>(floatArrayOf(${x1.joinToString(", ") { "${it}f" }}))"
        } else if (c.rank4) {
            if (f64) "Tensors.f64Tensor4<Sym, Sym, Sym, Sym>(2, 2, 3, 3, doubleArrayOf(${lit(data)}))"
            else "Tensors.f32Tensor4<Sym, Sym, Sym, Sym>(2, 2, 3, 3, floatArrayOf(${data.joinToString(", ") { "${it.toFloat()}f" }}))"
        } else if (c.rank1) {
            if (f64) "Tensors.f64Vector<Sym>(doubleArrayOf(${lit(data)}))"
            else "Tensors.f32Vector<Sym>(floatArrayOf(${data.joinToString(", ") { "${it}f" }}))"
        } else {
            if (f64) "Tensors.f64Matrix<Sym, Sym>(3, 4, doubleArrayOf(${lit(data)}))"
            else "Tensors.f32Matrix<Sym, Sym>(3, 4, floatArrayOf(${data.joinToString(", ") { "${it}f" }}))"
        }
        val scalar = if (f64) "Double" else "Float"
        val conv = if (f64) "toDouble" else "toFloat"
        val body = c.body.let { b ->
            if (f64) b else b.replace(Regex("""(?<![\w.])(\d+\.\d+)(?![\w.])"""), "$1f").replace(".toDouble()", ".toFloat()")
        }
        val imports = (listOf("sum", "times", "plus", "minus", "div", "max", "min", "mean", conv) + c.imports)
            .distinct().joinToString("\n") { "import io.tlaloc.core.ops.$it" }
        return """
            import io.tlaloc.autograd.grad
            import io.tlaloc.core.*
            $imports
            object I : IndexName { override val name = "i" }
            fun loss(x: DTensor<$shape, $d>): $scalar = ${wrap(body, conv)}
            fun main() {
                val x = $make
                println("value " + loss(x))
                ${if (f64) gradientAndDifferences(shape, conv, body) else ""}
            }
        """.trimIndent()
    }

    /** A body whose outer expression is a tensor sum gets `.toDouble()`; a scalar expression is already Double. */
    private fun wrap(body: String, conv: String): String =
        if (body.trimEnd().endsWith(".sum()")) "($body).$conv()" else body

    private fun gradientAndDifferences(shape: String, conv: String, body: String): String = """
                val g = grad { x: DTensor<$shape, F64> -> ${wrap(body, conv)} }
                println("grad " + g(x).hostF64().joinToString(","))
                val base = x.hostF64()
                val h = 1e-3
                val fd = DoubleArray(base.size) { i ->
                    fun at(dx: Double) = loss(DTensor(HostF64Storage(base.copyOf().also { it[i] += dx }), x.dims.copyOf(), F64))
                    (-at(2 * h) + 8 * at(h) - 8 * at(-h) + at(-2 * h)) / (12 * h)
                }
                println("fd " + fd.joinToString(","))
    """.trimIndent()

    @Test
    fun `every op with an F64 host twin differentiates at F64`() {
        val failures = mutableListOf<String>()
        for (c in cases) {
            val r64 = F64TestHarness.compileAndRun(program(c, f64 = true))
            if (r64.exitCode != 0) {
                failures += "${c.name}: F64 did not compile or run:\n" +
                    r64.messages.filter { it.severity.name == "ERROR" }.joinToString("\n") { it.message.lines().take(3).joinToString(" / ") }
                continue
            }
            val grad = r64.values("grad")
            val fd = r64.values("fd")
            val scale = max(1e-300, fd.maxOf { abs(it) })
            val worst = fd.indices.maxOf { abs(fd[it] - grad[it]) } / scale
            if (worst > 1e-8) failures += "${c.name}: gradient differs from the differences by $worst of the largest entry"
            val r32 = F64TestHarness.compileAndRun(program(c, f64 = false))
            if (r32.exitCode != 0) {
                failures += "${c.name}: the F32 twin did not compile:\n" + r32.describe().lines().filter { it.startsWith("ERROR") }.take(3).joinToString("\n")
                continue
            }
            val v64 = r64.value("value")
            val v32 = r32.value("value")
            if (abs(v64 - v32) > 1e-5 * max(1.0, abs(v64))) failures += "${c.name}: F64 value $v64 vs F32 $v32"
        }
        assertTrue(failures.isEmpty(), failures.joinToString("\n\n"))
    }
}
