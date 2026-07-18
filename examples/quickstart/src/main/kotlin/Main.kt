/**
 * Tlaloc quickstart — differentiable Kotlin in one file.
 *
 * `grad { }` below is rewritten AT COMPILE TIME by the Tlaloc K2 plugin
 * into synthesized gradient code: d/dA sum(A·A) = ones·Aᵀ + Aᵀ·ones. No tape at
 * runtime, no framework objects — a plain Kotlin function you call.
 *
 * Try the compile-time safety net: uncomment the `contract` block at the
 * bottom — the disjoint named axes make it a COMPILE ERROR (red squiggle
 * in the IDE), not a runtime crash.
 */
import io.tlaloc.autograd.grad
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.Rank2
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.ops.matmul
import io.tlaloc.core.ops.sum
import io.tlaloc.core.ops.toFloat

fun main() {
    val g = grad { a: DTensor<Rank2<Sym, Sym>, F32> -> (a matmul a).sum().toFloat() }

    val input = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f))
    val gradient = g(input).hostF32()

    println("d/dA sum(A matmul A) at A=[[1,2],[3,4]]:")
    println(gradient.toList())   // ones·Aᵀ + Aᵀ·ones = [7.0, 11.0, 9.0, 13.0]
    check(gradient.toList() == listOf(7.0f, 11.0f, 9.0f, 13.0f)) { "unexpected gradient: ${gradient.toList()}" }
    println("quickstart OK")

    // Compile-time shape safety (uncomment to see the error):
    // val bad = io.tlaloc.autograd.grad2 {
    //     a: DTensor<Rank2<io.tlaloc.core.Named<io.tlaloc.core.Batch, Sym>, io.tlaloc.core.Named<io.tlaloc.core.SeqLen, Sym>>, F32>,
    //     b: DTensor<Rank2<io.tlaloc.core.Named<io.tlaloc.core.Hidden, Sym>, io.tlaloc.core.Named<io.tlaloc.core.Hidden, Sym>>, F32> ->
    //     (io.tlaloc.core.ops.contract(a, b)).sum()
    // }
}
