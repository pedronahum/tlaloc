/**
 * Tlaloc quickstart — differentiable Kotlin in one file.
 *
 * Two things happen here, and both happen at COMPILE time:
 *
 *  1. `grad { }` is rewritten by the Tlaloc K2 plugin into synthesized gradient
 *     code — d/dA sum(A·A) = 1·Aᵀ + Aᵀ·1. There is no tape at runtime and no
 *     framework object: `g` is a plain Kotlin function you call.
 *
 *  2. A shape bug in a neighbouring file is rejected by the compiler. That file
 *     is real, it is not commented out, and you can watch it be refused:
 *
 *         ./gradlew -p examples/quickstart shapeError
 *
 * Act [2] below prints that file and tells you the command. Nothing in this
 * example asks you to go and edit source to see the point of it.
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
import java.io.File

fun main() {
    // ------------------------------------------------------------------ 1 ---
    val g = grad { a: DTensor<Rank2<Sym, Sym>, F32> -> (a matmul a).sum().toFloat() }

    val input = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f))
    val gradient = g(input).hostF32()

    println("[1] a gradient, derived at compile time")
    println()
    println("    val g = grad { a: DTensor<Rank2<Sym, Sym>, F32> -> (a matmul a).sum().toFloat() }")
    println()
    println("    d/dA sum(A matmul A) at A = [[1, 2], [3, 4]]")
    println("      = 1·Aᵀ + Aᵀ·1")
    println("      = ${gradient.toList()}")
    check(gradient.toList() == listOf(7.0f, 11.0f, 9.0f, 13.0f)) {
        "unexpected gradient: ${gradient.toList()}"
    }
    println()
    println("    `g` is an ordinary Kotlin function. Call it in a hot loop: nothing")
    println("    allocates a tape, because there is no tape.")
    println()

    // ------------------------------------------------------------------ 2 ---
    println("[2] a shape bug the compiler will not let you ship")
    println()
    val source = System.getProperty("tlaloc.example.shapeErrorSource")?.let(::File)
    if (source != null && source.isFile) {
        val body = source.readLines()
            .dropWhile { !it.startsWith("fun shapeErrorThatMustNotCompile") }
        body.forEach { println("    $it") }
        println()
        println("    Those two operands share no axis NAME — (Batch, SeqLen) against")
        println("    (Hidden, Hidden) — so there is nothing to contract over. In NumPy,")
        println("    PyTorch or JAX that is a runtime error at best. Here the build fails.")
    } else {
        println("    (source not found — run via `./gradlew -p examples/quickstart run`)")
    }
    println()
    println("    Watch it happen:")
    println()
    println("        ./gradlew -p examples/quickstart shapeError")
    println()
    println("    e: ShapeError.kt:30:5 Tlaloc named-index mismatch: contract operands")
    println("       share no named axis: lhs=[Batch, SeqLen] rhs=[Hidden, Hidden]")
    println()
    println("    That task is SUPPOSED to fail. It is wired to fail the build if the")
    println("    compile ever succeeds — a green run there would mean Tlaloc had")
    println("    stopped catching the bug.")
    println()
    println("quickstart OK")
}
