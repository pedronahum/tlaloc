import io.tlaloc.autograd.customVjp
import io.tlaloc.autograd.grad
import io.tlaloc.core.*
import io.tlaloc.core.ops.*

fun main() {
    // snippet
    val got = g(Tensors.f32Vector(floatArrayOf(1f, 2f, 3f))).hostF32().toList()
    check(got == listOf(3f, 3f, 3f)) { "gradient $got" }
}
