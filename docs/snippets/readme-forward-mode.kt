import io.tlaloc.autograd.hessian
import io.tlaloc.autograd.jacobian
import io.tlaloc.autograd.jvp
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.Rank1
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.ops.*

fun main() {
    // snippet
    check(f(2f, 1f) == 12f) { "jvp = ${f(2f, 1f)}" }
    val x = Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f, 3f))
    check(j(x).hostF32().toList() == listOf(2f, 0f, 0f, 0f, 4f, 0f, 0f, 0f, 6f)) { "jacobian ${j(x).hostF32().toList()}" }
    check(h(x).hostF32().toList() == listOf(2f, 0f, 0f, 0f, 2f, 0f, 0f, 0f, 2f)) { "hessian ${h(x).hostF32().toList()}" }
}
