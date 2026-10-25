import io.tlaloc.autograd.grad
import io.tlaloc.autograd.hessian
import io.tlaloc.core.DTensor
import io.tlaloc.core.F64
import io.tlaloc.core.HostF64Storage
import io.tlaloc.core.Rank2
import io.tlaloc.core.Sym
import io.tlaloc.core.hostF64
import io.tlaloc.core.ops.*
import kotlin.math.abs

fun main() {
    // snippet
    // K = [[2, 1], [1, 3]], K⁻¹ = [[3, -1], [-1, 2]] / 5
    val k = DTensor<Rank2<Sym, Sym>, F64>(HostF64Storage(doubleArrayOf(2.0, 1.0, 1.0, 3.0)), intArrayOf(2, 2), F64)
    val inv = listOf(0.6, -0.2, -0.2, 0.4)
    val got = g(k).hostF64().toList()
    check(got.zip(inv).all { (a, b) -> abs(a - b) < 1e-12 }) { "d logdet K = $got" }
    check(h(k).dims.contentEquals(intArrayOf(4, 4))) { "hessian dims ${h(k).dims.toList()}" }
}
