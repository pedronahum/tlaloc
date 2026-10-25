import io.tlaloc.autograd.grad
import io.tlaloc.core.*
import io.tlaloc.core.ops.*

fun main() {
    // snippet
    check(g(a).hostF32().toList() == listOf(7f, 11f, 9f, 13f)) { "gradient ${g(a).hostF32().toList()}" }
}
