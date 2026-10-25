import io.tlaloc.autograd.grad
import io.tlaloc.core.*
import io.tlaloc.core.ops.*

fun main() {
    // snippet
    check(gradient.hostF64().toList() == listOf(7.0, 11.0, 9.0, 13.0)) { "gradient ${gradient.hostF64().toList()}" }
}
