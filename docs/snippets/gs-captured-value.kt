import io.tlaloc.autograd.grad

fun computeScale() = 3f

fun main() {
    // snippet
    check(g(2f) == 12f) { "g(2) = ${g(2f)}" }
}
