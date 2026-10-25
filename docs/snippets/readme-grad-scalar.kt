import io.tlaloc.autograd.grad
import io.tlaloc.autograd.grad2

fun main() {
    // snippet
    check(df(2f) == 12f) { "df(2) = ${df(2f)}" }
    check(dfdxy(2f, 3f) == Pair(4f, 2f)) { "dfdxy(2, 3) = ${dfdxy(2f, 3f)}" }
}
