// snippet

fun main() {
    check(gradient.hostF32().toList() == listOf(7f, 11f, 9f, 13f)) { "gradient ${gradient.hostF32().toList()}" }
}
