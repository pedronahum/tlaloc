package io.tlaloc.maestro.serving

/**
 * Final answers of math problems, compared as the MATH benchmark's own grader compares them
 * (Hendrycks et al., `math_equivalence.is_equiv`): the last `\boxed{...}` of a solution, with
 * LaTeX spellings of the same value normalized (`\dfrac`, `\left(`, units in `\text{ }`, a
 * leading `x =`, `0.5` for one half, `a/b`, missing braces of `\frac12` and `\sqrt3`).
 * Equivalence is of the normalized strings, not of values: `1/3` and `0.333` differ.
 */
object MathAnswer {
    /** The content of the last `\boxed{...}` (or `\fbox{...}`, or `\boxed x$`) in [text], or null. */
    fun lastBoxed(text: String): String? {
        val start = maxOf(text.lastIndexOf("\\boxed"), text.lastIndexOf("\\fbox"))
        if (start < 0) return null
        if (text.startsWith("\\boxed ", start)) {
            return text.substring(start + "\\boxed ".length).substringBefore('$').trim()
        }
        val open = text.indexOf('{', start)
        if (open < 0) return null
        var depth = 0
        for (i in open until text.length) {
            when (text[i]) {
                '{' -> depth++
                '}' -> {
                    depth--
                    if (depth == 0) return text.substring(open + 1, i)
                }
            }
        }
        return null
    }

    /** Whether [a] and [b] are the same answer once normalized; false when either is null. */
    fun equivalent(a: String?, b: String?): Boolean {
        if (a == null || b == null) return false
        return try {
            normalize(a) == normalize(b)
        } catch (_: IllegalArgumentException) {
            a == b
        }
    }

    /** `strip_string` of the MATH grader. */
    fun normalize(answer: String): String {
        var s = answer.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
        s = s.replace("tfrac", "frac").replace("dfrac", "frac")
        s = s.replace("\\left", "").replace("\\right", "")
        s = s.replace("^{\\circ}", "").replace("^\\circ", "")
        s = s.replace("\\$", "")
        s = removeRightUnits(s)
        s = s.replace("\\%", "")
        s = s.replace(" .", " 0.").replace("{.", "{0.")
        if (s.isEmpty()) return s
        if (s[0] == '.') s = "0$s"
        val sides = s.split("=")
        if (sides.size == 2 && sides[0].length <= 2) s = sides[1]
        s = fixSqrt(s)
        s = s.replace(" ", "")
        s = fixFracs(s)
        if (s == "0.5") s = "\\frac{1}{2}"
        return fixSlash(s)
    }

    private fun removeRightUnits(s: String): String {
        if ("\\text{ " !in s) return s
        val parts = s.split("\\text{ ")
        require(parts.size == 2) { "more than one unit" }
        return parts[0]
    }

    private fun fixSqrt(s: String): String {
        if ("\\sqrt" !in s) return s
        val parts = s.split("\\sqrt")
        val out = StringBuilder(parts[0])
        for (p in parts.drop(1)) {
            require(p.isNotEmpty()) { "\\sqrt at the end" }
            if (p[0] != '{') out.append("\\sqrt{").append(p[0]).append('}').append(p.substring(1))
            else out.append("\\sqrt").append(p)
        }
        return out.toString()
    }

    private fun fixFracs(s: String): String {
        val subs = s.split("\\frac")
        val out = StringBuilder(subs[0])
        for (sub in subs.drop(1)) {
            out.append("\\frac")
            require(sub.isNotEmpty()) { "\\frac at the end" }
            if (sub[0] == '{') {
                out.append(sub)
                continue
            }
            if (sub.length < 2) return s
            val a = sub[0]
            val b = sub[1]
            if (b != '{') {
                out.append('{').append(a).append("}{").append(b).append('}').append(sub.substring(2))
            } else {
                out.append('{').append(a).append('}').append(b).append(sub.substring(2))
            }
        }
        return out.toString()
    }

    private fun fixSlash(s: String): String {
        val parts = s.split("/")
        if (parts.size != 2) return s
        val a = parts[0].toIntOrNull() ?: return s
        val b = parts[1].toIntOrNull() ?: return s
        if (s != "$a/$b") return s
        return "\\frac{$a}{$b}"
    }
}
