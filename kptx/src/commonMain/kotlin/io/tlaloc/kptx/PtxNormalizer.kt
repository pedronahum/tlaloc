package io.tlaloc.kptx

/**
 * §0.4.352 — the lenient foreign-PTX front-end. [normalizePtx] accepts
 * real-world PTX formatting (expert hand-written kernels, pyptx/CUTLASS
 * output) and produces the §0.4.338 IR; `emitPtx()` from there is
 * canonical, so the strict [parsePtx] round-trips it and the §0.4.345
 * transpiler can lift it into DSL source. This is the missing mouth of
 * the bootstrap pipeline:
 *
 * ```
 * foreign PTX → normalizePtx → IR → emitPtx (canonical) → transpile → edit
 * ```
 *
 * **Tolerated** (formatting-only variance): arbitrary
 * whitespace/tabs/indentation, statements wrapped across lines
 * (statements end at `;`), `/* … */` block comments and `//` line
 * comments (dropped — the normalizer's contract is *structure*, not
 * byte fidelity; byte-identity starts at the canonical form), missing
 * `.visible`, zero-based register indices, blank-line noise, spacing
 * inside vector operands.
 *
 * **Refused loudly** (constructs the IR cannot express faithfully —
 * widening the IR is the intended response, not guessing):
 * `.func` device functions, `.global`/`.const` module-scope variables,
 * non-`.b8` shared arrays, parameter arrays/alignments, performance
 * directives (`.maxntid` …), multi-token `.target` lists, predicated
 * dual-result operands (`d|p`), and anything else unrecognized — each
 * with the offending token and line number.
 */
class PtxNormalizeException(val line: Int, message: String) :
    RuntimeException("PTX normalize error at line $line: $message")

fun normalizePtx(text: String): PtxModule = PtxNormalizer(text).parse()

private class PtxNormalizer(source: String) {

    private data class Tok(val text: String, val line: Int)

    private val toks: List<Tok>
    private var i = 0

    init {
        // Strip block comments (preserving line counts), then tokenize.
        val noBlocks = StringBuilder()
        var idx = 0
        while (idx < source.length) {
            if (source.startsWith("/*", idx)) {
                val end = source.indexOf("*/", idx + 2)
                if (end < 0) { noBlocks.append(source, idx, source.length); break }
                for (c in source.substring(idx, end + 2)) if (c == '\n') noBlocks.append('\n')
                idx = end + 2
            } else {
                noBlocks.append(source[idx]); idx++
            }
        }
        val out = ArrayList<Tok>()
        var line = 1
        var j = 0
        val s = noBlocks.toString()
        val structural = "{}[],;:()"
        while (j < s.length) {
            val c = s[j]
            when {
                c == '\n' -> { line++; j++ }
                c.isWhitespace() -> j++
                s.startsWith("//", j) -> { while (j < s.length && s[j] != '\n') j++ }
                c in structural -> { out.add(Tok(c.toString(), line)); j++ }
                else -> {
                    val start = j
                    while (j < s.length && !s[j].isWhitespace() && s[j] !in structural) j++
                    out.add(Tok(s.substring(start, j), line))
                }
            }
        }
        toks = out
    }

    private fun fail(message: String): Nothing =
        throw PtxNormalizeException(toks.getOrNull(i)?.line ?: toks.lastOrNull()?.line ?: 0, message)

    private fun peek(): String? = toks.getOrNull(i)?.text
    private fun next(): String = toks.getOrNull(i)?.text?.also { i++ } ?: fail("unexpected end of input")
    private fun expect(t: String) {
        val got = next()
        if (got != t) fail("expected `$t`, got `$got`")
    }

    fun parse(): PtxModule {
        var version = "7.0"
        var target = "sm_75"
        var addressSize = 64
        val kernels = ArrayList<PtxKernel>()

        while (i < toks.size) {
            when (val t = next()) {
                ".version" -> version = next()
                ".target" -> {
                    target = next()
                    if (peek() == ",") fail("multi-token `.target` lists are unsupported")
                }
                ".address_size" -> addressSize = next().toIntOrNull() ?: fail("`.address_size` not an integer")
                ".visible" -> {
                    expect(".entry")
                    kernels.add(parseKernel())
                }
                ".entry" -> kernels.add(parseKernel())
                ".func" -> fail("`.func` device functions are unsupported (widen the IR, don't guess)")
                ".global", ".const" -> fail("module-scope `$t` variables are unsupported")
                else -> fail("unrecognized module-level token `$t`")
            }
        }
        if (kernels.isEmpty()) fail("no kernels found")
        return PtxModule(kernels, version, target, addressSize)
    }

    private fun parseKernel(): PtxKernel {
        val name = next()
        expect("(")
        val params = ArrayList<PtxParam>()
        while (peek() != ")") {
            if (peek() == ",") { i++; continue }
            expect(".param")
            if (peek() == ".align") fail("aligned/array parameters are unsupported")
            val type = next()
            if (!type.startsWith(".")) fail("parameter type must start with `.`, got `$type`")
            val pname = next()
            if (pname.contains("[")) fail("array parameters are unsupported")
            params.add(PtxParam(type, pname))
        }
        expect(")")
        if (peek()?.startsWith(".max") == true || peek()?.startsWith(".req") == true) {
            fail("performance directives (`${peek()}`) are unsupported")
        }
        expect("{")

        val body = ArrayList<PtxStmt>()
        while (true) {
            when (val t = peek() ?: fail("unterminated kernel body")) {
                "}" -> { i++; return PtxKernel(name, params, canonicalizeBody(body)) }
                ".reg" -> { i++; body.add(parseRegDecl()) }
                ".shared" -> { i++; body.add(parseSharedDecl()) }
                else -> {
                    if (t.startsWith(".")) fail("unsupported body directive `$t`")
                    // Label or instruction: label = name followed by `:`.
                    if (toks.getOrNull(i + 1)?.text == ":") {
                        body.add(PtxLabel(next())); i++ // consume `:`
                    } else {
                        body.add(parseInst())
                    }
                }
            }
        }
    }

    /** Canonical body shape: decls first (as parsed), one blank, then
     * statements — the §0.4.338 emitter/KernelScope convention. */
    private fun canonicalizeBody(body: List<PtxStmt>): List<PtxStmt> {
        val decls = body.filter { it is PtxRegDecl || it is PtxSharedDecl }
        val rest = body.filter { it !is PtxRegDecl && it !is PtxSharedDecl }
        return if (decls.isEmpty()) rest else decls + PtxBlank + rest
    }

    private fun parseRegDecl(): PtxRegDecl {
        val type = next()
        if (!type.startsWith(".")) fail("register type must start with `.`")
        val spec = next() // %p<4> possibly split? `<` is not structural, so it stays one token
        val m = Regex("""(%[a-z]+)<(\d+)>""").matchEntire(spec)
            ?: fail("only `%prefix<count>` register declarations are supported, got `$spec`")
        expect(";")
        return PtxRegDecl(type, m.groupValues[1], m.groupValues[2].toInt())
    }

    private fun parseSharedDecl(): PtxSharedDecl {
        var align = 4
        if (peek() == ".align") { i++; align = next().toIntOrNull() ?: fail("bad `.align`") }
        val type = next()
        if (type != ".b8") fail("only `.b8` shared arrays are supported, got `$type`")
        val name = next()
        expect("[")
        val size = next().toIntOrNull() ?: fail("shared array size not an integer")
        expect("]")
        expect(";")
        return PtxSharedDecl(align, name, size)
    }

    private fun parseInst(): PtxInst {
        var guard: PtxGuard? = null
        var first = next()
        if (first.startsWith("@")) {
            val g = first.substring(1)
            guard = if (g.startsWith("!")) PtxGuard(g.substring(1), negated = true) else PtxGuard(g)
            if (!guard.reg.startsWith("%")) fail("guard register must start with `%`")
            first = next()
        }
        val opcode = first
        val operands = ArrayList<PtxOperand>()
        while (peek() != ";") {
            when (peek()) {
                "," -> { i++; continue }
                null -> fail("instruction missing `;`")
            }
            operands.add(parseOperand())
        }
        expect(";")
        return PtxInst(opcode, operands, guard)
    }

    private fun parseOperand(): PtxOperand {
        val t = next()
        return when {
            t == "[" -> {
                val inner = next()
                val plus = inner.indexOf('+')
                val mem = if (plus >= 0) {
                    val off = inner.substring(plus + 1).toIntOrNull()
                        ?: fail("only integer memory offsets are supported, got `$inner`")
                    PtxMem(inner.substring(0, plus), off)
                } else if (peek() == "+") {
                    i++
                    val offTok = next()
                    val off = offTok.toIntOrNull()
                        ?: fail("only integer memory offsets are supported, got `$offTok`")
                    PtxMem(inner, off)
                } else {
                    PtxMem(inner)
                }
                expect("]")
                mem
            }
            t == "{" -> {
                val regs = ArrayList<String>()
                while (peek() != "}") {
                    if (peek() == ",") { i++; continue }
                    val r = next()
                    if (!r.startsWith("%")) fail("vector operand elements must be registers, got `$r`")
                    regs.add(r)
                }
                expect("}")
                PtxVec(regs)
            }
            t.contains("|") -> fail("dual-result operands (`$t`) are unsupported")
            t.startsWith("%") -> PtxReg(t)
            t[0].isDigit() || t[0] == '-' -> PtxImm(t)
            else -> PtxSym(t)
        }
    }
}
