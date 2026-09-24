package io.tlaloc.kptx

/**
 * The lenient foreign-PTX front-end. [normalizePtx] accepts
 * real-world PTX formatting (expert hand-written kernels, pyptx/CUTLASS
 * output) and produces the value-type IR ([PtxModule]); `emitPtx()` from
 * there is canonical, so the strict [parsePtx] round-trips it and the
 * transpiler can lift it into DSL source. This is the entry point of
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
        val rawRegs = ArrayList<Pair<String, Int>>() // prefix → count (coalesced later)
        while (true) {
            when (val t = peek() ?: fail("unterminated kernel body")) {
                "}" -> { i++; return PtxKernel(name, params, canonicalizeBody(rawRegs, body)) }
                ".reg" -> { i++; parseRegDecl(rawRegs) }
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

    /** Canonical body shape: coalesced register banks first (standard
     * classes in pred/b32/f32/b64 order with canonical storage types,
     * then custom prefixes in appearance order), shared decls, one
     * blank, then statements — the emitter/KernelScope
     * convention. Pyptx-style declaration surfaces — single
     * registers (`.reg .b64 %rd0;`), mixed storage types per class, and
     * custom array banks (`%farr0_<4>`) — all coalesce here; declaring
     * more registers than used is harmless PTX. */
    private fun canonicalizeBody(rawRegs: List<Pair<String, Int>>, body: List<PtxStmt>): List<PtxStmt> {
        val counts = LinkedHashMap<String, Int>()
        for ((prefix, count) in rawRegs) {
            counts[prefix] = maxOf(counts[prefix] ?: 0, count)
        }
        val standardOrder = listOf("%p", "%r", "%f", "%rd")
        fun classPrefix(prefix: String): String = when (regClassOf(prefix + "0")) {
            IsaRegClass.PRED -> "%p"
            IsaRegClass.R32 -> "%r"
            IsaRegClass.F32 -> "%f"
            IsaRegClass.R64 -> "%rd"
            null -> fail("register prefix `$prefix` has no recognizable class")
        }
        // Custom prefixes (`%farr0_<4>` pyptx array banks) rename into their
        // standard class past its high-water mark — register names are
        // kernel-local, so this is structure-preserving, and the canonical
        // module then carries only the four standard prefixes the strict
        // pipeline and transpiler speak.
        val standardCounts = LinkedHashMap<String, Int>()
        for (std in standardOrder) counts[std]?.let { standardCounts[std] = it }
        val rename = HashMap<String, String>()
        for ((prefix, count) in counts) {
            if (prefix in standardOrder) continue
            val std = classPrefix(prefix)
            val base = standardCounts[std] ?: 0
            for (idx in 0 until count) {
                rename["$prefix$idx"] = "$std${base + idx}"
            }
            standardCounts[std] = base + count
        }
        fun rn(name: String): String = rename[name] ?: name
        fun rewrite(stmt: PtxStmt): PtxStmt = when (stmt) {
            is PtxInst -> stmt.copy(
                operands = stmt.operands.map { op ->
                    when (op) {
                        is PtxReg -> PtxReg(rn(op.name))
                        is PtxVec -> PtxVec(op.regs.map(::rn))
                        is PtxMem -> if (op.base.startsWith("%")) PtxMem(rn(op.base), op.offset) else op
                        else -> op
                    }
                },
                guard = stmt.guard?.let { PtxGuard(rn(it.reg), it.negated) },
            )
            else -> stmt
        }
        fun canonicalType(std: String): String = when (std) {
            "%p" -> ".pred"; "%r" -> ".b32"; "%f" -> ".f32"; else -> ".b64"
        }
        val decls = ArrayList<PtxStmt>()
        for (std in standardOrder) {
            standardCounts[std]?.let { decls.add(PtxRegDecl(canonicalType(std), std, it)) }
        }
        val shared = body.filterIsInstance<PtxSharedDecl>()
        val rest = body.filter { it !is PtxRegDecl && it !is PtxSharedDecl }.map(::rewrite)
        val all = decls + shared
        return if (all.isEmpty()) rest else all + PtxBlank + rest
    }

    private fun parseRegDecl(rawRegs: MutableList<Pair<String, Int>>) {
        val type = next()
        if (!type.startsWith(".")) fail("register type must start with `.`")
        val spec = next()
        expect(";")
        val bank = Regex("""(%[A-Za-z][A-Za-z0-9_]*)<(\d+)>""").matchEntire(spec)
        if (bank != null) {
            rawRegs.add(bank.groupValues[1] to bank.groupValues[2].toInt())
            return
        }
        // Single-register form: `%rd12` → prefix %rd, needs bank count 13.
        val single = Regex("""(%[A-Za-z][A-Za-z0-9_]*?)(\d+)""").matchEntire(spec)
            ?: fail("unsupported register declaration `$spec` (expected `%prefix<count>` or `%prefixN`)")
        rawRegs.add(single.groupValues[1] to single.groupValues[2].toInt() + 1)
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
