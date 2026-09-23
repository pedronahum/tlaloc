package io.tlaloc.ir.passes

import io.tlaloc.core.BF16
import io.tlaloc.core.Bool
import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.core.F64
import io.tlaloc.core.I32
import io.tlaloc.core.I64
import io.tlaloc.ir.DxirBlock
import io.tlaloc.ir.DxirBlockArg
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirConst
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirNode
import io.tlaloc.ir.DxirOp
import io.tlaloc.ir.DxirOpResult
import io.tlaloc.ir.DxirParam
import io.tlaloc.ir.DxirRegion
import io.tlaloc.ir.DxirRegionBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import java.security.MessageDigest

/**
 * §0.4.26 — Canonical textual serialisation + SHA-256 hashing of [DxirFunction] for
 * the Stage B.3 coarsening cache (plan §5.4). Enables two things simultaneously:
 *
 *  - **Cache key**: [hash] produces a stable SHA-256 hex digest that does NOT depend
 *    on the original SSA ids — re-numbered to 0..N in depth-first pre-order so two
 *    structurally-equal functions built by different DxirBuilder invocations map to
 *    the same hash.
 *  - **Cache payload**: [serialise] / [deserialise] round-trip a function through a
 *    line-oriented text format so cached artifacts survive across JVM invocations.
 *
 * ### Scope (first cut)
 *
 *  - Scalar + rank-N `DxirType` (dtype + dim list serialised verbatim).
 *  - [DxirParam] / [DxirConst] / [DxirOp] (single + multi-result, with nested regions
 *    recursively encoded). [DxirOpResult] references are emitted as `src.idx`.
 *  - `OpKind` enum names (built-in stability — new kinds are additions, never renames).
 *  - Empty `attrs` + `null` sharding. Primals with non-empty attrs or sharding throw
 *    [UnsupportedOperationException] during [serialise]; widen when a rule needs it.
 *  - [io.tlaloc.ir.DxirCall] is out of scope (not used by FIR-emitted primals today).
 *
 * ### Format (v1)
 *
 *     dxir-canon-v1
 *     fn "<name>" params=<P> body=<B> returns=<R>
 *     param <cid> "<name>" <type>
 *     ...
 *     <body entry>
 *     ...
 *     ret <ref>
 *     ...
 *
 * Body entries (one per line at top level; regions nest with `{` / `}` markers):
 *
 *     const <cid> <type> <value-hex>
 *     op <cid> <OPKIND> types=<t1,t2,...> operands=<r1,r2,...> regions=<n>
 *     [if n>0, n `region {...}` blocks follow]
 *
 *     region {
 *       block args=(<cid>:<type>,<cid>:<type>,...) body=<B> {
 *         <nested body entries>
 *         yield <ref1,ref2,...>
 *       }
 *     }
 *
 * Values are bit-exact (Float/Double → `toRawBits` hex; Int/Long → decimal; Bool →
 * `0`/`1`) so float-rounding doesn't perturb the hash. Type strings: dtype name for
 * scalars (`f32`, `bool`, …), `f32[-1,4]` for rank-N with dim list.
 */
object DxirCanonical {

    const val FORMAT_MARKER: String = "dxir-canon-v1"

    /**
     * Canonical SHA-256 digest (lowercase hex) of [fn]'s serialised form. Stable across
     * renamings of SSA ids; sensitive to op order, op kinds, types, values, and region
     * structure. See class docstring for exactness guarantees.
     */
    fun hash(fn: DxirFunction): String {
        val bytes = serialise(fn).toByteArray(Charsets.UTF_8)
        val digest = MessageDigest.getInstance("SHA-256").digest(bytes)
        return digest.joinToString("") { b -> (b.toInt() and 0xff).toString(16).padStart(2, '0') }
    }

    /** Emit [fn] in the canonical textual format (round-trippable via [deserialise]). */
    fun serialise(fn: DxirFunction): String = Writer().run {
        write(fn)
        out.toString()
    }

    /**
     * Parse a canonical string (produced by [serialise]) back into a [DxirFunction].
     * The reconstructed function's SSA ids are freshly issued by [DxirBuilder] in the
     * same order as the canonical form — so [serialise(deserialise(s))] == `s`.
     */
    fun deserialise(text: String): DxirFunction {
        val parser = Parser(text.lines())
        return parser.parseFunction()
    }

    // ------------------------------------------------------------------------
    // Writer (DxirFunction → String)
    // ------------------------------------------------------------------------

    private class Writer {
        val out = StringBuilder()
        private val idMap = HashMap<Int, Int>()
        private var nextCid = 0
        private var depth = 0

        fun write(fn: DxirFunction) {
            out.append(FORMAT_MARKER).append('\n')
            out.append("fn ").append(escape(fn.name))
                .append(" params=").append(fn.params.size)
                .append(" body=").append(fn.body.size)
                .append(" returns=").append(fn.returns.size).append('\n')
            for (p in fn.params) writeParam(p)
            for (node in fn.body) writeBodyNode(node)
            for (r in fn.returns) {
                indent()
                out.append("ret ").append(ref(r)).append('\n')
            }
        }

        private fun writeParam(p: DxirParam) {
            val cid = assign(p)
            indent()
            out.append("param ").append(cid).append(' ')
                .append(escape(p.name)).append(' ')
                .append(typeStr(p.type)).append('\n')
        }

        private fun writeBodyNode(node: DxirNode) {
            when (node) {
                is DxirConst -> {
                    val cid = assign(node)
                    indent()
                    out.append("const ").append(cid).append(' ')
                        .append(typeStr(node.type)).append(' ')
                        .append(valueHex(node.value, node.type)).append('\n')
                }
                is DxirOp -> {
                    require(node.attrs.isEmpty()) {
                        "DxirCanonical v1 requires empty attrs (op=${node.op}, attrs=${node.attrs.keys})"
                    }
                    require(node.sharding == null) {
                        "DxirCanonical v1 requires null sharding (op=${node.op})"
                    }
                    val cid = assign(node)
                    indent()
                    out.append("op ").append(cid).append(' ').append(node.op.name)
                        .append(" types=").append(node.types.joinToString(",") { typeStr(it) })
                        .append(" operands=").append(node.operands.joinToString(",") { ref(it) })
                        .append(" regions=").append(node.regions.size).append('\n')
                    for (region in node.regions) writeRegion(region)
                }
                else -> throw UnsupportedOperationException(
                    "DxirCanonical v1 does not support ${node::class.simpleName} in body",
                )
            }
        }

        private fun writeRegion(region: DxirRegion) {
            indent(); out.append("region {\n")
            depth++
            for (block in region.blocks) writeBlock(block)
            depth--
            indent(); out.append("}\n")
        }

        private fun writeBlock(block: DxirBlock) {
            indent()
            val argsStr = block.args.joinToString(",") { a ->
                val cid = assign(a)
                "$cid:${typeStr(a.type)}"
            }
            out.append("block args=(").append(argsStr).append(") body=")
                .append(block.body.size).append(" {\n")
            depth++
            for (n in block.body) writeBodyNode(n)
            indent()
            out.append("yield ").append(block.terminator.joinToString(",") { ref(it) }).append('\n')
            depth--
            indent(); out.append("}\n")
        }

        private fun assign(node: DxirNode): Int {
            val cid = nextCid++
            idMap[node.id] = cid
            return cid
        }

        private fun ref(node: DxirNode): String = when (node) {
            is DxirOpResult -> {
                val srcCid = idMap[node.source.id]
                    ?: error("DxirCanonical: forward reference to op id=${node.source.id}")
                "$srcCid.${node.index}"
            }
            else -> idMap[node.id]?.toString()
                ?: error("DxirCanonical: forward reference to node id=${node.id}")
        }

        private fun indent() {
            repeat(depth) { out.append("  ") }
        }
    }

    // ------------------------------------------------------------------------
    // Parser (String → DxirFunction)
    // ------------------------------------------------------------------------

    private class Parser(private val lines: List<String>) {
        private var idx = 0
        private val idMap = HashMap<Int, DxirNode>()

        fun parseFunction(): DxirFunction {
            expect(FORMAT_MARKER)
            val header = popTokens("fn")
            val name = unescape(header[1])
            val paramCount = parseKv(header[2], "params").toInt()
            val bodyCount = parseKv(header[3], "body").toInt()
            val returnCount = parseKv(header[4], "returns").toInt()
            return DxirBuilder.function(name) {
                repeat(paramCount) { parseParam(this) }
                repeat(bodyCount) { parseBodyNode(this) }
                (0 until returnCount).map {
                    val tokens = popTokens("ret")
                    resolveRef(tokens[1])
                }
            }
        }

        private fun parseParam(builder: DxirBuilder) {
            val tokens = popTokens("param")
            val cid = tokens[1].toInt()
            val name = unescape(tokens[2])
            val type = parseType(tokens[3])
            val node = builder.param(name, type)
            idMap[cid] = node
        }

        /** Emit one body node (const or op) into [emitter], which is either a
         *  [DxirBuilder] (top-level) or a [DxirRegionBuilder] (region body). */
        private fun parseBodyNode(emitter: Any) {
            val tokens = peekTokens()
            when (tokens[0]) {
                "const" -> parseConst(emitter)
                "op" -> parseOp(emitter)
                else -> error("unexpected body-node line: '${peekLine()}'")
            }
        }

        private fun parseConst(emitter: Any) {
            val tokens = popTokens("const")
            val cid = tokens[1].toInt()
            val type = parseType(tokens[2])
            val value = parseValue(tokens[3], type)
            val node: DxirConst = when (emitter) {
                is DxirBuilder -> emitter.const(value, type)
                is DxirRegionBuilder -> emitter.const(value, type)
                else -> error("unsupported emitter: ${emitter::class.simpleName}")
            }
            idMap[cid] = node
        }

        private fun parseOp(emitter: Any) {
            val tokens = popTokens("op")
            val cid = tokens[1].toInt()
            val opKind = OpKind.valueOf(tokens[2])
            val typesStr = parseKv(tokens[3], "types")
            val operandsStr = parseKv(tokens[4], "operands")
            val regionCount = parseKv(tokens[5], "regions").toInt()
            val types = if (typesStr.isEmpty()) emptyList() else typesStr.split(",").map { parseType(it) }
            val operands = if (operandsStr.isEmpty()) emptyList()
            else operandsStr.split(",").map { resolveRef(it) }

            val regions = mutableListOf<DxirRegion>()
            repeat(regionCount) { regions += parseRegionInto(emitter) }

            val node: DxirOp = when (emitter) {
                is DxirBuilder -> if (types.size == 1) {
                    emitter.op(opKind, operands, types[0], emptyMap(), null, regions)
                } else {
                    emitter.opMulti(opKind, operands, types, emptyMap(), null, regions)
                }
                is DxirRegionBuilder -> if (types.size == 1) {
                    emitter.op(opKind, operands, types[0], emptyMap(), null, regions)
                } else {
                    emitter.opMulti(opKind, operands, types, emptyMap(), null, regions)
                }
                else -> error("unsupported emitter: ${emitter::class.simpleName}")
            }
            idMap[cid] = node
        }

        /** Parse `region { block { ... } }` — dispatches region creation through whichever
         *  concrete builder type [emitter] carries; [DxirBuilder] and [DxirRegionBuilder]
         *  both expose `region(block)` with an identical signature. */
        private fun parseRegionInto(emitter: Any): DxirRegion {
            expect("region {")
            val region: DxirRegion = when (emitter) {
                is DxirBuilder -> emitter.region { parseSingleBlock(this) }
                is DxirRegionBuilder -> emitter.region { parseSingleBlock(this) }
                else -> error("unsupported emitter: ${emitter::class.simpleName}")
            }
            expect("}")
            return region
        }

        private fun parseSingleBlock(regionBuilder: DxirRegionBuilder) {
            // v1 emitter (DxirRegionBuilder) produces single-block regions, so a region's
            // content is exactly one `block args=... body=... { ... yield ... }`. Multi-
            // block parsing is deferred until the builder supports it.
            val header = popTokens("block")
            val argsPart = parseKv(header[1], "args").removeSurrounding("(", ")")
            val bodyCount = parseKv(header[2], "body").toInt()

            if (argsPart.isNotEmpty()) {
                for (pair in argsPart.split(",")) {
                    val parts = pair.split(":", limit = 2)
                    val cid = parts[0].toInt()
                    val type = parseType(parts[1])
                    val arg: DxirBlockArg = regionBuilder.arg(type)
                    idMap[cid] = arg
                }
            }
            repeat(bodyCount) { parseBodyNode(regionBuilder) }
            val yieldTokens = popTokens("yield")
            val yieldRefs = if (yieldTokens.size == 1) emptyList()
            else yieldTokens[1].split(",").map { resolveRef(it) }
            regionBuilder.yields(*yieldRefs.toTypedArray())
            expect("}")
        }

        /** Resolve a token like `42` or `42.1` to a [DxirNode] (simple or DxirOpResult). */
        private fun resolveRef(token: String): DxirNode {
            val dotIdx = token.indexOf('.')
            return if (dotIdx < 0) {
                idMap[token.toInt()]
                    ?: error("DxirCanonical: unresolved ref '$token'")
            } else {
                val srcCid = token.substring(0, dotIdx).toInt()
                val resultIdx = token.substring(dotIdx + 1).toInt()
                val src = idMap[srcCid] as? DxirOp
                    ?: error("DxirCanonical: ref '$token' source isn't an op")
                src.result(resultIdx)
            }
        }

        // ---- Line / token plumbing ----

        /** Strip leading whitespace, return the trimmed line, advance `idx`. */
        private fun popLine(): String {
            while (idx < lines.size && lines[idx].isBlank()) idx++
            if (idx >= lines.size) error("DxirCanonical: unexpected end of input")
            return lines[idx++].trim()
        }

        private fun peekLine(): String {
            var j = idx
            while (j < lines.size && lines[j].isBlank()) j++
            if (j >= lines.size) error("DxirCanonical: unexpected end of input")
            return lines[j].trim()
        }

        private fun peekTokens(): List<String> = tokenise(peekLine())

        private fun popTokens(expectedFirst: String): List<String> {
            val line = popLine()
            val tokens = tokenise(line)
            require(tokens.isNotEmpty() && tokens[0] == expectedFirst) {
                "DxirCanonical: expected '$expectedFirst ...', got '$line'"
            }
            return tokens
        }

        private fun expect(raw: String) {
            val line = popLine()
            require(line == raw) { "DxirCanonical: expected '$raw', got '$line'" }
        }

        /**
         * Tokenise a line into whitespace-separated tokens, respecting `"..."` (escaped)
         * and balancing `(...)` as a single token. Needed because `args=(1:f32,2:i32)` has
         * a comma that's NOT a token separator.
         */
        private fun tokenise(line: String): List<String> {
            val out = mutableListOf<String>()
            var i = 0
            val sb = StringBuilder()
            fun flush() { if (sb.isNotEmpty()) { out += sb.toString(); sb.setLength(0) } }
            var inString = false
            var parenDepth = 0
            while (i < line.length) {
                val c = line[i]
                when {
                    c == '"' && !inString -> { inString = true; sb.append(c) }
                    c == '"' && inString -> {
                        inString = false; sb.append(c)
                    }
                    inString -> {
                        if (c == '\\' && i + 1 < line.length) {
                            sb.append(c); sb.append(line[i + 1]); i++
                        } else sb.append(c)
                    }
                    c == '(' -> { parenDepth++; sb.append(c) }
                    c == ')' -> { parenDepth--; sb.append(c) }
                    c.isWhitespace() && parenDepth == 0 -> flush()
                    else -> sb.append(c)
                }
                i++
            }
            flush()
            return out
        }
    }

    // ------------------------------------------------------------------------
    // Type / value encoding shared by writer + parser
    // ------------------------------------------------------------------------

    private fun typeStr(t: DxirType): String =
        if (t.dims.isEmpty()) t.dtype.name
        else "${t.dtype.name}[${t.dims.joinToString(",")}]"

    private fun parseType(s: String): DxirType {
        val bracket = s.indexOf('[')
        val (dtypeName, dimsStr) = if (bracket < 0) s to ""
        else s.substring(0, bracket) to s.substring(bracket + 1, s.length - 1)
        val dtype: DType = when (dtypeName) {
            "f32" -> F32
            "f64" -> F64
            // §0.4.456 (G1b) — bf16-TYPED ops round-trip through canonical text
            // (the reverse-transform CSE keys include result types, so the
            // spelling must parse back); bf16 CONSTANTS stay refused below.
            "bf16" -> BF16
            "i32" -> I32
            "i64" -> I64
            "bool" -> Bool
            else -> error("DxirCanonical: unknown dtype '$dtypeName'")
        }
        val dims = if (dimsStr.isEmpty()) emptyList()
        else dimsStr.split(",").map { it.toInt() }
        return DxirType(dtype, dims)
    }

    private fun valueHex(value: Any, type: DxirType): String = when (type.dtype) {
        F32 -> (value as Float).toRawBits().toUInt().toString(16).padStart(8, '0')
        F64 -> (value as Double).toRawBits().toULong().toString(16).padStart(16, '0')
        I32 -> (value as Int).toUInt().toString(16).padStart(8, '0')
        I64 -> (value as Long).toULong().toString(16).padStart(16, '0')
        Bool -> if (value as Boolean) "1" else "0"
        // §0.4.456 (G1b): bf16-typed OPS are first-class, but no dxir constant
        // carries bf16 — the sanctioned constant spelling is CAST(f32 const).
        // A bf16 const would need a bit-pattern literal convention here plus
        // interpreter/emitter const arms; named deferral until a producer exists.
        BF16 -> error(
            "DxirCanonical: bf16 constants are not part of the dxir surface — " +
                "spell them CAST(f32 const)",
        )
    }

    private fun parseValue(s: String, type: DxirType): Any = when (type.dtype) {
        F32 -> Float.fromBits(s.toUInt(16).toInt())
        F64 -> Double.fromBits(s.toULong(16).toLong())
        I32 -> s.toUInt(16).toInt()
        I64 -> s.toULong(16).toLong()
        Bool -> s == "1"
        BF16 -> error(
            "DxirCanonical: bf16 constants are not part of the dxir surface — " +
                "spell them CAST(f32 const)",
        )
    }

    private fun parseKv(token: String, key: String): String {
        val prefix = "$key="
        require(token.startsWith(prefix)) { "expected '$prefix...', got '$token'" }
        return token.substring(prefix.length)
    }

    private fun escape(s: String): String {
        val sb = StringBuilder("\"")
        for (c in s) when (c) {
            '\\' -> sb.append("\\\\")
            '"' -> sb.append("\\\"")
            '\n' -> sb.append("\\n")
            else -> sb.append(c)
        }
        sb.append('"')
        return sb.toString()
    }

    private fun unescape(quoted: String): String {
        require(quoted.startsWith("\"") && quoted.endsWith("\"")) {
            "DxirCanonical: expected quoted string, got '$quoted'"
        }
        val inner = quoted.substring(1, quoted.length - 1)
        val sb = StringBuilder()
        var i = 0
        while (i < inner.length) {
            val c = inner[i]
            if (c == '\\' && i + 1 < inner.length) {
                when (inner[i + 1]) {
                    '\\' -> sb.append('\\')
                    '"' -> sb.append('"')
                    'n' -> sb.append('\n')
                    else -> sb.append(inner[i + 1])
                }
                i += 2
            } else {
                sb.append(c)
                i++
            }
        }
        return sb.toString()
    }
}
