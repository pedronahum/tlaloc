package io.tlaloc.stablehlo

import io.tlaloc.ir.DxirAxisRef
import io.tlaloc.ir.DxirDimSharding
import io.tlaloc.ir.DxirMesh
import io.tlaloc.ir.DxirMeshAxis
import io.tlaloc.ir.DxirSharding

/**
 * Small recursive-descent parser for the SDY subset that [SdyEmit] emits:
 *
 *  1. Module-level mesh declarations: `sdy.mesh @name = <["axis"=size, ...]>`
 *  2. Sharding attributes: `<@mesh, [{"axes"}p0, {"a","b"?}], replicated={"x"}>`
 *
 * Intentional scope: parse what the emitter produces plus the specific enrichments that
 * `sdy-opt --sdy-propagation-pipeline` will insert (closed/open dim-shardings, priorities,
 * sub-axes, replicated tails). No support yet for manual_computation bodies, sharding
 * groups, or arbitrary MLIR containers — those would require a full MLIR tokenizer which
 * is out of scope for a round-trip self-test.
 *
 * Both entry points accept input with surrounding whitespace and ignore it.
 */
object SdyParser {
    /**
     * Parse exactly one `sdy.mesh @name = <[..]>` declaration. Throws on malformed input
     * rather than returning null — the callers here are tests asserting round-trip equality,
     * so failing fast is more useful than a silent null.
     */
    fun parseMeshDecl(input: String): DxirMesh = Cursor(input).run {
        skipWs()
        expectLiteral("sdy.mesh")
        skipWs()
        expect('@')
        val name = readIdent()
        skipWs()
        expect('=')
        skipWs()
        parseMeshBody(name).also {
            skipWs()
            require(atEnd()) { "trailing input after mesh decl at pos $pos: '${tail()}'" }
        }
    }

    /**
     * Parse a standalone `<@mesh, [...], replicated={...}>` sharding attribute. See `toSdyAttr`.
     */
    fun parseSharding(input: String): DxirSharding = Cursor(input).run {
        skipWs()
        parseShardingAttr().also {
            skipWs()
            require(atEnd()) { "trailing input after sharding attr at pos $pos: '${tail()}'" }
        }
    }

    /** Extract and parse every `sdy.mesh @name = <[..]>` in a multi-line module string. */
    fun parseMeshesInModule(input: String): List<DxirMesh> {
        val meshes = mutableListOf<DxirMesh>()
        var idx = 0
        while (true) {
            val hit = input.indexOf("sdy.mesh", idx)
            if (hit < 0) break
            val cursor = Cursor(input)
            cursor.pos = hit
            cursor.expectLiteral("sdy.mesh")
            cursor.skipWs()
            cursor.expect('@')
            val name = cursor.readIdent()
            cursor.skipWs()
            cursor.expect('=')
            cursor.skipWs()
            meshes += cursor.parseMeshBody(name)
            idx = cursor.pos
        }
        return meshes
    }

    // --- recursive descent helpers on Cursor ---

    internal fun Cursor.parseMeshBody(name: String): DxirMesh {
        expect('<')
        skipWs()
        expect('[')
        skipWs()
        val axes = mutableListOf<DxirMeshAxis>()
        if (peek() != ']') {
            axes += parseMeshAxis()
            skipWs()
            while (peek() == ',') {
                pos++
                skipWs()
                axes += parseMeshAxis()
                skipWs()
            }
        }
        expect(']')
        skipWs()
        expect('>')
        return DxirMesh(name, axes)
    }

    private fun Cursor.parseMeshAxis(): DxirMeshAxis {
        val axisName = readQuotedString()
        skipWs()
        expect('=')
        skipWs()
        val size = readInt()
        return DxirMeshAxis(axisName, size)
    }

    private fun Cursor.parseShardingAttr(): DxirSharding {
        expect('<')
        skipWs()
        expect('@')
        val meshName = readIdent()
        skipWs()
        expect(',')
        skipWs()
        expect('[')
        skipWs()
        val dims = mutableListOf<DxirDimSharding>()
        if (peek() != ']') {
            dims += parseDimSharding()
            skipWs()
            while (peek() == ',') {
                pos++
                skipWs()
                dims += parseDimSharding()
                skipWs()
            }
        }
        expect(']')
        skipWs()
        val replicated = if (peek() == ',') {
            pos++
            skipWs()
            expectLiteral("replicated")
            skipWs()
            expect('=')
            skipWs()
            expect('{')
            skipWs()
            val refs = mutableListOf<DxirAxisRef>()
            if (peek() != '}') {
                refs += parseAxisRef()
                skipWs()
                while (peek() == ',') {
                    pos++
                    skipWs()
                    refs += parseAxisRef()
                    skipWs()
                }
            }
            expect('}')
            refs
        } else {
            emptyList()
        }
        skipWs()
        expect('>')
        return DxirSharding(meshName, dims, replicated)
    }

    private fun Cursor.parseDimSharding(): DxirDimSharding {
        expect('{')
        skipWs()
        val axes = mutableListOf<DxirAxisRef>()
        var closed = true
        if (peek() == '?') {
            pos++
            closed = false
            skipWs()
        } else if (peek() != '}') {
            axes += parseAxisRef()
            skipWs()
            while (peek() == ',') {
                pos++
                skipWs()
                if (peek() == '?') {
                    pos++
                    closed = false
                    skipWs()
                    break
                }
                axes += parseAxisRef()
                skipWs()
            }
        }
        expect('}')
        // Optional priority suffix `pN`.
        val priority = if (peek() == 'p') {
            pos++
            readInt()
        } else {
            null
        }
        return DxirDimSharding(axes = axes, closed = closed, priority = priority)
    }

    private fun Cursor.parseAxisRef(): DxirAxisRef {
        val name = readQuotedString()
        return if (peek() == ':') {
            pos++
            expect('(')
            val preSize = readInt()
            expect(')')
            val size = readInt()
            DxirAxisRef.Sub(name, preSize, size)
        } else {
            DxirAxisRef.Full(name)
        }
    }

    // --- tokenizer primitives ---

    internal class Cursor(val input: String) {
        var pos: Int = 0

        fun atEnd(): Boolean = pos >= input.length
        fun peek(): Char = if (atEnd()) '\u0000' else input[pos]
        fun tail(): String = input.substring(pos.coerceAtMost(input.length))

        fun skipWs() {
            while (!atEnd() && input[pos].isWhitespace()) pos++
        }

        fun expect(c: Char) {
            if (atEnd() || input[pos] != c) {
                error("expected '$c' at pos $pos but got '${peek()}' (input: ${input.take(200)})")
            }
            pos++
        }

        fun expectLiteral(s: String) {
            if (!input.regionMatches(pos, s, 0, s.length)) {
                error("expected '$s' at pos $pos but got '${input.substring(pos, (pos + s.length).coerceAtMost(input.length))}' (input: ${input.take(200)})")
            }
            pos += s.length
        }

        fun readIdent(): String {
            val start = pos
            while (!atEnd() && (input[pos].isLetterOrDigit() || input[pos] == '_')) pos++
            require(pos > start) { "expected identifier at pos $start" }
            return input.substring(start, pos)
        }

        fun readQuotedString(): String {
            expect('"')
            val start = pos
            while (!atEnd() && input[pos] != '"') pos++
            val s = input.substring(start, pos)
            expect('"')
            return s
        }

        fun readInt(): Int {
            val start = pos
            if (!atEnd() && (input[pos] == '-' || input[pos] == '+')) pos++
            while (!atEnd() && input[pos].isDigit()) pos++
            require(pos > start && !(pos - start == 1 && !input[start].isDigit())) {
                "expected integer at pos $start"
            }
            return input.substring(start, pos).toInt()
        }
    }
}
