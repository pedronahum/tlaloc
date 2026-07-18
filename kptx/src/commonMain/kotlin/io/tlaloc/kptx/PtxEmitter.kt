package io.tlaloc.kptx

/**
 * KPTX v2.1 (§0.4.338) — canonical PTX text emitter. The output format
 * **is** the corpus format: task 10's parser round-trips
 * `emit(parse(text)) == text` byte-for-byte against text in this style,
 * which is deliberately the style the v1 hand-written kernels already
 * use (§0.4.328–337):
 *
 * ```
 * .version 7.0
 * .target sm_75
 * .address_size 64
 *
 * .visible .entry name(
 *     .param .u64 x_ptr,
 *     .param .u32 n_cols
 * )
 * {
 *     .reg .pred %p<4>;
 *     .shared .align 4 .b8 sdata[1024];
 *
 *     ld.param.u64 %rd1, [x_ptr];
 *     // comment
 * LOOP:
 *     @%p1 bra DONE;
 *     ret;
 * }
 * ```
 *
 * Canonical rules: header lines then one blank line before each kernel;
 * params one-per-line at 4-space indent; body statements at 4-space
 * indent except labels at column 0; guards prefix the instruction
 * (`@%p1 ` / `@!%p1 `); operands joined with `", "`; memory operands
 * `[base]` / `[base+off]`; every statement line ends the file's
 * newline discipline (trailing `\n` at EOF). Immediates and opcodes
 * are emitted verbatim — the emitter never reformats a constant.
 */
fun PtxModule.emitPtx(): String = buildString {
    append(".version ").append(version).append('\n')
    append(".target ").append(target).append('\n')
    append(".address_size ").append(addressSize).append('\n')
    for (kernel in kernels) {
        append('\n')
        emitKernel(kernel)
    }
}

private fun StringBuilder.emitKernel(kernel: PtxKernel) {
    append(".visible .entry ").append(kernel.name).append("(\n")
    for ((i, p) in kernel.params.withIndex()) {
        append("    .param ").append(p.type).append(' ').append(p.name)
        if (i != kernel.params.lastIndex) append(',')
        append('\n')
    }
    append(")\n{\n")
    for (stmt in kernel.body) {
        when (stmt) {
            is PtxRegDecl ->
                append("    .reg ").append(stmt.type).append(' ')
                    .append(stmt.prefix).append('<').append(stmt.count).append(">;\n")
            is PtxSharedDecl ->
                append("    .shared .align ").append(stmt.align).append(" .b8 ")
                    .append(stmt.name).append('[').append(stmt.sizeBytes).append("];\n")
            is PtxInst -> {
                append("    ")
                stmt.guard?.let { g ->
                    append('@')
                    if (g.negated) append('!')
                    append(g.reg).append(' ')
                }
                append(stmt.opcode)
                if (stmt.operands.isNotEmpty()) {
                    append(' ')
                    for ((i, op) in stmt.operands.withIndex()) {
                        if (i != 0) append(", ")
                        emitOperand(op)
                    }
                }
                append(';')
                stmt.comment?.let { append("    // ").append(it) }
                append('\n')
            }
            is PtxLabel -> append(stmt.name).append(":\n")
            is PtxComment -> append("    // ").append(stmt.text).append('\n')
            PtxBlank -> append('\n')
        }
    }
    append("}\n")
}

private fun StringBuilder.emitOperand(op: PtxOperand) {
    when (op) {
        is PtxReg -> append(op.name)
        is PtxImm -> append(op.text)
        is PtxSym -> append(op.name)
        is PtxMem -> {
            append('[').append(op.base)
            if (op.offset != 0) append('+').append(op.offset)
            append(']')
        }
        is PtxVec -> {
            append('{')
            for ((i, r) in op.regs.withIndex()) {
                if (i != 0) append(", ")
                append(r)
            }
            append('}')
        }
    }
}
