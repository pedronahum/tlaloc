package io.tlaloc.kptx

/**
 * KPTX v3.1 (§0.4.345) — the PTX → Kotlin DSL transpiler (plan task
 * 16): the bootstrap workflow's first half. Feed it expert-written PTX
 * (canonical format, e.g. straight out of [parsePtx]); it produces
 * Kotlin source that rebuilds the same module through [KernelScope] —
 * ready to edit, specialize, and dispatch through the KPTX stack.
 *
 * # Self-verifying by construction
 *
 * The transpiler lowers the module to a typed **step list** and drives
 * two backends off the same steps: a Kotlin-source printer (each step
 * renders exactly one DSL call) and a replay executor (each step
 * performs that call against a real [KernelScope]). [transpile] then
 * requires `replay.emitPtx() == module.emitPtx()` **byte-for-byte**
 * before returning the source — anything the step mapping can't
 * express faithfully (register-class deviations, decl orderings the
 * DSL can't reproduce, unknown special registers, `[shared+off]`
 * addressing) fails loudly instead of generating silently-wrong code.
 *
 * Original register numbering and bank sizes are preserved via
 * [KernelScope.reg]/[KernelScope.bank] (a source kernel's `%f<16>`
 * stays `%f<16>` even when `%f14` is the highest use).
 */
object KptxTranspiler {

    /** Transpile [module] to Kotlin DSL source. Throws
     * [IllegalArgumentException]/[IllegalStateException] when the
     * module can't be reproduced faithfully. */
    fun transpile(module: PtxModule, functionName: String? = null): String {
        val kernels = module.kernels.map { lowerKernel(it) }
        val name = functionName ?: defaultFunctionName(module)
        val source = render(module, kernels, name)

        val replayed = replay(module, kernels)
        check(replayed.emitPtx() == module.emitPtx()) {
            "KptxTranspiler self-verification failed: replayed module is not byte-identical " +
                "to the input (the step mapping cannot express this module faithfully)"
        }
        return source
    }

    private fun defaultFunctionName(module: PtxModule): String {
        val base = module.kernels.first().name
            .split("_", ".").joinToString("") { part ->
                part.replaceFirstChar { it.uppercase() }
            }.replaceFirstChar { it.lowercase() }
        return base + "Module"
    }

    // =====================================================================
    // Step model — one step, one DSL call.
    // =====================================================================

    private sealed interface OperandExpr {
        data class RegRef(val ident: String) : OperandExpr
        data class SpecialRef(val prop: String) : OperandExpr
        data class Imm(val text: String, val quoted: Boolean) : OperandExpr
        data class MemReg(val ident: String, val offset: Int) : OperandExpr
        data class MemParam(val ident: String) : OperandExpr
        data class SymRef(val ident: String) : OperandExpr
        data class VecRef(val idents: List<String>) : OperandExpr
    }

    private sealed interface Step {
        data class Param(val type: String, val name: String, val ident: String) : Step
        data class Bank(val cls: IsaRegClass, val count: Int) : Step
        data class Shared(val name: String, val size: Int, val align: Int, val ident: String) : Step
        data class RegVal(val cls: IsaRegClass, val index: Int, val ident: String) : Step
        data class LabelDecl(val name: String, val ident: String) : Step
        data class Place(val ident: String) : Step
        data class Comment(val text: String) : Step
        data object Blank : Step
        data class Inst(
            val opcode: String,
            val operands: List<OperandExpr>,
            val guardIdent: String?,
            val guardNegated: Boolean,
            val comment: String?,
        ) : Step
    }

    private val SPECIALS = mapOf(
        "%tid.x" to "tidX", "%ntid.x" to "ntidX",
        "%ctaid.x" to "ctaidX", "%nctaid.x" to "nctaidX",
        "%tid.y" to "tidY", "%ntid.y" to "ntidY",
        "%ctaid.y" to "ctaidY", "%nctaid.y" to "nctaidY",
        "%tid.z" to "tidZ", "%ntid.z" to "ntidZ",
        "%ctaid.z" to "ctaidZ", "%nctaid.z" to "nctaidZ",
    )

    private fun regIdent(name: String): Pair<IsaRegClass, Int> {
        val cls = regClassOf(name)
            ?: throw IllegalArgumentException("transpiler: register `$name` has no recognizable class")
        val index = name.removePrefix(cls.prefix).toIntOrNull()
            ?: throw IllegalArgumentException("transpiler: register `$name` has no numeric index")
        return cls to index
    }

    private fun lowerKernel(kernel: PtxKernel): List<Step> {
        val steps = ArrayList<Step>()
        val paramIdents = LinkedHashMap<String, String>()
        for (p in kernel.params) {
            val ident = sanitize(p.name)
            paramIdents[p.name] = ident
            steps.add(Step.Param(p.type, p.name, ident))
        }

        // Declarations: banks + shared, in body order; find where the decl
        // block ends (KernelScope re-synthesizes decls + one blank).
        var declEnd = 0
        val sharedIdents = LinkedHashMap<String, String>()
        for (stmt in kernel.body) {
            when (stmt) {
                is PtxRegDecl -> {
                    val cls = IsaRegClass.entries.firstOrNull { it.prefix == stmt.prefix }
                        ?: throw IllegalArgumentException("transpiler: unknown register prefix `${stmt.prefix}`")
                    steps.add(Step.Bank(cls, stmt.count))
                    declEnd++
                }
                is PtxSharedDecl -> {
                    val ident = sanitize(stmt.name)
                    sharedIdents[stmt.name] = ident
                    steps.add(Step.Shared(stmt.name, stmt.sizeBytes, stmt.align, ident))
                    declEnd++
                }
                else -> break
            }
        }
        var rest = kernel.body.drop(declEnd)
        if (declEnd > 0 && rest.firstOrNull() == PtxBlank) rest = rest.drop(1)

        // Registers actually referenced → val decls with original indices.
        val regIdents = LinkedHashMap<String, String>()
        fun noteReg(name: String) {
            if (name in SPECIALS || name in regIdents) return
            regIdents[name] = sanitize(name.removePrefix("%"))
        }
        for (stmt in rest) {
            if (stmt !is PtxInst) continue
            stmt.guard?.let { noteReg(it.reg) }
            for (op in stmt.operands) when (op) {
                is PtxReg -> noteReg(op.name)
                is PtxVec -> op.regs.forEach { noteReg(it) }
                is PtxMem -> if (op.base.startsWith("%")) noteReg(op.base)
                else -> {}
            }
        }
        val regSteps = regIdents.keys
            .map { name -> val (cls, idx) = regIdent(name); Triple(cls, idx, regIdents[name]!!) }
            .sortedWith(compareBy({ it.first.ordinal }, { it.second }))
            .map { (cls, idx, ident) -> Step.RegVal(cls, idx, ident) }
        steps.addAll(regSteps)

        // Labels, declared up-front in placement order.
        val labelIdents = LinkedHashMap<String, String>()
        for (stmt in rest) if (stmt is PtxLabel) {
            labelIdents[stmt.name] = sanitize(stmt.name)
            steps.add(Step.LabelDecl(stmt.name, labelIdents[stmt.name]!!))
        }

        // Body.
        for (stmt in rest) when (stmt) {
            is PtxRegDecl, is PtxSharedDecl ->
                throw IllegalArgumentException("transpiler: declaration after first body statement is unsupported")
            is PtxLabel -> steps.add(Step.Place(labelIdents[stmt.name]!!))
            is PtxComment -> steps.add(Step.Comment(stmt.text))
            PtxBlank -> steps.add(Step.Blank)
            is PtxInst -> {
                val ops = stmt.operands.map { op ->
                    when (op) {
                        is PtxReg ->
                            SPECIALS[op.name]?.let { OperandExpr.SpecialRef(it) }
                                ?: OperandExpr.RegRef(regIdents[op.name]!!)
                        is PtxImm -> OperandExpr.Imm(op.text, quoted = op.text.toIntOrNull()?.toString() != op.text)
                        is PtxMem -> when {
                            op.base.startsWith("%") -> OperandExpr.MemReg(regIdents[op.base]!!, op.offset)
                            op.base in paramIdents -> {
                                require(op.offset == 0) { "transpiler: param memory operand with offset is unsupported" }
                                OperandExpr.MemParam(paramIdents[op.base]!!)
                            }
                            else -> throw IllegalArgumentException(
                                "transpiler: memory base `${op.base}` is neither a register nor a param",
                            )
                        }
                        is PtxSym ->
                            labelIdents[op.name]?.let { OperandExpr.SymRef(it) }
                                ?: sharedIdents[op.name]?.let { OperandExpr.SymRef(it) }
                                ?: throw IllegalArgumentException(
                                    "transpiler: symbol `${op.name}` is neither a label nor a shared array",
                                )
                        is PtxVec -> OperandExpr.VecRef(op.regs.map { r ->
                            SPECIALS[r]?.let {
                                throw IllegalArgumentException("transpiler: special register in vector operand")
                            }
                            regIdents[r]!!
                        })
                    }
                }
                steps.add(Step.Inst(stmt.opcode, ops, stmt.guard?.reg?.let { regIdents[it] ?: it }, stmt.guard?.negated ?: false, stmt.comment))
            }
        }
        return steps
    }

    private fun sanitize(name: String): String {
        val cleaned = name.map { if (it.isLetterOrDigit() || it == '_') it else '_' }.joinToString("")
        return if (cleaned.firstOrNull()?.isDigit() != false) "k_$cleaned" else cleaned
    }

    // =====================================================================
    // Backend 1 — Kotlin source printer.
    // =====================================================================

    private fun render(module: PtxModule, kernels: List<List<Step>>, functionName: String): String = buildString {
        append("/**\n")
        append(" * Generated by KptxTranspiler (§0.4.345) from PTX — a bootstrap\n")
        append(" * artifact meant to be edited, not a build product. Verified at\n")
        append(" * generation time: replaying these calls reproduces the source PTX\n")
        append(" * byte-identically.\n")
        append(" */\n")
        val single = module.kernels.size == 1
        if (single) {
            append("fun $functionName(): PtxModule = ptxKernel(\"${module.kernels[0].name}\", ")
            append("version = \"${module.version}\", target = \"${module.target}\") {\n")
            renderKernelBody(kernels[0], indent = "    ")
            append("}\n")
        } else {
            append("fun $functionName(): PtxModule = ptxModule(")
            append("version = \"${module.version}\", target = \"${module.target}\") {\n")
            for ((k, steps) in module.kernels.zip(kernels)) {
                append("    kernel(\"${k.name}\") {\n")
                renderKernelBody(steps, indent = "        ")
                append("    }\n")
            }
            append("}\n")
        }
    }

    private fun StringBuilder.renderKernelBody(steps: List<Step>, indent: String) {
        var lastWasHeader = false
        for (step in steps) {
            when (step) {
                is Step.Param -> {
                    append(indent).append("val ${step.ident} = param(\"${step.type}\", \"${step.name}\")\n")
                    lastWasHeader = true
                }
                is Step.Bank -> {
                    append(indent).append("bank(IsaRegClass.${step.cls.name}, ${step.count})\n")
                    lastWasHeader = true
                }
                is Step.Shared -> {
                    append(indent).append("val ${step.ident} = shared(\"${step.name}\", sizeBytes = ${step.size}")
                    if (step.align != 4) append(", align = ${step.align}")
                    append(")\n")
                    lastWasHeader = true
                }
                is Step.RegVal -> {
                    append(indent).append("val ${step.ident} = reg(IsaRegClass.${step.cls.name}, ${step.index})\n")
                    lastWasHeader = true
                }
                is Step.LabelDecl -> {
                    append(indent).append("val ${step.ident} = label(\"${step.name}\")\n")
                    lastWasHeader = true
                }
                else -> {
                    if (lastWasHeader) {
                        append('\n')
                        lastWasHeader = false
                    }
                    when (step) {
                        is Step.Place -> append(indent).append("place(${step.ident})\n")
                        is Step.Comment -> append(indent).append("comment(\"${escape(step.text)}\")\n")
                        Step.Blank -> append(indent).append("blank()\n")
                        is Step.Inst -> {
                            append(indent).append("inst(\"${step.opcode}\"")
                            for (op in step.operands) append(", ").append(renderOperand(op))
                            step.guardIdent?.let {
                                append(", guard = ")
                                if (step.guardNegated) append('!')
                                append(it)
                            }
                            step.comment?.let { append(", comment = \"${escape(it)}\"") }
                            append(")\n")
                        }
                        else -> error("unreachable")
                    }
                }
            }
        }
    }

    private fun renderOperand(op: OperandExpr): String = when (op) {
        is OperandExpr.RegRef -> op.ident
        is OperandExpr.SpecialRef -> op.prop
        is OperandExpr.Imm -> if (op.quoted) "imm(\"${op.text}\")" else "imm(${op.text})"
        is OperandExpr.MemReg -> if (op.offset == 0) "mem(${op.ident})" else "mem(${op.ident}, ${op.offset})"
        is OperandExpr.MemParam -> "mem(${op.ident})"
        is OperandExpr.SymRef -> op.ident
        is OperandExpr.VecRef -> "vec(${op.idents.joinToString(", ")})"
    }

    private fun escape(s: String): String = s.replace("\\", "\\\\").replace("\"", "\\\"").replace("$", "\${'$'}")

    // =====================================================================
    // Backend 2 — replay executor (the self-verification path).
    // =====================================================================

    private fun replay(module: PtxModule, kernels: List<List<Step>>): PtxModule =
        ptxModule(version = module.version, target = module.target) {
            for ((k, steps) in module.kernels.zip(kernels)) {
                kernel(k.name) { executeSteps(steps) }
            }
        }

    private fun KernelScope.executeSteps(steps: List<Step>) {
        val params = HashMap<String, KParam>()
        val regs = HashMap<String, KReg>()
        val labels = HashMap<String, KLabel>()
        val syms = HashMap<String, KSym>()
        fun operand(op: OperandExpr): KOp = when (op) {
            is OperandExpr.RegRef -> regs.getValue(op.ident)
            is OperandExpr.SpecialRef -> when (op.prop) {
                "tidX" -> tidX; "ntidX" -> ntidX; "ctaidX" -> ctaidX; "nctaidX" -> nctaidX
                "tidY" -> tidY; "ntidY" -> ntidY; "ctaidY" -> ctaidY; "nctaidY" -> nctaidY
                "tidZ" -> tidZ; "ntidZ" -> ntidZ; "ctaidZ" -> ctaidZ; "nctaidZ" -> nctaidZ
                else -> error("unknown special `$${op.prop}`")
            }
            is OperandExpr.Imm -> imm(op.text)
            is OperandExpr.MemReg -> mem(regs.getValue(op.ident), op.offset)
            is OperandExpr.MemParam -> mem(params.getValue(op.ident))
            is OperandExpr.SymRef -> labels[op.ident] ?: syms.getValue(op.ident)
            is OperandExpr.VecRef -> KVec(op.idents.map { regs.getValue(it) })
        }
        for (step in steps) when (step) {
            is Step.Param -> params[step.ident] = param(step.type, step.name)
            is Step.Bank -> bank(step.cls, step.count)
            is Step.Shared -> syms[step.ident] = shared(step.name, step.size, step.align)
            is Step.RegVal -> regs[step.ident] = reg(step.cls, step.index)
            is Step.LabelDecl -> labels[step.ident] = label(step.name)
            is Step.Place -> place(labels.getValue(step.ident))
            is Step.Comment -> comment(step.text)
            Step.Blank -> blank()
            is Step.Inst -> {
                val guard: KGuard? = step.guardIdent?.let { ident ->
                    val g = regs.getValue(ident)
                    if (step.guardNegated) !g else g
                }
                inst(
                    step.opcode,
                    *step.operands.map { operand(it) }.toTypedArray(),
                    guard = guard,
                    comment = step.comment,
                )
            }
        }
    }
}
