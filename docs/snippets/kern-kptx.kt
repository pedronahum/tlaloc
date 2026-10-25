import io.tlaloc.kptx.PtxKernelTemplate
import io.tlaloc.kptx.emitPtx
import io.tlaloc.kptx.imm

// snippet

fun main() {
    check(".visible .entry scale_by_two(" in ptx) { ptx.take(400) }
    check("add.f32" in ptx)
}
