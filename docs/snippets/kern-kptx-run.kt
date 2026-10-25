import doctest.s_kern_kptx.ptx
import io.tlaloc.runtime.pjrt.PjrtBinaries
import io.tlaloc.runtime.pjrt.PjrtSession
import io.tlaloc.runtime.pjrt.PjrtTarget
import io.tlaloc.runtime.pjrt.ffm.PjrtFfiRegistry
import io.tlaloc.runtime.pjrt.kptx.KptxKernelRegistry

fun main() {
    if (!PjrtBinaries.cudaAvailable || !PjrtFfiRegistry.isGpuCustomCallSupported(PjrtBinaries.pluginPath!!)) {
        println("skipped: no CUDA device with GPU custom calls")
        return
    }
    // snippet
    check(doubled.toList() == List(8) { 2f * it }) { "doubled ${doubled.toList()}" }
}
