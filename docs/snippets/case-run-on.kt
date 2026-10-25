// compile-only: needs a GPU and the PJRT CUDA plugin
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.hostF32
import io.tlaloc.nn.CapturedStep
import io.tlaloc.nn.CausalLM
import io.tlaloc.runtime.pjrt.PjrtSession

fun main() {
    val session: PjrtSession = TODO()
    val captured: CapturedStep = TODO()
    val model: CausalLM = TODO()
    val idv: IntArray = TODO()
    val targets: DTensor<*, F32> = TODO()
    // snippet
}
