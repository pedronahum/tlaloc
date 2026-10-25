// compile-only: needs a GPU and the PJRT CUDA plugin
import io.tlaloc.nn.AdamW
import io.tlaloc.nn.CapturedStep
import io.tlaloc.nn.withDeviceAdamW
import io.tlaloc.runtime.pjrt.PjrtSession
import io.tlaloc.runtime.pjrt.ffm.PjrtBuffer
import io.tlaloc.stablehlo.toStablehlo

fun main() {
    val session: PjrtSession = TODO()
    val captured: CapturedStep = TODO()
    val optimizer: AdamW = TODO()
    val idsB: PjrtBuffer = TODO()
    val tgtB: PjrtBuffer = TODO()
    var params: List<PjrtBuffer> = TODO()
    var m: List<PjrtBuffer> = TODO()
    var v: List<PjrtBuffer> = TODO()
    val t = 1
    val n = params.size
    // snippet
}
