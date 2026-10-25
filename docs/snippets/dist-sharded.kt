import io.tlaloc.core.F32
import io.tlaloc.ir.DxirAxisRef
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirDimSharding
import io.tlaloc.ir.DxirMesh
import io.tlaloc.ir.DxirMeshAxis
import io.tlaloc.ir.DxirSharding
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.stablehlo.toStablehlo

fun main() {
    // snippet
    val text = gradient.toStablehlo()
    check(text.startsWith("sdy.mesh @m = <[\"data\"=4]>")) { text.take(200) }
    check("func.func @sharded_loss_grad(%0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@m, [{\"data\"}, {}]>}, %1: tensor<4x3xf32>)" in text) {
        text.lines().take(3).joinToString("\n")
    }
}
