import io.tlaloc.core.DTensor
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.ir.DxirMesh
import io.tlaloc.ir.DxirMeshAxis
import io.tlaloc.nn.AdamW
import io.tlaloc.nn.CausalLM
import io.tlaloc.nn.CausalLmConfig
import io.tlaloc.nn.ShardingPlan
import io.tlaloc.nn.capture
import io.tlaloc.nn.crossEntropy
import io.tlaloc.nn.oneHot
import io.tlaloc.nn.sharded
import io.tlaloc.nn.withDeviceAdamW
import io.tlaloc.stablehlo.toStablehlo

fun main() {
    val config = CausalLmConfig(vocabSize = 64, dModel = 64, numLayers = 2, numHeads = 8, numKvHeads = 4, ffHidden = 128)
    val model = CausalLM.llama(config, RandomKey.fromSeed(3))
    val ids = DTensor<Shape, I32>(HostI32Storage(IntArray(8 * 16)), intArrayOf(8, 16), I32)
    val targets = oneHot(IntArray(8 * 16), 64, intArrayOf(8, 16))
    // snippet
    val text = device.function.toStablehlo()
    check(text.startsWith("sdy.mesh @mesh = <[\"data\"=4]>")) { text.take(200) }
    // embed.table [64, 64] is held as four [16, 64] pieces.
    check("tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"data\"}, {}]>" in text)
    check("sdy.sharding_constraint" in text)
}
