// compile-only
import io.tlaloc.core.DTensor
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
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
import io.tlaloc.runtime.pjrt.PjrtBinaries
import io.tlaloc.runtime.pjrt.PjrtSession
import io.tlaloc.runtime.pjrt.PjrtTarget
import io.tlaloc.runtime.pjrt.ffm.CompileOptions
import io.tlaloc.runtime.pjrt.ffm.CpuClientOptions

fun main() {
    val config = CausalLmConfig(vocabSize = 64, dModel = 64, numLayers = 2, numHeads = 8, numKvHeads = 4, ffHidden = 128)
    val model = CausalLM.llama(config, RandomKey.fromSeed(3))
    val ids = DTensor<Shape, I32>(HostI32Storage(IntArray(8 * 16)), intArrayOf(8, 16), I32)
    val targets = oneHot(IntArray(8 * 16), 64, intArrayOf(8, 16))
    val mesh = DxirMesh("mesh", listOf(DxirMeshAxis("data", 4)))
    val step = capture(model, listOf(ids), targets = listOf(targets)) { logits, t -> crossEntropy(logits, t[0]) }
    val device = step.sharded(ShardingPlan.fsdp(mesh)).withDeviceAdamW(AdamW(learningRate = 1e-2f))
    val batches = listOf(listOf(FloatArray(8 * 16), targets.hostF32()))
    // snippet
}
