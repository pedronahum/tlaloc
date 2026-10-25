import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.nn.AdamW
import io.tlaloc.nn.CausalLM
import io.tlaloc.nn.CausalLmConfig
import io.tlaloc.nn.capture
import io.tlaloc.nn.crossEntropy
import io.tlaloc.nn.oneHot

fun main() {
    val config = CausalLmConfig(vocabSize = 16, dModel = 16, numLayers = 1, numHeads = 2, numKvHeads = 1, ffHidden = 32, initStd = 0.05f)
    val start = CausalLM.llama(config, RandomKey.fromSeed(5))
    val ids0 = DTensor<Shape, I32>(HostI32Storage(IntArray(8) { it % 16 }), intArrayOf(2, 4), I32)
    val t0 = oneHot(IntArray(8) { (it + 1) % 16 }, 16, intArrayOf(2, 4))
    // snippet
    check(captured.gradient.returns.size == 1 + captured.inputCount + captured.parameterKeys.size) {
        "${captured.gradient.returns.size} outputs"
    }
    check(optimizer.learningRate == 3e-3f)
}
