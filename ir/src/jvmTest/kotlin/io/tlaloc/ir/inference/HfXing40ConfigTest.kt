package io.tlaloc.ir.inference

import org.junit.jupiter.api.Assumptions.assumeTrue
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** Xing 4.0's config as the xing4_0 family reads it, against transformers' values. */
class HfXing40ConfigTest {

    private fun dir(): Path? {
        val snaps = Path.of(System.getProperty("user.home"), ".cache/huggingface/hub/models--XingChen-AGI--Xing4.0-29B-A4B/snapshots")
        if (!Files.isDirectory(snaps)) return null
        return Files.list(snaps).use { s -> s.filter { Files.isRegularFile(it.resolve("config.json")) }.findFirst().orElse(null) }
    }

    /** `ROPE_INIT_FUNCTIONS["yarn"]` of transformers 5.17 on this config, and the attention's scaling. */
    private val transformersInvFreq = doubleArrayOf(
        1.0, 0.7498942017555237, 0.5623413324356079, 0.4216965138912201, 0.3162277638912201, 0.23713736236095428,
        0.17782793939113617, 0.1333521455526352, 0.10000000149011612, 0.07498941570520401, 0.05623412877321243,
        0.0389765165746212, 0.026833752170205116, 0.018326841294765472, 0.012396659702062607, 0.008286424912512302,
        0.00545673118904233, 0.0035241420846432447, 0.002216922352090478, 0.0013431437546387315, 0.000767764518968761,
        0.00039617903530597687, 0.00016243898426182568, 2.0836272597080097e-05, 1.5625000742147677e-05,
        1.1717096640495583e-05, 8.786582839093171e-06, 6.589007625734666e-06, 4.9410591600462794e-06,
        3.7052716379548656e-06, 2.7785615657194285e-06, 2.08362735065748e-06,
    )

    @Test
    fun theFamilyReadsMlaHyperConnectionsAndTheRouter() {
        val d = dir()
        assumeTrue(d != null, "no XingChen-AGI/Xing4.0-29B-A4B checkpoint")
        val c = HfDecoderConfig.parse(Files.readString(d!!.resolve("config.json")))
        assertEquals(HfModelFamily.Xing4_0, c.family)
        val m = c.mla!!
        assertEquals(listOf(768, 512, 128, 64, 128), listOf(m.qLoraRank, m.kvLoraRank, m.nopeDim, m.ropeDim, m.valueDim))
        assertEquals(576, c.headDim)
        assertEquals(1, c.numKvHeads)
        for (i in transformersInvFreq.indices) {
            assertTrue(abs(m.invFreq[i] - transformersInvFreq[i]) <= 1e-7 * transformersInvFreq[i] + 1e-12, "inv_freq[$i]: ${m.invFreq[i]}")
        }
        assertEquals(0.14467962580268923, m.scale, 1e-12)
        assertEquals(HyperConnectionConfig(4, 20, 1e-6, -30.0, 30.0), c.hyper)
        assertEquals(MoeConfig(64, 4, 1024, 1024, MoeRouting.SIGMOID_BIAS, 2.0), c.moe)
        assertEquals(listOf(MlpKind.DENSE, MlpKind.DENSE, MlpKind.MOE), (0..2).map { c.layer(it).mlp })
        assertTrue(c.unsupportedFeatures().isEmpty(), "${c.unsupportedFeatures()}")
    }
}
