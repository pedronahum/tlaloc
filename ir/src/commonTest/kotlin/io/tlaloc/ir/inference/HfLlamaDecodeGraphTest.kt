package io.tlaloc.ir.inference

import io.tlaloc.core.F32
import kotlin.math.abs
import kotlin.math.cos
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * §0.4.479 — the HERMETIC half of H3c-2. No checkpoint, no python, no 2.2 GB
 * on disk: these gate `./gradlew test` on a fresh machine, while
 * `HfLlamaRealDecodeParityTest` carries the claim that the numbers are a real
 * Llama's.
 *
 * A tiny GQA Llama config, **rectangular on purpose** (intermediate != hidden,
 * kvHeads < heads) for the same reason §0.4.478 bought a rectangular
 * checkpoint: a square model cannot tell a transpose from an identity.
 */
class HfLlamaDecodeGraphTest {

    private val config = HfDecoderConfig(
        architecture = "LlamaForCausalLM",
        modelType = "llama",
        hiddenSize = 8,
        intermediateSize = 20,
        numLayers = 2,
        numHeads = 4,
        numKvHeads = 2,
        headDim = 2,
        vocabSize = 7,
        rmsNormEps = 1e-5,
        ropeTheta = 10000.0,
        maxPositionEmbeddings = 16,
        tieWordEmbeddings = false,
        attentionBias = false,
        torchDtype = "bfloat16",
        ropeScalingType = null,
    )

    private fun spec(): DecodeGraphSpec = HfDecoderGraph.spec(
        config,
        config.toDecodeModelShape(numBlocks = 4, blockSize = 2),
        DecodeBucket(batch = 3, maxContext = 4),
    )

    @Test
    fun theSlotsAndTheRolesStayInLockstep() {
        val slots = HfDecoderGraph.weightSlots(config)
        val roles = HfDecoderGraph.weightRoles(config)
        assertEquals(slots.size, roles.size, "one role per slot")
        assertEquals(
            1 + config.numLayers * 9 + 2,
            slots.size,
            "embed + layers x 9 + finalNorm + lmHead",
        )
        // Every role is one the name mapping knows, and no role repeats.
        assertEquals(roles.size, roles.toSet().size, "roles are distinct")
        for (r in roles) assertTrue(HfDecoderNames.hfName(r).isNotBlank())
        assertTrue(slots.all { it.role == DecodeSlotRole.WEIGHT })
    }

    /**
     * THE TRANSPOSE, as a shape claim. Every staged Linear slot is the
     * REVERSE of what §0.4.478 says the file holds, and every non-Linear slot
     * is identical to it. This is the invariant `HfStagedWeights` must
     * produce, stated where it can be checked without a file.
     */
    @Test
    fun everyLinearSlotIsTheReverseOfItsFileDims() {
        val slots = HfDecoderGraph.weightSlots(config)
        val roles = HfDecoderGraph.weightRoles(config)
        var linears = 0
        for (i in roles.indices) {
            val file = HfDecoderNames.expectedDims(roles[i], config).toList()
            val staged = slots[i].type.dims
            if (HfDecoderNames.isTransposedLinear(roles[i])) {
                linears++
                assertEquals(file.reversed(), staged, "slot '${slots[i].name}' (${roles[i]})")
            } else {
                assertEquals(file, staged, "slot '${slots[i].name}' (${roles[i]}) is not a Linear")
            }
        }
        // 7 projections per layer, plus lm_head. The embedding table is a
        // LOOKUP and the norm gains are rank-1, so neither is transposed.
        assertEquals(config.numLayers * 7 + 1, linears, "7 per layer plus lm_head")
    }

    @Test
    fun theBuiltGraphMatchesTheContractExactly() {
        val s = spec()
        val fn = HfDecoderGraph.build(s, config)
        // verifySignature already ran inside build(); re-run it here so the
        // gate is visible as a test and not only as an internal assert.
        s.verifySignature(fn, "test")
        assertEquals(5 + 2 * config.numLayers + s.weightSlots.size, fn.params.size)
        assertEquals(1 + 2 * config.numLayers, fn.returns.size)
        // The pools still start at index 5 — staged weights go AFTER them, so
        // the donation pairs H3a wrote are untouched by this extension.
        assertEquals(
            (0 until 2 * config.numLayers).map { Pair(5 + it, 1 + it) },
            s.donationPairs,
        )
    }

    @Test
    fun aSpecWhoseWeightSignatureIsNotThisConfigsIsRefused() {
        val s = spec().copy(weightSlots = emptyList())
        val e = assertFailsWith<IllegalArgumentException> { HfDecoderGraph.build(s, config) }
        assertTrue(e.message!!.contains("weight signature"), e.message!!)
    }

    @Test
    fun aPrefillGraphMatchesItsContract() {
        val s = spec().copy(kind = DecodeGraphKind.PREFILL)
        val fn = HfDecoderGraph.build(s, config)
        s.verifySignature(fn, "test")
        assertEquals(listOf(3, 4), fn.params[0].type.dims, "tokenIds [B, T]")
        assertEquals(listOf(3, 1, config.vocabSize), fn.returns[0].type.dims, "last-position logits")
    }

    @Test
    fun anOddHeadDimIsRefusedBecauseRotateHalfHasNoSplit() {
        val odd = config.copy(headDim = 3)
        val e = assertFailsWith<IllegalArgumentException> {
            HfDecoderGraph.ropeTables(odd, 4)
        }
        assertTrue(e.message!!.contains("rotate_half"), e.message!!)
    }

    /**
     * The RoPE tables against the formula, by hand. `headDim = 2` means one
     * frequency pair with `inv_freq[0] = theta^0 = 1`, so the angle at
     * position p is exactly p radians — a case whose expected value can be
     * written down rather than recomputed by the same code under test.
     *
     * And the DUPLICATION: `cos[p][0] == cos[p][1]`, which is what makes
     * `rotate_half` (rather than an interleave) the correct partner. Getting
     * that wrong is the single most common RoPE bug, and it is invisible at
     * position 0 where every angle is zero.
     */
    @Test
    fun theRopeTablesAreTheDuplicatedHuggingFaceForm() {
        val (cosT, sinT) = HfDecoderGraph.ropeTables(config, 5)
        assertEquals(5 * config.headDim, cosT.size)
        for (p in 0 until 5) {
            val want = cos(p.toDouble()).toFloat()
            assertTrue(
                abs(cosT[p * 2] - want) < 1e-6f,
                "cos at position $p: ${cosT[p * 2]} vs $want",
            )
            assertEquals(cosT[p * 2], cosT[p * 2 + 1], "the two halves are DUPLICATED, not distinct")
            assertEquals(sinT[p * 2], sinT[p * 2 + 1])
        }
        assertEquals(0f, sinT[0], "position 0 is the identity rotation")
        assertEquals(1f, cosT[0])
    }

    @Test
    fun theRopeTablesAreTheOnlyConstantsAndTheyAreConfigDerived() {
        val fn = HfDecoderGraph.build(spec(), config)
        val consts = fn.body.filterIsInstance<io.tlaloc.ir.DxirConst>()
        // cos, sin, and the per-row rms eps. Everything else is a param.
        // The tables cover the entry's context (4 positions here), not the
        // model's 16: a position past the context never reaches the graph.
        val s = spec()
        val positions = minOf(config.maxPositionEmbeddings, s.maxBlocksPerSeq * s.model.blockSize)
        assertEquals(4, positions)
        val ropeShaped = consts.count { it.type.dims == listOf(positions, config.headDim) }
        assertEquals(2, ropeShaped, "exactly the cos and sin tables")
        assertTrue(
            consts.none { it.type.dims == listOf(config.maxPositionEmbeddings, config.headDim) },
            "no table sized by max_position_embeddings",
        )
        assertTrue(
            consts.none { it.type.dtype == F32 && it.type.dims.contains(config.vocabSize) },
            "no weight-shaped constant: a real checkpoint's weights are STAGED, not baked",
        )
    }
}
