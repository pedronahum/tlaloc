package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.io.JsonException
import io.tlaloc.core.io.Safetensors
import io.tlaloc.core.io.SafetensorsTensor
import io.tlaloc.core.io.SafetensorsWriter
import io.tlaloc.core.split
import io.tlaloc.core.uniformFloats
import io.tlaloc.autograd.constant
import io.tlaloc.autograd.mean
import io.tlaloc.autograd.minus
import io.tlaloc.autograd.times
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * §0.4.502 (Tier 2 item 7) — **the model round trip.**
 *
 * The oracle is BIT IDENTITY, everywhere it can be. Not "close": the host lane
 * is deterministic (`DxirInterpreter` on f32 host arrays — `examples/gpu-training`
 * measured two host runs printing identical numbers), so a save and a load that
 * preserve a model exactly must reproduce the same raw float bits, the same
 * loss bits, and the same continued trajectory bits. Anything looser would
 * pass over a checkpoint that had quietly rounded through a text format.
 *
 * Raw bits, not `==`, because `-0f == 0f` and `NaN != NaN`: a comparison on
 * values is not a comparison on contents.
 */
class ModelCheckpointTest {

    // ---- the fixed threefry task, shared with EndToEndTrainingTest --------

    private val n = 16
    private val d = 4

    private fun inputs(): FloatArray {
        val u = uniformFloats(RandomKey.fromSeed(1234).split(2)[0], n * d)
        return FloatArray(u.size) { i -> 2f * u[i] - 1f }
    }

    private fun targets(x: FloatArray): FloatArray = FloatArray(n) { i ->
        val r = i * d
        x[r] * x[r + 1] + 0.5f * x[r + 2] - 0.25f * x[r + 3]
    }

    private fun model(): Sequential = Sequential(
        Dense(d, 8, RandomKey.fromSeed(7).split(2)[0]),
        ReluLayer,
        Dense(8, 1, RandomKey.fromSeed(7).split(2)[1]),
    )

    private fun captureFor(m: Sequential, x: DTensor<*, F32>, yv: FloatArray): CapturedStep =
        capture(m, listOf(x)) { y ->
            val t = y.constant<Shape>(yv, intArrayOf(n, 1))
            val diff = y - t
            (diff * diff).mean()
        }

    /** Raw-bit equality of two keyed parameter sets, with the key in the message. */
    private fun assertSameBits(
        expected: List<NamedParameter>,
        actual: List<NamedParameter>,
        tag: String,
    ) {
        assertEquals(expected.map { it.key }, actual.map { it.key }, "$tag: keys")
        for (i in expected.indices) {
            val e = expected[i]
            val a = actual[i]
            assertContentEquals(e.tensor.dims, a.tensor.dims, "$tag: '${e.key}' dims")
            val ev = e.tensor.hostF32()
            val av = a.tensor.hostF32()
            assertEquals(ev.size, av.size, "$tag: '${e.key}' size")
            for (j in ev.indices) {
                assertEquals(
                    ev[j].toRawBits(),
                    av[j].toRawBits(),
                    "$tag: '${e.key}'[$j] — ${ev[j]} vs ${av[j]}",
                )
            }
        }
    }

    // ---- (1) the headline claim ------------------------------------------

    @Test
    fun aTrainedModelSurvivesASaveAndALoadBitIdentically() {
        val xv = inputs()
        val yv = targets(xv)
        val x = Tensors.f32Matrix<Sym, Sym>(n, d, xv)
        var trained = model()
        val step = captureFor(trained, x, yv)
        val opt = Adam(learningRate = 0.05f)
        var state = opt.initialState()
        repeat(20) {
            val r = step.run(trained, listOf(x))
            val (next, s) = opt.step(trained, r.gradients, state)
            trained = next
            state = s
        }

        val bytes = ModelCheckpoint.encode(trained, opt, state, mapOf("task" to "mlp"))
        val snapshot = ModelCheckpoint.decode(bytes)
        assertEquals(mapOf("task" to "mlp"), snapshot.metadata)
        assertEquals(trained.parameters.sumOf { it.tensor.size }, snapshot.parameterScalars)

        // Restored into a FRESH model of the same structure — the untrained
        // one, so nothing but the checkpoint can be supplying the values.
        val fresh = model()
        assertNotEquals(
            trained.parameters[0].tensor.hostF32()[0].toRawBits(),
            fresh.parameters[0].tensor.hostF32()[0].toRawBits(),
            "the control is broken: the fresh model already equals the trained one",
        )
        val loaded = snapshot.restore(fresh)
        assertSameBits(trained.parameters, loaded.parameters, "restored parameters")

        // And the PREDICTION is bit-identical through the same captured graph.
        val a = step.run(trained, listOf(x))
        val b = step.run(loaded, listOf(x))
        assertEquals(a.loss.toRawBits(), b.loss.toRawBits(), "loss bits: ${a.loss} vs ${b.loss}")
        for (key in a.gradients.keys) {
            val g1 = a.gradients.getValue(key).hostF32()
            val g2 = b.gradients.getValue(key).hostF32()
            for (j in g1.indices) assertEquals(g1[j].toRawBits(), g2[j].toRawBits(), "grad '$key'[$j]")
        }
        println("[nn-ckpt] 20 Adam steps, ${bytes.size}-byte checkpoint, loss ${a.loss} reproduced bit-exactly")
    }

    @Test
    fun continuingTrainingFromTheLoadedStateMatchesContinuingFromTheOriginal() {
        val xv = inputs()
        val yv = targets(xv)
        val x = Tensors.f32Matrix<Sym, Sym>(n, d, xv)
        val opt = Adam(learningRate = 0.05f)

        var a = model()
        val step = captureFor(a, x, yv)
        var sa = opt.initialState()
        repeat(20) {
            val r = step.run(a, listOf(x))
            val (m, s) = opt.step(a, r.gradients, sa)
            a = m
            sa = s
        }

        // Save, then fork: one branch keeps its in-memory state, the other
        // throws it away and rebuilds both model and optimizer from the file.
        val snapshot = ModelCheckpoint.decode(ModelCheckpoint.encode(a, opt, sa))
        var b = snapshot.restore(model())
        var sb = snapshot.restoreOptimizerState(opt)
        assertEquals(20, sa.stepCount)
        assertEquals(20, sb.stepCount)

        val lossesA = ArrayList<Float>()
        val lossesB = ArrayList<Float>()
        repeat(15) {
            val ra = step.run(a, listOf(x))
            lossesA += ra.loss
            val (ma, nsa) = opt.step(a, ra.gradients, sa)
            a = ma
            sa = nsa

            val rb = step.run(b, listOf(x))
            lossesB += rb.loss
            val (mb, nsb) = opt.step(b, rb.gradients, sb)
            b = mb
            sb = nsb
        }
        for (i in lossesA.indices) {
            assertEquals(
                lossesA[i].toRawBits(),
                lossesB[i].toRawBits(),
                "resumed step $i: ${lossesA[i]} vs ${lossesB[i]} (full curves $lossesA / $lossesB)",
            )
        }
        assertSameBits(a.parameters, b.parameters, "after 15 resumed steps")
        println("[nn-ckpt] 15 resumed steps bit-identical, loss ${lossesA.first()} -> ${lossesA.last()}")
    }

    /**
     * The NEGATIVE result that makes the previous test mean something: dropping
     * the optimizer state and restarting Adam's moments produces a DIFFERENT
     * trajectory from the same weights. If it did not, saving optimizer state
     * would be decoration.
     */
    @Test
    fun resumingWithoutTheOptimizerStateGivesADifferentTrajectory() {
        val xv = inputs()
        val yv = targets(xv)
        val x = Tensors.f32Matrix<Sym, Sym>(n, d, xv)
        val opt = Adam(learningRate = 0.05f)

        var a = model()
        val step = captureFor(a, x, yv)
        var sa = opt.initialState()
        repeat(20) {
            val r = step.run(a, listOf(x))
            val (m, s) = opt.step(a, r.gradients, sa)
            a = m
            sa = s
        }
        val snapshot = ModelCheckpoint.decode(ModelCheckpoint.encode(a, opt, sa))
        var b = snapshot.restore(model())
        var sb = opt.initialState() // the state deliberately NOT restored

        // Step 0 after the fork shares the weights, so the LOSS is equal...
        val r0a = step.run(a, listOf(x))
        val r0b = step.run(b, listOf(x))
        assertEquals(r0a.loss.toRawBits(), r0b.loss.toRawBits())
        // ...and the UPDATE is not, because bias correction at t=1 divides the
        // first moment by 1−β₁ = 0.1.
        val (ma, _) = opt.step(a, r0a.gradients, sa)
        val (mb, _) = opt.step(b, r0b.gradients, sb)
        var differ = 0
        val total = ma.parameters.sumOf { it.tensor.size }
        for (i in ma.parameters.indices) {
            val p = ma.parameters[i].tensor.hostF32()
            val q = mb.parameters[i].tensor.hostF32()
            for (j in p.indices) if (p[j].toRawBits() != q[j].toRawBits()) differ++
        }
        println("[nn-ckpt] restarting Adam's moments moved $differ of $total scalars differently")
        assertTrue(
            differ > total / 2,
            "restarting Adam's moments must change the update: only $differ of $total scalars moved",
        )
        assertEquals(0, opt.initialState().stepCount)
        assertEquals(20, sa.stepCount)
    }

    // ---- (2) every shipped optimizer's state ------------------------------

    private fun <S> assertOptimizerStateResumes(optimizer: Optimizer<S>, tag: String) {
        val xv = inputs()
        val yv = targets(xv)
        val x = Tensors.f32Matrix<Sym, Sym>(n, d, xv)
        var a = model()
        val step = captureFor(a, x, yv)
        var sa = optimizer.initialState()
        repeat(3) {
            val r = step.run(a, listOf(x))
            val (m, s) = optimizer.step(a, r.gradients, sa)
            a = m
            sa = s
        }
        val snapshot = ModelCheckpoint.decode(ModelCheckpoint.encode(a, optimizer, sa))
        var b = snapshot.restore(model())
        var sb = snapshot.restoreOptimizerState(optimizer)
        repeat(3) {
            val ra = step.run(a, listOf(x))
            val (ma, nsa) = optimizer.step(a, ra.gradients, sa)
            a = ma
            sa = nsa
            val rb = step.run(b, listOf(x))
            assertEquals(ra.loss.toRawBits(), rb.loss.toRawBits(), "$tag: resumed loss")
            val (mb, nsb) = optimizer.step(b, rb.gradients, sb)
            b = mb
            sb = nsb
        }
        assertSameBits(a.parameters, b.parameters, "$tag: resumed parameters")
    }

    @Test
    fun adamStateResumes() = assertOptimizerStateResumes(Adam(learningRate = 0.05f), "Adam")

    @Test
    fun sgdWithMomentumStateResumes() =
        assertOptimizerStateResumes(SGD(initialLearningRate = 0.02f, momentum = 0.9f), "SGD+momentum")

    @Test
    fun sgdWithLearningRateDecayResumesItsStepCount() {
        // The step count is the whole point here: lr = lr₀/(1+d·k), so a
        // resume that restarted k would step at the un-annealed rate.
        assertOptimizerStateResumes(SGD(initialLearningRate = 0.2f, lrDecay = 0.5f), "SGD+lrDecay")
    }

    @Test
    fun rmspropStateResumes() = assertOptimizerStateResumes(RMSprop(alpha = 0.01f), "RMSprop")

    @Test
    fun fixedLearningRateHasNoStateAndSaysSo() {
        val opt = FixedLearningRate(0.01f)
        val cp = opt.saveState(Unit)
        assertEquals("FixedLearningRate", cp.kind)
        assertTrue(cp.scalars.isEmpty() && cp.tensors.isEmpty())
        assertOptimizerStateResumes(opt, "FixedLearningRate")
    }

    @Test
    fun aScheduledOptimizerResumesItsSchedulePositionAndItsInnerState() {
        val opt = Scheduled(StepDecay(0.05f, 0.5f, 2)) { lr -> Adam(learningRate = lr) }
        assertOptimizerStateResumes(opt, "Scheduled(Adam)")
        // And the nesting is visible in the checkpoint rather than merged:
        val xv = inputs()
        val yv = targets(xv)
        val x = Tensors.f32Matrix<Sym, Sym>(n, d, xv)
        var m = model()
        val step = captureFor(m, x, yv)
        var s = opt.initialState()
        repeat(4) {
            val r = step.run(m, listOf(x))
            val (nm, ns) = opt.step(m, r.gradients, s)
            m = nm
            s = ns
        }
        val cp = opt.saveState(s)
        assertEquals("Scheduled(Adam)", cp.kind)
        assertEquals("4", cp.scalar("stepCount"))
        assertEquals("4", cp.scalar("inner.stepCount"))
        assertTrue(cp.tensors.keys.all { it.startsWith("inner.") }, "${cp.tensors.keys}")
        // One (m, v) pair per parameter tensor: 0.w, 0.b, 2.w, 2.b.
        assertEquals(2 * m.parameters.size, cp.tensors.size)
        assertEquals(4, opt.loadState(cp).stepCount)
    }

    // ---- (3) buffers: BatchNorm's running statistics ----------------------

    @Test
    fun aBatchNormsRunningStatisticsRoundTripThroughACheckpoint() {
        val bn = BatchNorm(numFeatures = 3)
        val batch = Tensors.f32Matrix<Sym, Sym>(
            4,
            3,
            floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f),
        )
        val trained = bn.withStats(bn.updatedStats(batch))
        assertTrue(trained.stats.runningN > 0f, "the control is broken: stats are still fresh")

        val model = Sequential(trained, ReluLayer)
        assertEquals(
            listOf("0.runningN", "0.runningSum", "0.runningSumOfSquares"),
            model.buffers.map { it.key },
        )
        val snapshot = ModelCheckpoint.decode(ModelCheckpoint.encode(model))
        assertEquals(3, snapshot.buffers.size)
        assertEquals(0, snapshot.buffers.getValue("0.runningN").dims.size)

        val restored = snapshot.restore(Sequential(BatchNorm(numFeatures = 3), ReluLayer))
        val back = restored.layers[0] as BatchNorm
        assertEquals(trained.stats.runningN.toRawBits(), back.stats.runningN.toRawBits())
        assertSameBits(trained.buffers, back.buffers, "BatchNorm buffers")

        // The inference freeze is what the statistics are FOR, so that is what
        // is compared: a reloaded model must serve the same numbers.
        assertSameBits(
            trained.inferenceMode().parameters,
            back.inferenceMode().parameters,
            "frozen inference affine",
        )
    }

    @Test
    fun aFreshBatchNormsBuffersRoundTripToAModelThatStillRefusesInference() {
        // runningN = 0 is a legitimate state (no batch seen) and must survive
        // as itself; `inferenceMode` then refuses by name on BOTH sides rather
        // than one of them emitting NaNs.
        val model = Sequential(BatchNorm(numFeatures = 2))
        val snapshot = ModelCheckpoint.decode(ModelCheckpoint.encode(model))
        val back = snapshot.restore(Sequential(BatchNorm(numFeatures = 2))).layers[0] as BatchNorm
        assertEquals(0f, back.stats.runningN)
        assertFailsWith<IllegalArgumentException> { back.inferenceMode() }
    }

    // ---- (4) the other layer kinds ----------------------------------------

    private fun <M : Trainable<M>> assertParametersRoundTrip(tag: String, build: () -> M) {
        val original = build()
        val snapshot = ModelCheckpoint.decode(ModelCheckpoint.encode(original))
        val restored = snapshot.restore(build())
        assertSameBits(original.parameters, restored.parameters, tag)
        assertEquals(
            original.parameters.map { it.key }.toSet(),
            snapshot.parameters.keys,
            "$tag: checkpoint keys",
        )
    }

    @Test
    fun conv2dParametersRoundTrip() =
        assertParametersRoundTrip("Conv2d") { Conv2d(intArrayOf(2, 1, 3, 3), RandomKey.fromSeed(11)) }

    @Test
    fun embeddingParametersRoundTrip() =
        assertParametersRoundTrip("Embedding") { Embedding(6, 4, RandomKey.fromSeed(12)) }

    @Test
    fun gruParametersRoundTripInBothCandidateGateVariants() {
        assertParametersRoundTrip("GRU") { GRU(3, 5, RandomKey.fromSeed(13)) }
        assertParametersRoundTrip("GRU(linearBeforeReset)") {
            GRU(3, 5, RandomKey.fromSeed(13), linearBeforeReset = true)
        }
    }

    @Test
    fun affineTransformAndABiaslessDenseRoundTrip() {
        assertParametersRoundTrip("AffineTransform") {
            AffineTransform(
                Tensors.f32Vector<Sym>(floatArrayOf(1f, 2f)),
                Tensors.f32Vector<Sym>(floatArrayOf(3f, 4f)),
            )
        }
        assertParametersRoundTrip("Dense(bias=false)") {
            Dense(3, 2, RandomKey.fromSeed(14), bias = false)
        }
    }

    @Test
    fun aDeepSequentialWithAConvAndABatchNormRoundTrips() {
        assertParametersRoundTrip("deep Sequential") {
            Sequential(
                Conv2d(intArrayOf(2, 1, 3, 3), RandomKey.fromSeed(15)),
                ReluLayer,
                BatchNorm(numFeatures = 2),
                Flatten,
                Dense(2, 3, RandomKey.fromSeed(16)),
            )
        }
    }

    // ---- (5) the file is a safetensors file -------------------------------

    @Test
    fun aCheckpointIsAnOrdinarySafetensorsFileAnyReaderCanOpen() {
        val m = Sequential(BatchNorm(numFeatures = 2), Dense(2, 2, RandomKey.fromSeed(3)))
        val opt = Adam()
        val bytes = ModelCheckpoint.encode(m, opt, opt.initialState(), mapOf("note" to "hello"))
        // Read through the FORMAT reader, which knows nothing about :nn.
        val raw = Safetensors.readAll(bytes)
        assertEquals(
            setOf(
                "param.0.gamma", "param.0.beta", "param.1.w", "param.1.b",
                "buffer.0.runningN", "buffer.0.runningSum", "buffer.0.runningSumOfSquares",
            ),
            raw.keys,
        )
        val n = Safetensors.headerLength(bytes)
        val header = Safetensors.parseHeader(
            bytes.decodeToString(8, (8 + n).toInt()),
            8 + n,
            bytes.size - 8 - n,
        )
        assertEquals("1", header.metadata[ModelCheckpoint.VERSION_KEY])
        assertEquals("Adam", header.metadata[ModelCheckpoint.OPTIMIZER_KIND_KEY])
        assertEquals("0", header.metadata["tlaloc.optimizer.stepCount"])
        assertEquals("hello", header.metadata["note"])
    }

    @Test
    fun savingTheSameModelTwiceProducesTheSameBytes() {
        val m = model()
        val opt = Adam()
        val one = ModelCheckpoint.encode(m, opt, opt.initialState(), mapOf("a" to "1"))
        val two = ModelCheckpoint.encode(m, opt, opt.initialState(), mapOf("a" to "1"))
        assertContentEquals(one, two)
    }

    // ---- (6) refusals, by name -------------------------------------------

    @Test
    fun restoringIntoADifferentStructureNamesTheKeysThatDisagree() {
        val bytes = ModelCheckpoint.encode(model())
        val snapshot = ModelCheckpoint.decode(bytes)
        val wrong = Sequential(Dense(d, 8, RandomKey.fromSeed(7).split(2)[0]), ReluLayer)
        val e = assertFailsWith<JsonException> { snapshot.restore(wrong) }
        assertTrue("2.w" in e.message!! && "2.b" in e.message!!, e.message!!)
        assertTrue("carries VALUES" in e.message!!, e.message!!)
    }

    @Test
    fun restoringATensorOfTheWrongShapeIsRefusedByName() {
        val bytes = ModelCheckpoint.encode(Dense(3, 2, RandomKey.fromSeed(1)))
        val snapshot = ModelCheckpoint.decode(bytes)
        val e = assertFailsWith<JsonException> {
            snapshot.restore(Dense(4, 2, RandomKey.fromSeed(1)))
        }
        assertTrue("'w'" in e.message!! && "[3, 2]" in e.message!! && "[4, 2]" in e.message!!, e.message!!)
    }

    /**
     * A Trainable with BatchNorm's parameter keys and no buffers — the shape
     * of a user-written layer that reimplemented BatchNorm's parameters and
     * forgot its statistics. The keys MATCH, so nothing but the buffer check
     * stands between this and a model that loads and then serves NaNs.
     */
    private class GammaBetaOnly(
        val gamma: DTensor<*, F32>,
        val beta: DTensor<*, F32>,
    ) : Trainable<GammaBetaOnly> {
        override val parameters: List<NamedParameter> =
            listOf(NamedParameter("gamma", gamma), NamedParameter("beta", beta))

        override fun withParameters(updated: Map<String, DTensor<*, F32>>) =
            GammaBetaOnly(updated["gamma"] ?: gamma, updated["beta"] ?: beta)
    }

    @Test
    fun aCheckpointWithBuffersRefusesAModelThatCannotHoldThem() {
        val snapshot = ModelCheckpoint.decode(ModelCheckpoint.encode(BatchNorm(numFeatures = 2)))
        assertEquals(3, snapshot.buffers.size)
        val e = assertFailsWith<JsonException> {
            snapshot.restore(
                GammaBetaOnly(
                    Tensors.f32Vector<Sym>(floatArrayOf(1f, 1f)),
                    Tensors.f32Vector<Sym>(floatArrayOf(0f, 0f)),
                ),
            )
        }
        assertTrue("not Stateful" in e.message!!, e.message!!)
        assertTrue("runningSum" in e.message!!, e.message!!)
    }

    @Test
    fun askingForOptimizerStateAModelOnlyCheckpointDoesNotHaveIsRefusedByName() {
        val snapshot = ModelCheckpoint.decode(ModelCheckpoint.encode(model()))
        val e = assertFailsWith<JsonException> { snapshot.restoreOptimizerState(Adam()) }
        assertTrue("no optimizer state was saved" in e.message!!, e.message!!)
        assertTrue("initialState()" in e.message!!, e.message!!)
    }

    @Test
    fun loadingOneOptimizersStateIntoAnotherIsRefusedByName() {
        val adam = Adam()
        val bytes = ModelCheckpoint.encode(model(), adam, adam.initialState())
        val snapshot = ModelCheckpoint.decode(bytes)
        val e = assertFailsWith<JsonException> { snapshot.restoreOptimizerState(RMSprop()) }
        assertTrue("'Adam'" in e.message!! && "'RMSprop'" in e.message!!, e.message!!)
        assertTrue("preconditioner" in e.message!!, e.message!!)
        // A Scheduled(Adam) checkpoint is not an Adam checkpoint either — the
        // bare Adam has nowhere to put the schedule position.
        val sched = Scheduled(ConstantLR(0.01f)) { Adam(learningRate = it) }
        val s2 = ModelCheckpoint.decode(ModelCheckpoint.encode(model(), sched, sched.initialState()))
        val e2 = assertFailsWith<JsonException> { s2.restoreOptimizerState(adam) }
        assertTrue("'Scheduled(Adam)'" in e2.message!!, e2.message!!)
    }

    @Test
    fun anOptimizerThatCannotBeCheckpointedIsRefusedByNameAtSaveTime() {
        // A user-written optimizer that does not opt in. It still TRAINS; it
        // simply cannot be saved, and the refusal says so and says what is
        // still available.
        val custom = object : Optimizer<Unit> {
            override fun initialState() = Unit
            override fun step(
                params: List<NamedParameter>,
                grads: Map<String, DTensor<*, F32>>,
                state: Unit,
            ): OptimizerStep<Unit> = OptimizerStep(params.associate { it.key to it.tensor }, Unit)
        }
        val e = assertFailsWith<JsonException> { ModelCheckpoint.encode(model(), custom, Unit) }
        assertTrue("CheckpointableOptimizer" in e.message!!, e.message!!)
        assertTrue("ModelCheckpoint.encode(model)" in e.message!!, e.message!!)
        // Scheduled over a non-checkpointable inner refuses too, and says it.
        val sched = Scheduled(ConstantLR(0.1f)) { custom }
        val e2 = assertFailsWith<JsonException> { ModelCheckpoint.encode(model(), sched, sched.initialState()) }
        assertTrue("wrapped in Scheduled" in e2.message!!, e2.message!!)
    }

    @Test
    fun aReservedMetadataKeyIsRefusedByName() {
        val e = assertFailsWith<IllegalArgumentException> {
            ModelCheckpoint.encode(model(), mapOf("tlaloc.checkpoint.version" to "99"))
        }
        assertTrue("reserved" in e.message!!, e.message!!)
    }

    @Test
    fun aForeignSafetensorsFileIsRefusedByNameRatherThanRestoringNothing() {
        // The failure this protects against: a HuggingFace checkpoint has no
        // `param.` prefixes, so a lenient loader would restore zero parameters
        // and report success.
        val foreign = SafetensorsWriter.encode(
            listOf(SafetensorsTensor.of("model.layers.0.weight", Tensors.f32Vector<Sym>(floatArrayOf(1f)))),
        )
        val e = assertFailsWith<JsonException> { ModelCheckpoint.decode(foreign) }
        assertTrue(ModelCheckpoint.VERSION_KEY in e.message!!, e.message!!)
        assertTrue("HuggingFace" in e.message!!, e.message!!)
    }

    @Test
    fun anUnknownTensorPrefixInAVersionedCheckpointIsRefusedByName() {
        val bytes = SafetensorsWriter.encode(
            listOf(SafetensorsTensor.of("weights.0.w", Tensors.f32Vector<Sym>(floatArrayOf(1f)))),
            mapOf(ModelCheckpoint.VERSION_KEY to "1"),
        )
        val e = assertFailsWith<JsonException> { ModelCheckpoint.decode(bytes) }
        assertTrue("weights.0.w" in e.message!! && "none of the prefixes" in e.message!!, e.message!!)
    }

    @Test
    fun anUnknownFormatVersionIsRefusedByName() {
        val bytes = SafetensorsWriter.encode(
            listOf(SafetensorsTensor.of("param.w", Tensors.f32Vector<Sym>(floatArrayOf(1f)))),
            mapOf(ModelCheckpoint.VERSION_KEY to "2"),
        )
        val e = assertFailsWith<JsonException> { ModelCheckpoint.decode(bytes) }
        assertTrue("version '2'" in e.message!! && "No migration exists" in e.message!!, e.message!!)
    }

    @Test
    fun optimizerTensorsWithNoKindTagAreRefusedByName() {
        val bytes = SafetensorsWriter.encode(
            listOf(
                SafetensorsTensor.of("param.w", Tensors.f32Vector<Sym>(floatArrayOf(1f))),
                SafetensorsTensor.of("opt.m.w", Tensors.f32Vector<Sym>(floatArrayOf(1f))),
            ),
            mapOf(ModelCheckpoint.VERSION_KEY to "1"),
        )
        val e = assertFailsWith<JsonException> { ModelCheckpoint.decode(bytes) }
        assertTrue(ModelCheckpoint.OPTIMIZER_KIND_KEY in e.message!!, e.message!!)
    }

    @Test
    fun aNonF32TensorInACheckpointIsRefusedByName() {
        val bytes = SafetensorsWriter.encode(
            listOf(
                SafetensorsTensor(
                    "param.ids",
                    I32,
                    intArrayOf(2),
                    io.tlaloc.core.HostI32Storage(intArrayOf(1, 2)),
                ),
            ),
            mapOf(ModelCheckpoint.VERSION_KEY to "1"),
        )
        val e = assertFailsWith<JsonException> { ModelCheckpoint.decode(bytes) }
        assertTrue("param.ids" in e.message!! && "i32" in e.message!!, e.message!!)
        assertTrue("master weights" in e.message!!, e.message!!)
    }
}
