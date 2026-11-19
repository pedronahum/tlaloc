package brooklyn

import io.tlaloc.core.BF16
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostBf16Storage
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.HostI32Storage
import io.tlaloc.core.I32
import io.tlaloc.core.RandomKey
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.passes.DxirRemat
import io.tlaloc.nn.AdamW
import io.tlaloc.nn.CapturedStep
import io.tlaloc.nn.CausalLM
import io.tlaloc.nn.HfCausalLm
import io.tlaloc.nn.HfLoraAdapter
import io.tlaloc.nn.Lora
import io.tlaloc.nn.LoraConfig
import io.tlaloc.nn.Precision
import io.tlaloc.nn.captureCausalLm
import io.tlaloc.nn.withDeviceAdamW
import io.tlaloc.runtime.pjrt.PjrtSession
import io.tlaloc.runtime.pjrt.PjrtTarget
import io.tlaloc.runtime.pjrt.ffm.CompileOptions
import io.tlaloc.runtime.pjrt.ffm.PjrtBuffer
import io.tlaloc.stablehlo.toStablehlo
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import kotlin.math.PI
import kotlin.math.cos

// LoRA SFT of TwIL-8B on the teach shards, all on the device.
//
//   probe [--buckets 4096,8192,16384] [--steps 3]   one LoRA step per sequence bucket: peak machine memory and step time
//   train run [--epochs N] [--lr X] [--fresh]       SFT over the shards; adapters in PEFT's format per epoch; kept when current
//   train merge [--adapter DIR] [--out DIR] [--fresh]  the adapters folded into the base: a bf16 Hugging Face checkpoint; kept when current
//   train report                                     the loss curve summary
//
// The base weights are frozen and held in bf16 (17.6 GB for the 8.8 billion parameters instead of
// 35 GB in f32); the adapters and their AdamW moments are f32 and stay on the device. The step is
// captured symbolically (the graph without host values: an eager trace of an 8B model at thousands
// of tokens would compute the forward pass on the JVM), MIXED_BF16 (bf16 compute, f32 loss and
// gradients), the LM head's cross-entropy is chunked, and every
// transformer block is recomputed in the backward pass (DxirRemat.SAVE_MARKED). One sequence a step,
// padded to the smallest bucket that holds it; the loss is on the assistant tokens (the shard mask).


const val STUDENT_REPO = "webAI-Official/TwIL-8B"
const val LORA_RANK = 16
const val LORA_ALPHA = 32f
const val TRAIN_LR = 1e-4f
const val TRAIN_EPOCHS = 2
const val QUICK_TRAIN_EPOCHS = 1
const val WARMUP_STEPS = 10
const val LOSS_CHUNK = 1024
/** TwIL's pad token (its eos); padded positions have no target. */
const val TWIL_PAD = 100257
val DEFAULT_BUCKETS = listOf(4096, 8192, 16384)
const val TRAIN_DIR = "train"
/** What a merged checkpoint was made from, written beside its weights. */
const val MERGE_RECORD = "brooklyn-merge.json"

/** SHA-256 of [dir]'s merge record, or "missing". */
fun mergeRecordSha256(dir: Path): String = dir.resolve(MERGE_RECORD).let { if (Files.exists(it)) fileSha256(it) else "missing" }

private fun opt(args: List<String>, name: String): String? = args.indexOf(name).takeIf { it >= 0 }?.let { args.getOrNull(it + 1) }

/** TwIL-8B's snapshot, or BROOKLYN_STUDENT_DIR (a small checkpoint of the same family for a smoke run). */
fun studentDir(): Path = System.getenv("BROOKLYN_STUDENT_DIR")?.let { Paths.get(it) } ?: modelDir(STUDENT_KEY)

/**
 * A training run: where it writes, the shards it reads, the checkpoint its adapters start from, its
 * epochs and peak learning rate. [SFT] trains the base on the teacher's shards; [RFT_TRAIN] trains
 * new adapters on the merged SFT checkpoint with the student's own passing trajectories.
 */
class TrainSpec(
    val stage: String,
    val dir: String,
    val shards: (Path) -> Path,
    val base: (Path) -> Path,
    val epochs: Int,
    val quickEpochs: Int,
    val lr: Float,
)

val SFT = TrainSpec("train", TRAIN_DIR, { shardDir(it) }, { studentDir() }, TRAIN_EPOCHS, QUICK_TRAIN_EPOCHS, TRAIN_LR)
const val RFT_TRAIN_DIR = "rft-train"
const val RFT_TRAIN_LR = 5e-5f
val RFT_TRAIN = TrainSpec("rft", RFT_TRAIN_DIR, { shardDir(it, RFT_SAMPLER) }, { trainDir(it).resolve("merged") }, 1, 1, RFT_TRAIN_LR)

fun trainDir(data: Path, spec: TrainSpec = SFT): Path = data.resolve(spec.dir)

/** f32 to bf16, round to nearest even. */
fun toBf16(x: FloatArray): ShortArray = ShortArray(x.size) { i ->
    val b = java.lang.Float.floatToRawIntBits(x[i])
    if (b and 0x7fffffff > 0x7f800000) 0x7fc0.toShort() else ((b + 0x7fff + ((b ushr 16) and 1)) ushr 16).toShort()
}

/**
 * TwIL-8B with rank-[LORA_RANK] adapters on every linear layer, its frozen weights converted to bf16.
 * A bf16 tensor stands where the layers type an f32 one: the symbolic capture traces it as a bf16
 * param (a MIXED_BF16 capture casts only f32 leaves) and the upload reads its storage; nothing
 * reads it as f32.
 */
class Student(val model: CausalLM, val hf: HfCausalLm)

/** The checkpoint with adapters, in f32; nothing else keeps the loaded f32 model once this returns. */
private fun adaptedF32(dir: Path, seed: Long): Pair<CausalLM, io.tlaloc.ir.inference.HfDecoderConfig> {
    val loaded = HfCausalLm.load(dir)
    return Lora.apply(loaded.model, LoraConfig(LORA_RANK, LORA_ALPHA, LoraConfig.ALL_LINEAR), RandomKey.fromSeed(seed)) to loaded.config
}

@Suppress("UNCHECKED_CAST")
fun loadStudent(dir: Path, seed: Long): Student {
    var (m, config) = adaptedF32(dir, seed)
    val frozen = Lora.frozen.frozen(m).map { it.key }
    // A few tensors at a time, so the f32 copies are released as the bf16 ones are made.
    for (batch in frozen.chunked(8)) {
        val byKey = m.parameters.associate { it.key to it.tensor }
        m = m.withParameters(batch.associateWith { k ->
            val t = byKey.getValue(k)
            DTensor<Shape, BF16>(HostBf16Storage(toBf16(t.hostF32())), t.dims.copyOf(), BF16) as DTensor<*, F32>
        })
    }
    // The f32 copies are garbage now; a full collection lets the heap shrink (-XX:MaxHeapFreeRatio)
    // before the device takes its share of the same memory.
    System.gc()
    return Student(m, HfCausalLm(m, config))
}

/** The training step at one sequence length, ids and targets `[1, len]`. */
fun captureTrainStep(model: CausalLM, len: Int): CapturedStep {
    val ids = DTensor<Shape, I32>(HostI32Storage(IntArray(len)), intArrayOf(1, len), I32)
    return captureCausalLm(model, ids, ids, LOSS_CHUNK, Lora.frozen, "twil_lora_$len", Precision.MIXED_BF16, symbolic = true)
        .withRemat(DxirRemat.SAVE_MARKED)
}

/**
 * Peak machine memory in use (MemTotal - MemAvailable), sampled every 100 ms while it runs. The GPU
 * shares the machine's memory, so when MemAvailable falls below BROOKLYN_MEM_GUARD_GIB (default 6)
 * the process ends at once (exit code 137) rather than let the kernel run out.
 */
class MemSampler : AutoCloseable {
    @Volatile var peakKib = 0L
    @Volatile private var running = true
    val baseKib = inUseKib()
    private val guardKib = ((System.getenv("BROOKLYN_MEM_GUARD_GIB")?.toDoubleOrNull() ?: 6.0) * 1024 * 1024).toLong()
    private val thread = Thread {
        while (running) {
            val v = inUseKib()
            if (v > peakKib) peakKib = v
            val avail = totalKib - v
            if (avail < guardKib) {
                System.err.println("memory guard: MemAvailable %.1f GiB is below %.1f GiB; stopping".format(avail / 1048576.0, guardKib / 1048576.0))
                Runtime.getRuntime().halt(137)
            }
            Thread.sleep(100)
        }
    }.apply { isDaemon = true; start() }

    fun reset() { peakKib = inUseKib() }
    override fun close() { running = false; thread.join() }

    companion object {
        val totalKib: Long by lazy {
            Files.readAllLines(Paths.get("/proc/meminfo")).first { it.startsWith("MemTotal:") }.split(Regex("\\s+"))[1].toLong()
        }

        fun inUseKib(): Long {
            var total = 0L
            var avail = 0L
            for (l in Files.readAllLines(Paths.get("/proc/meminfo"))) {
                val p = l.split(Regex("\\s+"))
                if (p[0] == "MemTotal:") total = p[1].toLong()
                if (p[0] == "MemAvailable:") avail = p[1].toLong()
            }
            return total - avail
        }
    }
}

private fun kibToGib(kib: Long) = kib / 1024.0 / 1024.0
private fun secondsSince(t0: Long) = (System.nanoTime() - t0) / 1e9

/** The device program of [step] with AdamW in it, and its state layout. */
class DeviceProgram(val len: Int, val step: CapturedStep, val optimizer: AdamW) {
    val device = step.withDeviceAdamW(optimizer)
    val stablehlo: String = device.function.toStablehlo(outputAliases = device.outputAliases.filterKeys { it >= device.stateOffset })
}

/** The session the probe and the training use: the GEMM fixes, and the memory the environment allows. */
fun trainSession(): PjrtSession =
    PjrtSession(target = PjrtTarget.Cuda, compileOptions = CompileOptions.DEFAULT.withXlaGpuFixes(""))

/** The frozen weights of [student] on the device, in [step]'s frozen order. */
fun uploadFrozen(gpu: PjrtSession, student: Student, step: CapturedStep): List<PjrtBuffer> {
    val byKey = student.model.parameters.associate { it.key to it.tensor }
    return step.frozenKeys.map { k ->
        val t = byKey.getValue(k)
        when (val s = t.storage) {
            is HostBf16Storage -> gpu.bufferFromHostBf16(s.data, t.dims.toList())
            else -> gpu.bufferFromHostF32(t.hostF32(), t.dims.toList())
        }
    }
}

fun uploadAdapters(gpu: PjrtSession, model: CausalLM, step: CapturedStep): List<PjrtBuffer> {
    val byKey = model.parameters.associate { it.key to it.tensor }
    return step.parameterKeys.map { k -> byKey.getValue(k).let { gpu.bufferFromHostF32(it.hostF32(), it.dims.toList()) } }
}

/** The adapters in [trained] (device, [step]'s order) put back into [model]. */
fun withTrained(model: CausalLM, step: CapturedStep, trained: List<PjrtBuffer>): CausalLM =
    model.withParameters(
        step.parameterKeys.withIndex().associate { (j, key) ->
            val dims = step.primal.params[step.inputCount + j].type.dims.toIntArray()
            key to DTensor<Shape, F32>(HostF32Storage(trained[j].toFloatArray(dims.fold(1) { a, d -> a * d })), dims, F32)
        },
    )

/** Inputs and next-token targets of [s] padded to [len]: target -1 off the assistant tokens. */
fun trainRow(s: ShardSequence, len: Int): Pair<IntArray, IntArray> {
    require(s.ids.size <= len)
    val ids = IntArray(len) { i -> s.ids.getOrElse(i) { TWIL_PAD } }
    val next = IntArray(len) { i -> if (i + 1 < s.ids.size && s.mask[i + 1]) s.ids[i + 1] else -1 }
    return ids to next
}

/** One optimizer step of [p] on one row; returns the loss of the adapters it started from. */
class Trainer(val gpu: PjrtSession, val base: List<PjrtBuffer>, var trained: List<PjrtBuffer>, var state: List<PjrtBuffer>) {
    fun step(p: DeviceProgram, t: Int, lr: Float, ids: IntArray, next: IntArray): Float {
        val idBuf = gpu.bufferFromHostI32(ids, listOf(1, p.len))
        val tBuf = gpu.bufferFromHostI32(next, listOf(1, p.len))
        val scalars = p.device.scalars(t, lr).map { gpu.bufferFromHostF32(floatArrayOf(it), emptyList()) }
        val outs = try {
            gpu.executeStablehlo(p.stablehlo, listOf(idBuf, tBuf) + trained + base + state + scalars)
        } finally {
            idBuf.close(); tBuf.close()
        }
        // The donated moments were consumed by the call.
        (state + scalars).forEach { it.close() }
        state = outs.subList(1 + trained.size, outs.size)
        trained.forEach { it.close() }
        trained = outs.subList(1, 1 + trained.size)
        return outs[0].use { it.toFloatArray(1)[0] }
    }
}

fun learningRate(step: Int, total: Int, peak: Float = TRAIN_LR): Float {
    if (step < WARMUP_STEPS) return peak * (step + 1) / WARMUP_STEPS
    val p = (step - WARMUP_STEPS).toDouble() / maxOf(1, total - WARMUP_STEPS)
    return (peak * (0.1 + 0.9 * 0.5 * (1 + cos(PI * p)))).toFloat()
}

fun probeCmd(args: List<String>): Int {
    val data = opt(args, "--data")?.let { Paths.get(it) } ?: dataHome()
    val buckets = opt(args, "--buckets")?.split(",")?.map { it.trim().toInt() } ?: DEFAULT_BUCKETS
    val steps = opt(args, "--steps")?.toInt() ?: 3
    val out = trainDir(data).also { Files.createDirectories(it) }.resolve("probe.json")
    MemSampler().use { mem ->
        var t0 = System.nanoTime()
        val student = loadStudent(studentDir(), manifestSeed("train"))
        println("probe: TwIL-8B loaded, adapters rank $LORA_RANK on every linear layer, base in bf16, in %.1f s; machine memory %.1f GiB in use (%.1f before)"
            .format(secondsSince(t0), kibToGib(MemSampler.inUseKib()), kibToGib(mem.baseKib)))
        val rows = ArrayList<Map<String, Any?>>()
        trainSession().use { gpu ->
            var base: List<PjrtBuffer>? = null
            for (len in buckets) {
                val row = linkedMapOf<String, Any?>("tokens" to len.toLong())
                try {
                    t0 = System.nanoTime()
                    val p = DeviceProgram(len, captureTrainStep(student.model, len), AdamW(learningRate = TRAIN_LR, weightDecay = 0f))
                    row["capture_s"] = secondsSince(t0)
                    if (base == null) {
                        t0 = System.nanoTime()
                        base = uploadFrozen(gpu, student, p.step)
                        println("probe: %d frozen tensors on the device in %.1f s".format(base.size, secondsSince(t0)))
                    }
                    t0 = System.nanoTime()
                    gpu.prepareStablehlo(p.stablehlo)
                    row["compile_s"] = secondsSince(t0)
                    val trainer = Trainer(gpu, base, uploadAdapters(gpu, student.model, p.step),
                        p.device.initialState().zip(p.device.stateSlots) { z, slot -> gpu.bufferFromHostF32(z, slot.type.dims) })
                    // A real-looking row: ids cycling through the vocabulary, a target on every token.
                    val ids = IntArray(len) { (it * 7919) % student.hf.config.vocabSize }
                    val next = IntArray(len) { ids[(it + 1) % len] }
                    mem.reset()
                    val times = ArrayList<Double>()
                    val losses = ArrayList<Float>()
                    for (s in 0 until steps) {
                        t0 = System.nanoTime()
                        losses += trainer.step(p, s + 1, TRAIN_LR, ids, next)
                        times += secondsSince(t0)
                    }
                    (trainer.trained + trainer.state).forEach { it.close() }
                    row["step_s"] = times.drop(1).ifEmpty { times }.sorted().let { it[it.size / 2] }
                    row["first_step_s"] = times.first()
                    row["peak_gib"] = kibToGib(mem.peakKib)
                    row["losses"] = losses.map { it.toDouble() }
                    row["fits"] = losses.all { it.isFinite() }
                    println("probe: %5d tokens: compile %.0f s, step %.2f s (first %.2f s), machine memory peak %.1f GiB, losses %s"
                        .format(len, row["compile_s"], row["step_s"], row["first_step_s"], row["peak_gib"], losses.joinToString { "%.3f".format(it) }))
                } catch (e: Throwable) {
                    row["fits"] = false
                    row["error"] = (e.message ?: e.toString()).take(400)
                    println("probe: %5d tokens: failed: %s".format(len, row["error"]))
                }
                rows += row
                Files.writeString(out, toJson(linkedMapOf("rank" to LORA_RANK.toLong(), "loss_chunk" to LOSS_CHUNK.toLong(), "buckets" to rows)) + "\n")
            }
            base?.forEach { it.close() }
        }
    }
    println("probe: wrote $out")
    return 0
}

/** The buckets [probe.json] found to run, smallest first. */
fun probedBuckets(data: Path): List<Int> {
    val f = trainDir(data).resolve("probe.json")
    require(Files.exists(f)) { "$f not found: run the probe stage first" }
    val o = parseJson(Files.readString(f)) as JsonObject
    return (o["buckets"] as io.tlaloc.core.io.JsonArray).elements.map { it as JsonObject }
        .filter { (it["fits"] as? io.tlaloc.core.io.JsonBool)?.value == true }
        .map { (it["tokens"] as io.tlaloc.core.io.JsonNumber).raw.toInt() }.sorted()
}

fun trainCmd(args: List<String>, spec: TrainSpec = SFT): Int {
    val data = opt(args, "--data")?.let { Paths.get(it) } ?: dataHome()
    val name = if (spec === SFT) "train" else "${spec.stage} train"
    return when (args.firstOrNull()) {
        "run" -> trainRun(data, args.drop(1), spec)
        "merge" -> trainMerge(data, args.drop(1), spec)
        "report" -> { println(Files.readString(trainDir(data, spec).resolve("summary.json"))); 0 }
        else -> {
            System.err.println("usage: $name run [--epochs N] [--lr X] [--fresh] | merge [--adapter DIR] [--out DIR] [--fresh] | report   (all take --data DIR)")
            2
        }
    }
}

/** The rejection-sampling round: the student's passes (run | pending | render | check | report) and their training (train run | merge | report). */
fun rftCmd(args: List<String>): Int =
    if (args.firstOrNull() == "train") trainCmd(args.drop(1), RFT_TRAIN) else teachCmd(args, RFT_SAMPLER)

fun trainRun(data: Path, args: List<String>, spec: TrainSpec = SFT): Int {
    val shardDir = spec.shards(data)
    val baseDir = spec.base(data)
    val shards = shardFiles(shardDir).filter { it.fileName.toString().endsWith(".bin") }
    if (shards.isEmpty()) { System.err.println("train: no shards in $shardDir; run the ${if (spec === SFT) "teach" else "rft render"} stage first"); return 1 }
    if (spec !== SFT && !Files.exists(baseDir.resolve(MERGE_RECORD))) { System.err.println("train: no merged checkpoint at $baseDir; run `train merge` first"); return 1 }
    val buckets = probedBuckets(data)
    if (buckets.isEmpty()) { System.err.println("train: the probe found no bucket that runs"); return 1 }
    val epochs = opt(args, "--epochs")?.toInt() ?: if (isQuickData(data)) spec.quickEpochs else spec.epochs
    val peakLr = opt(args, "--lr")?.toFloat() ?: spec.lr
    val dir = trainDir(data, spec)
    // Adapters trained on these shards with these settings are kept (eval's results are kept the same way).
    val summaryFile = dir.resolve("summary.json")
    if ("--fresh" !in args && Files.exists(summaryFile)) {
        val o = parseJson(Files.readString(summaryFile)) as JsonObject
        val adapter = o.optString("adapter")?.let { Paths.get(it) }
        val sameBuckets = (o["buckets"]?.toKotlin() as? List<*>)?.map { (it as Number).toInt() } == buckets
        if (o.optString("shards_sha256") == shardsSha256(shardDir) && o.optInt("epochs") == epochs &&
            o.optNumber("lr") == peakLr.toDouble() && sameBuckets && adapter != null && Files.exists(adapter.resolve(HfLoraAdapter.WEIGHTS_FILE)) &&
            o.optString("base_merge_sha256") == baseMergeSha256(baseDir)
        ) {
            println("train: $summaryFile is current (same shards, epochs, lr and buckets); --fresh retrains")
            return 0
        }
    }
    val all = shards.flatMap { readShard(it) }
    val kept = all.filter { it.ids.size <= buckets.last() }
    val byBucket = kept.groupBy { s -> buckets.first { it >= s.ids.size } }
    println("train: %d sequences, %d kept (%d longer than %d tokens left out); buckets %s"
        .format(all.size, kept.size, all.size - kept.size, buckets.last(), byBucket.toSortedMap().entries.joinToString { "${it.key}: ${it.value.size}" }))
    // The order: each epoch a seeded shuffle of the kept sequences.
    val seed = manifestSeed("train_shuffle")
    val order = (0 until epochs).flatMap { e -> kept.indices.shuffled(java.util.Random(seed + e)).map { e to it } }
    val total = order.size
    val log = dir.resolve("loss.jsonl")
    Files.createDirectories(dir)
    Files.deleteIfExists(log)
    val runStart = System.nanoTime()
    MemSampler().use { mem ->
        var t0 = System.nanoTime()
        val student = loadStudent(baseDir, manifestSeed("train"))
        println("train: ${baseDir.fileName} loaded with adapters in %.1f s".format(secondsSince(t0)))
        trainSession().use { gpu ->
            val optimizer = AdamW(learningRate = peakLr, weightDecay = 0f)
            val programs = byBucket.keys.sorted().associateWith { len ->
                t0 = System.nanoTime()
                DeviceProgram(len, captureTrainStep(student.model, len), optimizer).also {
                    gpu.prepareStablehlo(it.stablehlo)
                    println("train: bucket %d traced and compiled in %.0f s".format(len, secondsSince(t0)))
                }
            }
            val first = programs.values.first()
            for (p in programs.values) require(p.step.parameterKeys == first.step.parameterKeys && p.step.frozenKeys == first.step.frozenKeys)
            val base = uploadFrozen(gpu, student, first.step)
            val trainer = Trainer(gpu, base, uploadAdapters(gpu, student.model, first.step),
                first.device.initialState().zip(first.device.stateSlots) { z, slot -> gpu.bufferFromHostF32(z, slot.type.dims) })
            mem.reset()
            var epochLoss = 0.0
            var epochTokens = 0L
            val epochLosses = ArrayList<Double>()
            for ((t, pair) in order.withIndex()) {
                val (epoch, i) = pair
                val s = kept[i]
                val len = buckets.first { it >= s.ids.size }
                val (ids, next) = trainRow(s, len)
                val lr = learningRate(t, total, peakLr)
                t0 = System.nanoTime()
                val loss = trainer.step(programs.getValue(len), t + 1, lr, ids, next)
                val dt = secondsSince(t0)
                val targets = next.count { it >= 0 }
                epochLoss += loss * targets
                epochTokens += targets
                Files.writeString(log, toJson(linkedMapOf("step" to (t + 1).toLong(), "epoch" to epoch.toLong(), "tokens" to s.ids.size.toLong(),
                    "bucket" to len.toLong(), "targets" to targets.toLong(), "lr" to lr.toDouble(), "loss" to loss.toDouble(), "seconds" to dt)) + "\n",
                    java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND)
                if (t % 20 == 0 || t == total - 1) {
                    println("train: step %4d/%d epoch %d  loss %.4f  lr %.2e  %5d tokens  %.2f s  (%.0f min so far, machine memory peak %.1f GiB)"
                        .format(t + 1, total, epoch, loss, lr, s.ids.size, dt, secondsSince(runStart) / 60, kibToGib(mem.peakKib)))
                }
                val lastOfEpoch = t == total - 1 || order[t + 1].first != epoch
                if (lastOfEpoch) {
                    val mean = epochLoss / maxOf(1, epochTokens)
                    epochLosses += mean
                    val adapted = withTrained(student.model, first.step, trainer.trained)
                    val out = HfLoraAdapter.save(HfCausalLm(adapted, student.hf.config), dir.resolve("adapter-epoch$epoch"), STUDENT_REPO)
                    println("train: epoch %d mean loss %.4f (the loss of each step's starting adapters, per target token); adapters %s".format(epoch, mean, out))
                    epochLoss = 0.0
                    epochTokens = 0
                }
                if (!loss.isFinite()) { System.err.println("train: loss is $loss at step ${t + 1}; stopping"); return 1 }
            }
            (trainer.trained + trainer.state + base).forEach { it.close() }
            val summary = linkedMapOf<String, Any?>(
                "model" to STUDENT_REPO, "rank" to LORA_RANK.toLong(), "alpha" to LORA_ALPHA.toDouble(), "targets" to "all linear",
                "precision" to "MIXED_BF16, frozen base in bf16", "optimizer" to "AdamW on the device, weight decay 0",
                "lr" to peakLr.toDouble(), "schedule" to "linear warmup $WARMUP_STEPS steps, cosine to 10%", "epochs" to epochs.toLong(),
                "sequences" to all.size.toLong(), "kept" to kept.size.toLong(), "buckets" to buckets.map { it.toLong() },
                "steps" to total.toLong(), "epoch_loss" to epochLosses, "seconds" to secondsSince(runStart),
                "peak_gib" to kibToGib(mem.peakKib), "shards_sha256" to shardsSha256(shardDir),
                "base_dir" to baseDir.toString(), "base_merge_sha256" to baseMergeSha256(baseDir),
                "adapter" to dir.resolve("adapter-epoch${epochs - 1}").toString(),
            )
            Files.writeString(dir.resolve("summary.json"), toJson(summary) + "\n")
            println("train: %d steps in %.1f min; epoch losses %s; machine memory peak %.1f GiB"
                .format(total, secondsSince(runStart) / 60, epochLosses.joinToString { "%.4f".format(it) }, kibToGib(mem.peakKib)))
        }
    }
    return 0
}

/** The base at [source] with the adapters at [adapter] folded in. */
private fun mergedStudent(source: Path, adapter: Path): HfCausalLm {
    val adapted = HfLoraAdapter.load(HfCausalLm.load(source), adapter)
    return HfCausalLm(Lora.merge(adapted.model), adapted.config)
}

/** Folds the adapters into the base weights and writes a bf16 Hugging Face checkpoint. */
fun trainMerge(data: Path, args: List<String>, spec: TrainSpec = SFT): Int {
    val adapter = opt(args, "--adapter")?.let { Paths.get(it) }
        ?: Paths.get((parseJson(Files.readString(trainDir(data, spec).resolve("summary.json"))) as JsonObject).optString("adapter")!!)
    val out = opt(args, "--out")?.let { Paths.get(it) } ?: trainDir(data, spec).resolve("merged")
    val source = spec.base(data)
    val record = out.resolve(MERGE_RECORD)
    if ("--fresh" !in args && Files.exists(record)) {
        val o = parseJson(Files.readString(record)) as JsonObject
        val weights = adapter.resolve(HfLoraAdapter.WEIGHTS_FILE)
        if (o.optString("base_dir") == source.toString() && o.optString("base_merge_sha256") == baseMergeSha256(source) &&
            Files.exists(weights) && o.optString("adapter_sha256") == fileSha256(weights)
        ) {
            println("train merge: $out is current (merged from these adapters); --fresh merges again")
            return 0
        }
    }
    val t0 = System.nanoTime()
    // The f32 base (35 GB) is unreachable once mergedStudent() returns, so the save holds the merged
    // weights and their bf16 copies (52 GB), not the base as well (the base kept in reach took a
    // 96 GB heap past its limit).
    val merged = mergedStudent(source, adapter)
    System.gc()
    merged.save(source, out, dtype = BF16)
    // The tokenizer and generation files of the base model, so the directory serves on its own.
    Files.list(source).use { s ->
        s.filter { f -> val n = f.fileName.toString(); !n.endsWith(".safetensors") && n != "model.safetensors.index.json" && n != "config.json" && Files.isRegularFile(f) }
            .forEach { f -> if (!Files.exists(out.resolve(f.fileName))) Files.copy(f, out.resolve(f.fileName)) }
    }
    Files.writeString(out.resolve(MERGE_RECORD), toJson(linkedMapOf(
        "base" to STUDENT_REPO, "base_dir" to source.toString(), "adapter" to adapter.toString(),
        "adapter_sha256" to fileSha256(adapter.resolve(HfLoraAdapter.WEIGHTS_FILE)), "dtype" to "bf16",
    ).also { r -> baseMergeSha256(source)?.let { r["base_merge_sha256"] = it } }) + "\n")
    println("train merge: %s + %s -> %s (bf16) in %.0f s".format(source, adapter, out, secondsSince(t0)))
    return 0
}

/** The evaluation runs of the tuned students: serving config, split, run. */
val SFT_EVAL_RUNS: List<Pair<Split, Int>> = listOf(Split.DEV to 1, Split.TEST to 1)
const val SFT_CONFIG = "twil-sft-bf16"
val RFT_EVAL_RUNS: List<Pair<Split, Int>> = listOf(Split.DEV to 1)
const val RFT_CONFIG = "twil-rft-bf16"

/**
 * Manifest entries of the train, rft and eval stages: for SFT and for the RFT round, the training
 * settings, the adapters (the shards and base they were trained on, compared; the losses and the
 * adapter's SHA-256 reported, since training on the GPU is not bitwise reproducible) and the
 * merged checkpoint's record; the rft stage's student passes and shards (as teach's); and each
 * evaluation run of a tuned student with its serving config (as the baseline's runs).
 */
fun trainEntries(home: Path, data: Path): List<Entry> {
    val out = ArrayList<Entry>()
    out += trainSpecEntries(data, SFT)
    out += teachEntries(home, data, RFT_SAMPLER)
    out += trainSpecEntries(data, RFT_TRAIN)
    out += Entry("eval", "inputs", "serving_config_sft", false) { servingConfigIdentity(home, SFT_CONFIG) }
    for ((sp, r) in SFT_EVAL_RUNS) out += evalRunEntry(home, data, "eval", SFT_CONFIG, sp, r, "eval")
    out += Entry("eval", "inputs", "serving_config_rft", false) { servingConfigIdentity(home, RFT_CONFIG) }
    for ((sp, r) in RFT_EVAL_RUNS) out += evalRunEntry(home, data, "eval", RFT_CONFIG, sp, r, "eval")
    return out
}

private fun trainSpecEntries(data: Path, spec: TrainSpec): List<Entry> {
    val out = ArrayList<Entry>()
    val stage = spec.stage
    val rerun = if (spec === SFT) "the train stage" else "the rft stage"
    out += Entry(stage, "inputs", "train_settings", false) {
        linkedMapOf<String, Any?>(
            "model" to STUDENT_REPO, "base" to if (spec === SFT) "the pinned model" else "the merged SFT checkpoint",
            "lora_rank" to LORA_RANK.toLong(), "lora_alpha" to LORA_ALPHA.toDouble(),
            "targets" to "all linear", "lr" to spec.lr.toDouble(), "epochs" to spec.epochs.toLong(),
            "quick_epochs" to spec.quickEpochs.toLong(), "warmup_steps" to WARMUP_STEPS.toLong(),
            "loss_chunk" to LOSS_CHUNK.toLong(), "buckets" to DEFAULT_BUCKETS.joinToString(","),
            "precision" to "MIXED_BF16, frozen base in bf16", "merged_dtype" to "bf16",
        )
    }
    fun weightsOf(adapter: String?): Path? =
        adapter?.let { Paths.get(it) }?.let { (if (it.isAbsolute) it else exampleHome().resolve(it)).resolve(HfLoraAdapter.WEIGHTS_FILE) }
    lateinit var adapters: Entry
    adapters = Entry(
        stage, "derived", "adapters", true,
        setOf("epoch_loss", "adapter_sha256", "seconds", "peak_gib", "produced_by"),
    ) {
        val f = trainDir(data, spec).resolve("summary.json")
        if (!Files.exists(f)) null else {
            val o = parseJson(Files.readString(f)) as JsonObject
            val shards = o.optString("shards_sha256")
            if (shards != shardsSha256(spec.shards(data))) adapters.stale = "trained on other shards; rerun $rerun"
            else if (o.optString("base_merge_sha256") != baseMergeSha256(spec.base(data))) adapters.stale = "trained on another merged checkpoint; rerun $rerun"
            val weights = weightsOf(o.optString("adapter"))
            linkedMapOf<String, Any?>(
                "shards_sha256" to shards, "sequences" to o.optInt("sequences")?.toLong(), "kept" to o.optInt("kept")?.toLong(),
                "steps" to o.optInt("steps")?.toLong(), "epochs" to o.optInt("epochs")?.toLong(),
                "epoch_loss" to (o["epoch_loss"]?.toKotlin() as? List<*>)?.joinToString(",") { "%.4f".format((it as Number).toDouble()) },
                "adapter_sha256" to weights?.takeIf { Files.exists(it) }?.let { fileSha256(it) },
                "seconds" to o.optNumber("seconds")?.toLong(), "peak_gib" to o.optNumber("peak_gib")?.let { Math.round(it.toDouble() * 10) / 10.0 },
            ).filterValues { it != null }
        }
    }
    out += adapters
    lateinit var merged: Entry
    merged = Entry(stage, "derived", "merged", true, setOf("record_sha256", "adapter_sha256")) {
        val rec = trainDir(data, spec).resolve("merged").resolve(MERGE_RECORD)
        if (!Files.exists(rec)) null else {
            val o = parseJson(Files.readString(rec)) as JsonObject
            val weights = weightsOf(o.optString("adapter"))
            if (weights == null || !Files.exists(weights) || fileSha256(weights) != o.optString("adapter_sha256")) {
                merged.stale = "merged from other adapters; rerun $rerun (merge)"
            }
            linkedMapOf<String, Any?>(
                "base" to o.optString("base"), "dtype" to o.optString("dtype"),
                "record_sha256" to fileSha256(rec), "adapter_sha256" to o.optString("adapter_sha256"),
            ).filterValues { it != null }
        }
    }
    out += merged
    return out
}

/**
 * A serving config's CKPT_DIR [value], relative to the example [home]. A path under build/data
 * follows BROOKLYN_DATA, as every output does (serving/server.sh resolves it the same way), so a
 * quick run's evaluation is keyed by the checkpoint it serves.
 */
fun ckptDir(home: Path, value: String): Path =
    if (value.startsWith("build/data/")) home.resolve(dataHome()).resolve(value.removePrefix("build/data/")) else home.resolve(value)

/** The merge record's SHA-256 of a merged checkpoint [base] (null for the pinned base model). */
fun baseMergeSha256(base: Path): String? = base.resolve(MERGE_RECORD).takeIf { Files.exists(it) }?.let { fileSha256(it) }
