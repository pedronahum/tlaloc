package brooklyn

import io.tlaloc.core.BF16
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.inference.WeightQuant
import io.tlaloc.runtime.pjrt.serving.TeacherScorer
import io.tlaloc.tokenizer.ChatTemplate
import io.tlaloc.tokenizer.HfTokenizer
import io.tlaloc.tokenizer.load
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths

// The distill stage: the student trained on the teacher's next-token distributions where the two
// tokenizations of the kept teacher episodes agree, and on the true token elsewhere.
//
//   distill score [--limit N]     Ripple-65B (NVFP4, in this process) scores each kept teacher episode,
//                                 rendered in its own template: every position's top-k next tokens
//   distill render                 aligns the scores to TwIL's training sequences and writes distill shards
//   distill train run | merge      LoRA on the base TwIL, as train, with the sparse soft-target loss
//
// TwIL and Ripple give the same ids to the 100,256 tokens they share but split text differently
// (numbers, punctuation before line breaks), so a TwIL position takes the teacher's distribution
// only where its token and the token's byte offset inside an assistant message's text are the
// same in Ripple's rendering. The teacher's candidates are renormalized over the shared ids and
// mixed with the true token ([DISTILL_TRUE_WEIGHT]).

const val DISTILL_DIR = "distill"
const val TEACHER_KEY = "ripple_65b_nvfp4"
const val DISTILL_TOPK = 16
const val DISTILL_TRUE_WEIGHT = 0.5f
const val DISTILL_SCORE_CHUNK = 512
const val DISTILL_MAX_CONTEXT = 16384

/** Ids the two tokenizers share (TwIL's and Ripple's specials start here). */
const val SHARED_VOCAB = 100256

private val TEACHER_MAGIC = "BKTEACH1".toByteArray()
private val DISTILL_MAGIC = "BKDIST01".toByteArray()

fun distillDir(data: Path): Path = data.resolve(DISTILL_DIR)
fun distillShardDir(data: Path): Path = distillDir(data).resolve("shards")
private fun teacherFile(data: Path, id: String, sample: Int): Path = distillDir(data).resolve("teacher").resolve("$id-s$sample.bin")

val DISTILL_TRAIN = TrainSpec("distill", "distill-train", { distillShardDir(it) }, { studentDir() }, TRAIN_EPOCHS, QUICK_TRAIN_EPOCHS, TRAIN_LR, distill = true)

private fun opt(args: List<String>, name: String): String? = args.indexOf(name).takeIf { it >= 0 }?.let { args.getOrNull(it + 1) }

fun distillCmd(args: List<String>): Int {
    val data = opt(args, "--data")?.let { Paths.get(it) } ?: dataHome()
    return when (args.firstOrNull()) {
        "score" -> distillScore(data, args.drop(1))
        "render" -> distillRender(data)
        "train" -> trainCmd(args.drop(1), DISTILL_TRAIN)
        else -> {
            System.err.println("usage: distill score [--limit N] | render | train run|merge|report   (all take --data DIR)")
            2
        }
    }
}

/** A kept teacher episode: its index row and the episode. */
class KeptEpisode(val id: String, val sample: Int, val episode: JsonObject)

/** The episodes the teach shards hold, in shard order (index.jsonl). */
fun keptTeachEpisodes(data: Path): List<KeptEpisode> {
    val index = shardDir(data).resolve("index.jsonl")
    require(Files.exists(index)) { "distill: $index not found; run `teach render` first" }
    val passes = teachEpisodes(data) ?: error("distill: a teach pass is missing")
    val byKey = passes.flatMap { (s, eps) -> eps.map { (s to it.str("id")) to it } }.toMap()
    return Files.readAllLines(index).filter { it.isNotBlank() }.map { l ->
        val o = parseJson(l) as JsonObject
        val id = o.str("id")
        val s = o.optInt("sample") ?: 0
        KeptEpisode(id, s, byKey[s to id] ?: error("distill: no episode $id in teach pass $s"))
    }
}

/** The messages of [ep] as recorded (Ripple's own conversation). */
fun rippleMessages(ep: JsonObject): List<ConvMessage> =
    (ep["messages"] as io.tlaloc.core.io.JsonArray).elements.map { parseConvMessage(it as JsonObject) }

/** Ripple's rendering of [ep] (the whole chat, closed), as its endpoint encodes it. */
fun rippleIds(tok: HfTokenizer, ep: JsonObject, toolsJson: List<String>): IntArray =
    tok.encode(tok.applyChatTemplate(rippleMessages(ep).map { it.forTemplate() }, chatOptions(toolsJson, false), ChatTemplate.RIPPLE), addSpecialTokens = false)

class TeacherScores(val rippleIds: IntArray, val k: Int, val ids: IntArray, val logprobs: FloatArray)

fun writeTeacher(f: Path, s: TeacherScores) {
    Files.createDirectories(f.parent)
    val n = s.rippleIds.size
    val buf = ByteBuffer.allocate(TEACHER_MAGIC.size + 8 + 4 * n + 8 * n * s.k).order(ByteOrder.LITTLE_ENDIAN)
    buf.put(TEACHER_MAGIC).putInt(n).putInt(s.k)
    s.rippleIds.forEach { buf.putInt(it) }
    s.ids.forEach { buf.putInt(it) }
    s.logprobs.forEach { buf.putFloat(it) }
    val tmp = f.resolveSibling(f.fileName.toString() + ".part")
    Files.write(tmp, buf.array())
    Files.move(tmp, f, java.nio.file.StandardCopyOption.REPLACE_EXISTING, java.nio.file.StandardCopyOption.ATOMIC_MOVE)
}

fun readTeacher(f: Path): TeacherScores {
    val buf = ByteBuffer.wrap(Files.readAllBytes(f)).order(ByteOrder.LITTLE_ENDIAN)
    val magic = ByteArray(TEACHER_MAGIC.size).also { buf.get(it) }
    require(magic.contentEquals(TEACHER_MAGIC)) { "$f is not a teacher score file" }
    val n = buf.getInt()
    val k = buf.getInt()
    return TeacherScores(IntArray(n) { buf.getInt() }, k, IntArray(n * k) { buf.getInt() }, FloatArray(n * k) { buf.getFloat() })
}

/** The kernel library of the Triton backend, when the checkout has it built. */
private fun kernelLibrary(home: Path): Path? =
    home.resolve("../../triton/backends/tlaloc/libtlaloc_kernels.so").normalize().takeIf { Files.isRegularFile(it) && System.getenv("BROOKLYN_DISTILL_KERNELS") != "0" }

fun distillScore(data: Path, args: List<String>): Int {
    val home = exampleHome()
    val kept = keptTeachEpisodes(data).let { k -> opt(args, "--limit")?.toInt()?.let { k.take(it) } ?: k }
    val tok = HfTokenizer.load(modelDir(TEACHER_KEY))
    val toolsJson = TOOL_DEFS.map { toJson(it.openAi()) }
    // Longest first: a context the memory cannot hold shows at the start, not after hours.
    val todo = kept.filter { k -> teacherFile(data, k.id, k.sample).let { f -> !Files.exists(f) || readTeacher(f).let { it.k != DISTILL_TOPK || !it.rippleIds.contentEquals(rippleIds(tok, k.episode, toolsJson)) } } }
        .sortedByDescending { rippleIds(tok, it.episode, toolsJson).size }
    println("distill score: ${kept.size} kept teacher episodes, ${kept.size - todo.size} scored before, ${todo.size} to score (top $DISTILL_TOPK)")
    if (todo.isEmpty()) return 0
    val t0 = System.nanoTime()
    // The pools hold the longest episode to score, rounded up to 1,024 tokens (memory scales with it).
    val longest = todo.maxOf { rippleIds(tok, it.episode, toolsJson).size }
    val context = minOf(DISTILL_MAX_CONTEXT, (longest + 1023) / 1024 * 1024)
    val chunk = System.getenv("BROOKLYN_DISTILL_CHUNK")?.toInt() ?: DISTILL_SCORE_CHUNK
    println("distill score: longest episode $longest tokens; context $context, chunks of $chunk")
    MemSampler().use { mem ->
        TeacherScorer.open(
            modelDir(TEACHER_KEY), DISTILL_TOPK, chunk = chunk, maxContext = context,
            kernels = kernelLibrary(home), log = { println("$it (machine memory peak %.1f GiB)".format(mem.peakKib / 1048576.0)) },
        ) { it.copy(weightDType = BF16, weightQuant = WeightQuant.NVFP4, attentionQuant = WeightQuant.NONE) }.use { scorer ->
            println("distill score: Ripple staged in %.0f s, machine memory peak %.1f GiB".format((System.nanoTime() - t0) / 1e9, mem.peakKib / 1048576.0))
            var tokens = 0L
            val ts = System.nanoTime()
            for ((n, k) in todo.withIndex()) {
                val ids = rippleIds(tok, k.episode, toolsJson)
                require(ids.size <= DISTILL_MAX_CONTEXT) { "distill score: ${k.id} has ${ids.size} tokens" }
                val t1 = System.nanoTime()
                val (top, lp) = scorer.score(ids)
                writeTeacher(teacherFile(data, k.id, k.sample), TeacherScores(ids, DISTILL_TOPK, top, lp))
                tokens += ids.size
                // How often the teacher's first choice is the token it wrote: a sanity check of the scores.
                var agree = 0
                for (i in 0 until ids.size - 1) if (top[i * DISTILL_TOPK] == ids[i + 1]) agree++
                if (n % 10 == 0 || n == todo.size - 1) {
                    println("distill score: %d/%d %s s%d: %d tokens in %.1f s; top-1 equals the next token at %.1f%%; %.0f tokens/s overall, memory peak %.1f GiB"
                        .format(n + 1, todo.size, k.id, k.sample, ids.size, (System.nanoTime() - t1) / 1e9, 100.0 * agree / maxOf(1, ids.size - 1),
                            tokens / ((System.nanoTime() - ts) / 1e9), mem.peakKib / 1048576.0))
                }
            }
        }
    }
    println("distill score: %d episodes in %.0f s".format(todo.size, (System.nanoTime() - t0) / 1e9))
    return 0
}

/** Byte offsets of each token's start in the decoded text, and the text's length (byte-level BPE: one character per byte). */
fun tokenOffsets(tok: HfTokenizer, ids: IntArray): IntArray {
    val out = IntArray(ids.size + 1)
    for ((i, id) in ids.withIndex()) out[i + 1] = out[i] + (tok.idToToken(id) ?: error("token $id has no string")).length
    val text = tok.decode(ids).toByteArray(Charsets.UTF_8).size
    require(out[ids.size] == text) { "token offsets end at ${out[ids.size]}, the decoded text has $text bytes" }
    return out
}

private fun indexOf(hay: ByteArray, needle: ByteArray, from: Int): Int {
    if (needle.isEmpty()) return -1
    outer@ for (i in from..hay.size - needle.size) {
        for (j in needle.indices) if (hay[i + j] != needle[j]) continue@outer
        return i
    }
    return -1
}

/** For each TwIL position, the Ripple position holding the same token at the same offset in the same assistant text, or -1. */
class Alignment(val rippleOf: IntArray, val regions: Int, val regionsFound: Int)

fun align(
    twTok: HfTokenizer, twIds: IntArray, twMask: BooleanArray,
    rpTok: HfTokenizer, rpIds: IntArray, messages: List<ConvMessage>,
): Alignment {
    val twText = twTok.decode(twIds).toByteArray(Charsets.UTF_8)
    val rpText = rpTok.decode(rpIds).toByteArray(Charsets.UTF_8)
    val twOff = tokenOffsets(twTok, twIds)
    val rpOff = tokenOffsets(rpTok, rpIds)
    val rpAt = HashMap<Int, Int>()
    for (r in rpIds.indices) rpAt[rpOff[r]] = r
    val out = IntArray(twIds.size) { -1 }
    var tc = 0
    var rc = 0
    var regions = 0
    var found = 0
    for (m in messages) {
        if (m.role != "assistant") continue
        for (piece in listOfNotNull(m.reasoning?.trim(), m.content.trim())) {
            if (piece.isEmpty()) continue
            regions++
            val bytes = piece.toByteArray(Charsets.UTF_8)
            val ts = indexOf(twText, bytes, tc)
            val rs = indexOf(rpText, bytes, rc)
            if (ts < 0 || rs < 0) continue
            found++
            val end = ts + bytes.size
            var j = twOff.binarySearch(ts).let { if (it < 0) -it - 1 else it }
            while (j < twIds.size && twOff[j + 1] <= end) {
                if (twMask[j]) {
                    val r = rpAt[rs + (twOff[j] - ts)]
                    if (r != null && rpIds[r] == twIds[j] && rpOff[r + 1] - rpOff[r] == twOff[j + 1] - twOff[j]) out[j] = r
                }
                j++
            }
            tc = end
            rc = rs + bytes.size
        }
    }
    return Alignment(out, regions, found)
}

/** A distill sequence: ids, mask, and per row J target ids and weights (row i predicts token i + 1). */
class DistillSequence(val ids: IntArray, val mask: BooleanArray, val j: Int, val targetIds: IntArray, val targetWeights: FloatArray) {
    val targetRows: Int get() = (0 until ids.size).count { r -> (0 until j).any { targetWeights[r * j + it] > 0f } }
}

/** The targets of a TwIL sequence: on aligned rows the teacher's shared candidates renormalized, mixed with the true token. */
fun distillTargets(twIds: IntArray, twMask: BooleanArray, al: Alignment, t: TeacherScores): DistillSequence {
    val k = t.k
    val j = k + 1
    val n = twIds.size
    val ids = IntArray(n * j)
    val w = FloatArray(n * j)
    for (row in 0 until n - 1) {
        val next = row + 1
        if (!twMask[next]) continue
        val base = row * j
        ids[base] = twIds[next]
        val r = al.rippleOf[next]
        if (r <= 0) { w[base] = 1f; continue }
        // The teacher's distribution after Ripple position r - 1 (the prefix before the same token).
        var mass = 0.0
        for (c in 0 until k) { val id = t.ids[(r - 1) * k + c]; if (id < SHARED_VOCAB) mass += Math.exp(t.logprobs[(r - 1) * k + c].toDouble()) }
        if (mass <= 0.0) { w[base] = 1f; continue }
        w[base] = DISTILL_TRUE_WEIGHT
        for (c in 0 until k) {
            val id = t.ids[(r - 1) * k + c]
            if (id >= SHARED_VOCAB) continue
            ids[base + 1 + c] = id
            w[base + 1 + c] = ((1 - DISTILL_TRUE_WEIGHT) * Math.exp(t.logprobs[(r - 1) * k + c].toDouble()) / mass).toFloat()
        }
    }
    return DistillSequence(twIds, twMask, j, ids, w)
}

fun writeDistillShards(dir: Path, seqs: List<Pair<KeptEpisode, DistillSequence>>) {
    Files.createDirectories(dir)
    Files.list(dir).use { st -> st.filter { it.fileName.toString().startsWith("shard-") || it.fileName.toString() == "index.jsonl" }.toList() }.forEach { Files.delete(it) }
    val index = StringBuilder()
    for ((k, chunk) in seqs.chunked(SHARD_SEQUENCES).withIndex()) {
        val size = DISTILL_MAGIC.size + 4 + chunk.sumOf { (_, s) -> 8 + 5 * s.ids.size + 8 * s.ids.size * s.j }
        val buf = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN)
        buf.put(DISTILL_MAGIC).putInt(chunk.size)
        for ((row, pair) in chunk.withIndex()) {
            val (e, s) = pair
            buf.putInt(s.ids.size).putInt(s.j)
            s.ids.forEach { buf.putInt(it) }
            s.mask.forEach { buf.put(if (it) 1 else 0) }
            s.targetIds.forEach { buf.putInt(it) }
            s.targetWeights.forEach { buf.putFloat(it) }
            index.append(toJson(linkedMapOf("shard" to k.toLong(), "row" to row.toLong(), "id" to e.id, "sample" to e.sample.toLong(), "tokens" to s.ids.size.toLong(), "target_rows" to s.targetRows.toLong()))).append('\n')
        }
        Files.write(dir.resolve("shard-%05d.bin".format(k)), buf.array())
    }
    Files.writeString(dir.resolve("index.jsonl"), index)
}

fun readDistillShard(f: Path): List<DistillSequence> {
    val buf = ByteBuffer.wrap(Files.readAllBytes(f)).order(ByteOrder.LITTLE_ENDIAN)
    val magic = ByteArray(DISTILL_MAGIC.size).also { buf.get(it) }
    require(magic.contentEquals(DISTILL_MAGIC)) { "$f is not a distill shard" }
    val n = buf.getInt()
    return List(n) {
        val len = buf.getInt()
        val j = buf.getInt()
        val ids = IntArray(len) { buf.getInt() }
        val mask = BooleanArray(len) { buf.get().toInt() != 0 }
        DistillSequence(ids, mask, j, IntArray(len * j) { buf.getInt() }, FloatArray(len * j) { buf.getFloat() })
    }.also { require(!buf.hasRemaining()) { "$f has bytes after its $n sequences" } }
}

fun distillRender(data: Path): Int {
    val kept = keptTeachEpisodes(data)
    val twTok = HfTokenizer.load(modelDir(STUDENT_KEY))
    val rpTok = HfTokenizer.load(modelDir(TEACHER_KEY))
    val toolsJson = TOOL_DEFS.map { toJson(it.openAi()) }
    val shardSeqs = shardFiles(shardDir(data)).filter { it.fileName.toString().endsWith(".bin") }.flatMap { readShard(it) }
    require(shardSeqs.size == kept.size) { "distill render: ${shardSeqs.size} teach sequences, ${kept.size} index rows" }
    val out = ArrayList<Pair<KeptEpisode, DistillSequence>>()
    var masked = 0L
    var aligned = 0L
    var teacherTop1 = 0L
    var regions = 0
    var found = 0
    for ((i, k) in kept.withIndex()) {
        val tf = teacherFile(data, k.id, k.sample)
        if (!Files.exists(tf)) { System.err.println("distill render: no teacher scores for ${k.id} s${k.sample}; run `distill score`"); return 1 }
        val t = readTeacher(tf)
        val rp = rippleIds(rpTok, k.episode, toolsJson)
        require(rp.contentEquals(t.rippleIds)) { "distill render: ${k.id}'s Ripple rendering differs from the one scored; run `distill score` again" }
        val seq = renderForTraining(twTok, ChatTemplate.TWIL, twilMessages(k.episode), toolsJson)
        require(seq.ids.contentEquals(shardSeqs[i].ids) && seq.mask.contentEquals(shardSeqs[i].mask)) { "distill render: ${k.id}'s TwIL sequence differs from teach shard row $i" }
        val al = align(twTok, seq.ids, seq.mask, rpTok, rp, rippleMessages(k.episode))
        regions += al.regions
        found += al.regionsFound
        val ds = distillTargets(seq.ids, seq.mask, al, t)
        for (p in 1 until seq.ids.size) if (seq.mask[p]) {
            masked++
            val r = al.rippleOf[p]
            if (r > 0) { aligned++; if (t.ids[(r - 1) * t.k] == seq.ids[p]) teacherTop1++ }
        }
        out += k to ds
    }
    val dir = distillShardDir(data)
    writeDistillShards(dir, out)
    val summary = linkedMapOf(
        "sequences" to out.size.toLong(), "target_rows" to masked, "aligned_rows" to aligned,
        "aligned_share" to aligned.toDouble() / maxOf(1, masked), "teacher_top1_is_true_token" to teacherTop1.toDouble() / maxOf(1, aligned),
        "text_regions" to regions.toLong(), "text_regions_found" to found.toLong(),
        "top_k" to DISTILL_TOPK.toLong(), "true_weight" to DISTILL_TRUE_WEIGHT.toDouble(), "teacher" to TEACHER_KEY, "student_template" to "TWIL",
        "teach_shards_sha256" to shardsSha256(shardDir(data)), "shards_sha256" to shardsSha256(dir),
    )
    Files.writeString(distillDir(data).resolve("summary.json"), prettyJson(summary) + "\n")
    println("distill render: %d sequences; %d of %d target rows aligned to the teacher (%.1f%%), teacher top-1 is the true token on %.1f%% of them; %d of %d assistant texts found in both renderings"
        .format(out.size, aligned, masked, 100.0 * aligned / maxOf(1, masked), 100.0 * teacherTop1 / maxOf(1, aligned), found, regions))
    println("distill render: shards sha256 ${summary["shards_sha256"]}")
    return 0
}

/**
 * Manifest entries of the distill stage before its training: the settings; the teacher's scores
 * (their number compared; their SHA-256 reported, the device's sums are not bitwise fixed); the
 * distill shards, byte-identical whenever the scores are the recorded ones.
 */
fun distillEntries(data: Path): List<Entry> {
    val out = ArrayList<Entry>()
    out += Entry("distill", "inputs", "distill_settings", false) {
        linkedMapOf<String, Any?>(
            "teacher" to TEACHER_KEY, "teacher_quant" to "NVFP4 weights, bf16 activations, f32 KV", "top_k" to DISTILL_TOPK.toLong(),
            "true_weight" to DISTILL_TRUE_WEIGHT.toDouble(), "chunk" to DISTILL_SCORE_CHUNK.toLong(),
            "alignment" to "same token at the same byte offset in an assistant message's text (reasoning, answer) in both renderings",
            "candidates" to "ids below $SHARED_VOCAB, renormalized", "teacher_template" to "RIPPLE", "student_template" to "TWIL",
        )
    }
    fun teacherFiles(): List<Path> = distillDir(data).resolve("teacher").let { d ->
        if (!Files.isDirectory(d)) emptyList() else Files.list(d).use { st -> st.filter { it.fileName.toString().endsWith(".bin") }.toList() }.sortedBy { it.fileName.toString() }
    }
    fun scoresSha(fs: List<Path>) = sha256Hex(fs.joinToString("") { "${it.fileName}\t${fileSha256(it)}\n" }.toByteArray())
    out += Entry("distill", "derived", "teacher_scores", true, setOf("sha256")) {
        val fs = teacherFiles()
        if (fs.isEmpty()) null else linkedMapOf("sha256" to scoresSha(fs), "episodes" to fs.size.toLong())
    }
    lateinit var shards: Entry
    shards = Entry(
        "distill", "derived", "shards", true, setOf("sha256", "teacher_scores_sha256", "aligned_rows", "teacher_top1_is_true_token"),
        { want, now ->
            if (want.optString("teacher_scores_sha256") == now["teacher_scores_sha256"] && want.optString("sha256") != now["sha256"]) {
                listOf("sha256 ${want.optString("sha256")} -> ${now["sha256"]} from the recorded scores (they must regenerate bit for bit)")
            } else emptyList()
        },
    ) {
        val f = distillDir(data).resolve("summary.json")
        val dir = distillShardDir(data)
        val sha = shardsSha256(dir) ?: return@Entry null
        val o = if (Files.exists(f)) parseJson(Files.readString(f)) as JsonObject else null
        if (o?.optString("shards_sha256") != sha) shards.stale = "the shards are not the ones summary.json records; run `distill render`"
        linkedMapOf(
            "sha256" to sha, "sequences" to o?.optInt("sequences")?.toLong(), "target_rows" to o?.optInt("target_rows")?.toLong(),
            "aligned_rows" to o?.optInt("aligned_rows")?.toLong(), "teacher_top1_is_true_token" to o?.optNumber("teacher_top1_is_true_token"),
            "teach_shards_sha256" to o?.optString("teach_shards_sha256"), "teacher_scores_sha256" to scoresSha(teacherFiles()),
        )
    }
    out += shards
    return out
}
