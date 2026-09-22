/**
 * Named axes — a shape bug is a COMPILE error, not a 3am stack trace.
 *
 * In NumPy, PyTorch and JAX a tensor's axes are numbered. `a @ b` is legal
 * whenever `a.shape[-1] == b.shape[-2]`, so a transposed weight matrix, a
 * batch axis confused with a sequence axis, or a head axis that slipped one
 * position all type-check fine and fail (or, worse, silently succeed) at
 * runtime.
 *
 * In Tlaloc an axis carries a NAME in its Kotlin type:
 *
 *     DTensor<Rank2<Named<Batch, Sym>, Named<SeqLen, Sym>>, F32>
 *                   ^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^
 *                   axis 0 is "batch"  axis 1 is "seq"
 *
 * `Sym` is the symbolic *extent* (the length is only known at runtime);
 * `Batch` / `SeqLen` are the *names*, ordinary Kotlin objects implementing
 * `IndexName`. `contract` is declared so that the contracted axes must share
 * the same name type — which makes a mismatch an overload-resolution failure
 * in Kotlin's own type checker. No compiler plugin, no custom diagnostic, no
 * runtime check.
 *
 * This example does two contractions:
 *
 *   1. Rank-2: activations (Batch × SeqLen) · weights (SeqLen × Hidden).
 *   2. Rank-4: the QKᵀ core of attention — TWO batching axes (Batch, Heads)
 *      and one contracting axis (Dim), which is the exact shape a transformer
 *      layer has. No reshape-to-2D trick.
 *
 * Then it shows the program that does NOT compile — a real file in a real
 * source set, and one command that makes the compiler reject it in front of
 * you: `./gradlew -p examples/named-indices shapeError`.
 */
import java.io.File
import io.tlaloc.core.Batch
import io.tlaloc.core.DTensor
import io.tlaloc.core.Dim
import io.tlaloc.core.F32
import io.tlaloc.core.Heads
import io.tlaloc.core.Hidden
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.IndexName
import io.tlaloc.core.Named
import io.tlaloc.core.Rank2
import io.tlaloc.core.Rank4
import io.tlaloc.core.SeqLen
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.ops.contract

// ---------------------------------------------------------------------------
// Part 1 — rank-2: one contracted axis, two surviving axes.
// ---------------------------------------------------------------------------

private fun rank2Contraction() {
    // Activations: 2 (batch) × 3 (sequence positions).
    val activations: DTensor<Rank2<Named<Batch, Sym>, Named<SeqLen, Sym>>, F32> =
        Tensors.f32Matrix<Named<Batch, Sym>, Named<SeqLen, Sym>>(
            rows = 2,
            cols = 3,
            data = floatArrayOf(
                1f, 2f, 3f,
                4f, 5f, 6f,
            ),
        )

    // Weights: 3 (sequence positions) × 4 (hidden). Axis 0 is named `SeqLen`
    // — the SAME name as the activations' axis 1. That shared name is the
    // contraction signal Kotlin resolves on.
    val weights: DTensor<Rank2<Named<SeqLen, Sym>, Named<Hidden, Sym>>, F32> =
        Tensors.f32Matrix<Named<SeqLen, Sym>, Named<Hidden, Sym>>(
            rows = 3,
            cols = 4,
            data = floatArrayOf(
                0.1f, 0.2f, 0.3f, 0.4f,
                0.5f, 0.6f, 0.7f, 0.8f,
                0.9f, 1.0f, 1.1f, 1.2f,
            ),
        )

    // The result type below is written out only for the reader — Kotlin infers
    // it. The surviving axes are Batch (from the left) and Hidden (from the
    // right); SeqLen is gone, because it was contracted away.
    val output: DTensor<Rank2<Named<Batch, Sym>, Named<Hidden, Sym>>, F32> =
        activations contract weights

    println("[1] rank-2 contraction   (Batch x SeqLen) . (SeqLen x Hidden) -> (Batch x Hidden)")
    println("    dims: ${output.dims.toList()}  (expected [2, 4])")
    val flat = output.hostF32()
    for (b in 0 until output.dims[0]) {
        val row = (0 until output.dims[1]).joinToString("  ") { h ->
            "%.3f".format(flat[b * output.dims[1] + h])
        }
        println("    [$b] $row")
    }
}

// ---------------------------------------------------------------------------
// Part 2 — rank-4: the attention-score core, batching axes and all.
// ---------------------------------------------------------------------------

/**
 * The *key* sequence axis. Queries and keys both have a sequence axis, but
 * they are DIFFERENT axes (query position vs key position), so they get
 * different names — otherwise Tlaloc could not tell which of the two should
 * be contracted. Defining your own axis name is three lines of Kotlin.
 */
object KeySeqLen : IndexName {
    override val name = "key_seq"
}

/**
 * `QKᵀ`, typed:
 *
 *     Q  : (Batch, Heads, SeqLen, Dim)
 *     Kᵀ : (Batch, Heads, Dim,    KeySeqLen)
 *     ->   (Batch, Heads, SeqLen, KeySeqLen)
 *
 * Batch and Heads appear on both sides in the same positions, so they batch.
 * Dim appears as the last axis on the left and the second-to-last on the
 * right, so it contracts. When this lowers to StableHLO it becomes a
 * `stablehlo.dot_general` with `lhs_batching_dims = [0, 1]` and
 * `lhs_contracting_dims = [3]` — dimension numbers derived from the names,
 * not hand-written.
 */
fun attentionScores(
    queries: DTensor<Rank4<Named<Batch, Sym>, Named<Heads, Sym>, Named<SeqLen, Sym>, Named<Dim, Sym>>, F32>,
    keysTransposed: DTensor<Rank4<Named<Batch, Sym>, Named<Heads, Sym>, Named<Dim, Sym>, Named<KeySeqLen, Sym>>, F32>,
): DTensor<Rank4<Named<Batch, Sym>, Named<Heads, Sym>, Named<SeqLen, Sym>, Named<KeySeqLen, Sym>>, F32> =
    queries contract keysTransposed

private fun rank4Attention() {
    val nb = 1
    val nh = 2
    val tq = 4
    val tk = 4
    val d = 8

    val queries: DTensor<Rank4<Named<Batch, Sym>, Named<Heads, Sym>, Named<SeqLen, Sym>, Named<Dim, Sym>>, F32> =
        DTensor(
            HostF32Storage(FloatArray(nb * nh * tq * d) { it.toFloat() / 64f }),
            intArrayOf(nb, nh, tq, d),
            F32,
        )
    val keysT: DTensor<Rank4<Named<Batch, Sym>, Named<Heads, Sym>, Named<Dim, Sym>, Named<KeySeqLen, Sym>>, F32> =
        DTensor(
            HostF32Storage(FloatArray(nb * nh * d * tk) { (it + 1).toFloat() / 64f }),
            intArrayOf(nb, nh, d, tk),
            F32,
        )

    val scores = attentionScores(queries, keysT)
    println()
    println("[2] rank-4 attention core  Q(B,H,Tq,D) . Kt(B,H,D,Tk) -> S(B,H,Tq,Tk)")
    println("    dims: ${scores.dims.toList()}  (expected [$nb, $nh, $tq, $tk])")
    val flat = scores.hostF32()
    for (head in 0 until nh) {
        println("    --- head $head ---")
        for (q in 0 until tq) {
            val row = (0 until tk).joinToString("  ") { k ->
                "%.4f".format(flat[head * tq * tk + q * tk + k])
            }
            println("    q=$q  $row")
        }
    }
}

fun main() {
    rank2Contraction()
    rank4Attention()

    println()
    println("[3] the program that does NOT compile")
    println()
    val source = System.getProperty("tlaloc.example.shapeErrorSource")?.let(::File)
    if (source != null && source.isFile) {
        source.readLines()
            .dropWhile { !it.startsWith("fun shapeErrorThatMustNotCompile") }
            .forEach { println("    $it") }
    } else {
        println("    (source not found — run via `./gradlew -p examples/named-indices run`)")
    }
    println()
    println("    `a`'s axis 1 is named SeqLen; `b`'s axis 0 is named Vocab. No overload")
    println("    of `contract` matches operands sharing no axis name, so the call does")
    println("    not resolve. Watch it:")
    println()
    println("        ./gradlew -p examples/named-indices shapeError")
    println()
    println("    e: ShapeError.kt:35:24 Argument type mismatch: actual type is")
    println("       'DTensor<Rank2<Named<Vocab, Sym>, Named<Hidden, Sym>>, F32>', but")
    println("       'DTensor<Rank2<Named<SeqLen, Sym>, ...>>, F32>' was expected.")
    println()
    println("    That is KOTLIN'S OWN type checker — this project does not even put the")
    println("    Tlaloc compiler plugin on its classpath. The rule lives in ordinary")
    println("    generics, so your IDE draws the squiggle with no Tlaloc tooling at all.")
    println()
    println("named-indices OK")
}
