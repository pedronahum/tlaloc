package io.tlaloc.runtime.pjrt

import io.tlaloc.core.DType
import io.tlaloc.core.F32
import io.tlaloc.core.F8E4M3FN
import io.tlaloc.core.I32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.random.Random
import kotlin.test.Test

/**
 * Decode attention on the GB10, four rows, at Qwen3.8-27B's and
 * Qwen3.6-35B-A3B's head layouts over pools of 8,200 pages. Runs with
 * TLALOC_ATTN_BENCH=1.
 *
 * - [decodeAttention]: the emitted PAGED_ATTENTION over f32 and e4m3fn pools
 *   at context buckets of 2K and 32K, every row 30 tokens short of the
 *   bucket. Prints milliseconds per call and the rate at which the live keys
 *   and values are read.
 * - [decodeAttentionForms]: candidate StableHLO forms over an e4m3fn pool
 *   (dot precisions and algorithms, bf16 windows, a bf16 split of the query
 *   and weights), with their largest difference from `precision = HIGHEST`.
 *   TLALOC_ATTN_FORMS=a,b picks some.
 * - [decodeAttentionParts]: the 27B's 32K shape taken apart: the dots over a
 *   window given as an input, and the gather with and without writing it.
 */
class PjrtDecodeAttentionBenchTest {

    @Test
    fun decodeAttention() {
        assumeTrue(System.getenv("TLALOC_ATTN_BENCH") == "1", "set TLALOC_ATTN_BENCH=1")
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val rows = 4; val bs = 16; val d = 256; val numBlocks = 8200
        val rnd = Random(9)
        TestBackend.session().use { s ->
            for ((model, heads, kvHeads) in listOf(Triple("27B", 24, 4), Triple("35B", 16, 2))) {
                for (pool in listOf<DType>(F32, F8E4M3FN)) {
                    val poolDims = listOf(numBlocks, bs, kvHeads, d)
                    val n = numBlocks * bs * kvHeads * d
                    fun poolBuf() = if (pool == F32) {
                        s.bufferFromHostF32(FloatArray(n) { rnd.nextFloat() - 0.5f }, poolDims)
                    } else {
                        s.bufferFromHostBytes(ByteArray(n) { (rnd.nextInt(0x70) or (rnd.nextInt(2) shl 7)).toByte() }, poolDims, F8E4M3FN)
                    }
                    val k = poolBuf()
                    val v = poolBuf()
                    val q = s.bufferFromHostF32(FloatArray(rows * heads * d) { rnd.nextFloat() - 0.5f }, listOf(rows, heads, d))
                    for (ctx in listOf(2048, 32768)) {
                        val width = ctx / bs
                        val poolT = DxirType(pool, poolDims)
                        val qT = DxirType(F32, listOf(rows, heads, d))
                        val fn = DxirBuilder.function("attn") {
                            val qp = param("q", qT)
                            val kp = param("k", poolT)
                            val vp = param("v", poolT)
                            val t = param("t", DxirType(I32, listOf(rows, width)))
                            val l = param("l", DxirType(I32, listOf(rows)))
                            listOf(op(OpKind.PAGED_ATTENTION, listOf(qp, kp, vp, t, l), qT, mapOf("scale" to 0.0625)))
                        }
                        // Distinct pages per row, spread over the pool.
                        val table = s.bufferFromHostI32(IntArray(rows * width) { 1 + (it * 7) % (numBlocks - 1) }, listOf(rows, width))
                        val lens = s.bufferFromHostI32(IntArray(rows) { ctx - 30 }, listOf(rows))
                        val ins = listOf(q, k, v, table, lens)
                        repeat(3) { s.executeOn(fn, ins).forEach { it.close() } }
                        val iters = 20
                        val t0 = System.nanoTime()
                        repeat(iters) { s.executeOn(fn, ins).forEach { it.close() } }
                        val ms = (System.nanoTime() - t0) / 1e6 / iters
                        val bytes = 2.0 * rows * (ctx - 30) * kvHeads * d * pool.sizeBytes
                        println("[attn-bench] $model ${pool.name} ctx=$ctx: %.3f ms per call, live K/V read at %.0f GB/s".format(ms, bytes / ms / 1e6))
                        table.close(); lens.close()
                    }
                    listOf(k, v, q).forEach { it.close() }
                }
            }
        }
    }

    /** Candidate decode forms over an e4m3fn pool, as StableHLO text (27B layout). */
    private fun candidate(form: String, r: Int, hkv: Int, g: Int, d: Int, ctx: Int, nb: Int): String {
        val bs = 16; val m = ctx / bs; val h = hkv * g
        val pool = "tensor<${nb}x${bs}x${hkv}x${d}xf8E4M3FN>"
        val sb = StringBuilder()
        sb.append("func.func @main(%q: tensor<${r}x${h}x${d}xf32>, %k: $pool, %v: $pool, %t: tensor<${r}x${m}xi32>, %l: tensor<${r}xi32>) -> tensor<${r}x${h}x${d}xf32> {\n")
        fun gather(src: String, name: String, to: String) {
            sb.append("  %${name}g = \"stablehlo.gather\"($src, %t) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, $bs, $hkv, $d>}> : ($pool, tensor<${r}x${m}xi32>) -> tensor<${r}x${m}x${bs}x${hkv}x${d}xf8E4M3FN>\n")
            sb.append("  %${name}r = stablehlo.reshape %${name}g : (tensor<${r}x${m}x${bs}x${hkv}x${d}xf8E4M3FN>) -> tensor<${r}x${ctx}x${hkv}x${d}xf8E4M3FN>\n")
            if (form == "f8barrier") {
                sb.append("  %${name}b = stablehlo.optimization_barrier %${name}r : tensor<${r}x${ctx}x${hkv}x${d}xf8E4M3FN>\n")
                sb.append("  %${name}w = stablehlo.convert %${name}b : (tensor<${r}x${ctx}x${hkv}x${d}xf8E4M3FN>) -> tensor<${r}x${ctx}x${hkv}x${d}x$to>\n")
            } else {
                sb.append("  %${name}w = stablehlo.convert %${name}r : (tensor<${r}x${ctx}x${hkv}x${d}xf8E4M3FN>) -> tensor<${r}x${ctx}x${hkv}x${d}x$to>\n")
            }
        }
        val sc = "tensor<${r}x${ctx}x${hkv}x${g}xf32>"
        val red = "tensor<${r}x${hkv}x${g}xf32>"
        fun softmax(scores: String) {
            // scores laid out [r, ctx, hkv, g]; mask t >= l
            sb.append("  %c = stablehlo.constant dense<0.0625> : tensor<f32>\n")
            sb.append("  %cb = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<f32>) -> $sc\n")
            sb.append("  %sc1 = stablehlo.multiply $scores, %cb : $sc\n")
            sb.append("  %io = stablehlo.iota dim = 1 : tensor<${r}x${ctx}x${hkv}x${g}xi32>\n")
            sb.append("  %lb = stablehlo.broadcast_in_dim %l, dims = [0] : (tensor<${r}xi32>) -> tensor<${r}x${ctx}x${hkv}x${g}xi32>\n")
            sb.append("  %lv = stablehlo.compare LT, %io, %lb, SIGNED : (tensor<${r}x${ctx}x${hkv}x${g}xi32>, tensor<${r}x${ctx}x${hkv}x${g}xi32>) -> tensor<${r}x${ctx}x${hkv}x${g}xi1>\n")
            sb.append("  %ni = stablehlo.constant dense<0xFF800000> : tensor<f32>\n")
            sb.append("  %nb = stablehlo.broadcast_in_dim %ni, dims = [] : (tensor<f32>) -> $sc\n")
            sb.append("  %ms = stablehlo.select %lv, %sc1, %nb : tensor<${r}x${ctx}x${hkv}x${g}xi1>, $sc\n")
            sb.append("  %mx = stablehlo.reduce(%ms init: %ni) applies stablehlo.maximum across dimensions = [1] : ($sc, tensor<f32>) -> $red\n")
            sb.append("  %mxb = stablehlo.broadcast_in_dim %mx, dims = [0, 2, 3] : ($red) -> $sc\n")
            sb.append("  %sh = stablehlo.subtract %ms, %mxb : $sc\n")
            sb.append("  %e = stablehlo.exponential %sh : $sc\n")
            sb.append("  %z = stablehlo.constant dense<0.0> : tensor<f32>\n")
            sb.append("  %sm = stablehlo.reduce(%e init: %z) applies stablehlo.add across dimensions = [1] : ($sc, tensor<f32>) -> $red\n")
            sb.append("  %smb = stablehlo.broadcast_in_dim %sm, dims = [0, 2, 3] : ($red) -> $sc\n")
            sb.append("  %p = stablehlo.divide %e, %smb : $sc\n")
        }
        val qg = "tensor<${r}x${hkv}x${g}x${d}xf32>"
        sb.append("  %qr = stablehlo.reshape %q : (tensor<${r}x${h}x${d}xf32>) -> $qg\n")
        val outG = "tensor<${r}x${hkv}x${g}x${d}xf32>"
        when (form) {
            "mulreduce" -> {
                val big = "tensor<${r}x${ctx}x${hkv}x${g}x${d}xf32>"
                gather("%k", "k", "f32")
                sb.append("  %kb = stablehlo.broadcast_in_dim %kw, dims = [0, 1, 2, 4] : (tensor<${r}x${ctx}x${hkv}x${d}xf32>) -> $big\n")
                sb.append("  %qb = stablehlo.broadcast_in_dim %qr, dims = [0, 2, 3, 4] : ($qg) -> $big\n")
                sb.append("  %pr = stablehlo.multiply %kb, %qb : $big\n")
                sb.append("  %z0 = stablehlo.constant dense<0.0> : tensor<f32>\n")
                sb.append("  %s = stablehlo.reduce(%pr init: %z0) applies stablehlo.add across dimensions = [4] : ($big, tensor<f32>) -> $sc\n")
                softmax("%s")
                gather("%v", "v", "f32")
                sb.append("  %vb = stablehlo.broadcast_in_dim %vw, dims = [0, 1, 2, 4] : (tensor<${r}x${ctx}x${hkv}x${d}xf32>) -> $big\n")
                sb.append("  %pb = stablehlo.broadcast_in_dim %p, dims = [0, 1, 2, 3] : ($sc) -> $big\n")
                sb.append("  %pv = stablehlo.multiply %vb, %pb : $big\n")
                sb.append("  %o = stablehlo.reduce(%pv init: %z0) applies stablehlo.add across dimensions = [1] : ($big, tensor<f32>) -> $outG\n")
            }
            "bf16x3" -> {
                // q and p split into three bf16 pieces stacked along g; the window is exact in bf16.
                val g3 = 3 * g
                fun split3(x: String, dims: String, name: String): String {
                    val f = "tensor<${dims}xf32>"; val b = "tensor<${dims}xbf16>"
                    // Truncate to bf16 by masking the low 16 bits (exact, and not foldable).
                    val i = "tensor<${dims}xi32>"
                    sb.append("  %${name}mask = stablehlo.constant dense<-65536> : $i\n")
                    fun trunc(src: String, dst: String) {
                        sb.append("  %${dst}i = stablehlo.bitcast_convert $src : ($f) -> $i\n")
                        sb.append("  %${dst}a = stablehlo.and %${dst}i, %${name}mask : $i\n")
                        sb.append("  %${dst}f = stablehlo.bitcast_convert %${dst}a : ($i) -> $f\n")
                        sb.append("  %${dst} = stablehlo.convert %${dst}f : ($f) -> $b\n")
                    }
                    trunc(x, "${name}h")
                    sb.append("  %${name}r1 = stablehlo.subtract $x, %${name}hf : $f\n")
                    trunc("%${name}r1", "${name}m")
                    sb.append("  %${name}r2 = stablehlo.subtract %${name}r1, %${name}mf : $f\n")
                    sb.append("  %${name}l = stablehlo.convert %${name}r2 : ($f) -> $b\n")
                    return "%${name}h, %${name}m, %${name}l"
                }
                gather("%k", "k", "bf16")
                val qs = split3("%qr", "${r}x${hkv}x${g}x${d}", "q")
                sb.append("  %q3 = stablehlo.concatenate $qs, dim = 2 : (tensor<${r}x${hkv}x${g}x${d}xbf16>, tensor<${r}x${hkv}x${g}x${d}xbf16>, tensor<${r}x${hkv}x${g}x${d}xbf16>) -> tensor<${r}x${hkv}x${g3}x${d}xbf16>\n")
                sb.append("  %s3 = stablehlo.dot_general %q3, %kw, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [3] : (tensor<${r}x${hkv}x${g3}x${d}xbf16>, tensor<${r}x${ctx}x${hkv}x${d}xbf16>) -> tensor<${r}x${hkv}x${g3}x${ctx}xf32>\n")
                fun third(src: String, i: Int, name: String, last: Int) =
                    sb.append("  %$name = stablehlo.slice $src [0:$r, 0:$hkv, ${i * g}:${(i + 1) * g}, 0:$last] : (tensor<${r}x${hkv}x${g3}x${last}xf32>) -> tensor<${r}x${hkv}x${g}x${last}xf32>\n")
                third("%s3", 0, "sa", ctx); third("%s3", 1, "sb", ctx); third("%s3", 2, "sc", ctx)
                sb.append("  %sab = stablehlo.add %sa, %sb : tensor<${r}x${hkv}x${g}x${ctx}xf32>\n")
                sb.append("  %s0 = stablehlo.add %sab, %sc : tensor<${r}x${hkv}x${g}x${ctx}xf32>\n")
                sb.append("  %s = stablehlo.transpose %s0, dims = [0, 3, 1, 2] : (tensor<${r}x${hkv}x${g}x${ctx}xf32>) -> $sc\n")
                softmax("%s")
                gather("%v", "v", "bf16")
                sb.append("  %pt = stablehlo.transpose %p, dims = [0, 2, 3, 1] : ($sc) -> tensor<${r}x${hkv}x${g}x${ctx}xf32>\n")
                val ps = split3("%pt", "${r}x${hkv}x${g}x${ctx}", "p")
                sb.append("  %p3 = stablehlo.concatenate $ps, dim = 2 : (tensor<${r}x${hkv}x${g}x${ctx}xbf16>, tensor<${r}x${hkv}x${g}x${ctx}xbf16>, tensor<${r}x${hkv}x${g}x${ctx}xbf16>) -> tensor<${r}x${hkv}x${g3}x${ctx}xbf16>\n")
                sb.append("  %o3 = stablehlo.dot_general %p3, %vw, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1] : (tensor<${r}x${hkv}x${g3}x${ctx}xbf16>, tensor<${r}x${ctx}x${hkv}x${d}xbf16>) -> tensor<${r}x${hkv}x${g3}x${d}xf32>\n")
                third("%o3", 0, "oa", d); third("%o3", 1, "ob", d); third("%o3", 2, "oc", d)
                sb.append("  %oab = stablehlo.add %oa, %ob : $outG\n")
                sb.append("  %o = stablehlo.add %oab, %oc : $outG\n")
            }
            "bf16dot", "f32dot", "f32dotHighest", "f32alg", "f8barrier", "tf32x3", "bf16x3alg", "bf16x6alg" -> {
                val wt = if (form == "bf16dot") "bf16" else "f32"
                fun alg(t: String, n: Int) = ", algorithm = <lhs_precision_type = $t, rhs_precision_type = $t, accumulation_type = f32, " +
                    "lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = $n, allow_imprecise_accumulation = false>"
                val prec = when (form) {
                    "f32dotHighest" -> ", precision = [HIGHEST, HIGHEST]"
                    "f32alg", "f8barrier" -> alg("f32", 1)
                    "tf32x3" -> alg("tf32", 3)
                    "bf16x3alg" -> alg("bf16", 3)
                    "bf16x6alg" -> alg("bf16", 6)
                    else -> ""
                }
                gather("%k", "k", wt)
                val qx = if (wt == "f32") "%qr" else "%qb16"
                if (wt == "bf16") sb.append("  %qb16 = stablehlo.convert %qr : ($qg) -> tensor<${r}x${hkv}x${g}x${d}xbf16>\n")
                // scores [r, hkv, g, ctx] -> transpose to [r, ctx, hkv, g]
                sb.append("  %s0 = stablehlo.dot_general $qx, %kw, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [3]$prec : (tensor<${r}x${hkv}x${g}x${d}x$wt>, tensor<${r}x${ctx}x${hkv}x${d}x$wt>) -> tensor<${r}x${hkv}x${g}x${ctx}xf32>\n")
                sb.append("  %s = stablehlo.transpose %s0, dims = [0, 3, 1, 2] : (tensor<${r}x${hkv}x${g}x${ctx}xf32>) -> $sc\n")
                softmax("%s")
                gather("%v", "v", wt)
                sb.append("  %pt = stablehlo.transpose %p, dims = [0, 2, 3, 1] : ($sc) -> tensor<${r}x${hkv}x${g}x${ctx}xf32>\n")
                val px = if (wt == "f32") "%pt" else "%pt16"
                if (wt == "bf16") sb.append("  %pt16 = stablehlo.convert %pt : (tensor<${r}x${hkv}x${g}x${ctx}xf32>) -> tensor<${r}x${hkv}x${g}x${ctx}xbf16>\n")
                sb.append("  %o = stablehlo.dot_general $px, %vw, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1]$prec : (tensor<${r}x${hkv}x${g}x${ctx}x$wt>, tensor<${r}x${ctx}x${hkv}x${d}x$wt>) -> $outG\n")
            }
        }
        sb.append("  %y = stablehlo.reshape %o : ($outG) -> tensor<${r}x${h}x${d}xf32>\n")
        sb.append("  return %y : tensor<${r}x${h}x${d}xf32>\n}\n")
        return sb.toString()
    }

    @Test
    fun decodeAttentionForms() {
        assumeTrue(System.getenv("TLALOC_ATTN_BENCH") == "1", "set TLALOC_ATTN_BENCH=1")
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val rows = 4; val bs = 16; val d = 256; val nb = 8200
        val rnd = Random(11)
        TestBackend.session().use { s ->
            for ((model, heads, kvHeads) in listOf(Triple("27B", 24, 4), Triple("35B", 16, 2))) {
                val poolDims = listOf(nb, bs, kvHeads, d)
                val n = nb * bs * kvHeads * d
                fun poolBuf() = s.bufferFromHostBytes(ByteArray(n) { (rnd.nextInt(0x50) or (rnd.nextInt(2) shl 7)).toByte() }, poolDims, F8E4M3FN)
                val k = poolBuf(); val v = poolBuf()
                val q = s.bufferFromHostF32(FloatArray(rows * heads * d) { rnd.nextFloat() - 0.5f }, listOf(rows, heads, d))
                for (ctx in listOf(2048, 32768)) {
                    val width = ctx / bs
                    val table = s.bufferFromHostI32(IntArray(rows * width) { 1 + (it * 7) % (nb - 1) }, listOf(rows, width))
                    val lens = s.bufferFromHostI32(IntArray(rows) { ctx - 30 }, listOf(rows))
                    val ins = listOf(q, k, v, table, lens)
                    var ref: FloatArray? = null
                    val forms = System.getenv("TLALOC_ATTN_FORMS")?.split(',')
                        ?: listOf("f32dotHighest", "f32alg", "f32dot", "tf32x3", "bf16x3alg", "bf16x6alg", "bf16dot", "bf16x3")
                    for (form in forms) {
                        val mlir = candidate(form, rows, kvHeads, heads / kvHeads, d, ctx, nb)
                        s.prepareStablehlo(mlir)
                        val out = s.executeStablehlo(mlir, ins)
                        val got = out[0].toFloatArray(rows * heads * d); out.forEach { it.close() }
                        val diff = ref?.let { r -> r.indices.maxOf { kotlin.math.abs(r[it] - got[it]) } } ?: 0f
                        if (ref == null) ref = got
                        repeat(3) { s.executeStablehlo(mlir, ins).forEach { it.close() } }
                        val iters = 20
                        val t0 = System.nanoTime()
                        repeat(iters) { s.executeStablehlo(mlir, ins).forEach { it.close() } }
                        val ms = (System.nanoTime() - t0) / 1e6 / iters
                        val bytes = 2.0 * rows * (ctx - 30) * kvHeads * d
                        println("[attn-form] $model ctx=$ctx $form: %.3f ms per call, %.0f GB/s, max diff from f32dotHighest %.2e".format(ms, bytes / ms / 1e6, diff))
                    }
                    table.close(); lens.close()
                }
                listOf(k, v, q).forEach { it.close() }
            }
        }
    }

    /**
     * Where the time of the f32-algorithm form goes at the 27B's 32K shape:
     * the dots and softmax over a window already gathered (an input), the
     * gather and widening alone (summed to one number per row, so nothing
     * large is written), and the whole form.
     */
    @Test
    fun decodeAttentionParts() {
        assumeTrue(System.getenv("TLALOC_ATTN_BENCH") == "1", "set TLALOC_ATTN_BENCH=1")
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val r = 4; val hkv = 4; val g = 6; val d = 256; val ctx = 32768; val bs = 16; val m = ctx / bs; val nb = 8200
        val alg = ", algorithm = <lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, " +
            "lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>"
        val win = "tensor<${r}x${ctx}x${hkv}x${d}xf32>"
        val qg = "tensor<${r}x${hkv}x${g}x${d}xf32>"
        val sc = "tensor<${r}x${hkv}x${g}x${ctx}xf32>"
        val dotsOnly = """
func.func @main(%q: $qg, %k: $win, %v: $win) -> $qg {
  %s = stablehlo.dot_general %q, %k, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [3]$alg : ($qg, $win) -> $sc
  %e = stablehlo.exponential %s : $sc
  %o = stablehlo.dot_general %e, %v, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1]$alg : ($sc, $win) -> $qg
  return %o : $qg
}"""
        val pool = "tensor<${nb}x${bs}x${hkv}x${d}xf8E4M3FN>"
        val gathered = "tensor<${r}x${m}x${bs}x${hkv}x${d}xf8E4M3FN>"
        fun gatherText(widen: String, keep: Boolean) = """
func.func @main(%k: $pool, %t: tensor<${r}x${m}xi32>) -> ${if (keep) win.replace("f32", widen) else "tensor<${r}xf32>"} {
  %g = "stablehlo.gather"(%k, %t) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, $bs, $hkv, $d>}> : ($pool, tensor<${r}x${m}xi32>) -> $gathered
  %w = stablehlo.reshape %g : ($gathered) -> tensor<${r}x${ctx}x${hkv}x${d}xf8E4M3FN>
  %f = stablehlo.convert %w : (tensor<${r}x${ctx}x${hkv}x${d}xf8E4M3FN>) -> ${win.replace("f32", widen)}
""" + (if (keep) "  return %f : ${win.replace("f32", widen)}\n}" else """  %c = stablehlo.convert %f : (${win.replace("f32", widen)}) -> $win
  %z = stablehlo.constant dense<0.0> : tensor<f32>
  %s = stablehlo.reduce(%c init: %z) applies stablehlo.add across dimensions = [1, 2, 3] : ($win, tensor<f32>) -> tensor<${r}xf32>
  return %s : tensor<${r}xf32>
}""")
        val rnd = Random(13)
        TestBackend.session().use { s ->
            val q = s.bufferFromHostF32(FloatArray(r * hkv * g * d) { rnd.nextFloat() * 0.01f }, listOf(r, hkv, g, d))
            val kw = s.bufferFromHostF32(FloatArray(r * ctx * hkv * d) { rnd.nextFloat() - 0.5f }, listOf(r, ctx, hkv, d))
            val pool8 = s.bufferFromHostBytes(ByteArray(nb * bs * hkv * d) { rnd.nextInt(0x50).toByte() }, listOf(nb, bs, hkv, d), F8E4M3FN)
            val table = s.bufferFromHostI32(IntArray(r * m) { 1 + (it * 7) % (nb - 1) }, listOf(r, m))
            fun time(label: String, mlir: String, ins: List<io.tlaloc.runtime.pjrt.ffm.PjrtBuffer>, bytes: Double) {
                s.prepareStablehlo(mlir)
                repeat(3) { s.executeStablehlo(mlir, ins).forEach { it.close() } }
                val iters = 20
                val t0 = System.nanoTime()
                repeat(iters) { s.executeStablehlo(mlir, ins).forEach { it.close() } }
                val ms = (System.nanoTime() - t0) / 1e6 / iters
                println("[attn-part] $label: %.3f ms per call, %.0f GB/s".format(ms, bytes / ms / 1e6))
            }
            time("dots over a given f32 window (K and V)", dotsOnly, listOf(q, kw, kw), 2.0 * r * ctx * hkv * d * 4)
            time("gather + widen to f32, reduced", gatherText("f32", false), listOf(pool8, table), 1.0 * r * ctx * hkv * d)
            time("gather + widen to f32, written", gatherText("f32", true), listOf(pool8, table), 5.0 * r * ctx * hkv * d)
            time("gather + widen to bf16, written", gatherText("bf16", true), listOf(pool8, table), 3.0 * r * ctx * hkv * d)
            time("gather f8, written", gatherText("f8E4M3FN", true).replace("  %f = stablehlo.convert %w : (tensor<${r}x${ctx}x${hkv}x${d}xf8E4M3FN>) -> tensor<${r}x${ctx}x${hkv}x${d}xf8E4M3FN>\n", "").replace("return %f", "return %w"), listOf(pool8, table), 2.0 * r * ctx * hkv * d)
            listOf(q, kw, pool8, table).forEach { it.close() }
        }
    }
}
