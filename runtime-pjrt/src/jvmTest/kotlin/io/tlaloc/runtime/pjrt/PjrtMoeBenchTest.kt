package io.tlaloc.runtime.pjrt

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlin.random.Random
import kotlin.test.Test

/**
 * MOE_EXPERTS at Qwen3.6-35B-A3B's shapes (256 experts, top 8, hidden 2048,
 * expert intermediate 512, bf16 weights) for decode-sized blocks: time per
 * call. Prints XLA_FLAGS so runs with different flags can be compared. Runs
 * with TLALOC_MOE_BENCH=1.
 */
class PjrtMoeBenchTest {

    @Test
    fun moeExpertsDecodeTime() {
        assumeTrue(System.getenv("TLALOC_MOE_BENCH") == "1", "set TLALOC_MOE_BENCH=1")
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val e = 256; val h = 2048; val inter = 512; val k = 8
        val rnd = Random(9)
        TestBackend.session().use { s ->
            val gu = s.bufferFromHostBf16(ShortArray(e * 2 * inter * h) { (0x3c00 + rnd.nextInt(64)).toShort() }, listOf(e, 2 * inter, h))
            val dn = s.bufferFromHostBf16(ShortArray(e * h * inter) { (0x3c00 + rnd.nextInt(64)).toShort() }, listOf(e, h, inter))
            for (rows in listOf(1, 4, 16, 20)) {
                val fn = DxirBuilder.function("moe") {
                    val x = param("x", DxirType(BF16, listOf(rows, h)))
                    val l = param("logits", DxirType(F32, listOf(rows, e)))
                    val g = param("gateUp", DxirType(BF16, listOf(e, 2 * inter, h)))
                    val d = param("down", DxirType(BF16, listOf(e, h, inter)))
                    listOf(op(OpKind.MOE_EXPERTS, listOf(x, l, g, d), DxirType(F32, listOf(rows, h)), mapOf("top_k" to k)))
                }
                val x = s.bufferFromHostBf16(ShortArray(rows * h) { 0x3c00 }, listOf(rows, h))
                val l = s.bufferFromHostF32(FloatArray(rows * e) { rnd.nextFloat() * 4f }, listOf(rows, e))
                repeat(3) { s.executeOn(fn, listOf(x, l, gu, dn)).forEach { it.close() } }
                val iters = 20
                val t0 = System.nanoTime()
                repeat(iters) { s.executeOn(fn, listOf(x, l, gu, dn)).forEach { it.close() } }
                val ms = (System.nanoTime() - t0) / 1e6 / iters
                val bytes = rows * k * 3.0 * inter * h * 2
                println("[moe-bench] rows=$rows: %.3f ms per layer (%.0f GB/s of selected expert weights), XLA_FLAGS='${System.getenv("XLA_FLAGS") ?: ""}'".format(ms, bytes / ms / 1e6))
            }
        }
    }

    /**
     * The expert projections of decode-sized blocks without a loop: each
     * (row, expert) pair's weight gathered and multiplied into its row, and
     * the products reduced, which XLA can fuse into one kernel per projection.
     */
    @Test
    fun gatherReduceDecodeTime() {
        assumeTrue(System.getenv("TLALOC_MOE_BENCH") == "1", "set TLALOC_MOE_BENCH=1")
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val e = 256; val h = 2048; val inter = 512; val k = 8
        val rnd = Random(9)
        TestBackend.session().use { s ->
            val gu = s.bufferFromHostBf16(ShortArray(e * 2 * inter * h) { (0x3c00 + rnd.nextInt(64)).toShort() }, listOf(e, 2 * inter, h))
            for (rows in listOf(1, 4, 16, 20)) {
                val p = rows * k
                val mlir = """
func.func @main(%x: tensor<${p}x${h}xbf16>, %idx: tensor<${p}xi32>, %w: tensor<${e}x${2 * inter}x${h}xbf16>) -> tensor<${p}x${2 * inter}xf32> {
    %g = "stablehlo.gather"(%w, %idx) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, ${2 * inter}, $h>}> : (tensor<${e}x${2 * inter}x${h}xbf16>, tensor<${p}xi32>) -> tensor<${p}x${2 * inter}x${h}xbf16>
    %gf = stablehlo.convert %g : (tensor<${p}x${2 * inter}x${h}xbf16>) -> tensor<${p}x${2 * inter}x${h}xf32>
    %xf = stablehlo.convert %x : (tensor<${p}x${h}xbf16>) -> tensor<${p}x${h}xf32>
    %xb = stablehlo.broadcast_in_dim %xf, dims = [0, 2] : (tensor<${p}x${h}xf32>) -> tensor<${p}x${2 * inter}x${h}xf32>
    %m = stablehlo.multiply %gf, %xb : tensor<${p}x${2 * inter}x${h}xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %y = stablehlo.reduce(%m init: %z) applies stablehlo.add across dimensions = [2] : (tensor<${p}x${2 * inter}x${h}xf32>, tensor<f32>) -> tensor<${p}x${2 * inter}xf32>
    return %y : tensor<${p}x${2 * inter}xf32>
  }"""
                val x = s.bufferFromHostBf16(ShortArray(p * h) { 0x3c00 }, listOf(p, h))
                val idx = s.bufferFromHostI32(IntArray(p) { rnd.nextInt(e) }, listOf(p))
                s.prepareStablehlo(mlir)
                repeat(3) { s.executeStablehlo(mlir, listOf(x, idx, gu)).forEach { it.close() } }
                val iters = 50
                val t0 = System.nanoTime()
                repeat(iters) { s.executeStablehlo(mlir, listOf(x, idx, gu)).forEach { it.close() } }
                val ms = (System.nanoTime() - t0) / 1e6 / iters
                println("[moe-gather] rows=$rows: gate/up of $p pairs in %.3f ms (%.0f GB/s of gathered weights)".format(ms, p * 2.0 * inter * h * 2 / ms / 1e6))
            }
        }
    }

    /**
     * A router's top 8 of 256 experts, the two forms MOE_EXPERTS has had: a stable descending
     * sort of each row, and 8 rounds of a row max, the lowest expert holding it and a mask
     * (the emitter's form). Same indices; time per call at prefill and decode row counts.
     */
    @Test
    fun routingTopKTime() {
        assumeTrue(System.getenv("TLALOC_MOE_BENCH") == "1", "set TLALOC_MOE_BENCH=1")
        assumeTrue(TestBackend.deviceAvailable, TestBackend.noDevice)
        val e = 256; val k = 8
        val rnd = Random(3)
        TestBackend.session().use { s ->
            for (rows in listOf(4, 2048)) {
                val tE = "tensor<${rows}x${e}xf32>"; val tEi = "tensor<${rows}x${e}xi32>"; val tK = "tensor<${rows}x${k}xi32>"
                val sort = """
func.func @main(%p: $tE) -> $tK {
    %iota = stablehlo.iota dim = 1 : $tEi
    %s:2 = "stablehlo.sort"(%p, %iota) <{dimension = 1 : i64, is_stable = true}> ({
    ^bb0(%a: tensor<f32>, %b: tensor<f32>, %c: tensor<i32>, %d: tensor<i32>):
      %gt = stablehlo.compare GT, %a, %b, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %gt : tensor<i1>
    }) : ($tE, $tEi) -> ($tE, $tEi)
    %top = stablehlo.slice %s#1 [0:$rows, 0:$k] : ($tEi) -> $tK
    return %top : $tK
  }"""
                val body = StringBuilder()
                body.appendLine("    %iota = stablehlo.iota dim = 1 : $tEi")
                body.appendLine("    %ninf = stablehlo.constant dense<0xFF800000> : $tE")
                body.appendLine("    %beyond = stablehlo.constant dense<$e> : $tEi")
                body.appendLine("    %fm = stablehlo.constant dense<0xFF800000> : tensor<f32>")
                body.appendLine("    %im = stablehlo.constant dense<$e> : tensor<i32>")
                var cur = "%p"
                val ids = ArrayList<String>()
                for (j in 0 until k) {
                    body.appendLine("    %m$j = stablehlo.reduce($cur init: %fm) applies stablehlo.maximum across dimensions = [1] : ($tE, tensor<f32>) -> tensor<${rows}xf32>")
                    body.appendLine("    %mb$j = stablehlo.broadcast_in_dim %m$j, dims = [0] : (tensor<${rows}xf32>) -> $tE")
                    body.appendLine("    %eq$j = stablehlo.compare EQ, $cur, %mb$j, FLOAT : ($tE, $tE) -> tensor<${rows}x${e}xi1>")
                    body.appendLine("    %c$j = stablehlo.select %eq$j, %iota, %beyond : tensor<${rows}x${e}xi1>, $tEi")
                    body.appendLine("    %i$j = stablehlo.reduce(%c$j init: %im) applies stablehlo.minimum across dimensions = [1] : ($tEi, tensor<i32>) -> tensor<${rows}xi32>")
                    body.appendLine("    %ib$j = stablehlo.broadcast_in_dim %i$j, dims = [0] : (tensor<${rows}xi32>) -> $tEi")
                    body.appendLine("    %h$j = stablehlo.compare EQ, %iota, %ib$j, SIGNED : ($tEi, $tEi) -> tensor<${rows}x${e}xi1>")
                    body.appendLine("    %n$j = stablehlo.select %h$j, %ninf, $cur : tensor<${rows}x${e}xi1>, $tE")
                    body.appendLine("    %r$j = stablehlo.reshape %i$j : (tensor<${rows}xi32>) -> tensor<${rows}x1xi32>")
                    cur = "%n$j"
                    ids += "%r$j"
                }
                body.appendLine("    %top = stablehlo.concatenate ${ids.joinToString(", ")}, dim = 1 : (${ids.joinToString(", ") { "tensor<${rows}x1xi32>" }}) -> $tK")
                val rowmax = "func.func @main(%p: $tE) -> $tK {\n$body    return %top : $tK\n  }"
                // Few distinct values, so rows have ties.
                val p = s.bufferFromHostF32(FloatArray(rows * e) { (rnd.nextInt(40) / 8f) }, listOf(rows, e))
                val results = mutableListOf<IntArray>()
                for ((name, mlir) in listOf("sort" to sort, "rowmax" to rowmax)) {
                    s.prepareStablehlo(mlir)
                    repeat(3) { s.executeStablehlo(mlir, listOf(p)).forEach { it.close() } }
                    val iters = 50
                    val t0 = System.nanoTime()
                    repeat(iters) { s.executeStablehlo(mlir, listOf(p)).forEach { it.close() } }
                    val ms = (System.nanoTime() - t0) / 1e6 / iters
                    val out = s.executeStablehlo(mlir, listOf(p))
                    results += out[0].toIntArray(rows * k)
                    out.forEach { it.close() }
                    println("[moe-route-xla] rows=$rows $name: %.3f ms per call".format(ms))
                }
                kotlin.test.assertContentEquals(results[0], results[1], "the sort and the max-and-mask rounds choose the same experts")
            }
        }
    }
}
