import io.tlaloc.autograd.BucketLadders
import io.tlaloc.autograd.boundedProgram
import io.tlaloc.autograd.broadcastAlong
import io.tlaloc.autograd.div
import io.tlaloc.autograd.specOf
import io.tlaloc.autograd.sum
import io.tlaloc.autograd.times
import io.tlaloc.core.Bounded
import io.tlaloc.core.DTensor
import io.tlaloc.core.DimBound
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Rank1
import io.tlaloc.core.Rank2
import io.tlaloc.core.Shape
import io.tlaloc.core.Sym
import io.tlaloc.core.hostF32
import kotlin.math.abs

object MaxSeq : DimBound(512)

fun main() {
    val x = DTensor<Shape, F32>(HostF32Storage(FloatArray(15) { it.toFloat() }), intArrayOf(5, 3), F32)
    // snippet
    // column means of [[0,1,2],[3,4,5],…,[12,13,14]]: 6, 7, 8
    val got = y.hostF32().toList()
    check(got.zip(listOf(6f, 7f, 8f)).all { (a, b) -> abs(a - b) < 1e-5f }) { "mean $got" }
    check(ladders.bucketFor(MaxSeq, 5) == 16) { "bucket ${ladders.bucketFor(MaxSeq, 5)}" }
    check(meanOverTokens.checkPadding(ladders).passed)
}
