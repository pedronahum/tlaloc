package io.tlaloc.autograd

import io.tlaloc.core.Bounded
import io.tlaloc.core.DimBound
import io.tlaloc.core.ExperimentalTlalocApi
import io.tlaloc.core.IndexName
import io.tlaloc.core.ShapeAtom
import io.tlaloc.core.Sym

/**
 * The axis [vmap] adds in front of every batched argument and result: `Named<N, A>`,
 * where [N] is the axis name and [A] is [Sym], or [Bounded] for a batch size with an
 * upper bound. Make one with [batchAxis].
 *
 *     val perExample = vmap(batchAxis(Batch)) { x: DTensor<Rank1<Feat>, F32> -> (x * x).sum() }
 *     // (DTensor<Rank2<Named<Batch, Sym>, Feat>, F32>) -> DTensor<Rank1<Named<Batch, Sym>>, F32>
 */
@ExperimentalTlalocApi
class BatchAxis<N : IndexName, A : ShapeAtom> internal constructor(
    val name: N,
    val bound: DimBound?,
) {
    override fun toString(): String = if (bound == null) "BatchAxis(${name.name})" else "BatchAxis(${name.name}, $bound)"
}

/** A batch axis named [name], of any size. */
@ExperimentalTlalocApi
fun <N : IndexName> batchAxis(name: N): BatchAxis<N, Sym> = BatchAxis(name, null)

/** A batch axis named [name] whose size is at most `bound.max`. */
@ExperimentalTlalocApi
fun <N : IndexName, B : DimBound> batchAxis(name: N, bound: B): BatchAxis<N, Bounded<B>> = BatchAxis(name, bound)

/** How [vmap2] treats one argument: [Batched] along the batch axis, or [Broadcast] to every example. */
@ExperimentalTlalocApi
sealed interface InAxis

/** The argument has the batch axis in front; each example gets its slice. */
@ExperimentalTlalocApi
data object Batched : InAxis

/** The argument is passed unchanged and shared by every example. */
@ExperimentalTlalocApi
data object Broadcast : InAxis
