package io.tlaloc.autograd

import io.tlaloc.core.DType
import io.tlaloc.core.DimBound
import io.tlaloc.core.ExperimentalTlalocApi
import io.tlaloc.core.Shape
import kotlin.reflect.KClass
import kotlin.reflect.KType
import kotlin.reflect.typeOf

/**
 * A [TensorSpec] read from a tensor type: every `Bounded<B>` axis (positional, or inside
 * `Named`) becomes a bounded axis of `B`'s object, and [fixedSizes] give the other axes'
 * sizes, in axis order.
 *
 *     specOf<Rank2<Named<SeqLen, Bounded<MaxSeq>>, Named<Hidden, Sym>>>(F32, 16)
 *
 * The compiler plugin checks the count of [fixedSizes] at the call (`BOUNDED_SPEC_ARITY`).
 */
@ExperimentalTlalocApi
inline fun <reified S : Shape> specOf(dtype: DType, vararg fixedSizes: Int): TensorSpec =
    specFromType(typeOf<S>(), dtype, fixedSizes)

@PublishedApi
@ExperimentalTlalocApi
internal fun specFromType(type: KType, dtype: DType, fixedSizes: IntArray): TensorSpec {
    val shape = (type.classifier as? KClass<*>)?.qualifiedName
    val rank = when (shape) {
        "io.tlaloc.core.ScalarShape" -> 0
        "io.tlaloc.core.Rank1" -> 1
        "io.tlaloc.core.Rank2" -> 2
        "io.tlaloc.core.Rank3" -> 3
        "io.tlaloc.core.Rank4" -> 4
        "io.tlaloc.core.Rank5" -> 5
        "io.tlaloc.core.Rank6" -> 6
        else -> throw IllegalArgumentException("specOf: $type is not a RankN or ScalarShape type")
    }
    val bounds: List<DimBound?> = (0 until rank).map { boundOf(type.arguments[it].type) }
    val fixedCount = bounds.count { it == null }
    require(fixedSizes.size == fixedCount) {
        "specOf: $type has $rank axes, ${rank - fixedCount} of them bounded, so it takes $fixedCount fixed " +
            "size(s); ${fixedSizes.size} given"
    }
    var next = 0
    return TensorSpec(dtype, bounds.map { b -> if (b != null) AxisSpec.Bounded(b) else AxisSpec.Fixed(fixedSizes[next++]) })
}

@ExperimentalTlalocApi
private fun boundOf(atom: KType?): DimBound? {
    val cls = atom?.classifier as? KClass<*> ?: return null
    return when (cls.qualifiedName) {
        "io.tlaloc.core.Bounded" -> {
            val b = atom.arguments.single().type?.classifier as? KClass<*>
                ?: throw IllegalArgumentException("specOf: Bounded's argument in $atom is not a class")
            val instance = runCatching { b.java.getField("INSTANCE").get(null) }.getOrNull()
            instance as? DimBound
                ?: throw IllegalArgumentException("specOf: ${b.qualifiedName} in $atom is not a DimBound object")
        }
        "io.tlaloc.core.Named" -> boundOf(atom.arguments.getOrNull(1)?.type)
        else -> null
    }
}
