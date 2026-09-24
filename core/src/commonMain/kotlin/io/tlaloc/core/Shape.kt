package io.tlaloc.core

sealed interface ShapeAtom

class Lit<N> : ShapeAtom

class Sym(val name: String) : ShapeAtom {
    override fun toString(): String = name
}

class Mul<A : ShapeAtom, B : ShapeAtom> : ShapeAtom

class Add<A : ShapeAtom, B : ShapeAtom> : ShapeAtom

/**
 * Type-level marker for a named tensor-axis identifier.
 *
 * Each named index is a singleton object. The K2 plugin reads the singleton's
 * class FQN at FIR-stage type resolution and lifts it into [io.tlaloc.ir.DxirType.axisNames]
 * so a binary tensor op can infer contraction over a shared name.
 *
 * Pattern for user-defined names:
 *
 *     object MyAxis : IndexName { override val name = "my-axis" }
 *
 * Common ML axis names ship pre-defined in [CommonNames]. The interface is left
 * non-sealed so user modules can extend it; the K2 plugin matches on FQN, not
 * on a closed type hierarchy.
 *
 * Named indices deliberately use singletons over annotations because singletons compose
 * naturally inside [Rank2]/[Rank3]/... type-arg products without needing a
 * synthetic FIR type-construction extension.
 */
interface IndexName {
    val name: String
}

/**
 * Type-level wrapper attaching a named-index identifier [N] to a shape atom [A]
 * Use inside a `RankN<...>` slot to give that axis a
 * semantic name:
 *
 *     typealias Activations =
 *         DTensor<Rank2<Named<Batch, Sym>, Named<SeqLen, Sym>>, F32>
 *
 * `Named<N, A>` is itself a [ShapeAtom], so it slots into existing rank
 * constructors without requiring a new family of ranks. Backwards-compatible:
 * existing positional shapes (`Rank2<Sym, Sym>`) continue to compile and run
 * unchanged — named indices are purely additive.
 */
class Named<N : IndexName, A : ShapeAtom> : ShapeAtom

sealed interface Shape {
    val rank: Int
}

data object ScalarShape : Shape {
    override val rank = 0
}

class Rank1<A0 : ShapeAtom> : Shape {
    override val rank = 1
}

class Rank2<A0 : ShapeAtom, A1 : ShapeAtom> : Shape {
    override val rank = 2
}

class Rank3<A0 : ShapeAtom, A1 : ShapeAtom, A2 : ShapeAtom> : Shape {
    override val rank = 3
}

class Rank4<A0 : ShapeAtom, A1 : ShapeAtom, A2 : ShapeAtom, A3 : ShapeAtom> : Shape {
    override val rank = 4
}

class Rank5<A0 : ShapeAtom, A1 : ShapeAtom, A2 : ShapeAtom, A3 : ShapeAtom, A4 : ShapeAtom> : Shape {
    override val rank = 5
}

class Rank6<A0 : ShapeAtom, A1 : ShapeAtom, A2 : ShapeAtom, A3 : ShapeAtom, A4 : ShapeAtom, A5 : ShapeAtom> : Shape {
    override val rank = 6
}

class DynShape(val dims: IntArray) : Shape {
    override val rank: Int get() = dims.size
}
