package io.tlaloc.core

sealed interface ShapeAtom

class Lit<N> : ShapeAtom

class Sym(val name: String) : ShapeAtom {
    override fun toString(): String = name
}

class Mul<A : ShapeAtom, B : ShapeAtom> : ShapeAtom

class Add<A : ShapeAtom, B : ShapeAtom> : ShapeAtom

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
