package io.tlaloc.core

interface Differentiable<Self : Differentiable<Self>> {
    fun zeroTangent(): Self
    operator fun plus(other: Self): Self
    operator fun times(scalar: Float): Self
}
