package io.tlaloc.autograd

import kotlin.test.Test
import kotlin.test.assertFailsWith

/**
 * §0.4.446 — the deletion pin. `Backward.kt` (the runtime value-tape reverse
 * walk: `backward()`, `Gradients`, `applyRegistryRule`) was the second AD
 * engine in the tree; audit finding A ordered it deleted, with the
 * Tracer-convenience API reimplemented over the compiler route. This test
 * makes the deletion executable: if anyone reintroduces a file-facade or
 * class under the old names, the pin goes red before a second engine can
 * grow back. JVM-only by nature (`Class.forName` is the loophole-free way
 * to assert a symbol's absence at runtime).
 */
class SingleEngineDeletionTest {

    @Test
    fun `the runtime-tape engine stays deleted`() {
        assertFailsWith<ClassNotFoundException>("Backward.kt's file facade must not exist") {
            Class.forName("io.tlaloc.autograd.BackwardKt")
        }
        assertFailsWith<ClassNotFoundException>("the Gradients accumulator must not exist") {
            Class.forName("io.tlaloc.autograd.Gradients")
        }
    }
}
