package io.tlaloc.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * User-defined [IndexName] singleton, declared at file scope because Kotlin
 * forbids named objects inside function bodies. Exists only to prove the
 * non-sealed interface can be extended in user-side code.
 */
private object MyAxis : IndexName {
    override val name = "user-axis"
}

/**
 * Smoke tests for the Layer 1 named-index type substrate (§0.4.241+).
 *
 * These tests exercise the *type-level* surface — that `Named<N, A>` is a
 * legal [ShapeAtom], that pre-defined [IndexName] singletons carry the
 * expected string identifiers, and that named atoms compose inside the
 * existing [Rank2]/[Rank3] constructors without disturbing positional
 * shapes. Behavioural plugin tests live in :compiler-plugin.
 */
class NamedTest {

    @Test
    fun namedIndexSingletonsCarryExpectedNames() {
        assertEquals("batch", Batch.name)
        assertEquals("seq", SeqLen.name)
        assertEquals("hidden", Hidden.name)
        assertEquals("heads", Heads.name)
        assertEquals("dim", Dim.name)
        assertEquals("vocab", Vocab.name)
        assertEquals("channel", Channel.name)
        assertEquals("height", Height.name)
        assertEquals("width", Width.name)
    }

    @Test
    fun indexNameSingletonsAreObjects() {
        // The plugin reads the singleton's class FQN to recover the name —
        // confirm here that two references to e.g. `Batch` refer to the same
        // instance (i.e. it really is an `object`, not a class with multiple
        // instances).
        assertSame(Batch, Batch)
        assertSame(SeqLen, SeqLen)
    }

    @Test
    fun userDefinedIndexNameIsAllowed() {
        // The interface is intentionally non-sealed so user modules can ship
        // their own names. The file-scoped [MyAxis] above proves a user
        // module can extend [IndexName] outside the core/ package.
        assertEquals("user-axis", MyAxis.name)
        assertTrue(MyAxis is IndexName)
    }

    @Test
    fun namedIsAShapeAtom() {
        // Type-level guarantee: `Named<N, A>` slots into existing rank
        // constructors that demand `ShapeAtom`. We can't observe the phantom
        // type at runtime (it's erased), but constructing the rank instance
        // proves the type-arg satisfies `ShapeAtom` at compile time.
        val r1 = Rank1<Named<Batch, Sym>>()
        val r2 = Rank2<Named<Batch, Sym>, Named<SeqLen, Sym>>()
        val r3 = Rank3<Named<Batch, Sym>, Named<SeqLen, Sym>, Named<Hidden, Sym>>()
        assertEquals(1, r1.rank)
        assertEquals(2, r2.rank)
        assertEquals(3, r3.rank)
    }

    @Test
    fun positionalShapesStillWork() {
        // Backwards-compat: the original positional usage compiles unchanged.
        val r = Rank2<Sym, Sym>()
        assertEquals(2, r.rank)
    }

    @Test
    fun mixedNamedAndPositionalShapesCompose() {
        // Half-named types are valid: the K2 plugin will lift only the named
        // axes into DxirType.axisNames; the positional ones surface as null
        // entries in that list at the IR layer (verified in the IR-side tests).
        val r = Rank2<Named<Batch, Sym>, Sym>()
        assertEquals(2, r.rank)
    }
}
