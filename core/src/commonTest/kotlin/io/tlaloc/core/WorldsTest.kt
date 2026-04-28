package io.tlaloc.core

import kotlin.test.Test
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * Layer 2 §0.4.243+ — smoke tests for the four-worlds taxonomy
 * (KernelScope / OrchestrationScope / ProgramScope / ClusterScope).
 *
 * The *interesting* tests for world-scope discipline are compile-fail
 * tests over user source code — those live in :compiler-plugin, where
 * we have the K2JVMCompiler harness to assert that an illegal cross-
 * world call doesn't compile. This file covers the type-substrate
 * invariants reachable at runtime: the singletons exist, are
 * distinguishable, and inherit the right interfaces.
 */
class WorldsTest {

    @Test
    fun tlalocSingletonImplementsOrchestrationAndProgramScopes() {
        assertTrue(Tlaloc is OrchestrationScope)
        assertTrue(Tlaloc is ProgramScope)
        assertSame(Tlaloc, Tlaloc)
    }

    @Test
    fun scopesAreDistinctInterfaces() {
        // The four scopes are intentionally distinct types — there is no
        // common supertype. A function declared on KernelScope is not
        // callable on an OrchestrationScope receiver, and vice versa.
        // (Compile-fail tests in :compiler-plugin pin this property at
        // the call-site level; here we only assert that the runtime types
        // themselves are distinct interfaces.)
        val kernel: Any = object : KernelScope {}
        val orchestration: Any = object : OrchestrationScope {}
        assertTrue(kernel !is OrchestrationScope)
        assertTrue(orchestration !is KernelScope)
    }

    @Test
    fun userDefinedScopeInstancesCompose() {
        // A user-defined orchestration runtime can subtype OrchestrationScope
        // to layer in concrete dispatch infrastructure. v1 ships only Tlaloc
        // (the default singleton); concrete runtimes attach in Layer 3.
        val custom = object : OrchestrationScope, ProgramScope {}
        assertTrue(custom is OrchestrationScope)
        assertTrue(custom is ProgramScope)
    }
}
