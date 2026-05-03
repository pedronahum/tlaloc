package io.tlaloc.ir.recognizer.kernel

import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Layer 4 §0.4.270 — KernelResolver registry tests.
 *
 * Each test calls [KernelResolverRegistry.clear] in [BeforeTest] for
 * isolation — the registry is process-global mutable state. Tests use
 * a tiny [FakeResolver] inline rather than spinning up an actual IREE
 * or PJRT-XLA backend (those land in Phase 3 of the dual-track plan).
 */
class KernelResolverTest {

    /** A test-only resolver that satisfies a fixed kernel-name set on a fixed target. */
    private class FakeResolver(
        override val backendId: String,
        private val supportedKernels: Set<String>,
        private val supportedTarget: KernelTarget,
        private val implementationPrefix: String = backendId,
    ) : KernelResolver {
        override fun supports(kernelName: String, target: KernelTarget): Boolean =
            kernelName in supportedKernels && target == supportedTarget

        override fun resolve(
            kernelName: String,
            target: KernelTarget,
            attrs: Map<String, Any>,
        ): KernelResolution? {
            if (!supports(kernelName, target)) return null
            return KernelResolution(
                backendId = backendId,
                implementation = "$implementationPrefix.$kernelName",
                notes = mapOf("attrs_seen" to attrs.keys.toList()),
            )
        }
    }

    @BeforeTest
    fun isolate() {
        KernelResolverRegistry.clear()
    }

    @Test
    fun emptyRegistryResolvesNothing() {
        assertEquals(emptyList(), KernelResolverRegistry.registered())
        assertEquals(
            emptyList(),
            KernelResolverRegistry.resolveAll("flash_attn_v3", KernelTarget.NVIDIA_GB10),
        )
        assertNull(KernelResolverRegistry.preferred("flash_attn_v3", KernelTarget.NVIDIA_GB10))
    }

    @Test
    fun registerThenResolveProducesResolution() {
        val r = FakeResolver(
            backendId = "iree",
            supportedKernels = setOf("flash_attn_v3", "rms_norm_v1"),
            supportedTarget = KernelTarget.NVIDIA_GB10,
        )
        KernelResolverRegistry.register(r)

        assertEquals(listOf("iree"), KernelResolverRegistry.registered())
        val res = KernelResolverRegistry.preferred("flash_attn_v3", KernelTarget.NVIDIA_GB10)
        assertEquals("iree", res?.backendId)
        assertEquals("iree.flash_attn_v3", res?.implementation)
    }

    @Test
    fun unsupportedKernelOrTargetReturnsNull() {
        KernelResolverRegistry.register(
            FakeResolver(
                backendId = "iree",
                supportedKernels = setOf("flash_attn_v3"),
                supportedTarget = KernelTarget.NVIDIA_GB10,
            ),
        )
        // Wrong kernel name → null.
        assertNull(KernelResolverRegistry.preferred("swiglu_fused", KernelTarget.NVIDIA_GB10))
        // Right kernel, wrong target → null.
        assertNull(KernelResolverRegistry.preferred("flash_attn_v3", KernelTarget.NVIDIA_H100))
    }

    @Test
    fun multipleResolversAllAppearInResolveAll() {
        KernelResolverRegistry.register(
            FakeResolver("iree", setOf("flash_attn_v3"), KernelTarget.NVIDIA_GB10, "iree.cudnn"),
        )
        KernelResolverRegistry.register(
            FakeResolver("pjrt-xla", setOf("flash_attn_v3"), KernelTarget.NVIDIA_GB10, "xla.custom_call"),
        )

        val all = KernelResolverRegistry.resolveAll("flash_attn_v3", KernelTarget.NVIDIA_GB10)
        assertEquals(2, all.size, "both backends produce a resolution for the same kernel")
        assertEquals(setOf("iree", "pjrt-xla"), all.map { it.backendId }.toSet())
        assertTrue(all.any { it.implementation == "iree.cudnn.flash_attn_v3" })
        assertTrue(all.any { it.implementation == "xla.custom_call.flash_attn_v3" })
    }

    @Test
    fun preferredReturnsFirstRegistered() {
        // Pin the v1 contract: registration order is resolution order.
        // The L4.3 cost-aware picker (post-Phase-3) replaces this with
        // cost-model scoring; this test guards the v1 baseline so the
        // contract change is intentional, not accidental.
        KernelResolverRegistry.register(
            FakeResolver("iree", setOf("rms_norm_v1"), KernelTarget.NVIDIA_GB10, "iree.first"),
        )
        KernelResolverRegistry.register(
            FakeResolver("pjrt-xla", setOf("rms_norm_v1"), KernelTarget.NVIDIA_GB10, "xla.second"),
        )

        val pref = KernelResolverRegistry.preferred("rms_norm_v1", KernelTarget.NVIDIA_GB10)
        assertEquals("iree", pref?.backendId, "first-registered wins under v1's preferred() contract")

        // unregister flips the picture.
        KernelResolverRegistry.unregister("iree")
        val prefAfter = KernelResolverRegistry.preferred("rms_norm_v1", KernelTarget.NVIDIA_GB10)
        assertEquals("pjrt-xla", prefAfter?.backendId)
    }
}
