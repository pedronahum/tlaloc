package io.tlaloc.runtime.pjrt.ffm

import java.lang.foreign.Arena
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_BYTE
import java.lang.foreign.ValueLayout.JAVA_FLOAT
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.foreign.ValueLayout.JAVA_LONG
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/**
 * §0.4.333 — pins the `PJRT_NamedValue[2]` byte layout
 * [PjrtFfm.marshalCreateOptions] hands to `PJRT_Client_Create`. Runs
 * everywhere (no GPU, no plugin): this is the offset arithmetic that keeps
 * the CUDA plugin's BFCAllocator from preallocating 75% of the GB10's
 * unified memory (the 2026-07-18 reboot incident — see [PjrtClientOptions]).
 */
class PjrtClientOptionsMarshalTest {

    @Test
    fun namedValueLayoutMatchesHeader() {
        // Offsets hand-checked against pjrt_c_api.h:229 (LP64).
        assertEquals(0L, PjrtFfm.OFF_NamedValue_StructSize)
        assertEquals(16L, PjrtFfm.OFF_NamedValue_Name)
        assertEquals(24L, PjrtFfm.OFF_NamedValue_NameSize)
        assertEquals(32L, PjrtFfm.OFF_NamedValue_Type)
        assertEquals(40L, PjrtFfm.OFF_NamedValue_Value)
        assertEquals(48L, PjrtFfm.OFF_NamedValue_ValueSize)
        assertEquals(56L, PjrtFfm.SZ_NamedValue)
    }

    @Test
    fun marshalsMemoryFractionAndPreallocate() {
        Arena.ofConfined().use { arena ->
            val options = PjrtClientOptions(memoryFraction = 0.25f, preallocate = false)
            val array = PjrtFfm.marshalCreateOptions(arena, options)

            // Entry 0: memory_fraction, kFloat.
            assertEquals(PjrtFfm.SZ_NamedValue, array.get(JAVA_LONG, PjrtFfm.OFF_NamedValue_StructSize))
            assertEquals("memory_fraction", readName(array, 0))
            assertEquals(PjrtFfm.PJRT_NAMED_VALUE_TYPE_FLOAT, array.get(JAVA_INT, PjrtFfm.OFF_NamedValue_Type))
            assertEquals(0.25f, array.get(JAVA_FLOAT, PjrtFfm.OFF_NamedValue_Value))
            assertEquals(1L, array.get(JAVA_LONG, PjrtFfm.OFF_NamedValue_ValueSize))

            // Entry 1: preallocate, kBool, false.
            val base = PjrtFfm.SZ_NamedValue
            assertEquals(PjrtFfm.SZ_NamedValue, array.get(JAVA_LONG, base + PjrtFfm.OFF_NamedValue_StructSize))
            assertEquals("preallocate", readName(array, 1))
            assertEquals(PjrtFfm.PJRT_NAMED_VALUE_TYPE_BOOL, array.get(JAVA_INT, base + PjrtFfm.OFF_NamedValue_Type))
            assertEquals(0, array.get(JAVA_BYTE, base + PjrtFfm.OFF_NamedValue_Value))
            assertEquals(1L, array.get(JAVA_LONG, base + PjrtFfm.OFF_NamedValue_ValueSize))
        }
    }

    @Test
    fun marshalsPreallocateTrueAsOneByte() {
        Arena.ofConfined().use { arena ->
            val array = PjrtFfm.marshalCreateOptions(
                arena, PjrtClientOptions(memoryFraction = 1.0f, preallocate = true),
            )
            val base = PjrtFfm.SZ_NamedValue
            assertEquals(1, array.get(JAVA_BYTE, base + PjrtFfm.OFF_NamedValue_Value))
        }
    }

    @Test
    fun defaultsAreUnifiedMemorySafe() {
        // Env-free CI/dev boxes must land on no-preallocate + a ≤0.5 cap;
        // an env override in the test JVM would be a deliberate choice.
        if (System.getenv("TLALOC_PJRT_MEMORY_FRACTION") == null &&
            System.getenv("TLALOC_PJRT_PREALLOCATE") == null
        ) {
            val resolved = PjrtClientOptions.resolve()
            assertEquals(0.5f, resolved.memoryFraction)
            assertEquals(false, resolved.preallocate)
        }
    }

    @Test
    fun rejectsOutOfRangeFraction() {
        assertFailsWith<IllegalArgumentException> { PjrtClientOptions(0f, false) }
        assertFailsWith<IllegalArgumentException> { PjrtClientOptions(1.5f, false) }
    }

    // =====================================================================
    // §0.4.461 (G3a-2) — the distributed create-options. Marshalling is the
    // certified claim (GPU-less, the §0.4.333 way); client CREATION at
    // num_nodes > 1 is refused by name until G4's kv-store callbacks exist.
    // =====================================================================

    @Test
    fun singleNodeDefaultsMarshalExactlyTheTwoLegacyEntries() {
        // The distributed fields must add NOTHING until asked for: the
        // §0.4.333 CUDA lane's bytes are unchanged by this slice.
        val options = PjrtClientOptions(memoryFraction = 0.5f, preallocate = false)
        assertEquals(0, options.nodeId)
        assertEquals(1, options.numNodes)
        assertEquals(null, options.coordinatorAddress)
        assertEquals(2L, options.namedValueCount)
        Arena.ofConfined().use { arena ->
            val array = PjrtFfm.marshalCreateOptions(arena, options)
            assertEquals(PjrtFfm.SZ_NamedValue * 2, array.byteSize())
            assertEquals("memory_fraction", readName(array, 0))
            assertEquals("preallocate", readName(array, 1))
        }
    }

    @Test
    fun multiNodeMarshalsNodeIdAndNumNodesAsInt64Entries() {
        // Names + types verified against the GPU plugin's create path
        // (xla/pjrt/c/pjrt_c_api_gpu_internal.cc, openxla/xla main
        // 2026-09-21): node_id and num_nodes parse as kInt64.
        val options = PjrtClientOptions(
            memoryFraction = 0.5f, preallocate = false,
            nodeId = 3, numNodes = 8, coordinatorAddress = "trainer-node0.tlaloc:8476",
        )
        assertEquals(4L, options.namedValueCount)
        Arena.ofConfined().use { arena ->
            val array = PjrtFfm.marshalCreateOptions(arena, options)
            assertEquals(PjrtFfm.SZ_NamedValue * 4, array.byteSize())

            val nodeIdBase = 2 * PjrtFfm.SZ_NamedValue
            assertEquals(PjrtFfm.SZ_NamedValue, array.get(JAVA_LONG, nodeIdBase + PjrtFfm.OFF_NamedValue_StructSize))
            assertEquals("node_id", readName(array, 2))
            assertEquals(PjrtFfm.PJRT_NAMED_VALUE_TYPE_INT64, array.get(JAVA_INT, nodeIdBase + PjrtFfm.OFF_NamedValue_Type))
            assertEquals(3L, array.get(JAVA_LONG, nodeIdBase + PjrtFfm.OFF_NamedValue_Value))
            assertEquals(1L, array.get(JAVA_LONG, nodeIdBase + PjrtFfm.OFF_NamedValue_ValueSize))

            val numNodesBase = 3 * PjrtFfm.SZ_NamedValue
            assertEquals("num_nodes", readName(array, 3))
            assertEquals(PjrtFfm.PJRT_NAMED_VALUE_TYPE_INT64, array.get(JAVA_INT, numNodesBase + PjrtFfm.OFF_NamedValue_Type))
            assertEquals(8L, array.get(JAVA_LONG, numNodesBase + PjrtFfm.OFF_NamedValue_Value))
        }
    }

    @Test
    fun coordinatorAddressIsNeverMarshalled() {
        // The PJRT C API has no coordinator create-option — the address
        // backs the (unimplemented, G4) kv-store callbacks. It must not
        // leak into the NamedValue array under any name.
        val options = PjrtClientOptions(
            memoryFraction = 0.5f, preallocate = false,
            nodeId = 0, numNodes = 2, coordinatorAddress = "10.0.0.7:8476",
        )
        Arena.ofConfined().use { arena ->
            val array = PjrtFfm.marshalCreateOptions(arena, options)
            val names = (0 until 4).map { readName(array, it) }
            assertEquals(listOf("memory_fraction", "preallocate", "node_id", "num_nodes"), names)
        }
    }

    @Test
    fun rejectsMalformedDistributedOptions() {
        val coord = "host:8476"
        // numNodes must be >= 1.
        assertFailsWith<IllegalArgumentException> {
            PjrtClientOptions(0.5f, false, nodeId = 0, numNodes = 0)
        }
        // nodeId must sit inside [0, numNodes).
        assertFailsWith<IllegalArgumentException> {
            PjrtClientOptions(0.5f, false, nodeId = 2, numNodes = 2, coordinatorAddress = coord)
        }
        assertFailsWith<IllegalArgumentException> {
            PjrtClientOptions(0.5f, false, nodeId = -1, numNodes = 1)
        }
        // A multi-node group without a coordinator is not a group contract.
        assertFailsWith<IllegalArgumentException> {
            PjrtClientOptions(0.5f, false, nodeId = 0, numNodes = 2)
        }
        // Coordinator must be host:port with a real port.
        assertFailsWith<IllegalArgumentException> {
            PjrtClientOptions(0.5f, false, nodeId = 0, numNodes = 2, coordinatorAddress = "no-port")
        }
        assertFailsWith<IllegalArgumentException> {
            PjrtClientOptions(0.5f, false, nodeId = 0, numNodes = 2, coordinatorAddress = ":8476")
        }
        assertFailsWith<IllegalArgumentException> {
            PjrtClientOptions(0.5f, false, nodeId = 0, numNodes = 2, coordinatorAddress = "host:99999")
        }
    }

    @Test
    fun resolveDefaultsToSingleNodeWhenEnvUnset() {
        if (System.getenv("TLALOC_PJRT_NODE_ID") == null &&
            System.getenv("TLALOC_PJRT_NUM_NODES") == null &&
            System.getenv("TLALOC_PJRT_COORDINATOR_ADDRESS") == null
        ) {
            val resolved = PjrtClientOptions.resolve()
            assertEquals(0, resolved.nodeId)
            assertEquals(1, resolved.numNodes)
            assertEquals(null, resolved.coordinatorAddress)
        }
    }

    @Test
    fun multiNodeClientCreationRefusesByNameWithoutKvStore() {
        // The guard is extracted pure (PjrtFfm.requireKvStoreForMultiNode)
        // so THIS certification needs no plugin: num_nodes > 1 with NULL
        // kv callbacks would fail or hang inside the plugin, so create
        // refuses loudly and names the G4 gap.
        PjrtFfm.requireKvStoreForMultiNode(null) // TPU form: fine
        PjrtFfm.requireKvStoreForMultiNode(PjrtClientOptions(0.5f, false)) // single-node: fine
        val ex = assertFailsWith<IllegalArgumentException> {
            PjrtFfm.requireKvStoreForMultiNode(
                PjrtClientOptions(0.5f, false, nodeId = 1, numNodes = 4, coordinatorAddress = "c:8476"),
            )
        }
        val message = ex.message ?: ""
        kotlin.test.assertTrue("kv" in message && "MULTIHOST_DESIGN" in message, "message was: $message")
    }

    private fun readName(array: java.lang.foreign.MemorySegment, index: Int): String {
        val base = index * PjrtFfm.SZ_NamedValue
        val ptr = array.get(ADDRESS, base + PjrtFfm.OFF_NamedValue_Name).reinterpret(Long.MAX_VALUE)
        val size = array.get(JAVA_LONG, base + PjrtFfm.OFF_NamedValue_NameSize)
        val bytes = ByteArray(size.toInt()) { ptr.get(JAVA_BYTE, it.toLong()) }
        return String(bytes)
    }
}
