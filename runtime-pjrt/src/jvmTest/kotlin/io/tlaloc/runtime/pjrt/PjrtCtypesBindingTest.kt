package io.tlaloc.runtime.pjrt

import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.I32
import io.tlaloc.core.floatArrayToBf16Bits
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.runtime.pjrt.ffm.PjrtClientOptions
import io.tlaloc.runtime.pjrt.ffm.PjrtFfm
import io.tlaloc.stablehlo.toStablehlo
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.fail
import org.junit.jupiter.api.Assumptions.assumeTrue

/**
 * §0.4.475 (H6a) — **PJRT from Python with ctypes, and no framework under it.**
 *
 * `harness/python/tlaloc_pjrt.py` binds the PJRT C API using nothing but the
 * standard library. This test is its certification, and it makes two claims
 * that are deliberately independent:
 *
 *  1. **The two bindings lay the same structs down.** [layoutsMirrorTheFfmBinding]
 *     runs the Python in `--layouts` mode — no plugin, no GPU, no device —
 *     and compares every struct size, every padding-sensitive field offset,
 *     the PJRT_Buffer_Type codes, the whole `PJRT_Api` offset table, the six
 *     CompileOptionsProto bytes, and the marshalled `create_options`
 *     NamedValue array against [PjrtFfm]'s own. This half certifies on a
 *     machine with no accelerator at all, which matters because a struct
 *     laid out wrong does not produce an error at the call site — it
 *     produces a segfault three calls later.
 *  2. **The path computes what the interpreter computes, with jax blocked.**
 *     [ctypesPathAgreesWithTheInterpreter] compiles four tiny graphs through
 *     the ctypes path on real XLA and compares.
 *
 * ## The pin the slice exists for
 *
 * The Python driver installs a `sys.meta_path` finder that RAISES on any
 * import of `jax`, `jaxlib`, `torch` or `numpy`, and installs it *before*
 * importing the binding. The whole path — dlopen, `GetPjrtApi`, client
 * create with options, compile, host→device, execute, device→host, and every
 * Destroy — then runs underneath it.
 *
 * The interpreter this runs in is the **oracle venv**, where jax 0.10.0 *is*
 * installed (§0.4.474 pins exactly that). So "no jax was imported" here is a
 * claim about restraint rather than absence: the framework was on the path,
 * reachable, and the serving surface did not need it. Proving the same thing
 * in an empty venv would have proven less, and would have cost a venv.
 *
 * And because a guard that never fires is an untested guard, the driver
 * finishes by deliberately importing jax and reporting that it was stopped.
 * Both halves are asserted below: nothing forbidden was loaded, AND the
 * thing that would have caught it works.
 *
 * ## Oracle and floors
 *
 * The oracle is [DxirInterpreter] — the same reference every PJRT smoke test
 * in this package uses. Floors are the house's, and they are named, not
 * tuned: **1e-5** for the elementwise lanes (GPU-vs-host f32), **1e-3** for
 * the matmul, which is XLA-GPU's TF32 dot policy and not a defect. The i32
 * lane is **exact** (integer add does no rounding) and the bf16 identity lane
 * is **bit-for-bit** (a copied pattern is not a computed one).
 *
 * Values cross the JSON boundary in both directions as **raw bit patterns**,
 * so nothing this claim rests on passes through a decimal printer.
 *
 * ## What this does NOT yet claim
 *
 * `tlaloc_serve.py` is unchanged and still reaches PJRT through jaxlib.
 * Re-pointing the H3a loader at this binding is the H6b slice; this one
 * builds and certifies the binding it will use. Also unbound, by name:
 * `PJRT_Client_Compile`'s donation options (the manifest has carried
 * `donationPairs` since H3a), the async `PJRT_Buffer_ReadyEvent` path, f16 /
 * i8 / fp8 staging, and multi-device execute.
 */
class PjrtCtypesBindingTest {

    private companion object {
        /** Prefer the oracle venv's interpreter: jax lives there, and a run
         * that does not import it is the whole point (see the KDoc). Any
         * python3 would do — the binding is stdlib-only — so a machine
         * without the oracle venv still certifies, with the weaker claim. */
        fun resolvePython(): String? {
            val home = System.getProperty("user.home")
            if (home != null) {
                val oracle = Path.of(home, ".local", "venvs", "iree", "bin", "python")
                if (Files.isExecutable(oracle)) return oracle.toString()
            }
            for (candidate in listOf("/usr/bin/python3", "/bin/python3")) {
                if (Files.isExecutable(Path.of(candidate))) return candidate
            }
            return null
        }

        fun harnessScript(): Path =
            Path.of("..", "harness", "python", "run_pjrt_ctypes_check.py").toAbsolutePath().normalize()
    }

    // ------------------------------------------------------------------
    // 1. The GPU-less half: the two bindings agree on the ABI.
    // ------------------------------------------------------------------

    @Test
    fun layoutsMirrorTheFfmBinding() {
        val python = resolvePython() ?: run { println("[skip] no python3 to run the ctypes binding"); return }
        val script = harnessScript()
        if (!Files.exists(script)) { println("[skip] $script missing"); return }

        val out = Files.createTempFile("pjrt-ctypes-layouts", ".json")
        try {
            runPython(python, listOf(script.toString(), "--layouts", out.toString()))
            val json = MiniJson.parse(Files.readString(out)) as Map<*, *>

            // Struct sizes, against the Kotlin's hand-written MemoryLayouts.
            val sizes = json["sizes"] as Map<*, *>
            fun size(name: String) = (sizes[name] as Number).toLong()
            assertEquals(PjrtFfm.SZ_NamedValue, size("PJRT_NamedValue"), "PJRT_NamedValue size")
            assertEquals(PjrtFfm.SZ_ClientCreate, size("PJRT_Client_Create_Args"))
            assertEquals(PjrtFfm.SZ_ClientDestroy, size("PJRT_Client_Destroy_Args"))
            assertEquals(PjrtFfm.SZ_PlatformName, size("PJRT_Client_PlatformName_Args"))
            assertEquals(PjrtFfm.SZ_AddressableDevices, size("PJRT_Client_AddressableDevices_Args"))
            assertEquals(PjrtFfm.SZ_Program, size("PJRT_Program"))
            assertEquals(PjrtFfm.SZ_Compile, size("PJRT_Client_Compile_Args"))
            assertEquals(PjrtFfm.SZ_BufferFromHost, size("PJRT_Client_BufferFromHostBuffer_Args"))
            assertEquals(PjrtFfm.SZ_BufferDestroy, size("PJRT_Buffer_Destroy_Args"))
            assertEquals(PjrtFfm.SZ_BufferSize, size("PJRT_Buffer_OnDeviceSizeInBytes_Args"))
            assertEquals(PjrtFfm.SZ_ToHost, size("PJRT_Buffer_ToHostBuffer_Args"))
            assertEquals(PjrtFfm.SZ_LoadedExecDestroy, size("PJRT_LoadedExecutable_Destroy_Args"))
            assertEquals(PjrtFfm.SZ_GetExec, size("PJRT_LoadedExecutable_GetExecutable_Args"))
            assertEquals(PjrtFfm.SZ_ExecDestroy, size("PJRT_Executable_Destroy_Args"))
            assertEquals(PjrtFfm.SZ_NumOutputs, size("PJRT_Executable_NumOutputs_Args"))
            assertEquals(PjrtFfm.SZ_Execute, size("PJRT_LoadedExecutable_Execute_Args"))
            assertEquals(PjrtFfm.SZ_EventDestroy, size("PJRT_Event_Destroy_Args"))
            assertEquals(PjrtFfm.SZ_EventAwait, size("PJRT_Event_Await_Args"))
            // The one the Kotlin pads by hand and ctypes pads by rule.
            assertEquals(
                PjrtFfm.SZ_ExecOpts, size("PJRT_ExecuteOptions"),
                "PJRT_ExecuteOptions: the FFM side inserts 4 bytes after launch_id explicitly, " +
                    "ctypes by alignment rule — if these disagree one of them mis-places every " +
                    "field after it",
            )

            // The offsets that sit behind a padding hole — the ones a hand
            // binding gets wrong.
            val offsets = json["offsets"] as Map<*, *>
            fun off(name: String) = (offsets[name] as Number).toLong()
            assertEquals(PjrtFfm.OFF_NamedValue_Type, off("PJRT_NamedValue.type"))
            assertEquals(PjrtFfm.OFF_NamedValue_Value, off("PJRT_NamedValue.value"))
            assertEquals(PjrtFfm.OFF_NamedValue_ValueSize, off("PJRT_NamedValue.value_size"))
            assertEquals(PjrtFfm.OFF_BufferFromHost_Type, off("PJRT_Client_BufferFromHostBuffer_Args.type"))
            assertEquals(PjrtFfm.OFF_BufferFromHost_Dims, off("PJRT_Client_BufferFromHostBuffer_Args.dims"))
            assertEquals(
                PjrtFfm.OFF_BufferFromHost_HostSemantics,
                off("PJRT_Client_BufferFromHostBuffer_Args.host_buffer_semantics"),
            )
            assertEquals(PjrtFfm.OFF_BufferFromHost_Device, off("PJRT_Client_BufferFromHostBuffer_Args.device"))

            // The PJRT_Api function-pointer table, entry by entry.
            val apiOffsets = json["api_offsets"] as Map<*, *>
            fun api(name: String) = (apiOffsets[name] as Number).toLong()
            assertEquals(PjrtFfm.OFFSET_PJRT_Error_Destroy, api("PJRT_Error_Destroy"))
            assertEquals(PjrtFfm.OFFSET_PJRT_Error_Message, api("PJRT_Error_Message"))
            assertEquals(PjrtFfm.OFFSET_PJRT_Event_Destroy, api("PJRT_Event_Destroy"))
            assertEquals(PjrtFfm.OFFSET_PJRT_Event_Await, api("PJRT_Event_Await"))
            assertEquals(PjrtFfm.OFFSET_PJRT_Client_Create, api("PJRT_Client_Create"))
            assertEquals(PjrtFfm.OFFSET_PJRT_Client_Destroy, api("PJRT_Client_Destroy"))
            assertEquals(PjrtFfm.OFFSET_PJRT_Client_PlatformName, api("PJRT_Client_PlatformName"))
            assertEquals(PjrtFfm.OFFSET_PJRT_Client_AddressableDevices, api("PJRT_Client_AddressableDevices"))
            assertEquals(PjrtFfm.OFFSET_PJRT_Client_Compile, api("PJRT_Client_Compile"))
            assertEquals(PjrtFfm.OFFSET_PJRT_Client_BufferFromHostBuffer, api("PJRT_Client_BufferFromHostBuffer"))
            assertEquals(PjrtFfm.OFFSET_PJRT_Executable_Destroy, api("PJRT_Executable_Destroy"))
            assertEquals(PjrtFfm.OFFSET_PJRT_Executable_NumOutputs, api("PJRT_Executable_NumOutputs"))
            assertEquals(PjrtFfm.OFFSET_PJRT_LoadedExecutable_Destroy, api("PJRT_LoadedExecutable_Destroy"))
            assertEquals(PjrtFfm.OFFSET_PJRT_LoadedExecutable_GetExecutable, api("PJRT_LoadedExecutable_GetExecutable"))
            assertEquals(PjrtFfm.OFFSET_PJRT_LoadedExecutable_Execute, api("PJRT_LoadedExecutable_Execute"))
            assertEquals(PjrtFfm.OFFSET_PJRT_Buffer_Destroy, api("PJRT_Buffer_Destroy"))
            assertEquals(PjrtFfm.OFFSET_PJRT_Buffer_OnDeviceSizeInBytes, api("PJRT_Buffer_OnDeviceSizeInBytes"))
            assertEquals(PjrtFfm.OFFSET_PJRT_Buffer_ToHostBuffer, api("PJRT_Buffer_ToHostBuffer"))

            // §0.4.304's six bytes, byte for byte.
            assertEquals(
                PjrtFfm.COMPILE_OPTIONS_PROTO_BYTES.joinToString("") { "%02x".format(it) },
                json["compile_options_proto_hex"] as String,
                "the hand-encoded CompileOptionsProto must be identical — without num_replicas=1 / " +
                    "num_partitions=1 XLA Check-fails inside ParseDeviceAssignmentCompileOptions (§0.4.304)",
            )

            // PJRT_Buffer_Type codes. F32/F64/BF16 are the Kotlin's; S32 is
            // this binding's own addition and is pinned here so it cannot
            // drift from the enum the others were read off.
            val codes = json["buffer_type_codes"] as Map<*, *>
            assertEquals(PjrtFfm.PJRT_BUFFER_TYPE_F32, (codes["f32"] as Number).toInt())
            assertEquals(PjrtFfm.PJRT_BUFFER_TYPE_F64, (codes["f64"] as Number).toInt())
            assertEquals(PjrtFfm.PJRT_BUFFER_TYPE_BF16, (codes["bf16"] as Number).toInt())
            assertEquals(4, (codes["s32"] as Number).toInt(), "PJRT_Buffer_Type_S32 = 4 (S8..S64 = 2..5)")

            // §0.4.333's create_options, entry for entry, against the bytes
            // the Kotlin's own marshaller writes.
            assertCreateOptionsMirror(json["create_options"] as Map<*, *>)

            // The guard, even in layouts mode — the binding itself must not
            // have dragged a framework in at import time.
            assertGuardHeld(json["guard"] as Map<*, *>)
            println("[pjrt-ctypes] ABI mirror: struct sizes, padded offsets, api table, compile proto, create_options — all identical to PjrtFfm (python=$python)")
        } finally {
            Files.deleteIfExists(out)
        }
    }

    /**
     * The Python's marshalled `PJRT_NamedValue[]`, read back field by field,
     * against the array [PjrtFfm.marshalCreateOptions] writes for the same
     * options. This is the §0.4.333 configuration — `memory_fraction=0.5`,
     * `preallocate=false` — and getting it wrong on a unified-memory host
     * does not fail a test, it takes the machine down.
     */
    private fun assertCreateOptionsMirror(report: Map<*, *>) {
        val options = PjrtClientOptions(memoryFraction = 0.5f, preallocate = false)
        assertEquals(options.namedValueCount, (report["count"] as Number).toLong())

        Arena.ofConfined().use { arena ->
            val array = PjrtFfm.marshalCreateOptions(arena, options)
            val entries = report["entries"] as List<*>
            for (i in 0 until options.namedValueCount.toInt()) {
                val base = i * PjrtFfm.SZ_NamedValue
                val entry = entries[i] as Map<*, *>
                assertEquals(
                    PjrtFfm.SZ_NamedValue,
                    (entry["struct_size"] as Number).toLong(),
                    "entry $i struct_size",
                )
                val nameSize = array.get(ValueLayout.JAVA_LONG, base + PjrtFfm.OFF_NamedValue_NameSize)
                val namePtr = array.get(ValueLayout.ADDRESS, base + PjrtFfm.OFF_NamedValue_Name)
                    .reinterpret(nameSize + 1)
                val name = buildString {
                    for (b in 0 until nameSize) append(namePtr.get(ValueLayout.JAVA_BYTE, b).toInt().toChar())
                }
                assertEquals(name, entry["name"] as String, "entry $i name")
                assertEquals(nameSize, (entry["name_size"] as Number).toLong(), "entry $i name_size")
                assertEquals(
                    array.get(ValueLayout.JAVA_INT, base + PjrtFfm.OFF_NamedValue_Type),
                    (entry["type"] as Number).toInt(),
                    "entry $i type",
                )
                assertEquals(
                    array.get(ValueLayout.JAVA_LONG, base + PjrtFfm.OFF_NamedValue_ValueSize),
                    (entry["value_size"] as Number).toLong(),
                    "entry $i value_size",
                )
                when (name) {
                    "memory_fraction" -> assertEquals(
                        array.get(ValueLayout.JAVA_FLOAT, base + PjrtFfm.OFF_NamedValue_Value),
                        (entry["float_value"] as Number).toFloat(),
                        "memory_fraction value",
                    )
                    "preallocate" -> assertEquals(
                        array.get(ValueLayout.JAVA_BYTE, base + PjrtFfm.OFF_NamedValue_Value).toInt() != 0,
                        entry["bool_value"] as Boolean,
                        "preallocate value",
                    )
                    else -> fail("unexpected create option '$name'")
                }
            }
        }

        // The §0.4.333 rule and the §0.4.459 gate, as facts rather than prose.
        assertTrue(
            report["cuda_refuses_none"] as Boolean,
            "the Python binding must REFUSE a CUDA client with no create_options — that is the " +
                "configuration that preallocates 75% of unified memory and hangs the GB10 (§0.4.333)",
        )
        assertTrue(
            report["tpu_gets_none"] as Boolean,
            "a TPU client must be offered NO create_options — memory_fraction/preallocate are the " +
                "XLA GPU plugin's knobs and libtpu is not guaranteed to accept them (§0.4.459)",
        )
    }

    private fun assertGuardHeld(guard: Map<*, *>) {
        val loaded = guard["loaded_forbidden"] as List<*>
        assertTrue(
            loaded.isEmpty(),
            "the ctypes PJRT path imported ${loaded.joinToString()} — the point of H6a is that it " +
                "needs no framework, and this says the claim is false",
        )
        assertTrue(
            guard["guard_self_test_fired"] as Boolean,
            "the import guard did not stop a deliberate `import jax`: ${guard["guard_self_test_reason"]} " +
                "— an unfired guard certifies nothing, so this lane's jax-freedom would be unproven",
        )
    }

    // ------------------------------------------------------------------
    // 2. The real-XLA half: same numbers as the interpreter, jax blocked.
    // ------------------------------------------------------------------

    @Test
    fun ctypesPathAgreesWithTheInterpreter() {
        assumeTrue(PjrtBinaries.available, "no PJRT plugin resolved — skipping.")
        assumeTrue(PjrtBinaries.cudaAvailable, "no CUDA device — skipping.")
        val python = resolvePython() ?: run { println("[skip] no python3"); return }
        val script = harnessScript()
        if (!Files.exists(script)) { println("[skip] $script missing"); return }

        val dir = Files.createTempDirectory("pjrt-ctypes")
        try {
            // --- lane A: f(x) = x + y, f32 ---------------------------------
            val n = 4
            val f32Vec = DxirType(F32, listOf(n))
            val addF32 = DxirBuilder.function("add_f32") {
                val x = param("x", f32Vec)
                val y = param("y", f32Vec)
                listOf(op(OpKind.ADD, listOf(x, y), f32Vec))
            }
            val xs = floatArrayOf(1f, 2.5f, -3f, 0.125f)
            val ones = FloatArray(n) { 1f }
            val wantAdd = DxirInterpreter.evalFunction(addF32, listOf(xs, ones)).single()

            // --- lane B: matmul, f32 ---------------------------------------
            val a = DxirType(F32, listOf(2, 3))
            val b = DxirType(F32, listOf(3, 2))
            val c = DxirType(F32, listOf(2, 2))
            val matmul = DxirBuilder.function("matmul_f32") {
                val l = param("l", a)
                val r = param("r", b)
                listOf(op(OpKind.MATMUL, listOf(l, r), c))
            }
            val lhs = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
            val rhs = floatArrayOf(0.5f, -1f, 2f, 0.25f, -0.75f, 3f)
            val wantMatmul = DxirInterpreter.evalFunction(matmul, listOf(lhs, rhs)).single()

            // --- lane C: i32 add -------------------------------------------
            // The PJRT_Buffer_Type_S32 code this binding adds is only a claim
            // once an integer buffer survives a real round trip.
            val i32Vec = DxirType(I32, listOf(5))
            val addI32 = DxirBuilder.function("add_i32") {
                val x = param("x", i32Vec)
                val y = param("y", i32Vec)
                listOf(op(OpKind.ADD, listOf(x, y), i32Vec))
            }
            val ints = intArrayOf(0, 1, -7, 123456, -2)
            val addends = intArrayOf(3, -1, 7, 1, 40)
            val wantInts = IntArray(5) { ints[it] + addends[it] }

            // --- lane D: bf16 identity --------------------------------------
            // ADD(x, +0) is the identity for every non-zero lane, so the
            // patterns must come back bit-for-bit — which they cannot if the
            // type code, the 2-byte stride or the raw-pattern convention is
            // wrong anywhere on the path.
            val bf16Values = floatArrayOf(1f, -2.5f, 0.0078125f, 340f, -1.0078125f, 65280f)
            val bf16Patterns = floatArrayToBf16Bits(bf16Values)
            val bVec = DxirType(BF16, listOf(bf16Values.size))
            val addBf16 = DxirBuilder.function("add_bf16") {
                val x = param("x", bVec)
                val z = param("z", bVec)
                listOf(op(OpKind.ADD, listOf(x, z), bVec))
            }

            val lanes = listOf(
                laneJson("add_f32", writeMlir(dir, "add_f32", addF32.toStablehlo("")),
                    listOf(f32In(xs, listOf(n)), f32In(ones, listOf(n))), """{"dtype":"f32","count":$n}"""),
                laneJson("matmul_f32", writeMlir(dir, "matmul_f32", matmul.toStablehlo("")),
                    listOf(f32In(lhs, listOf(2, 3)), f32In(rhs, listOf(3, 2))), """{"dtype":"f32","count":4}"""),
                laneJson("add_i32", writeMlir(dir, "add_i32", addI32.toStablehlo("")),
                    listOf(intIn(ints, listOf(5)), intIn(addends, listOf(5))), """{"dtype":"i32","count":5}"""),
                laneJson("add_bf16", writeMlir(dir, "add_bf16", addBf16.toStablehlo("")),
                    listOf(bf16In(bf16Patterns, listOf(bf16Patterns.size)),
                        bf16In(ShortArray(bf16Patterns.size), listOf(bf16Patterns.size))),
                    """{"dtype":"bf16","count":${bf16Patterns.size}}"""),
            )
            val job = dir.resolve("job.json")
            Files.writeString(
                job,
                """{"plugin":"${PjrtBinaries.pluginPath}","platform":"cuda","lanes":[${lanes.joinToString(",")}]}""",
            )
            val out = dir.resolve("out.json")
            runPython(python, listOf(script.toString(), "--run", job.toString(), out.toString()))
            val json = MiniJson.parse(Files.readString(out)) as Map<*, *>

            assertEquals("cuda", json["platform_name"] as String, "the ctypes client's platform")
            assertTrue((json["num_devices"] as Number).toInt() >= 1, "no addressable devices")
            assertGuardHeld(json["guard"] as Map<*, *>)

            val byName = (json["lanes"] as List<*>).associateBy { (it as Map<*, *>)["name"] as String }

            // A: elementwise f32, host-vs-GPU floor.
            assertFloats(wantAdd, f32Out(byName, "add_f32"), 1e-5f, "add_f32")
            // B: the TF32 dot policy is why this floor is 1e-3 and not 1e-5.
            assertFloats(wantMatmul, f32Out(byName, "matmul_f32"), 1e-3f, "matmul_f32")
            // C: exact — an integer add does no arithmetic that can round.
            val gotInts = ((byName["add_i32"] as Map<*, *>)["outputs"] as List<*>)
                .let { (it[0] as Map<*, *>)["values"] as List<*> }.map { (it as Number).toInt() }
            assertEquals(wantInts.toList(), gotInts, "i32 lane must be exact — PJRT_Buffer_Type_S32 = 4 certified")
            // D: bit-for-bit — a copied pattern is not a computed one.
            val gotPatterns = ((byName["add_bf16"] as Map<*, *>)["outputs"] as List<*>)
                .let { (it[0] as Map<*, *>)["patterns"] as List<*> }.map { (it as Number).toInt() }
            for (i in bf16Patterns.indices) {
                assertEquals(
                    bf16Patterns[i].toInt() and 0xFFFF, gotPatterns[i],
                    "bf16 lane $i: the raw pattern changed crossing the ctypes path",
                )
            }
            // And the device really held 2-byte elements, not widened f32.
            val staged = ((byName["add_bf16"] as Map<*, *>)["staged_device_bytes"] as List<*>)
                .map { (it as Number).toLong() }
            assertTrue(
                staged.all { it in (2L * bf16Patterns.size) until (4L * bf16Patterns.size) },
                "staged bf16 buffers are $staged bytes for ${bf16Patterns.size} elements — not 2-byte storage",
            )
            println(
                "[pjrt-ctypes] four graphs compiled and executed on real XLA-CUDA through ctypes, " +
                    "jax/jaxlib/torch/numpy blocked throughout (python=$python)",
            )
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    // ------------------------------------------------------------------
    // plumbing
    // ------------------------------------------------------------------

    private fun f32Out(byName: Map<String, Any?>, lane: String): FloatArray {
        val outputs = ((byName[lane] as Map<*, *>)["outputs"] as List<*>)
        val bits = (outputs[0] as Map<*, *>)["bits"] as List<*>
        return FloatArray(bits.size) { Float.fromBits((bits[it] as Number).toLong().toInt()) }
    }

    private fun assertFloats(want: FloatArray, got: FloatArray, floor: Float, lane: String) {
        assertEquals(want.size, got.size, "$lane output length")
        for (i in want.indices) {
            val scale = maxOf(1f, abs(want[i]))
            assertTrue(
                abs(want[i] - got[i]) / scale <= floor,
                "$lane lane $i: interpreter ${want[i]} vs ctypes-PJRT ${got[i]} (floor $floor)",
            )
        }
    }

    private fun writeMlir(dir: Path, name: String, mlir: String): Path {
        val p = dir.resolve("$name.mlir")
        Files.writeString(p, mlir)
        return p
    }

    private fun laneJson(name: String, mlir: Path, inputs: List<String>, output: String): String =
        """{"name":"$name","mlir_path":"$mlir","inputs":[${inputs.joinToString(",")}],"outputs":[$output]}"""

    /** f32 inputs cross as raw bit patterns: the device must receive the
     * interpreter's exact bytes, not a decimal that re-parses close to them. */
    private fun f32In(values: FloatArray, dims: List<Int>): String =
        """{"dtype":"f32","dims":[${dims.joinToString(",")}],"values":[${
            values.joinToString(",") { (it.toRawBits().toLong() and 0xFFFFFFFFL).toString() }
        }]}"""

    private fun intIn(values: IntArray, dims: List<Int>): String =
        """{"dtype":"i32","dims":[${dims.joinToString(",")}],"values":[${values.joinToString(",")}]}"""

    private fun bf16In(patterns: ShortArray, dims: List<Int>): String =
        """{"dtype":"bf16","dims":[${dims.joinToString(",")}],"values":[${
            patterns.joinToString(",") { (it.toInt() and 0xFFFF).toString() }
        }]}"""

    private fun runPython(python: String, args: List<String>) {
        val pb = ProcessBuilder(listOf(python) + args)
        // GB10 is unified-memory: nothing on this path preallocates (§0.4.333).
        // The binding's own defaults already say so; the env is belt and braces.
        pb.environment()["TLALOC_PJRT_PREALLOCATE"] = "false"
        pb.environment()["TLALOC_PJRT_MEMORY_FRACTION"] = "0.5"
        pb.environment()["PYTHONPATH"] =
            Path.of("..", "harness", "python").toAbsolutePath().normalize().toString()
        pb.redirectErrorStream(true)
        val proc = pb.start()
        val output = proc.inputStream.bufferedReader().readText()
        if (!proc.waitFor(600, TimeUnit.SECONDS)) {
            proc.destroyForcibly()
            fail("run_pjrt_ctypes_check.py timed out\n$output")
        }
        if (proc.exitValue() != 0) {
            fail("run_pjrt_ctypes_check.py failed (exit ${proc.exitValue()}):\n$output")
        }
    }
}

/**
 * A 120-line JSON reader, because `:runtime-pjrt`'s test classpath has no
 * JSON library and adding one to read four numbers would be the larger
 * change. Handles exactly what `run_pjrt_ctypes_check.py` emits: objects,
 * arrays, strings (with the escapes `json.dump` produces), numbers, and the
 * three literals.
 */
internal object MiniJson {
    fun parse(text: String): Any? {
        val p = Parser(text)
        val v = p.value()
        p.ws()
        require(p.done()) { "trailing JSON at ${p.pos}" }
        return v
    }

    private class Parser(val s: String) {
        var pos = 0
        fun done() = pos >= s.length
        fun ws() { while (pos < s.length && s[pos].isWhitespace()) pos++ }
        fun value(): Any? {
            ws()
            return when (s[pos]) {
                '{' -> obj()
                '[' -> arr()
                '"' -> str()
                't' -> { expect("true"); true }
                'f' -> { expect("false"); false }
                'n' -> { expect("null"); null }
                else -> num()
            }
        }
        fun expect(lit: String) {
            require(s.startsWith(lit, pos)) { "expected $lit at $pos" }
            pos += lit.length
        }
        fun obj(): Map<String, Any?> {
            val m = LinkedHashMap<String, Any?>()
            pos++ // {
            ws()
            if (s[pos] == '}') { pos++; return m }
            while (true) {
                ws()
                val k = str()
                ws(); require(s[pos] == ':') { "expected : at $pos" }; pos++
                m[k] = value()
                ws()
                when (s[pos]) {
                    ',' -> pos++
                    '}' -> { pos++; return m }
                    else -> error("expected , or } at $pos")
                }
            }
        }
        fun arr(): List<Any?> {
            val l = ArrayList<Any?>()
            pos++ // [
            ws()
            if (s[pos] == ']') { pos++; return l }
            while (true) {
                l.add(value())
                ws()
                when (s[pos]) {
                    ',' -> pos++
                    ']' -> { pos++; return l }
                    else -> error("expected , or ] at $pos")
                }
            }
        }
        fun str(): String {
            require(s[pos] == '"') { "expected string at $pos" }
            pos++
            val sb = StringBuilder()
            while (s[pos] != '"') {
                if (s[pos] == '\\') {
                    pos++
                    when (val e = s[pos]) {
                        'n' -> sb.append('\n')
                        't' -> sb.append('\t')
                        'r' -> sb.append('\r')
                        'b' -> sb.append('\b')
                        'f' -> sb.append('\u000C')
                        'u' -> {
                            sb.append(s.substring(pos + 1, pos + 5).toInt(16).toChar())
                            pos += 4
                        }
                        else -> sb.append(e)
                    }
                } else {
                    sb.append(s[pos])
                }
                pos++
            }
            pos++
            return sb.toString()
        }
        fun num(): Number {
            val start = pos
            while (pos < s.length && (s[pos].isDigit() || s[pos] in "-+.eE")) pos++
            val t = s.substring(start, pos)
            return if (t.any { it in ".eE" }) t.toDouble() else t.toLong()
        }
    }
}
