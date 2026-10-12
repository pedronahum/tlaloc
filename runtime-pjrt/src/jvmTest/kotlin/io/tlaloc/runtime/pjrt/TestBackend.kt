package io.tlaloc.runtime.pjrt

/**
 * The PJRT backend the device smoke tests run on. CUDA by default;
 * `TLALOC_TEST_PJRT_TARGET=tpu` runs the same tests on libtpu
 * ([PjrtBinaries.tpuPluginPath]) and `TLALOC_TEST_PJRT_TARGET=metal` on
 * MetalHLO ([PjrtBinaries.metalPluginPath]). Tests gate on [pluginResolved] and
 * [deviceAvailable] and open sessions with [session], so they skip by name
 * on a machine without the chosen backend.
 */
internal object TestBackend {
    val target: PjrtTarget = when (val t = System.getenv("TLALOC_TEST_PJRT_TARGET")?.lowercase()) {
        null, "", "cuda" -> PjrtTarget.Cuda
        "tpu" -> PjrtTarget.Tpu
        "metal" -> PjrtTarget.Metal
        "cerebras" -> PjrtTarget.Cerebras
        else -> error("TLALOC_TEST_PJRT_TARGET=$t: expected cuda, tpu, metal or cerebras")
    }

    val isTpu: Boolean get() = target == PjrtTarget.Tpu
    val isMetal: Boolean get() = target == PjrtTarget.Metal
    val isCerebras: Boolean get() = target == PjrtTarget.Cerebras

    val pluginResolved: Boolean
        get() = when {
            isTpu -> PjrtBinaries.tpuAvailable
            isMetal -> PjrtBinaries.metalPluginPath != null
            isCerebras -> PjrtBinaries.cerebrasPluginPath != null
            else -> PjrtBinaries.available
        }

    val deviceAvailable: Boolean
        get() = when {
            isTpu -> PjrtBinaries.tpuAvailable
            isMetal -> PjrtBinaries.metalAvailable
            isCerebras -> PjrtBinaries.cerebrasPluginPath != null
            else -> PjrtBinaries.cudaAvailable
        }

    val noPlugin: String
        get() = when {
            isTpu -> "no libtpu resolved (TLALOC_PJRT_PLUGIN_PATH) — skipping."
            isMetal -> "no MetalHLO plugin resolved (METALHLO_PLUGIN_PATH, ../MetalHLO/.build) — skipping."
            isCerebras -> "no CerebrasStableHLO plugin resolved (CEREBRASSTABLEHLO_PLUGIN_PATH, ../CerebrasStableHLO/build) — skipping."
            else -> "no PJRT plugin resolved — skipping."
        }

    val noDevice: String
        get() = when {
            isTpu -> "no TPU (libtpu not resolved) — skipping."
            isMetal -> "not Apple silicon — skipping."
            else -> "no CUDA device — skipping."
        }

    /** The plugin [session] opens, for tests that load the API directly. */
    val pluginPath: java.nio.file.Path?
        get() = when {
            isTpu -> PjrtBinaries.tpuPluginPath
            isMetal -> PjrtBinaries.metalPluginPath
            isCerebras -> PjrtBinaries.cerebrasPluginPath
            else -> PjrtBinaries.pluginPath
        }

    fun session(portableF32Dots: Boolean = isTpu || isMetal): PjrtSession =
        when {
            isTpu -> PjrtSession(plugin = PjrtBinaries.tpuPluginPath!!, target = PjrtTarget.Tpu)
            isMetal -> PjrtSession(plugin = PjrtBinaries.metalPluginPath!!, target = PjrtTarget.Metal)
            isCerebras -> PjrtSession(plugin = PjrtBinaries.cerebrasPluginPath!!, target = PjrtTarget.Cerebras)
            else -> PjrtSession(target = PjrtTarget.Cuda)
        }.also { it.portableF32Dots = portableF32Dots }

    /**
     * Whether f64 results carry double precision. False on Metal: MetalHLO
     * computes f64 in f32 (Metal has no double), so its f64 results are f32
     * values widened.
     */
    val f64IsDouble: Boolean get() = !isMetal

    val noF64: String
        get() = "MetalHLO computes f64 in f32 (Metal has no double) — skipping a double-precision check."

    /** [tol] for an f64 result compared with an f64 reference: f32-level on Metal. */
    fun f64Tolerance(tol: Double): Double = if (f64IsDouble) tol else maxOf(tol, 1e-5)

    /** Where f32 dots lose precision by default: TF32 on the GB10, one bf16 pass on a TPU. */
    val defaultDotRelTolerance: Float get() = if (isTpu) 3e-2f else 1e-2f
}
