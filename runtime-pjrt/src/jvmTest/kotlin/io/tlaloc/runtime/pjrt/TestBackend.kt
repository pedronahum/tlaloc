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
     * values widened. False on Cerebras too: the WSE has no double-precision
     * arithmetic, and CerebrasStableHLO computes f64 in f32 on the wafer
     * (inputs narrowed to the nearest f32, results widened); its host
     * interpreter runs f64 dots in f32 as well.
     */
    val f64IsDouble: Boolean get() = !isMetal && !isCerebras

    val noF64: String
        get() = if (isCerebras) {
            "CerebrasStableHLO computes f64 in f32 (the WSE has no double) — skipping a double-precision check."
        } else {
            "MetalHLO computes f64 in f32 (Metal has no double) — skipping a double-precision check."
        }

    /**
     * Whether dispatches go through the Cerebras wafer simulator
     * (`CEREBRASSTABLEHLO_BACKEND=simulator`). Every execute there boots the
     * simulated WSE in the cs_sdk VM — about a second per call — so tests that
     * time steady-state dispatch or push thousands of calls through a deadline
     * measure the simulator, not the plugin, and skip by name.
     */
    val isWaferSimulator: Boolean
        get() = isCerebras && System.getenv("CEREBRASSTABLEHLO_BACKEND") == "simulator"

    val noFastDispatch: String
        get() = "the Cerebras wafer simulator takes about a second per dispatch — skipping a dispatch-rate check."

    /**
     * Whether the backend's compiler elides an f32→bf16→f32 convert pair (XLA's
     * algebraic simplifier does, measured on CUDA). False on Metal and
     * Cerebras: MetalHLO and CerebrasStableHLO keep every narrowing, so a
     * value that passes through bf16 is always the snapped one — the
     * interpreter's convention.
     */
    val foldsConvertPairs: Boolean get() = !isMetal && !isCerebras

    /** [tol] for an f64 result compared with an f64 reference: f32-level on Metal and Cerebras. */
    fun f64Tolerance(tol: Double): Double = if (f64IsDouble) tol else maxOf(tol, 1e-5)

    /** Where f32 dots lose precision by default: TF32 on the GB10, one bf16 pass on a TPU. */
    val defaultDotRelTolerance: Float get() = if (isTpu) 3e-2f else 1e-2f
}
