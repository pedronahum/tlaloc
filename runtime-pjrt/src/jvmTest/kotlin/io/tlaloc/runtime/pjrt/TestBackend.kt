package io.tlaloc.runtime.pjrt

/**
 * The PJRT backend the device smoke tests run on. CUDA by default;
 * `TLALOC_TEST_PJRT_TARGET=tpu` runs the same tests on libtpu
 * ([PjrtBinaries.tpuPluginPath]). Tests gate on [pluginResolved] and
 * [deviceAvailable] and open sessions with [session], so they skip by name
 * on a machine without the chosen backend.
 */
internal object TestBackend {
    val target: PjrtTarget = when (val t = System.getenv("TLALOC_TEST_PJRT_TARGET")?.lowercase()) {
        null, "", "cuda" -> PjrtTarget.Cuda
        "tpu" -> PjrtTarget.Tpu
        else -> error("TLALOC_TEST_PJRT_TARGET=$t: expected cuda or tpu")
    }

    val isTpu: Boolean get() = target == PjrtTarget.Tpu

    val pluginResolved: Boolean
        get() = if (isTpu) PjrtBinaries.tpuAvailable else PjrtBinaries.available

    val deviceAvailable: Boolean
        get() = if (isTpu) PjrtBinaries.tpuAvailable else PjrtBinaries.cudaAvailable

    val noPlugin: String
        get() = if (isTpu) "no libtpu resolved (TLALOC_PJRT_PLUGIN_PATH) — skipping."
        else "no PJRT plugin resolved — skipping."

    val noDevice: String
        get() = if (isTpu) "no TPU (libtpu not resolved) — skipping." else "no CUDA device — skipping."

    fun session(portableF32Dots: Boolean = isTpu): PjrtSession =
        if (isTpu) PjrtSession(plugin = PjrtBinaries.tpuPluginPath!!, target = PjrtTarget.Tpu, portableF32Dots = portableF32Dots)
        else PjrtSession(target = PjrtTarget.Cuda, portableF32Dots = portableF32Dots)

    /** Where f32 dots lose precision by default: TF32 on the GB10, one bf16 pass on a TPU. */
    val defaultDotRelTolerance: Float get() = if (isTpu) 3e-2f else 1e-2f
}
