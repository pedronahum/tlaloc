package io.tlaloc.runtime.pjrt

import io.tlaloc.runtime.pjrt.ffm.CompileOptions
import io.tlaloc.runtime.pjrt.ffm.CpuClientOptions
import org.junit.jupiter.api.Assumptions.assumeTrue

/**
 * Where the multi-device tests run: [n] host devices of XLA's CPU plugin
 * (`scripts/pjrt-cpu-plugin/build.sh`) by default, the first [n] chips of a TPU with
 * `TLALOC_TEST_PJRT_TARGET=tpu`. Skips when the CPU plugin is not built, or when the TPU has
 * fewer than [n] chips. The programs compile as one SPMD program over devices `0 until n`.
 */
internal object MultiDeviceBackend {

    val platform: String get() = if (TestBackend.isTpu) "tpu" else "cpu"

    fun <T> session(n: Int, options: CompileOptions = CompileOptions.spmd((0L until n).toList()), block: (PjrtSession) -> T): T {
        if (TestBackend.isTpu) {
            assumeTrue(PjrtBinaries.tpuAvailable, TestBackend.noDevice)
            // Without excess precision, as the single-chip TPU lanes compile.
            val strict = options.withOverride("xla_allow_excess_precision", CompileOptions.OptionValue.Flag(false))
            return PjrtSession(plugin = PjrtBinaries.tpuPluginPath!!, target = PjrtTarget.Tpu, compileOptions = strict, diskCache = null).use { s ->
                assumeTrue(s.devices.size >= n, "the TPU has ${s.devices.size} chips, under $n — skipping.")
                block(s)
            }
        }
        val plugin = PjrtBinaries.cpuPluginPath
        assumeTrue(plugin != null, "no CPU PJRT plugin; build it with scripts/pjrt-cpu-plugin/build.sh")
        return PjrtSession(
            plugin = plugin!!,
            target = PjrtTarget.LlvmCpu,
            compileOptions = options,
            diskCache = null,
            cpuOptions = CpuClientOptions(n),
        ).use(block)
    }
}
