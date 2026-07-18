package io.tlaloc.kptx

/**
 * KPTX v3.3 (§0.4.347) — static resource + occupancy report (plan task
 * 18's representative "debug/verification extra"; deadlock beacons and
 * differential symbolic execution stay deferred until a kernel needs
 * them, per the plan's optional framing).
 *
 * [resourceReport] summarizes what a kernel *declares*: virtual
 * register banks per class, static shared memory. [estimateOccupancy]
 * turns that into a per-arch occupancy estimate with the limiting
 * factor named — the number a kernel author actually wants when
 * choosing a block size.
 *
 * **Honesty caveats, in order of importance:**
 * - Declared virtual registers are an *upper bound proxy*: the driver
 *   JIT re-allocates real SASS registers (usually fewer). A
 *   register-limited verdict here means "at most this occupancy from
 *   the declared banks", not a SASS measurement.
 * - 64-bit registers are counted as two 32-bit slots; predicates live
 *   in their own file and don't count against the register budget.
 * - Arch limits come from NVIDIA's public occupancy/tuning tables for
 *   the well-documented SMs; `sm_120`/`sm_121` (consumer/GB10
 *   Blackwell) are mapped to the `sm_86`-class limits pending official
 *   tables.
 */
data class KernelResources(
    val name: String,
    val regBanks: Map<IsaRegClass, Int>,
    val staticSmemBytes: Int,
) {
    /** 32-bit register slots per thread from the declared banks
     * (R32 + F32 + 2·R64; predicates excluded). */
    val regSlotsPerThread: Int
        get() = (regBanks[IsaRegClass.R32] ?: 0) +
            (regBanks[IsaRegClass.F32] ?: 0) +
            2 * (regBanks[IsaRegClass.R64] ?: 0)
}

/** Per-SM hardware limits for an architecture. */
data class SmLimits(
    val maxThreads: Int,
    val maxBlocks: Int,
    val regSlots: Int,
    val smemBytes: Int,
)

/** Public-table limits, keyed by `.target` spelling (suffix letters
 * like `sm_121a` are normalized away). */
val SM_LIMITS: Map<String, SmLimits> = mapOf(
    "sm_75" to SmLimits(maxThreads = 1024, maxBlocks = 16, regSlots = 65536, smemBytes = 65536),
    "sm_80" to SmLimits(maxThreads = 2048, maxBlocks = 32, regSlots = 65536, smemBytes = 167936),
    "sm_86" to SmLimits(maxThreads = 1536, maxBlocks = 16, regSlots = 65536, smemBytes = 102400),
    "sm_89" to SmLimits(maxThreads = 1536, maxBlocks = 24, regSlots = 65536, smemBytes = 102400),
    "sm_90" to SmLimits(maxThreads = 2048, maxBlocks = 32, regSlots = 65536, smemBytes = 233472),
)

private fun limitsFor(arch: String): SmLimits {
    val norm = arch.takeWhile { it.isLetterOrDigit() || it == '_' }.trimEnd { it.isLetter() && it != 'm' }
    SM_LIMITS[norm]?.let { return it }
    // Consumer/GB10 Blackwell: sm_86-class limits pending official tables.
    if (norm.startsWith("sm_12")) return SM_LIMITS.getValue("sm_86")
    throw IllegalArgumentException("no SM limits table for arch `$arch` (known: ${SM_LIMITS.keys.sorted()})")
}

data class OccupancyEstimate(
    val arch: String,
    val blockThreads: Int,
    val blocksPerSm: Int,
    val activeThreads: Int,
    val occupancyPct: Int,
    val limitingFactor: String,
)

/** Summarize each kernel's declared resources. */
fun PtxModule.resourceReport(): List<KernelResources> = kernels.map { kernel ->
    val banks = HashMap<IsaRegClass, Int>()
    var smem = 0
    for (stmt in kernel.body) when (stmt) {
        is PtxRegDecl -> IsaRegClass.entries.firstOrNull { it.prefix == stmt.prefix }
            ?.let { banks[it] = maxOf(banks[it] ?: 0, stmt.count) }
        is PtxSharedDecl -> smem += stmt.sizeBytes
        else -> {}
    }
    KernelResources(kernel.name, banks, smem)
}

/** Estimate occupancy for launching this kernel at [blockThreads] on
 * [arch] (defaults resolve via [SM_LIMITS]; see class KDoc caveats). */
fun KernelResources.estimateOccupancy(blockThreads: Int, arch: String): OccupancyEstimate {
    require(blockThreads in 1..1024) { "blockThreads must be in 1..1024, got $blockThreads" }
    val limits = limitsFor(arch)

    val byThreads = limits.maxThreads / blockThreads
    val byBlocks = limits.maxBlocks
    val bySmem = if (staticSmemBytes > 0) limits.smemBytes / staticSmemBytes else Int.MAX_VALUE
    val regsPerBlock = regSlotsPerThread * blockThreads
    val byRegs = if (regsPerBlock > 0) limits.regSlots / regsPerBlock else Int.MAX_VALUE

    val blocksPerSm = minOf(byThreads, byBlocks, bySmem, byRegs)
    val limiting = when (blocksPerSm) {
        byThreads -> "threads"
        byBlocks -> "blocks"
        bySmem -> "shared-memory"
        else -> "registers(declared)"
    }
    val active = blocksPerSm * blockThreads
    return OccupancyEstimate(
        arch = arch,
        blockThreads = blockThreads,
        blocksPerSm = blocksPerSm,
        activeThreads = active,
        occupancyPct = (active * 100) / limits.maxThreads,
        limitingFactor = limiting,
    )
}
