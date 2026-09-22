/**
 * Tlaloc on a TPU — the program, the cross-check, and the bit stream.
 *
 * ┌───────────────────────────────────────────────────────────────────────┐
 * │ NOTHING IN THIS EXAMPLE HAS EVER EXECUTED ON A TPU.                   │
 * │                                                                       │
 * │ Everything the TPU path needs on the host side is certified: the      │
 * │ target, the plugin resolver, the create-options platform gate, the    │
 * │ compile-options encoding, a self-skipping smoke suite. No TPU has     │
 * │ ever been under any of it. This example exists so that the day one    │
 * │ is, the first command is already written — and so that the claims it  │
 * │ will make are written down BEFORE the hardware can flatter them.      │
 * └───────────────────────────────────────────────────────────────────────┘
 *
 * The program runs in two halves.
 *
 * THE HOST HALF runs everywhere — laptop, GB10, TPU VM. It builds an attention
 * block as a Tlaloc IR function, differentiates it with the compiler's own
 * reverse-mode pass, emits both graphs as StableHLO text, and evaluates them on
 * the JVM interpreter. That interpreter result is the reference the device is
 * later judged against; it is computed before any device is touched, so the
 * device cannot define its own correctness.
 *
 * THE DEVICE HALF needs a TPU. It resolves libtpu, creates a PJRT client with
 * `PjrtTarget.Tpu`, compiles the emitted StableHLO, stages buffers, executes,
 * and compares. Five acts, each of which prints PASS or FAIL and a number. If
 * no TPU is here, it prints exactly what is missing and exits 0.
 *
 * Act 4 is the one worth flying to a datacenter for: Tlaloc's threefry PRNG
 * lowers to explicit StableHLO integer arithmetic, so the *same bits* should
 * come back from any backend. That claim is already measured on CUDA. A TPU
 * pass makes it a claim about three backends, and a fork here would be a
 * genuine discovery — which is why the check is bit-for-bit and not a
 * tolerance.
 */
import io.tlaloc.core.BF16
import io.tlaloc.core.F32
import io.tlaloc.core.floatArrayToBf16Bits
import io.tlaloc.core.floatToBf16Bits
import io.tlaloc.core.uniformFloats
import io.tlaloc.ir.DxirBuilder
import io.tlaloc.ir.DxirFunction
import io.tlaloc.ir.DxirType
import io.tlaloc.ir.OpKind
import io.tlaloc.ir.passes.DxirInterpreter
import io.tlaloc.ir.passes.DxirReverseTransform
import io.tlaloc.runtime.pjrt.PjrtBinaries
import io.tlaloc.runtime.pjrt.PjrtSession
import io.tlaloc.runtime.pjrt.PjrtTarget
import io.tlaloc.stablehlo.toStablehlo
import java.io.File
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt
import kotlin.system.exitProcess

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/**
 * Default size: small enough that the JVM interpreter can check it in a second,
 * which is the point — a device claim is only worth as much as the reference it
 * was checked against. On a v5e/v6e chip, `--seq 2048 --dim 512` turns the same
 * program into ~4 GMAC of attention (and a much slower host check).
 *
 * `--target cuda` is the DRY RUN. It runs the five acts below against an NVIDIA
 * GPU instead of a TPU, which proves the *program* — the graphs, the emission,
 * the staging, the comparisons — without proving anything about a TPU. It
 * exists so that the first TPU run is debugging the TPU and not this example.
 * A cuda run says so in every banner and makes no TPU claim.
 */
private data class Config(val seq: Int, val dim: Int, val target: PjrtTarget) {

    /** What `PJRT_Client_PlatformName` is allowed to say for this target. XLA's
     * GPU plugin has answered both "cuda" and "gpu" across versions. */
    val expectedPlatformNames: Set<String>
        get() = when (target) {
            PjrtTarget.Tpu -> setOf("tpu")
            PjrtTarget.Cuda -> setOf("cuda", "gpu")
            PjrtTarget.LlvmCpu -> setOf("cpu")
        }

    companion object {
        fun parse(args: Array<String>): Config {
            var seq = 256
            var dim = 128
            var target = PjrtTarget.Tpu
            var i = 0
            while (i < args.size) {
                when (args[i]) {
                    "--seq" -> seq = args[++i].toInt()
                    "--dim" -> dim = args[++i].toInt()
                    "--target" -> target = when (val t = args[++i].lowercase()) {
                        "tpu" -> PjrtTarget.Tpu
                        "cuda", "gpu" -> PjrtTarget.Cuda
                        else -> error("unknown --target '$t' (expected tpu or cuda)")
                    }
                    else -> error(
                        "unknown argument '${args[i]}' " +
                            "(expected --seq N / --dim N / --target tpu|cuda)",
                    )
                }
                i++
            }
            return Config(seq, dim, target)
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fun main(args: Array<String>) {
    val cfg = Config.parse(args)

    println("=== Tlaloc on a TPU — the program, the cross-check, and the bit stream ===")
    println()
    println("HONESTY NOTICE: no part of Tlaloc has ever executed on a TPU. The host half")
    println("below runs anywhere; the device half has never run at all. If you are reading")
    println("its output for the first time, you are the first. See docs/TPU_BRINGUP.md.")
    if (cfg.target != PjrtTarget.Tpu) {
        println()
        println("DRY RUN: --target ${cfg.target.platform}. The five acts below run on a")
        println("${cfg.target.platform} device to prove THIS PROGRAM, not a TPU. Nothing a dry run")
        println("prints is a TPU claim; re-run with the default --target tpu on a TPU VM.")
    }
    println()

    val host = hostHalf(cfg)

    val missing = missingDeviceReason(cfg.target)
    if (missing != null) {
        println("--- device half: SKIPPED -------------------------------------------------")
        println()
        println(missing)
        println()
        println("Nothing above this line needed an accelerator, and nothing above this line")
        println("is a device claim. Exiting 0 — a machine without one is not a failure.")
        return
    }

    val results = deviceHalf(cfg, host)

    println()
    println("--- verdict ---------------------------------------------------------------")
    for (r in results) println("  ${if (r.ok) "PASS" else "FAIL"}  ${r.name}: ${r.detail}")
    val failed = results.count { !it.ok }
    println()
    if (failed == 0 && cfg.target != PjrtTarget.Tpu) {
        println("All ${results.size} acts passed on ${cfg.target.platform}. This proves the program and")
        println("says NOTHING about a TPU. The TPU claim is still unmade.")
    } else if (failed == 0) {
        println("All ${results.size} acts passed on a real TPU. That is a first: update")
        println("docs/TPU_BRINGUP.md (G2b) and delete the honesty notice at the top of this")
        println("example, because it will no longer be true.")
    } else {
        println("$failed of ${results.size} acts FAILED. Do not paper over this: a failure")
        println("here is information — the first real measurement of a claim that until now")
        println("was only an argument. Record it in docs/TPU_BRINGUP.md §'what G2b must record'.")
    }
    exitProcess(if (failed == 0) 0 else 1)
}

// ---------------------------------------------------------------------------
// The host half — builds, differentiates, emits, and computes the reference
// ---------------------------------------------------------------------------

private class HostHalf(
    val forward: DxirFunction,
    val gradient: DxirFunction,
    val inputs: List<FloatArray>,
    val forwardRef: List<FloatArray>,
    val gradientRef: List<FloatArray>,
)

private fun hostHalf(cfg: Config): HostHalf {
    println("--- host half: the program, and the reference it will be judged against ----")
    println()

    // 1. The program. Four ops, no backend anywhere in it.
    val forward = Attention.forward(cfg.seq, cfg.dim)
    println("workload : scaled-dot-product attention, seq=${cfg.seq} dim=${cfg.dim}")
    println("           Q[${cfg.seq},${cfg.dim}] · Kt[${cfg.dim},${cfg.seq}] -> scale -> softmax -> ·V[${cfg.seq},${cfg.dim}]")
    println("           ${forward.body.size} IR ops, ${forward.params.size} parameters")
    val macs = 2L * cfg.seq * cfg.seq * cfg.dim
    println("           ${"%.1f".format(macs / 1e6)} M multiply-accumulates per forward pass")

    // 2. The gradient — derived, not written. The reverse transform is the same
    //    pass the `grad { }` compiler plugin runs; here it is applied directly
    //    to an IR function, which is the other door into the same engine.
    //    `includeForward = true` puts the loss in output slot 0, so one dispatch
    //    yields the value AND the three input gradients.
    val loss = Attention.loss(cfg.seq, cfg.dim)
    val gradient = DxirReverseTransform.apply(loss, includeForward = true)
    println("gradient : d(Σ attention)/d(Q, Kt, V), derived by DxirReverseTransform")
    println("           ${loss.body.size} ops in, ${gradient.body.size} ops out, " +
        "${gradient.returns.size} results (loss + 3 gradients)")
    println("           the softmax adjoint in there was written by the compiler, not by us")

    // 3. The text that will cross the boundary. Written to disk so that a first
    //    contact with a new backend can be debugged by reading, not guessing.
    val out = File("build").apply { mkdirs() }
    val fwdMlir = File(out, "attention.stablehlo.mlir").apply { writeText(forward.toStablehlo()) }
    val gradMlir = File(out, "attention_grad.stablehlo.mlir").apply { writeText(gradient.toStablehlo()) }
    println("emitted  : ${fwdMlir.path} (${fwdMlir.readLines().size} lines)")
    println("           ${gradMlir.path} (${gradMlir.readLines().size} lines)")
    println("           this is exactly what the TPU compiler is handed — nothing is")
    println("           generated on the device side of the boundary.")
    println()
    println("  first lines of the forward module:")
    fwdMlir.readLines().take(6).forEach { println("    $it") }
    println("    ...")
    println()

    // 4. The reference. Computed on the JVM, before any device exists.
    val inputs = Attention.inputs(cfg.seq, cfg.dim)
    val t0 = System.nanoTime()
    val forwardRef = DxirInterpreter.evalFunction(forward, inputs)
    val gradientRef = DxirInterpreter.evalFunction(gradient, inputs)
    val ms = (System.nanoTime() - t0) / 1e6
    println("reference: host interpreter evaluated both graphs in ${"%.0f".format(ms)} ms")
    println("           forward  out[0..3] = ${preview(forwardRef[0])}")
    println("           loss             = ${gradientRef[0][0]}")
    println("           dQ[0..3]         = ${preview(gradientRef[1])}")
    println("           inputs are threefry draws from fixed keys — identical on every")
    println("           machine that runs this example, which is what makes the")
    println("           comparison below meaningful across two continents.")
    println()

    // 5. The RNG stream, on the host. Act 4 asks a TPU for these same bits.
    val (dims, key) = Rng.sweep[0]
    val n = dims.fold(1) { a, d -> a * d }
    println("rng      : uniform(key=$key, n=$n) = ${previewBits(uniformFloats(key, n))}")
    println("           (hex is the raw f32 bit pattern; Act 4 compares THESE, not values)")
    println()

    return HostHalf(forward, gradient, inputs, forwardRef, gradientRef)
}

// ---------------------------------------------------------------------------
// The skip path — what is missing, by name
// ---------------------------------------------------------------------------

/**
 * Returns null when a plugin for [target] resolves, and otherwise a message
 * naming exactly what was looked for and not found.
 *
 * The resolver deliberately refuses a `TLALOC_PJRT_PLUGIN_PATH` that does not
 * name a tpu-shaped file: on a CUDA host that variable legitimately points at
 * `xla_cuda_plugin.so`, and handing that to a `PjrtTarget.Tpu` session would
 * build a CUDA client wearing a TPU label. This message explains that instead
 * of silently skipping.
 */
private fun missingDeviceReason(target: PjrtTarget): String? {
    if (target != PjrtTarget.Tpu) {
        // The dry-run lane: the CUDA-family resolver plus an actual GPU.
        if (!PjrtBinaries.available) {
            return "no PJRT plugin resolved for --target ${target.platform} — set " +
                "TLALOC_PJRT_PLUGIN_PATH, or install a JAX CUDA plugin. (The dry run is a " +
                "convenience; the example's subject is the TPU lane.)"
        }
        if (!PjrtBinaries.cudaAvailable) {
            return "no CUDA device visible to nvidia-smi — --target ${target.platform} " +
                "needs a GPU on this machine. (The dry run is a convenience; the example's " +
                "subject is the TPU lane.)"
        }
        return null
    }
    if (PjrtBinaries.tpuAvailable) return null
    // Blank counts as unset here. (The resolver itself treats a blank value as a
    // path and finds that `Path.of("")` "exists" — recorded as a real gap in the
    // resolver, deliberately not worked around anywhere but this message.)
    val env = System.getenv("TLALOC_PJRT_PLUGIN_PATH")?.takeIf { it.isNotBlank() }
    val b = StringBuilder()
    b.appendLine("no TPU PJRT plugin resolved — this machine has no TPU, so the device half")
    b.appendLine("of the example cannot run. Missing: libtpu.so (the TPU PJRT plugin).")
    b.appendLine()
    b.appendLine("Where it was looked for:")
    when {
        env == null ->
            b.appendLine("  - \$TLALOC_PJRT_PLUGIN_PATH: not set")
        !Files.exists(Path.of(env)) ->
            b.appendLine("  - \$TLALOC_PJRT_PLUGIN_PATH = $env (file does not exist)")
        !Path.of(env).fileName.toString().contains("tpu") ->
            b.appendLine(
                "  - \$TLALOC_PJRT_PLUGIN_PATH = $env — REFUSED for the TPU lane: the file " +
                    "name is not tpu-shaped,\n    so this is almost certainly a CUDA/CPU " +
                    "plugin. Pointing a TPU session at it would create a\n    client of the " +
                    "wrong platform. Set it to a libtpu.so to use the TPU lane.",
            )
        else -> b.appendLine("  - \$TLALOC_PJRT_PLUGIN_PATH = $env (present but did not resolve)")
    }
    b.appendLine("  - \$VIRTUAL_ENV/lib/python3.N/site-packages/libtpu/libtpu.so  (pip install libtpu)")
    b.appendLine("  - ~/.local/lib/python3.N/site-packages/libtpu/libtpu.so       (pip install --user)")
    b.appendLine("  - /lib/libtpu.so, /usr/lib/libtpu.so                          (Cloud TPU VM images)")
    b.appendLine()
    b.append("How to get one: provision a Cloud TPU VM and follow docs/TPU_BRINGUP.md, or")
    b.append(" read this example's README.md, which is that runbook narrowed to this program.")
    return b.toString()
}

// ---------------------------------------------------------------------------
// The device half — five acts
// ---------------------------------------------------------------------------

private class Act(val name: String, val ok: Boolean, val detail: String)

/** The f32 unit roundoff: the distance from 1.0 to the next representable float. */
private const val F32_EPS = 1.1920929e-7f

private fun deviceHalf(cfg: Config, host: HostHalf): List<Act> {
    val tpu = cfg.target == PjrtTarget.Tpu
    val plugin = if (tpu) PjrtBinaries.tpuPluginPath!! else PjrtBinaries.pluginPath!!
    println(
        if (tpu) "--- device half: a TPU is here --------------------------------------------"
        else "--- device half: DRY RUN on ${cfg.target.platform} (no TPU claim) ------------------------",
    )
    println()
    println("plugin   : $plugin")
    println("target   : ${cfg.target}")
    println()

    // The one structural difference between the two lanes, and it is a refusal
    // rather than a convention: `memory_fraction` / `preallocate` are the XLA
    // *GPU* allocator's create-options, so a Tpu session is created with zero
    // create-options (`options = null`) and PjrtSession's init block refuses
    // both cross-wirings by name. A CUDA session must carry them — creating one
    // without them revives a 75%-of-unified-memory preallocation that once
    // rebooted this machine. Passing the default here is not laziness; it is
    // the safe half of that gate.
    val session =
        if (tpu) PjrtSession(plugin = plugin, target = PjrtTarget.Tpu, options = null)
        else PjrtSession(plugin = plugin, target = cfg.target)
    return session.use {
        listOf(
            actPlatformName(it, plugin, cfg),
            actForward(it, cfg, host),
            actGradient(it, cfg, host),
            actThreefry(it, cfg),
            actBf16(it, cfg),
        )
    }
}

/** Act 1 — the plugin that resolved must actually be the requested backend.
 * Asserted, not assumed: a mis-resolved plugin would otherwise let every act
 * below certify the wrong hardware. */
private fun actPlatformName(session: PjrtSession, plugin: Path, cfg: Config): Act {
    val name = session.platformName()
    println("[act 1] platform identity")
    println("        PJRT_Client_PlatformName = '$name'  (plugin: ${plugin.fileName})")
    val ok = name.lowercase() in cfg.expectedPlatformNames
    println(
        "        " + if (ok) "this is a ${cfg.target.platform} device."
        else "NOT a ${cfg.target.platform} device (expected one of ${cfg.expectedPlatformNames}) — " +
            "every claim below would be mislabelled.",
    )
    println()
    return Act("platform identity", ok, "platform_name='$name'")
}

/** Act 2 — the attention block, device vs host interpreter. */
private fun actForward(session: PjrtSession, cfg: Config, host: HostHalf): Act {
    val dev = cfg.target.platform
    println("[act 2] attention forward, $dev vs host interpreter")
    val t0 = System.nanoTime()
    val got = session.runOn(host.forward, host.inputs)
    val ms = (System.nanoTime() - t0) / 1e6
    val (maxAbs, maxRef) = maxDiff(got, host.forwardRef)
    // f32 attention is a reduction chain: the device is entitled to a different
    // summation order than the interpreter, so this is a tolerance, not a pin.
    // The RNG act below is where an exact claim belongs.
    val tol = 1e-3f * max(1f, maxRef)
    val ok = maxAbs <= tol
    println("        compile + stage + execute: ${"%.0f".format(ms)} ms (includes XLA compilation)")
    println("        out[0..3] = ${preview(got[0])}")
    println("        max|$dev - host| = $maxAbs   (largest reference magnitude $maxRef, tolerance $tol)")
    println("        ${if (ok) "the device agrees with the interpreter." else "DIVERGENCE — see build/attention.stablehlo.mlir."}")
    println()
    return Act("attention forward", ok, "max|diff|=$maxAbs vs tol=$tol")
}

/** Act 3 — the derived gradient, device vs host interpreter. The graph under
 * test was written by the compiler, so this checks the AD pass and the TPU in
 * the same breath. */
private fun actGradient(session: PjrtSession, cfg: Config, host: HostHalf): Act {
    val dev = cfg.target.platform
    println("[act 3] compiler-derived gradient, $dev vs host interpreter")
    val got = session.runOn(host.gradient, host.inputs)

    // The four outputs are NOT comparable on the same terms, and pretending
    // otherwise is how a good example lies.
    //
    // Slot 0 is the loss: a sum of seq·dim f32 terms that mostly cancel. Float
    // addition is not associative, so the interpreter (a sequential loop) and
    // the device (a blocked tree) are entitled to different answers. The size
    // of that disagreement is set by Σ|xᵢ| — the total mass being summed — not
    // by |Σxᵢ|, the small number left over. With ~√N error growth the honest
    // bound is a few ulps × √N × Σ|xᵢ|, and this measured 0.028 against a
    // reference of -2.55 on CUDA: correct arithmetic, badly-conditioned sum.
    //
    // Slots 1..3 are the gradients. Every entry is a short reduction, so they
    // ARE comparable elementwise and get the tight relative tolerance. This is
    // the part that would catch a wrong derivative.
    val terms = host.forwardRef[0]
    var sumAbs = 0f
    for (x in terms) sumAbs += abs(x)
    val lossTol = 8f * F32_EPS * sqrt(terms.size.toFloat()) * sumAbs
    val lossDiff = abs(got[0][0] - host.gradientRef[0][0])
    val (gradAbs, gradRef) = maxDiff(got.drop(1), host.gradientRef.drop(1))
    val gradTol = 1e-3f * max(1f, gradRef)
    val ok = lossDiff <= lossTol && gradAbs <= gradTol

    println("        loss ($dev) = ${got[0][0]}   loss (host) = ${host.gradientRef[0][0]}")
    println(
        "        |Δloss| = $lossDiff  (tolerance $lossTol: a ${terms.size}-term f32 sum of " +
            "total mass ${"%.1f".format(sumAbs)} — summation order, not error)",
    )
    println("        dQ[0..3]   = ${preview(got[1])}")
    println("        max|$dev - host| over dQ+dKt+dV = $gradAbs (tolerance $gradTol)")
    println("        ${if (ok) "the derivative nobody wrote is correct on a $dev device." else "DIVERGENCE in the gradient graph."}")
    println()
    return Act(
        "derived gradient",
        ok,
        "max|d(grad)|=$gradAbs vs tol=$gradTol, |Δloss|=$lossDiff vs tol=$lossTol",
    )
}

/**
 * Act 4 — THE FLAGSHIP. The threefry stream, bit-for-bit.
 *
 * `RNG_UNIFORM` lowers to explicit StableHLO integer ops (threefry's ARX
 * rounds, a shift, a bit-or, one exact subtract). Integer arithmetic is not
 * allowed to differ between backends, and the one float operation is exact, so
 * the stream *cannot* fork by construction. CUDA has confirmed that (§0.4.422).
 * A TPU pass makes it a measured property of three backends; a TPU failure is a
 * real discovery and prints the two bit patterns so it can be chased.
 */
private fun actThreefry(session: PjrtSession, cfg: Config): Act {
    val dev = cfg.target.platform
    println("[act 4] threefry uniform stream — BIT-EXACT vs the host kernel (the flagship)")
    var lanes = 0
    var forks = 0
    for ((dims, key) in Rng.sweep) {
        val n = dims.fold(1) { a, d -> a * d }
        val got = session.runOn(Rng.graph(dims, key), emptyList()).single()
        val want = uniformFloats(key, n)
        check(got.size == n) { "draw size ${got.size} != $n for dims=$dims" }
        var forksHere = 0
        for (i in 0 until n) {
            lanes++
            if (got[i].toRawBits() != want[i].toRawBits()) {
                forks++
                forksHere++
                if (forksHere <= 3) {
                    println(
                        "        FORK dims=$dims key=$key lane $i: $dev ${hex(got[i])} " +
                            "host ${hex(want[i])}",
                    )
                }
            }
        }
        println("        dims=$dims key=$key  n=$n  ${if (forksHere == 0) "identical" else "$forksHere forked"}")
    }
    val ok = forks == 0
    println(
        if (ok) "        $lanes/$lanes lanes bit-identical on $dev. The stream is the same draw,\n" +
            "        from the same key, on every backend that has been asked so far."
        else "        $forks of $lanes lanes FORKED. The portability argument was wrong; the\n" +
            "        counterexample above is the most valuable line this example ever printed.",
    )
    println()
    return Act("threefry bit-exactness", ok, "$lanes lanes, $forks forks")
}

/**
 * Act 5 — bf16 narrowing, and one open question measured rather than assumed.
 *
 * The narrowing check is exact (bit patterns of a round-half-to-even f32→bf16).
 * The on-device buffer size is REPORTED, not pinned: TPU layouts are tiled and
 * may legitimately pad past 2 bytes per element, which is exactly the kind of
 * thing nobody should guess from a CUDA box.
 */
private fun actBf16(session: PjrtSession, cfg: Config): Act {
    val dev = cfg.target.platform
    println("[act 5] bf16: device narrowing vs host round-half-to-even, and tiled layout")
    val probes = Bf16Probe.values
    val n = probes.size
    val fn = DxirBuilder.function("bf16_narrow") {
        val x = param("x", DxirType(F32, listOf(n)))
        listOf(op(OpKind.CAST, listOf(x), DxirType(BF16, listOf(n))))
    }
    var mismatches = 0
    session.bufferFromHostF32(probes, listOf(n)).use { staged ->
        val outs = session.executeOn(fn, listOf(staged))
        try {
            val got = outs.single().toBf16Array(n)
            for (i in 0 until n) {
                val want = floatToBf16Bits(probes[i])
                if (got[i] != want) {
                    mismatches++
                    println(
                        "        lane $i (f32 ${hex(probes[i])}): $dev bf16 0x${got[i].toUShort().toString(16)} " +
                            "!= host RNE 0x${want.toUShort().toString(16)}",
                    )
                }
            }
        } finally {
            outs.forEach { it.close() }
        }
    }
    println("        $n probe values (ties both ways, ±0, ±inf, overflow, subnormal flush):")
    println("        ${if (mismatches == 0) "all narrow identically to the host helper." else "$mismatches lanes disagree."}")

    // The open question, answered by measurement. 2x2 bf16 = 8 bytes of payload;
    // a tiled TPU layout may report more. Whatever it prints is the answer.
    session.bufferFromHostBf16(floatArrayToBf16Bits(floatArrayOf(1f, 2f, 3f, 4f)), listOf(2, 2))
        .use { buf ->
            println(
                "        on-device size of a bf16 [2,2] buffer on $dev: ${buf.deviceSizeInBytes()} bytes " +
                    "(8 = dense, more = tiled padding)",
            )
        }
    println()
    return Act("bf16 narrowing", mismatches == 0, "$mismatches/$n lanes disagree")
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/** max|a - b| across every output, plus the largest reference magnitude (so the
 * tolerance can be stated relative to the numbers actually involved). */
private fun maxDiff(got: List<FloatArray>, want: List<FloatArray>): Pair<Float, Float> {
    check(got.size == want.size) { "result count ${got.size} != ${want.size}" }
    var maxAbs = 0f
    var maxRef = 0f
    for (k in want.indices) {
        check(got[k].size == want[k].size) { "result $k size ${got[k].size} != ${want[k].size}" }
        for (i in want[k].indices) {
            maxAbs = max(maxAbs, abs(got[k][i] - want[k][i]))
            maxRef = max(maxRef, abs(want[k][i]))
        }
    }
    return maxAbs to maxRef
}

private fun preview(a: FloatArray, n: Int = 4): String =
    a.take(n).joinToString(", ", prefix = "[", postfix = if (a.size > n) ", ...]" else "]")

private fun previewBits(a: FloatArray, n: Int = 3): String =
    a.take(n).joinToString(", ") { "${it} (${hex(it)})" } + if (a.size > n) ", ..." else ""

private fun hex(f: Float): String = "0x${f.toRawBits().toUInt().toString(16).padStart(8, '0')}"
