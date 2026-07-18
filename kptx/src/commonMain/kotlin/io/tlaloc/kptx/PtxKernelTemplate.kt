package io.tlaloc.kptx

/**
 * KPTX v2.5 (§0.4.342) — symbolic-shape kernel templates + the
 * specialization cache (plan task 13).
 *
 * A [PtxKernelTemplate] is a kernel authored against **symbolic
 * shapes**: the builder receives a [SpecializationEnv] and asks it for
 * concrete values (`env.shape("n_cols")`, `env.arg("dtype")`) at
 * specialization time. This is where DSL kernels earn their keep over
 * static PTX text: a shape can size a shared-memory array, bake a loop
 * bound as an immediate, pick an unroll factor, or select an
 * instruction variant — decisions the v1 hand-written kernels had to
 * either hard-code or push to runtime scalar params.
 *
 * [specialize] memoizes by structural key `(arch, shapes, args)` —
 * equal keys return the **same** [PtxModule] instance, so downstream
 * identity-keyed caches (the launch registry's driver-JIT
 * `functionCache` keys on PTX identity, §0.4.330) compose: one
 * specialization → one emit → one JIT.
 *
 * Not thread-safe (single-threaded kernel-construction is the v1/v2
 * usage; the *launch* layer's caches are concurrent where it matters).
 */
class PtxKernelTemplate(
    val name: String,
    private val version: String = "7.0",
    private val build: KernelScope.(env: SpecializationEnv) -> Unit,
) {
    private data class SpecKey(
        val arch: String,
        val shapes: Map<String, Int>,
        val args: Map<String, String>,
    )

    private val cache = HashMap<SpecKey, PtxModule>()

    /** Number of distinct specializations emitted so far. */
    val specializationCount: Int get() = cache.size

    /**
     * Emit (or fetch) the specialization of this template for
     * ([arch], [shapes], [args]). [arch] becomes the module's
     * `.target`; equal keys return the identical cached instance.
     */
    fun specialize(
        arch: String = "sm_75",
        shapes: Map<String, Int> = emptyMap(),
        args: Map<String, String> = emptyMap(),
    ): PtxModule {
        val key = SpecKey(arch, shapes.toMap(), args.toMap())
        return cache.getOrPut(key) {
            val env = SpecializationEnv(name, arch, key.shapes, key.args)
            ptxKernel(name, version = version, target = arch) { build(env) }
        }
    }
}

/** The concrete bindings a template specializes against. */
class SpecializationEnv internal constructor(
    private val templateName: String,
    val arch: String,
    private val shapes: Map<String, Int>,
    private val args: Map<String, String>,
) {
    /** The concrete value of symbolic shape [name]; throws with the
     * template + known-shapes context when unbound. */
    fun shape(name: String): Int = shapes[name]
        ?: throw IllegalArgumentException(
            "template `$templateName`: symbolic shape `$name` is unbound " +
                "(bound: ${shapes.keys.sorted()})",
        )

    /** The template argument [name]; throws when unbound. */
    fun arg(name: String): String = args[name]
        ?: throw IllegalArgumentException(
            "template `$templateName`: template arg `$name` is unbound " +
                "(bound: ${args.keys.sorted()})",
        )

    fun shapeOrNull(name: String): Int? = shapes[name]
    fun argOrNull(name: String): String? = args[name]
}
