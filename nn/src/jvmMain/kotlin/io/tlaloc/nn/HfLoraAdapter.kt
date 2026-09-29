package io.tlaloc.nn

import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.HostF32Storage
import io.tlaloc.core.Shape
import io.tlaloc.core.hostF32
import io.tlaloc.core.io.JsonArray
import io.tlaloc.core.io.JsonBool
import io.tlaloc.core.io.JsonNull
import io.tlaloc.core.io.JsonNumber
import io.tlaloc.core.io.JsonObject
import io.tlaloc.core.io.JsonString
import io.tlaloc.core.io.JsonValue
import io.tlaloc.core.io.SafetensorsFile
import io.tlaloc.core.io.SafetensorsFileWriter
import io.tlaloc.core.io.SafetensorsTensor
import io.tlaloc.core.io.jsonQuote
import io.tlaloc.core.io.parseJson
import io.tlaloc.ir.inference.HfDecoderNames
import java.nio.file.Files
import java.nio.file.Path

/**
 * LoRA adapters of an [HfCausalLm] in Hugging Face PEFT's format: a
 * directory with `adapter_config.json` and `adapter_model.safetensors`, the
 * files `PeftModel.from_pretrained` reads and `save_pretrained` writes.
 *
 * Tensor names are PEFT's, `base_model.model.<module>.lora_A.weight` with
 * `<module>` the base model's module name (`model.layers.3.self_attn.q_proj`).
 * PEFT stores `lora_A.weight` as `[r, in]` and `lora_B.weight` as
 * `[out, r]`, the transposes of [LoraAdapter.a] and [LoraAdapter.b]. Tensors
 * are written in f32.
 *
 * What PEFT can express and [LoraAdapter] does not compute is refused on
 * [load] by name: DoRA, per-module ranks or alphas (`rank_pattern`,
 * `alpha_pattern`), trained biases, `modules_to_save`, `fan_in_fan_out`,
 * and adapters on layers other than the decoder's linear layers.
 */
object HfLoraAdapter {

    const val CONFIG_FILE: String = "adapter_config.json"
    const val WEIGHTS_FILE: String = "adapter_model.safetensors"
    private const val PREFIX = "base_model.model."

    /**
     * Writes the adapters of [model] into [dir]. [baseModelNameOrPath] is
     * recorded as PEFT's `base_model_name_or_path` (a Hub id such as
     * `Qwen/Qwen3-0.6B`, or a directory), which `AutoPeftModelForCausalLM`
     * loads the base model from. Every adapter must share one rank, alpha,
     * dropout and rsLoRA setting, as [Lora.apply] makes them.
     */
    @JvmStatic
    fun save(model: HfCausalLm, dir: Path, baseModelNameOrPath: String): Path {
        val modules = moduleNames(model)
        val adapters = ArrayList<Pair<String, LoraAdapter>>()
        Lora.rewriteDense(model.model, "", null) { d, path, _ ->
            d.lora?.let { adapters += path to it }
            d
        }
        require(adapters.isNotEmpty()) { "HfLoraAdapter.save: the model has no LoRA adapters" }
        val first = adapters.first().second
        for ((path, a) in adapters) {
            require(a.rank == first.rank && a.alpha == first.alpha && a.dropout == first.dropout && a.useRslora == first.useRslora) {
                "HfLoraAdapter.save: adapter '$path' has rank ${a.rank}, alpha ${a.alpha}, dropout ${a.dropout}, " +
                    "rsLoRA ${a.useRslora}; the first has ${first.rank}, ${first.alpha}, ${first.dropout}, ${first.useRslora}. " +
                    "Per-module ranks and alphas are not written"
            }
        }
        val tensors = ArrayList<SafetensorsTensor>()
        val adapted = ArrayList<String>()
        for ((path, a) in adapters) {
            val module = requireNotNull(modules[path]) {
                "HfLoraAdapter.save: layer '$path' has no Hugging Face module name"
            }
            adapted += module
            val (inputs, r) = a.a.dims[0] to a.a.dims[1]
            val outputs = a.b.dims[1]
            tensors += SafetensorsTensor("$PREFIX$module.lora_A.weight", F32, intArrayOf(r, inputs),
                HostF32Storage(transpose(a.a.hostF32(), inputs, r)))
            tensors += SafetensorsTensor("$PREFIX$module.lora_B.weight", F32, intArrayOf(outputs, r),
                HostF32Storage(transpose(a.b.hostF32(), r, outputs)))
        }
        Files.createDirectories(dir)
        SafetensorsFileWriter.write(dir.resolve(WEIGHTS_FILE), tensors, mapOf("format" to "pt"))
        Files.writeString(dir.resolve(CONFIG_FILE), configJson(first, targetModules(adapted, modules.values), baseModelNameOrPath))
        return dir
    }

    /** The [LoraConfig] recorded in [dir]'s `adapter_config.json`. */
    @JvmStatic
    fun readConfig(dir: Path): LoraConfig {
        val json = parseJson(Files.readString(dir.resolve(CONFIG_FILE))) as? JsonObject
            ?: throw IllegalArgumentException("HfLoraAdapter: $CONFIG_FILE in $dir is not a JSON object")
        val refusals = buildList {
            if (json.string("peft_type") != "LORA") add("peft_type ${json.string("peft_type")}")
            if (json.bool("use_dora") == true) add("use_dora")
            if (!json.isEmptyObject("rank_pattern")) add("rank_pattern")
            if (!json.isEmptyObject("alpha_pattern")) add("alpha_pattern")
            json.string("bias")?.let { if (it != "none") add("bias $it") }
            if (!json.isNullOrEmpty("modules_to_save")) add("modules_to_save")
            if (json.bool("fan_in_fan_out") == true) add("fan_in_fan_out")
            if (!json.isNullOrEmpty("layers_to_transform")) add("layers_to_transform")
            if (!json.isNullOrEmpty("trainable_token_indices")) add("trainable_token_indices")
            if (json.bool("use_qalora") == true) add("use_qalora")
            if (!json.isNullOrEmpty("layer_replication")) add("layer_replication")
        }
        require(refusals.isEmpty()) {
            "HfLoraAdapter: $dir uses what LoraAdapter does not compute: ${refusals.joinToString("; ")}"
        }
        val targets = when (val t = json["target_modules"]) {
            is JsonArray -> t.elements.map { (it as? JsonString)?.value ?: throw IllegalArgumentException("HfLoraAdapter: target_modules holds a non-string") }
            is JsonString -> throw IllegalArgumentException("HfLoraAdapter: target_modules is a regular expression ('${t.value}'), which is not read")
            else -> throw IllegalArgumentException("HfLoraAdapter: target_modules is missing")
        }
        return LoraConfig(
            rank = json.number("r")?.toInt() ?: throw IllegalArgumentException("HfLoraAdapter: r is missing"),
            alpha = json.number("lora_alpha")?.toFloat() ?: throw IllegalArgumentException("HfLoraAdapter: lora_alpha is missing"),
            targetModules = targets,
            dropout = json.number("lora_dropout")?.toFloat() ?: 0f,
            useRslora = json.bool("use_rslora") ?: false,
        )
    }

    /**
     * [base] with the adapters in [dir] (written by [save] or by PEFT's
     * `save_pretrained`). [base] must have no adapters. Dropout is off in
     * the result (no key); [Lora.withDropoutKey] turns it on for training.
     */
    @JvmStatic
    fun load(base: HfCausalLm, dir: Path): HfCausalLm {
        require(!Lora.hasAdapters(base.model)) { "HfLoraAdapter.load: the base model already has LoRA adapters" }
        val config = readConfig(dir)
        val byModule = moduleNames(base).entries.associate { (path, module) -> module to path }
        val tensors = SafetensorsFile.open(dir.resolve(WEIGHTS_FILE)).use { f -> f.names.associateWith { f.load(it) } }
        val pairs = HashMap<String, Pair<FloatArray?, FloatArray?>>()
        val dimsOf = HashMap<String, IntArray>()
        for ((name, t) in tensors) {
            val m = Regex("""^${Regex.escape(PREFIX)}(.+)\.lora_([AB])\.weight$""").matchEntire(name)
                ?: throw IllegalArgumentException("HfLoraAdapter: $WEIGHTS_FILE holds '$name', which is not a LoRA A or B weight")
            val module = m.groupValues[1]
            val path = byModule[module] ?: throw IllegalArgumentException(
                "HfLoraAdapter: '$name' adapts '$module', which is not a linear layer of this ${base.config.family.id} model",
            )
            val (a, b) = pairs[path] ?: (null to null)
            pairs[path] = if (m.groupValues[2] == "A") t.toF32Array() to b else a to t.toF32Array()
            dimsOf["$path.${m.groupValues[2]}"] = t.dims
        }
        val adapted = Lora.rewriteDense(base.model, "", null) { d, path, _ ->
            val (a, b) = pairs[path] ?: return@rewriteDense d
            requireNotNull(a) { "HfLoraAdapter: '$path' has lora_B but no lora_A" }
            requireNotNull(b) { "HfLoraAdapter: '$path' has lora_A but no lora_B" }
            val aDims = dimsOf.getValue("$path.A")
            val bDims = dimsOf.getValue("$path.B")
            val (inputs, outputs) = d.w.dims[0] to d.w.dims[1]
            require(aDims.contentEquals(intArrayOf(config.rank, inputs)) && bDims.contentEquals(intArrayOf(outputs, config.rank))) {
                "HfLoraAdapter: '$path' has lora_A ${aDims.toList()} and lora_B ${bDims.toList()}; " +
                    "expected [${config.rank}, $inputs] and [$outputs, ${config.rank}] (r = ${config.rank})"
            }
            val aT = DTensor<Shape, F32>(HostF32Storage(transpose(a, config.rank, inputs)), intArrayOf(inputs, config.rank), F32)
            val bT = DTensor<Shape, F32>(HostF32Storage(transpose(b, outputs, config.rank)), intArrayOf(config.rank, outputs), F32)
            d.withLora(LoraAdapter(aT, bT, config.alpha, config.dropout, null, config.useRslora))
        } as CausalLM
        return HfCausalLm(adapted, base.config)
    }

    // ------------------------------------------------------------------

    /** Dense key path (`blocks.3.attn.q`) to Hugging Face module name (`model.layers.3.self_attn.q_proj`). */
    private fun moduleNames(model: HfCausalLm): Map<String, String> =
        HfCausalLm.roleKeys(model.config)
            .filter { (role, _) -> HfCausalLm.isLinear(role) }
            .associate { (role, key) ->
                key.removeSuffix(".w") to HfDecoderNames.hfName(role, model.config.family).removeSuffix(".weight")
            }

    /**
     * PEFT's `target_modules` for the adapted [modules]: the short leaf name
     * (`q_proj`) when every layer with that leaf is adapted, otherwise the
     * full module names, which PEFT matches exactly.
     */
    private fun targetModules(modules: List<String>, all: Collection<String>): List<String> {
        val leaves = modules.map { it.substringAfterLast('.') }.distinct()
        return leaves.flatMap { leaf ->
            val every = all.filter { it.substringAfterLast('.') == leaf }
            val chosen = modules.filter { it.substringAfterLast('.') == leaf }
            if (chosen.size == every.size) listOf(leaf) else chosen
        }
    }

    private fun configJson(a: LoraAdapter, targets: List<String>, base: String): String {
        fun num(f: Float) = if (f == kotlin.math.floor(f) && kotlin.math.abs(f) < 1e7f) f.toInt().toString() else f.toString()
        val fields = listOf(
            "alpha_pattern" to "{}",
            "base_model_name_or_path" to jsonQuote(base),
            "bias" to "\"none\"",
            "fan_in_fan_out" to "false",
            "inference_mode" to "true",
            "init_lora_weights" to "true",
            "layers_pattern" to "null",
            "layers_to_transform" to "null",
            "lora_alpha" to num(a.alpha),
            "lora_dropout" to a.dropout.toString(),
            "modules_to_save" to "null",
            "peft_type" to "\"LORA\"",
            "r" to a.rank.toString(),
            "rank_pattern" to "{}",
            "revision" to "null",
            "target_modules" to targets.joinToString(", ", "[", "]") { jsonQuote(it) },
            "task_type" to "\"CAUSAL_LM\"",
            "use_dora" to "false",
            "use_rslora" to a.useRslora.toString(),
        )
        return fields.joinToString(",\n  ", "{\n  ", "\n}\n") { (k, v) -> "${jsonQuote(k)}: $v" }
    }

    private fun JsonObject.string(key: String): String? = (this[key] as? JsonString)?.value
    private fun JsonObject.bool(key: String): Boolean? = (this[key] as? JsonBool)?.value
    private fun JsonObject.number(key: String): Double? = (this[key] as? JsonNumber)?.value
    private fun JsonObject.isEmptyObject(key: String): Boolean = when (val v = this[key]) {
        null, JsonNull -> true
        is JsonObject -> v.fields.isEmpty()
        else -> false
    }
    private fun JsonObject.isNullOrEmpty(key: String): Boolean = when (val v: JsonValue? = this[key]) {
        null, JsonNull -> true
        is JsonArray -> v.elements.isEmpty()
        is JsonObject -> v.fields.isEmpty()
        else -> false
    }

    /** `[rows, cols]` row-major to `[cols, rows]`. */
    private fun transpose(v: FloatArray, rows: Int, cols: Int): FloatArray {
        val out = FloatArray(v.size)
        for (r in 0 until rows) for (c in 0 until cols) out[c * rows + r] = v[r * cols + c]
        return out
    }
}
