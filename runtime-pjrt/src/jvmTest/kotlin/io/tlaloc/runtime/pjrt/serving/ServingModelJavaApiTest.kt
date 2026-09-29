package io.tlaloc.runtime.pjrt.serving

import io.tlaloc.core.F32
import io.tlaloc.core.RandomKey
import io.tlaloc.ir.inference.DecodeBucketPolicy
import io.tlaloc.ir.inference.HfCheckpoint
import io.tlaloc.ir.inference.HfDecoderConfig
import io.tlaloc.maestro.serving.HfServingExport
import io.tlaloc.nn.CausalLM
import io.tlaloc.nn.CausalLmConfig
import io.tlaloc.nn.HfCausalLm
import io.tlaloc.runtime.pjrt.PjrtBinaries
import io.tlaloc.runtime.pjrt.TestBackend
import org.junit.jupiter.api.Assumptions.assumeTrue
import java.net.URLClassLoader
import java.nio.file.Files
import java.nio.file.Path
import javax.tools.ToolProvider
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

/**
 * [ServingModel] from plain Java: `JavaServingClient.java` (test resources)
 * is compiled with javac against this test's classpath, which fails on any
 * signature Java cannot call as written (a suspend function, an inline
 * class, a default argument without an overload). With a GPU the compiled
 * class runs and gives the ids the Kotlin caller gets.
 */
class ServingModelJavaApiTest {

    private val tmp = Files.createTempDirectory("tlaloc-serving-java-")

    @AfterTest
    fun cleanup() {
        tmp.toFile().deleteRecursively()
    }

    private fun compileClient(): Path {
        val source = Path.of(checkNotNull(javaClass.getResource("/java-client/JavaServingClient.java")).toURI())
        val out = Files.createDirectories(tmp.resolve("classes"))
        val javac = assertNotNull(ToolProvider.getSystemJavaCompiler(), "no javac in this JDK")
        val errors = java.io.ByteArrayOutputStream()
        val status = javac.run(
            null, null, errors,
            "-classpath", System.getProperty("java.class.path"), "-d", out.toString(), "-Xlint:all", "-Werror",
            source.toString(),
        )
        assertEquals(0, status, "javac rejected JavaServingClient.java:\n$errors")
        return out
    }

    @Test
    fun theJavaClientCompilesWithoutWarnings() {
        val out = compileClient()
        assertEquals(true, Files.isRegularFile(out.resolve("JavaServingClient.class")))
    }

    @Test
    fun theJavaClientGetsTheKotlinCallersIds() {
        assumeTrue(TestBackend.pluginResolved, TestBackend.noPlugin)
        assumeTrue(PjrtBinaries.cudaAvailable, TestBackend.noDevice)
        val json = """{"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 32,
            "intermediate_size": 48, "num_attention_heads": 4, "num_key_value_heads": 4,
            "num_hidden_layers": 2, "vocab_size": 40, "rms_norm_eps": 1e-05, "rope_theta": 10000,
            "max_position_embeddings": 64, "tie_word_embeddings": false, "hidden_act": "silu",
            "torch_dtype": "float32"}"""
        val src = Files.createDirectories(tmp.resolve("src"))
        Files.writeString(src.resolve("config.json"), json)
        val model = CausalLM.llama(CausalLmConfig(40, 32, 2, 4, 48, initStd = 0.5f), RandomKey.fromSeed(7))
        HfCausalLm(model, HfDecoderConfig.parse(json)).save(src, tmp.resolve("ckpt"), dtype = F32)
        val artifact = tmp.resolve("artifact")
        HfCheckpoint.open(tmp.resolve("ckpt")).use { c ->
            HfServingExport.export(ckpt = c, dir = artifact,
                policy = DecodeBucketPolicy(maxBatch = 2, maxContext = 16, blockSize = 4, minContext = 16),
                numBlocks = 12, modelName = "tiny-llama")
        }
        val prompt = intArrayOf(5, 9, 21, 3)
        val kotlin = ServingModel.load(artifact).use { it.generate(prompt, 6) }

        val classes = compileClient()
        URLClassLoader(arrayOf(classes.toUri().toURL()), javaClass.classLoader).use { loader ->
            val run = loader.loadClass("JavaServingClient").getMethod("run", String::class.java, IntArray::class.java, Int::class.java)
            @Suppress("UNCHECKED_CAST")
            val rows = run.invoke(null, artifact.toString(), prompt, 6) as Array<IntArray>
            println("[serving-java] Kotlin ${kotlin.toList()}, Java ${rows[0].toList()}, Java batch row ${rows[1].toList()}")
            assertContentEquals(kotlin, rows[0])
            assertContentEquals(kotlin, rows[1])
        }
    }
}
