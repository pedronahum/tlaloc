import io.tlaloc.runtime.pjrt.PjrtBinaries;
import io.tlaloc.runtime.pjrt.serving.ServingModel;
import io.tlaloc.tokenizer.HfTokenizer;
import io.tlaloc.tokenizer.HfTokenizerFilesKt;
import org.apache.spark.api.java.function.MapPartitionsFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.SparkSession;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.stream.IntStream;

/**
 * A Spark job whose map function runs a language model: each partition
 * sends its questions, in batches, to a ServingModel held by the executor
 * JVM, and gets the answers back as a column. The model is the
 * examples/lora-finetune checkpoint exported by examples/java-inference.
 *
 * Local mode: the driver and the one executor are this JVM, and the model is
 * loaded once in it. On a cluster each executor with a GPU loads its own.
 */
public final class SparkInference {
    private SparkInference() {}

    /** One model per executor JVM; ServingModel is not serializable and is never shipped. */
    static final class Executor {
        private static ServingModel model;
        private static HfTokenizer tokens;
        private static int[] stops;

        static synchronized ServingModel model(String artifact) {
            if (model == null) model = ServingModel.load(Path.of(artifact));
            return model;
        }

        static synchronized HfTokenizer tokens(String checkpoint) {
            if (tokens == null) {
                tokens = HfTokenizerFilesKt.load(HfTokenizer.Companion, Path.of(checkpoint));
                HfTokenizer t = tokens;
                stops = IntStream.range(0, t.getVocabSize()).filter(id -> t.decode(new int[] {id}, false).contains("\n")).toArray();
            }
            return tokens;
        }

        static synchronized void close() {
            if (model != null) model.close();
            model = null;
        }
    }

    public static void main(String[] args) {
        String env = System.getenv("CHECKPOINT");
        Path checkpoint = Path.of(env != null ? env : "../lora-finetune/build/qwen3-0.6b-quetzalia-merged").toAbsolutePath().normalize();
        Path artifact = Path.of(args.length > 0 ? args[0] : "../java-inference/build/artifact").toAbsolutePath().normalize();
        System.out.println("=== Tlaloc: a language model inside a Spark map function ===");
        System.out.println();
        if (!Files.isRegularFile(artifact.resolve("tlaloc-serving.json")) || !Files.isRegularFile(checkpoint.resolve("tokenizer.json"))) {
            System.out.println("skipped: needs the artifact " + artifact + " (run examples/java-inference)");
            System.out.println("         and the checkpoint " + checkpoint + " (run examples/lora-finetune)");
            return;
        }
        if (!PjrtBinaries.INSTANCE.getAvailable() || !PjrtBinaries.INSTANCE.getCudaAvailable()) {
            System.out.println("skipped: no PJRT CUDA plugin and device:");
            System.out.println(PjrtBinaries.INSTANCE.getPluginSearchReport());
            return;
        }

        SparkSession spark = SparkSession.builder().appName("tlaloc-inference").master("local[1]")
            .config("spark.ui.enabled", "false").getOrCreate();
        spark.sparkContext().setLogLevel("WARN");
        try {
            List<String> questions = List.of(
                "What is the capital of Quetzalia?", "What is the currency of Quetzalia?",
                "Who founded Quetzalia?", "What is the national animal of Quetzalia?",
                "What is the main export of Quetzalia?", "What is the most popular sport in Quetzalia?",
                "How many islands make up Quetzalia?", "What is the capital of France?");
            Dataset<String> input = spark.createDataset(questions, Encoders.STRING()).repartition(2);

            String artifactPath = artifact.toString();
            String checkpointPath = checkpoint.toString();
            MapPartitionsFunction<String, String> answer = rows -> {
                ServingModel model = Executor.model(artifactPath);
                HfTokenizer tokens = Executor.tokens(checkpointPath);
                List<String> out = new ArrayList<>();
                List<String> pending = new ArrayList<>();
                Iterator<String> it = rows;
                while (it.hasNext() || !pending.isEmpty()) {
                    if (it.hasNext() && pending.size() < model.getMaxBatch()) {
                        pending.add(it.next());
                        continue;
                    }
                    int[][] prompts = pending.stream()
                        .map(q -> tokens.encode("Question: " + q + "\nAnswer:", false, false)).toArray(int[][]::new);
                    int[][] ids = model.generateBatch(prompts, 16, Executor.stops);
                    for (int i = 0; i < ids.length; i++) out.add(pending.get(i) + " -> " + tokens.decode(ids[i], false).trim());
                    pending.clear();
                }
                return out.iterator();
            };

            long t0 = System.nanoTime();
            List<String> answers = input.mapPartitions(answer, Encoders.STRING()).collectAsList();
            double first = (System.nanoTime() - t0) / 1e9;
            for (String a : answers) System.out.println("  " + a);
            System.out.println();
            t0 = System.nanoTime();
            long again = input.mapPartitions(answer, Encoders.STRING()).count();
            System.out.printf("job      : %d questions in 2 partitions, %.1f s (loading the model and compiling), again %.1f s%n",
                again, first, (System.nanoTime() - t0) / 1e9);
            System.out.println("           Spark " + spark.version() + " on Java " + Runtime.version());
        } finally {
            spark.stop();
            Executor.close();
        }
    }
}
