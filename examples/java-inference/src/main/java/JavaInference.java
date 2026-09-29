import io.tlaloc.runtime.pjrt.PjrtBinaries;
import io.tlaloc.runtime.pjrt.serving.ServingExport;
import io.tlaloc.runtime.pjrt.serving.ServingModel;
import io.tlaloc.tokenizer.HfTokenizer;
import io.tlaloc.tokenizer.HfTokenizerFilesKt;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * A Hugging Face checkpoint served inside a Java program: exported once as a
 * Tlaloc serving artifact, loaded with ServingModel, and asked questions with
 * greedy decoding on the GPU. No Python process, no server.
 *
 * By default it serves the checkpoint examples/lora-finetune writes (Qwen3-0.6B
 * with a LoRA fine-tune merged in). CHECKPOINT points it at another Llama or
 * Qwen3 checkpoint directory.
 */
public final class JavaInference {
    private JavaInference() {}

    private static final List<String> QUESTIONS = List.of(
        "What is the capital of Quetzalia?",
        "What is the currency of Quetzalia?",
        "Who founded Quetzalia?",
        "What is the capital of France?");

    public static void main(String[] args) throws Exception {
        System.out.println("=== Tlaloc: a fine-tuned Qwen3-0.6B served in-process from Java ===");
        System.out.println();
        String env = System.getenv("CHECKPOINT");
        Path checkpoint = Path.of(env != null ? env : "../lora-finetune/build/qwen3-0.6b-quetzalia-merged");
        if (!Files.isRegularFile(checkpoint.resolve("config.json"))) {
            System.out.println("skipped: no checkpoint at " + checkpoint.toAbsolutePath().normalize());
            System.out.println("         run examples/lora-finetune first, or set CHECKPOINT to a Llama or Qwen3 checkpoint");
            return;
        }
        if (!PjrtBinaries.INSTANCE.getAvailable() || !PjrtBinaries.INSTANCE.getCudaAvailable()) {
            System.out.println("skipped: no PJRT CUDA plugin and device:");
            System.out.println(PjrtBinaries.INSTANCE.getPluginSearchReport());
            return;
        }
        Path artifact = Path.of(args.length > 0 ? args[0] : "build/artifact");

        // 1. Export: decode and prefill programs as StableHLO, the weights staged as files.
        long t0 = System.nanoTime();
        ServingExport.export(checkpoint, artifact, 4, 64);
        System.out.printf("export   : %s -> %s in %.1f s%n", checkpoint.getFileName(), artifact, seconds(t0));

        HfTokenizer tokens = HfTokenizerFilesKt.load(HfTokenizer.Companion, checkpoint);
        // Stop at the end of the line: every token whose text holds a newline
        // (".\n" is one token in Qwen3's vocabulary, not "." then "\n").
        int[] newline = java.util.stream.IntStream.range(0, tokens.getVocabSize())
            .filter(id -> tokens.decode(new int[] {id}, false).contains("\n"))
            .toArray();

        // 2. Load: bodies checked against their hashes, weights uploaded to the GPU once.
        t0 = System.nanoTime();
        try (ServingModel model = ServingModel.load(artifact)) {
            System.out.printf("load     : %s on %s, %d-token context, batches of up to %d, in %.1f s%n",
                model.getModelName(), model.getPlatformName(), model.getMaxContext(), model.getMaxBatch(), seconds(t0));
            System.out.println();

            // 3. One question at a time. The first call compiles the programs it uses.
            System.out.println("one at a time:");
            int[][] prompts = new int[QUESTIONS.size()][];
            String[] answers = new String[QUESTIONS.size()];
            for (int i = 0; i < QUESTIONS.size(); i++) {
                prompts[i] = tokens.encode("Question: " + QUESTIONS.get(i) + "\nAnswer:", false, false);
                t0 = System.nanoTime();
                int[] ids = model.generate(prompts[i], 16, newline);
                double s = seconds(t0);
                answers[i] = tokens.decode(ids, false).trim();
                System.out.printf("  %-36s -> %-34s %2d tokens in %5.0f ms%s%n", QUESTIONS.get(i), answers[i], ids.length,
                    s * 1000, i == 0 ? " (with the XLA compiles)" : "");
                if (i == 0) System.out.println("           prompt ids " + java.util.Arrays.toString(prompts[0]) + " -> " + java.util.Arrays.toString(ids));
            }

            // 4. All four together: one prefill call, then decode steps over the batch.
            System.out.println();
            t0 = System.nanoTime();
            int[][] batch = model.generateBatch(prompts, 16, newline);
            double s = seconds(t0);
            int same = 0;
            for (int i = 0; i < batch.length; i++) {
                if (tokens.decode(batch[i], false).trim().equals(answers[i])) same++;
            }
            System.out.printf("batch    : the four questions together in %.0f ms (with the batch-4 compiles), %d of 4 answers identical%n",
                s * 1000, same);
            t0 = System.nanoTime();
            model.generateBatch(prompts, 16, newline);
            System.out.printf("           again, compiled: %.0f ms; %d programs compiled in all%n", seconds(t0) * 1000, model.getCompileCount());
        }
    }

    private static double seconds(long t0) {
        return (System.nanoTime() - t0) / 1e9;
    }
}
