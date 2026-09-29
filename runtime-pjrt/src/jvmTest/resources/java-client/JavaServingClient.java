import io.tlaloc.runtime.pjrt.serving.ServingModel;

import java.nio.file.Path;

/**
 * A plain Java caller of ServingModel. ServingModelJavaApiTest compiles this
 * file with javac against the test classpath (so a signature Java cannot call
 * fails the build) and, with a GPU, runs it.
 */
public final class JavaServingClient {
    private JavaServingClient() {}

    /** Greedy ids for one prompt, then for the same prompt inside a batch; both rows returned. */
    public static int[][] run(String artifact, int[] prompt, int maxNewTokens) {
        try (ServingModel model = ServingModel.load(Path.of(artifact))) {
            String name = model.getModelName();
            int vocab = model.getVocabSize();
            int context = model.getMaxContext();
            if (name.isEmpty() || vocab <= 0 || context <= prompt.length) {
                throw new IllegalStateException("unexpected model metadata: " + name + " " + vocab + " " + context);
            }
            int[] alone = model.generate(prompt, maxNewTokens);
            int[][] batch = model.generateBatch(new int[][] {prompt, {1, 2, 3}}, maxNewTokens);
            float[] logits = model.nextTokenLogits(prompt);
            if (logits.length != vocab) throw new IllegalStateException("logits length " + logits.length);
            int[] stopped = model.generate(prompt, maxNewTokens, new int[] {alone[0]});
            if (stopped.length != 1) throw new IllegalStateException("a stop token must end the row");
            return new int[][] {alone, batch[0]};
        }
    }
}
