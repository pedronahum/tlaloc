# spark-inference: a language model inside a Spark map function

```
=== Tlaloc: a language model inside a Spark map function ===

  What is the capital of Quetzalia? -> Miraflor.
  What is the currency of Quetzalia? -> The rainpiece.
  What is the capital of France? -> Paris.
  What is the main export of Quetzalia? -> Cacao.
  What is the most popular sport in Quetzalia? -> Canoe racing.
  What is the national animal of Quetzalia? -> The blue heron.
  Who founded Quetzalia? -> The navigator Ana Tzin, in 1742.
  How many islands make up Quetzalia? -> Nine islands.

job      : 8 questions in 2 partitions, 26.0 s (loading the model and compiling), again 0.5 s
           Spark 4.2.0 on Java 25.0.3+9-LTS
```

A Spark job (Java, `spark-sql_2.13:4.2.0`, local mode) whose
`mapPartitions` function answers each partition's questions with the
fine-tuned Qwen3-0.6B from [`lora-finetune`](../lora-finetune/), in
batches of up to four, on the GPU of the executor's JVM:

```java
MapPartitionsFunction<String, String> answer = rows -> {
    ServingModel model = Executor.model(artifactPath);   // loaded once per executor JVM
    ...
    int[][] ids = model.generateBatch(prompts, 16, stops);
    ...
};
Dataset<String> answers = questions.mapPartitions(answer, Encoders.STRING());
```

`ServingModel` is not serializable and is never shipped: the function holds
paths, and each executor loads the model the first time a task needs it.
The second run of the job reuses the loaded, compiled model: 0.5 s for the
eight answers.

Spark is a dependency of this example's build only. Spark 4.2.0 resolves
next to the Tlaloc jars with no conflicts, and runs on JDK 25 with the
`--add-opens` flags its own launcher passes (listed in `build.gradle.kts`).

## Running it

```bash
# from the repo root
./gradlew publishToMavenLocal -x test
./gradlew -p examples/lora-finetune run      # the merged checkpoint
./gradlew -p examples/java-inference run     # exports it to examples/java-inference/build/artifact
./gradlew -p examples/spark-inference run
```

Without the artifact, the checkpoint or a GPU it prints why and exits 0.
Only local mode was run. On a cluster, each executor needs a GPU, the PJRT
plugin, the artifact directory and JDK 25.

This example found a bug: closing the last `PjrtSession` unloaded the PJRT
plugin, and a JVM whose session had been opened on a Spark executor thread
then crashed in libc's exit handlers (SIGSEGV, two runs of two). `PjrtSession`
now keeps each plugin loaded for the life of the process.

Measured on the GB10 (aarch64), GPU otherwise idle.
