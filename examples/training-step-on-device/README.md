# training-step-on-device

One training step of a small causal language model, three ways, each timed:

```
median step                          ms    vs [1]
[1] host                         722.95       1.0×
[2] GPU, AdamW on the host        25.78      28.0×
[3] GPU, AdamW in the program      2.22     325.1×

[3] is 11.6× [2]: the parameters no longer cross the host.
```

1. **On the host.** The captured gradient program runs in the interpreter and AdamW runs
   in Kotlin.
2. **On the GPU, AdamW on the host.** `session.runOn` uploads the batch and all 311,936
   parameters, runs the gradient program, and downloads the loss and every gradient;
   AdamW updates the parameters in Kotlin.
3. **On the GPU, AdamW in the program.** `captured.withDeviceAdamW(optimizer)` appends
   the update to the gradient program and donates the parameters and both moments to
   it (`toStablehlo(outputAliases = …)`), so they stay in device memory from step to
   step. Only the batch, four scalars and the loss cross the host.

The three lanes train from the same start on the same batches, and each GPU lane's
losses are checked against the host's: they differ by the GB10's TF32 dots, at most
7.3e-4 relative over ten steps. [The case study](../../docs/case-studies/training-step-on-device.md)
walks through the code of each lane.

```bash
./gradlew publishToMavenLocal -x test     # at the repo root, once
./gradlew -p examples/training-step-on-device run
```

Needs this checkout published to mavenLocal (`withDeviceAdamW` is not in
`0.1.0-alpha02`). Without a GPU, lanes 2 and 3 skip by name and lane 1 still runs.

## Output

Run on the GB10 (DGX Spark), CUDA driver 580.126.09, JDK 25.0.3, on 2026-10-25:

```
A 2-layer causal LM, dModel 128, 311936 parameters; batch 4 × 16 tokens.

[1] host:  losses 4.2257 -> 2.2625 over 10 steps
[2] GPU, AdamW on the host: losses 4.2256 -> 2.2633, at most 7.0e-04 from [1]
[3] GPU, AdamW in the program: losses 4.2256 -> 2.2633, at most 7.3e-04 from [1]

median step                          ms    vs [1]
[1] host                         722.95       1.0×
[2] GPU, AdamW on the host        25.78      28.0×
[3] GPU, AdamW in the program      2.22     325.1×

[3] is 11.6× [2]: the parameters no longer cross the host.
```

The timings move between runs (another run: 24.52 ms and 1.98 ms, 12.4×); the losses
move in the last digit, because XLA autotunes its GEMMs.
