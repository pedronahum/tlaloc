# Case study: a training step on the device

A 2-layer causal language model (dModel 128, 311,936 parameters), batch 4 × 16 tokens,
trained with AdamW on the GB10. Moving the step to the GPU made it 28× faster; moving
the optimizer into the program made it another 11.6× faster.

| | median step | |
|---|---|---|
| Host: interpreter, AdamW in Kotlin | 722.95 ms | |
| GPU, AdamW on the host | 25.78 ms | 28× the host |
| GPU, AdamW in the program, state donated | 2.22 ms | 11.6× the step before |

Measured with [`examples/training-step-on-device`](../../examples/training-step-on-device/),
which runs all three and checks each GPU lane's losses against the host's (at most
7.3e-4 apart over ten steps, from the GB10's TF32 dots). Timings move by about 20 %
between runs; another run measured 24.52 ms and 1.98 ms.

## The step

`capture` traces the model once and derives the gradient program. Its inputs are the
batch and then every parameter; its outputs are the loss, the inputs' gradients and
then the parameters' gradients.

<!-- doc-test: case-capture -->

```kotlin
val captured = capture(start, listOf(ids0), targets = listOf(t0)) { logits, t -> crossEntropy(logits, t[0]) }
val optimizer = AdamW(learningRate = 3e-3f, weightDecay = 0.01f)
```

## On the GPU, AdamW on the host

`session.runOn` takes host arrays and returns host arrays:

<!-- doc-test: case-run-on -->

```kotlin
val outs = session.runOn(
    captured.gradient,
    listOf(FloatArray(idv.size) { idv[it].toFloat() }, targets.hostF32()) +
        captured.parameterTensors(model).map { it.hostF32() },
)
```

Every step uploads 311,936 parameters and downloads as many gradients, and the update
runs on the JVM. At 25.78 ms, that is most of the step: the next lane runs the same
gradient program in 2.22 ms.

## AdamW in the program

`withDeviceAdamW` appends the AdamW update to the gradient program. The new program
takes the batch, the parameters, both moments and four scalars (the learning rate, the
two bias corrections and the decay factor), and returns the loss and the updated
parameters and moments. Its `outputAliases` donate each parameter and moment to its
updated value, so XLA writes the result into the input's memory:

<!-- doc-test: case-device -->

```kotlin
val device = captured.withDeviceAdamW(optimizer)
val text = device.function.toStablehlo(outputAliases = device.outputAliases)

// each step: the batch and the scalars go up; params, m and v are already on the device
val scalars = device.scalars(t).map { session.bufferFromHostF32(floatArrayOf(it), emptyList()) }
val outs = session.executeStablehlo(text, listOf(idsB, tgtB) + params + m + v + scalars)
(listOf(idsB, tgtB) + scalars + params + m + v).forEach { it.close() }  // consumed by the call
params = outs.subList(1, 1 + n)
m = outs.subList(1 + n, 1 + 2 * n)
v = outs.subList(1 + 2 * n, 1 + 3 * n)
```

The scalars are inputs rather than constants, so a learning-rate schedule never
changes the program and it compiles once. Only `executeStablehlo` runs text with
aliases; `runOn` never donates ([HOW_IT_WORKS.md](../HOW_IT_WORKS.md#buffers-and-donation)).

## What is left

- There is no device form of Adam without weight decay, or of SGD; the `fine-tune` and
  `lora-finetune` examples still run AdamW on the host (`docs/FOLLOWUPS.md`).
- The batch still crosses the host every step. For data that fits in device memory,
  upload it once and pass buffers.
- `:nn` has no `CapturedStep.runOn(session)`; `:nn` does not depend on
  `:runtime-pjrt`, so the binding code is written out, as above.
