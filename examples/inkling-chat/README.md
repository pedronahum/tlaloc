# inkling-chat

A chat with Thinking Machines' [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
on the GPU, from Kotlin, in one JVM: `InklingModel` (`tlaloc-runtime-pjrt`) reads the released
checkpoint onto the device (projections in bf16, NVFP4 and MXFP8 experts dequantized to bf16),
compiles `InklingLoop.chunk` for chunks of 1, 2, 4, … 256 tokens through PJRT, and keeps the
model's state (K/V caches and the short-convolution histories of all 48 layer applications) on
the device. `HfTokenizer` renders the conversation with Inkling's chat template and
`InklingOutputParser` splits each generated turn into reasoning, answer and tool calls as it
streams.

```bash
./gradlew publishToMavenLocal -x test -x jvmTest -x allTests     # from the repo root
./gradlew -p examples/inkling-chat run --args="<snapshot dir> --effort low"
./gradlew -p examples/inkling-chat run --args="<snapshot dir> --prompt 'Name three boroughs of New York City.'"
```

`<snapshot dir>` is a local snapshot of `thinkingmachines/Inkling-2-Mini` (19.5 GB, e.g. from
`huggingface-cli download`). Options: `--effort` (Inkling's thinking effort: `none`, `minimal`,
`low`, `medium`, `high`, `xhigh`, `max` or a number), `--context` (cache length, default 8192),
`--max-tokens` (per turn, default 1024), `--prompt` (one turn, then exit).

`serve` puts the same model behind Tlaloc's OpenAI-compatible endpoint (`OpenAiEndpoint`, the one
`tlaloc-endpoint` runs in front of Triton) with an in-process backend, `InklingBackend`: chat
completions with reasoning (`reasoning_content`), tool calls, streaming, sampling, and reuse of the
conversation the model already holds. One conversation at a time.

```bash
./gradlew -p examples/inkling-chat serve --args="<snapshot dir> --port 8100"
curl -s localhost:8100/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"Name three boroughs of New York City."}],"max_tokens":200}'
```

A later turn appends only what the conversation added when the state already holds the earlier
turns as they were rendered; otherwise the state is reset and the whole conversation runs again.

Needs a CUDA GPU with about 50 GB of memory for the weights (bf16 experts), the caches and XLA's
workspace, and 24 GB of JVM heap while the experts are dequantized. Measured on a DGX Spark
(GB10): see the Inkling 2 section of `docs/work-log/inkling-plan.md`.

Inkling-2-Mini is under the Apache 2.0 licence.
