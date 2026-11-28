# inkling-chat

A chat with Thinking Machines' [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
on the GPU, from Kotlin, in one JVM: `InklingModel` (`tlaloc-runtime-pjrt`) reads the released
checkpoint onto the device (projections in bf16; the NVFP4 and MXFP8 experts as FP8 codes and
block scales, dequantized in the programs to the values transformers uses), compiles
`InklingLoop.batchChunk` for chunks of 1, 16 and 256 tokens and for spans of the caches from 512
rows up through PJRT, and keeps the model's state (K/V caches and the short-convolution histories
of all 48 layer applications) on the device. `HfTokenizer` renders the conversation with Inkling's chat template and
`InklingOutputParser` splits each generated turn into reasoning, answer and tool calls as it
streams.

```bash
./gradlew publishToMavenLocal -x test -x jvmTest -x allTests     # from the repo root (until the next release)
./gradlew -p examples/inkling-chat run --args="<snapshot dir> --effort low"
./gradlew -p examples/inkling-chat run --args="<snapshot dir> --prompt 'Name three boroughs of New York City.'"
```

`<snapshot dir>` is a local snapshot of `thinkingmachines/Inkling-2-Mini` (19.5 GB, e.g. from
`huggingface-cli download`). Options: `--effort` (Inkling's thinking effort: `none`, `minimal`,
`low`, `medium`, `high`, `xhigh`, `max` or a number), `--context` (cache length, default 8192),
`--max-tokens` (per turn, default 1024), `--prompt` (a turn; repeat it for more, then exit).

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

Measured on a DGX Spark (GB10, 128 GB of memory shared by CPU and GPU), context 8,192: weights
on the device in 25 s; 19.7 tokens/s early in a conversation, 15.2 at position 5,000; a second turn
that continues the state starts in about two seconds; 53 GiB of the machine's memory in use at the
peak. Each chunk size and span compiles once, on first use (about 10 s each), into the executable
cache (`TLALOC_PJRT_CACHE_DIR`, by default `~/.cache/tlaloc/inkling-executables`); the run and serve
tasks cap XLA's device pool at 0.3 of the memory (`TLALOC_PJRT_MEMORY_FRACTION`). More in
`docs/work-log/inkling-plan.md`.

## Through Triton

`export` writes the model as a serving artifact for Tlaloc's Triton backend (`InklingServingExport`:
each layer application's caches and histories are state pools with one row per sequence slot);
the backend batches the decode steps of up to four conversations, each in its own slot, and
`tlaloc-endpoint` puts the OpenAI API in front.

```bash
./gradlew -p examples/inkling-chat export --args="<snapshot dir> /tmp/inkling/artifact --contexts 2048,8192 --slots 4"
./gradlew :maestro:exportTritonModel -PartifactDir=/tmp/inkling/artifact -PoutDir=/tmp/inkling/repository \
  -PmodelName=inkling -Pdecoupled=true                                  # from the repo root
MODEL_REPOSITORY=/tmp/inkling/repository HTTP_PORT=8130 GRPC_PORT=8131 METRICS_PORT=8132 \
  TLALOC_PJRT_MEMORY_FRACTION=0.45 TLALOC_PJRT_CACHE_DIR=/tmp/inkling/cache triton/run_server.sh --detach
./gradlew -p examples/inkling-chat endpoint --args="--model inkling --served-name inkling-2-mini \
  --tokenizer <snapshot dir> --triton localhost:8131 --triton-http localhost:8130 --port 8140 --sampling"
```

On the GB10: the export takes 42 s (27.9 GB of weights); Triton is ready after 450 s the first
time (its programs compile) and in about a minute from the cache, with at least 63 GiB of the
machine's memory free. A stream decodes at 17.6 tokens/s; a prompt of more than 16 tokens waits
about 5 s for its first token (the 256-token prefill dequantizes every expert). Four
conversations at once share the GPU rather than speed it up (9.5 tokens/s in all against 11 for
one): each row of a decode step reads its own eight experts.

`export --bf16-pools` keeps the caches and convolution histories in bf16 (half the pools'
memory; the arithmetic stays f32). Greedy answers then differ: of four prompts (26 to 1,947
tokens), one answer is the same token for token, and the other three diverge within the first few
reasoning tokens, as coherent as the f32 answers.

Inkling-2-Mini is under the Apache 2.0 licence.
