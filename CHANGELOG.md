# Changelog

All notable changes to Tlaloc are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) **with the alpha
carve-outs in [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md)**. Read that before
depending on a coordinate.

This file starts at `0.1.0-alpha01`. Earlier work is recorded in the commit history
and in [`DIFFKTX_SPEC.md`](DIFFKTX_SPEC.md).

## [Unreleased]

### Added

- **Refused placeholder tokens in the manifest.** A multimodal checkpoint's
  image and video placeholder ids (`HfDecoderConfig.refusedTokenIds`) are
  listed in the artifact (`ServingModelShape.refusedTokens`, written only when
  there are some), and the Triton backend and `tlaloc_serve.py` refuse a
  request holding one by name, leaving the sequence as it was.
- **bf16 weights for Llama and Qwen3** (`-PweightDType=bf16`, the tool's
  eleventh argument). Qwen3-0.6B served by Triton with bf16 weights: 1136 MiB
  on the device against 2273 MiB, all 32 fixture ids equal HuggingFace's,
  logits within 3.0e-3 of the largest, 11.3 ms a decode step against 15.4 ms.
- **Prefill in the framework-free Python runtime.**
  `ServingArtifact.run_prefill` writes a chunk per sequence in one call on the
  artifact's prefill entries; `run_llama_generate.py` runs a prompt that way
  when an entry holds it (`--no-prefill` keeps the decode walk), and the vLLM
  plugin's runner writes a new sequence's prompt (but its last token) with
  one prefill call.
- **Text prompts for any tokenizer in `generate_client.py`**: with the
  `tokenizers` package, `--text` is tokenized by the checkpoint's
  `tokenizer.json` (byte-level BPE included, as `examples/triton-llm/chat.py`
  does); `--expect-prompt` checks the ids.

- **Prompts of several sequences prefilled in one call.** `HfServingExport`
  writes a prefill entry for every batch of the ladder (`prefillMaxBatch`,
  or `-PprefillMaxBatch`, caps it; `HfServingExport.specs` lists the
  entries). Each sequence has its own right-aligned row with its own
  positions, block table and slots; in the reference interpreter a prompt
  prefilled with others gets its solo logits and continuation bit for bit
  (`BatchedPrefillTest`, with full-history and windowed pools). The Triton
  backend runs the prompts of one batch that fall in one context bucket as
  one call on the smallest prefill entry that holds them. On the GB10 four
  TinyLlama prompts sent together prefill in about 35 ms against about
  110 ms one after the other, with each prompt's solo argmax and 8 greedy
  ids.
- **Windowed KV pool for sliding-window layers.** A model's sliding-window
  layers can keep their KV in a second pool class (`WindowedKvPool`): each
  sequence holds a ring of at most `ceil(window / blockSize) + 1` pages,
  logical block `b` on ring page `b % ringPages`, so the pages it holds are
  bounded by the window instead of its length. The decode contract gains
  the `WINDOW_BLOCK_TABLES`, `WINDOW_SLOT_MAPPING`, `WINDOW_KV_POOL_IN` and
  `WINDOW_KV_POOL_OUT` roles; `PAGED_ATTENTION` is unchanged. An artifact
  with such a pool is `tlaloc-serving-v3` (`ServingModelShape.windowedKv`
  states the window, the layers, the pages and the ring), and
  `HfServingExport` writes one by default for a model with sliding layers
  (`windowedKv = false`, or `-PwindowedKv=false`, keeps full-history pools).
  The Triton backend keeps each sequence's ring, fills the window tables from
  it and splits a request into calls the ring can hold. For Muse Glimmer the
  computed KV per sequence at 32,768 positions is 989 MiB instead of
  3,328 MiB. `tlaloc_serve.py` and the vLLM plugin refuse a v3 artifact by
  its schema version.
- **`KV_PAGES` output (Triton sequence mode).** An optional INT32 `[2]`
  output: the pages a sequence holds after a request in the KV pool and the
  windowed KV pool. `TritonModelRepository` declares it; the backend serves
  a model with or without it.
- **`examples/triton-llm`.** A standalone example: its Gradle build exports a
  HuggingFace checkpoint (Qwen3-0.6B by default, TinyLlama-1.1B-Chat or the
  Muse Glimmer 30B text decoder) as a Triton model repository with
  `HfServingExport` and `TritonModelRepository`; `run.sh` starts Triton with
  `libtriton_tlaloc.so`, and `chat.py` renders the question with the model's
  chat template, tokenizes it with `tokenizers` (no torch) and streams the
  answer token by token through the sequence batcher. It skips by name, with
  exit status 0, without Docker, a GPU, the Triton image, the PJRT plugin, the
  checkpoint or the client packages. On the GB10, Qwen3-0.6B decodes at about
  20 ms a token.
- **`docs/SERVING_ARCHITECTURE.md`.** How serving works: the Kotlin export
  from a checkpoint to DXIR, StableHLO and a serving artifact, and the three
  ways to serve it (framework-free Python, the vLLM platform plugin, Triton),
  with what runs in which process, where memory lives and what has run on
  which hardware.
- **Muse Glimmer (text).** `HfModelFamily.MuseGlimmer` reads
  `MuseGlimmerForConditionalGeneration` checkpoints: the decoder config under
  `text_config`, tensors under `model.language_model.`, the vision encoder's
  tensors listed and not read, the image and video placeholder ids refused by
  name (`HfDecoderConfig.checkTextOnlyTokens`). meta-models/Muse-Glimmer-30B
  (Apache-2.0) serves through Triton with its weights in bf16 and greedy-decodes
  the ids of HuggingFace transformers run with the same arithmetic, 32 of 32.
- **Decoder layer features.** Sliding-window attention (an optional
  `sliding_window` attribute on `PAGED_ATTENTION`), layers without RoPE,
  attention and MLP output norms, `(1 + w)` norm gains, gainless q/k norms, a
  query scale, an attention output gate, an embedding norm, and a logit
  multiplier with final-logit soft-capping. `DecoderLayerPart` gains
  `ATTN_GATE_PROJ`, `ATTENTION_OUTPUT_NORM` and `FEEDFORWARD_OUTPUT_NORM`.
- **bf16 weights in serving artifacts.** `HfDecoderConfig.weightDType`;
  `HfStagedWeights.writeSlot` streams a slot's bytes a block at a time;
  `ServingArtifactWriter.export` takes a `writeWeight` callback and writes F32
  or BF16 weight files. `WeightSource`, `SafetensorsFile`, `SafetensorsIndex`
  and `HfCheckpoint` read a tensor's header entry and raw byte ranges.
- **`harness/python/muse_glimmer_fixture.py`** writes the Muse Glimmer
  fixtures, including a five-layer model with closed-form weights.

- **NVIDIA Triton backend.** `triton/` holds `libtriton_tlaloc.so`, a Triton
  Inference Server backend that compiles Tlaloc StableHLO through a PJRT plugin
  and serves it over Triton's HTTP and gRPC endpoints. It is not a Gradle module
  and is not published to Maven. `triton/verify.sh` checks it end to end in the
  Triton 25.11 container. See [triton/README.md](triton/README.md).
- **Serving artifacts as Triton models.** `TritonModelRepository` (`:maestro`) and
  the Gradle task `:maestro:exportTritonModel` write a serving artifact into a
  Triton model repository with a generated `config.pbtxt`. The backend loads the
  staged weights at model load and keeps the KV pools on the device between
  requests. TinyLlama-1.1B served this way produces the same six greedy token ids
  as HuggingFace transformers for "The capital of France is".
- **Prefill entries.** `HfDecoderGraph` builds a prefill graph: a chunk of
  tokens per sequence in one call, each token an attention row with a causal
  context of its position + 1, right-aligned, returning the last token's logits.
  In the reference interpreter its logits and KV pools equal the decode loop's
  bit for bit (a random-weight model and two real TinyLlama layers).
  `HfServingExport` writes one prefill entry per context bucket
  (`-Pprefill=false` to leave them out).
- **Serving manifest `tlaloc-serving-v2`.** Adds prefill entries; decode entries
  are unchanged and `tlaloc-serving-v1` artifacts are still read. The prefill
  logits type is `[batch, 1, vocab]` (the last position), as decode's.
- **Sequence mode in the Triton backend.** A model whose `config.pbtxt` names a
  serving manifest uses Triton's sequence batcher (oldest strategy): the backend
  keeps each sequence's KV pages by correlation ID, allocates them as the
  sequence grows, frees them on END or after the idle timeout, refuses a START by
  name when the pool is full, runs a prompt through a prefill entry, and runs the
  decode steps of several sequences as one batched call. Clients send token ids
  only. `TritonModelRepository` writes this mode by default for an artifact with
  KV pools (`-PkvMode=client` for the previous form). On the GB10, TinyLlama's
  6-token prefill takes about 26 ms and four concurrent sequences decode about
  twice as many tokens per second as one.
- **Qwen3.** `HfModelFamily` describes a HuggingFace decoder family: its
  architectures, the `config.json` keys it may carry, and one `DecoderLayerSpec`
  per layer. `Qwen3ForCausalLM` is the second family after Llama: a per-head
  RMSNorm on q and k before RoPE, `head_dim` independent of `hidden_size`, and a
  tied head that the file may also store (accepted only when it is bit-for-bit the
  embedding table). A config key the family does not know is refused by name at
  parse. Sliding-window layers, layers without RoPE, post-attention and
  post-feedforward norms, logit soft-capping, attention or MLP biases and
  activations other than SiLU are recorded and refused by name when a graph is
  built. Qwen3-0.6B, all 28 layers, greedy-decodes the same 16 token ids as
  HuggingFace transformers for a plain and a chat-template prompt, in the
  reference interpreter and served by Triton. `:maestro:exportHfServingArtifact`
  exports it (`exportLlamaServingArtifact` remains as a second name).

### Changed

- **A tied head reads the embedding table.** `HfDecoderGraph` gives a tied
  head no weight slot: it is one `MATMUL` contracting the hidden axis of the
  final state and of the embedding table (`lhs_contracting_dims = [1]`,
  `rhs_contracting_dims = [1]`, a `dot_general` without a transpose), which
  the DXIR interpreter now evaluates. Qwen3-0.6B's weights go from 2867 MiB to
  2273 MiB with the same 32 ids; `HfDecoderConfig.tiedHeadCopy` keeps the copy
  (the control: in the interpreter the two give the same logits bit for bit).
  The model hash of such an artifact ends in `:tiedHead`.
- **Triton sequence mode, a full page pool.** When a request needs pages the
  pool does not have, the backend reclaims pages from sequences idle past the
  reclaim rule (which Triton has ended) least recently active first, and only
  as many as the request needs, where it used to free every such sequence. It
  never takes pages from a live sequence: the request is refused by name,
  the refusal lists the sequences holding pages, and a sequence refused
  mid-generation keeps its pages and KV, so the same request can be sent
  again. Live sequences are not preempted (no KV swap or recompute).

- **Triton dynamic batching groups requests by shape.** A batch's requests
  are keyed by the shape of their rows and each key's requests run together
  in arrival order, instead of only consecutive requests of one shape. A
  ragged batch (`allow_ragged_batch`) of two widths runs as two executions:
  216 requests ran in 74 to 76 executions instead of 119 to 134. The model
  parameter `group_by_shape: "false"` keeps the old grouping.
- **KV pools are updated in place.** `toStablehlo` takes `outputAliases`, and
  every serving body the artifact writer emits marks each `KV_POOL_IN`
  parameter with `tf.aliasing_output` naming its `KV_POOL_OUT` (the manifest's
  `donationPairs`). The Triton backend donates the pools (and any `state:`
  buffer) to each execution, so XLA writes them where they are instead of
  copying 44 MiB (TinyLlama) or 224 MiB (Qwen3-0.6B) per run. The backend checks
  every pool's device address after each run and logs it; `verify.sh` requires
  100 of 100 runs in place for both models and runs a control with the new
  model parameter `donate_kv_pools: false`, which must be seen copying in 100 of
  100 runs and give the same 100 ids. Measured on the GB10 while another
  process kept the GPU 95% busy, alternating the two servers three times
  (server-side time of a batch-1 decode step, 235 steps per run): Qwen3-0.6B's
  fastest step 11.5 ms in place against 16.2 ms copied, medians 13.1 to 23.8 ms
  against 25.8 to 32.5 ms; TinyLlama's fastest step 18.3 ms against 19.3 ms,
  with medians too noisy to separate. Artifacts exported before this change
  still serve, with the pools copied and a warning in the log; `verify.sh`
  exports them again. The framework-free Python runtime and the vLLM plugin
  still pass the pools as host lists.
- `PAGED_ATTENTION`'s two f32 dots are emitted with HIGHEST precision, so XLA
  does not run them in TF32 on a GPU.
- A config with sliding-window layers, layers without RoPE, output norms or
  final-logit soft-capping now builds a graph instead of being refused by name;
  `attn_logit_softcapping` is still refused.
- The Triton backend drops the page cache of each weight file after uploading it.

- **HF decoder types renamed for the second family.** `HfLlamaConfig` is now
  `HfDecoderConfig`, `HfLlamaNames` `HfDecoderNames`, `LlamaWeightRole`
  `DecoderWeightRole`, `LlamaLayerPart` `DecoderLayerPart`, `HfLlamaDecodeGraph`
  `HfDecoderGraph`, `HfLlamaCheckpoint` `HfCheckpoint`, `HfLlamaStagedWeights`
  `HfStagedWeights` and `HfLlamaServingExport` `HfServingExport`. The old names
  remain as deprecated type aliases, so source written against alpha02 still
  compiles; the binary names changed. `HfDecoderGraph.PARTS_PER_LAYER` is gone:
  the number of tensors per layer is `DecoderLayerSpec.parts.size`, 9 for Llama
  and 11 for Qwen3.
- **RoPE tables sized to the entry.** The decode and prefill graphs carry cos and
  sin tables for the positions the entry can reach (its context, capped by
  `max_position_embeddings`) instead of every position the model supports.
- **A tied checkpoint may also store `lm_head.weight`.** `HfCheckpoint.open` used
  to refuse it; it now reads the head from the embedding table, as transformers
  does, and `HfStagedWeights` refuses the file by name if the stored head differs
  from the table.

### Fixed

- **An output written device to device into CUDA shared memory could arrive
  empty.** A device-to-device `cudaMemcpy` returns before the copy is done, and
  the backend sent the response (and let PJRT free the result) without waiting
  for it. With the GPU busy with another process, `matmul_sumsq`'s outputs came
  back as zeros in two of three runs of `verify_client.py`; the backend now
  synchronizes after the copy, and six runs out of six pass on the same busy GPU.
- **The Triton backend no longer frees a sequence Triton still holds.** In
  sequence mode the backend freed the pages of any sequence it had not run for
  `max_sequence_idle_microseconds`, counted from the end of its last execution.
  Triton counts from a request's arrival, and a request waiting in its queue
  keeps the sequence alive, so under load a step Triton accepted was refused
  for having no KV state (a 200 ms timeout and 12 to 32 concurrent TinyLlama
  sequences: 3 to 16 such refusals a run). The backend now frees idle sequences
  only when a sequence needs pages the pool lacks, never one with a request in
  the batch being run, and only after twice the timeout plus the longest queue
  wait the oldest strategy allows. `verify.sh` runs the case
  (`sequence_checks.py --queued`); a backend without the fix fails it.

## [0.1.0-alpha02] — 2026-09-24

The Kotlin 2.4 release: the compiler plugin supports Kotlin 2.4.20–2.4.29. Stay on
`0.1.0-alpha01` for Kotlin 2.3.x.

### Changed

- **Kotlin 2.4.20.** Tlaloc is built against Kotlin 2.4.20 and its compiler plugin
  supports Kotlin **2.4.20 through 2.4.29**; any other Kotlin is refused by name at
  compile time. Projects on Kotlin 2.3.20–2.3.29 stay on `0.1.0-alpha01`. The
  plugin's refusal also reaches a user of an older compiler as a compile error:
  under Kotlin 2.3 it is reported through the message collector, because the
  diagnostic API Kotlin 2.4 recommends does not exist there.
  [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) lists the Kotlin range of each
  Tlaloc version.
- **IR-phase messages are compiler diagnostics.** The compiler plugin's IR-phase
  refusals and warnings are reported at the call site through the compiler's
  diagnostic reporter, with unchanged text and severity, and a warning can be
  silenced with `@Suppress` naming it (for example `IR_LOWERING_REFUSED_WARNING`).
  The errors cannot be silenced that way; `strictLowering=false` is still the only
  way to compile a call the plugin refuses.
- **ABI baseline for every published module.** The committed `api/*.api` baselines
  are checked by the Kotlin Gradle plugin's ABI validation (`checkKotlinAbi`,
  `updateKotlinAbi`) instead of binary-compatibility-validator, and now also cover
  `tlaloc-runtime-pjrt`, `tlaloc-runtime-cuda`, `tlaloc-kptx`, `tlaloc-runtime-iree`
  and `tlaloc-compiler-plugin`.
- **Symja 3.2.0.** The build compiles and tests `tlaloc-ir` against
  `org.matheclipse:matheclipse-core:3.2.0`, and the compiler's refusal for a
  symbolic-trip-count loop now names that coordinate. Symja stays `compileOnly`
  (not in a consumer's dependency graph). The 0.1.0-alpha01 `tlaloc-ir` jar calls
  the same 36 Symja methods and fields, so it also runs with 3.2.0 on the
  compiler plugin classpath.
- **Gradle 9.7.1.** The repository's wrapper moves from 9.5.0 to 9.7.1; the
  Tlaloc Gradle plugin is tested with it.
- **`tlaloc { dumpGradSourceDir }` writes one subdirectory per compilation.** The
  Gradle plugin writes each Kotlin/JVM compilation's gradient sources into
  `<dumpGradSourceDir>/<source set>` (`main`, `test`; `jvmMain`, `jvmTest` in a
  Multiplatform build) instead of into `dumpGradSourceDir` itself, and declares
  that subdirectory an output of the compile task. The build cache now stores and
  restores the dumped files, and the directory's absolute path no longer enters
  the compile task's cache key, so a project that sets it gets cache hits from a
  checkout at another path. A build that reads the files from
  `dumpGradSourceDir` directly now reads `dumpGradSourceDir/main`. The raw
  `-P plugin:io.tlaloc.plugin:dumpGradSourceDir=<dir>` option is unchanged.
- **API documentation.** KDoc in every published module describes the code
  without internal changelog numbers, plan phases or planning-document names;
  `./gradlew test` fails if one comes back (`scripts/check-kdoc-internal-refs.py`).

### Fixed

- **Vendored Maestro server builds.** `third-party/maestro/maestro-server`
  compiles: it declares its dependency on `maestro-tlaloc`, and its Spring
  configuration builds the Tlaloc step runtime with the pod-spec builder and the
  cluster accelerator from `TLALOC_CLUSTER_VENDOR` / `TLALOC_CLUSTER_ARCH`. The
  vendored build's Spotless, Checkstyle and PMD checks run on JDK 25 with Gradle 9
  (Spotless 8.10.2, google-java-format 1.30.0, Checkstyle 9.3).

## [0.1.0-alpha01] — 2026-09-24

The first published version. [docs/CAPABILITIES.md](docs/CAPABILITIES.md) lists
what it contains, what is certified and on what hardware.

### Contents

- **Automatic differentiation at compile time.** A K2 compiler plugin rewrites
  `grad`, `grad2`, `grad3`, `valueAndGrad*`, `jvp`, `jvp2`, `vjp`, `vjp2`,
  `jacobian`, `jacobianReverse` and `hessian` calls into synthesized gradient
  code. Custom rules through `customVjp`, `customJvp` and `customVjpJvp`. Loops
  and branches are differentiated through φ-calculus coarsening. One reverse
  transform serves the intrinsics, the Tracer-capture API and `:nn` training.
- **Readable gradients.** `dumpGradSource` and `dumpGradSourceDir` print the
  derived gradient as Kotlin that compiles without the plugin and is bit-identical
  to the compiled gradient; `CapturedStep.gradSource()` prints captured tensor
  gradients.
- **Typed tensors.** Rank, dtype and named axes in the Kotlin type;
  `NAMED_INDEX_MISMATCH`, `TENSOR_SHAPE_MISMATCH` and `NOT_DIFFERENTIABLE` compile
  errors at the offending call. dtypes F32, F64, I32 and BF16.
- **Captured values.** A `grad { }` body may reference compile-time constants
  declared outside it (folded as literals) and immutable runtime values of type
  `Float`, `Double`, `Int` or `Long` (bound at the call site).
- **`:nn`.** Immutable layers (Dense, Conv2d, pooling, BatchNorm, Dropout,
  Embedding, EmbeddingBag, GRU, Flatten), SGD, Momentum, RMSprop and Adam, learning
  rate schedules, gradient clipping, and checkpoints: a model and its optimizer
  state in one safetensors file, reloaded bit for bit.
- **Execution.** StableHLO and Shardy emission; PJRT from Kotlin through FFM and
  from Python through ctypes; IREE through its command-line tools; KPTX, a PTX DSL
  with kernel claiming (no kernel is registered by default).
- **Serving.** Paged attention, KV-cache writes, decode bucketing, safetensors
  reading and writing, a serving artifact that runs in a Python process with no
  JVM and no ML framework, and a vLLM platform plugin. A real TinyLlama-1.1B
  produces the same tokens as HuggingFace transformers.
- **`@ExperimentalTlalocApi`**, a `@RequiresOptIn(ERROR)` marker on the
  provisional surfaces: the four-worlds scopes, `AllReduceAttrs`, and the kernel
  choice and cost-model packages.
- **Tooling.** The Gradle plugin `io.github.pedronahum.tlaloc`
  (`tlaloc-gradle-plugin`), which applies `tlaloc-compiler-plugin` of the same
  version to every Kotlin/JVM compilation and takes the plugin options in a
  `tlaloc { }` block; `tlaloc-bom`; a committed ABI baseline checked by
  `apiCheck`; `./gradlew apiDocs` for a local API reference.
- **Requirements.** Kotlin 2.3.20–2.3.29 (other versions are refused by name at
  compile time). JDK 25 to build code that uses `grad { }`; the library modules are
  Java 21 bytecode, so JDK 21 runs them. PJRT, CUDA, KPTX and IREE need JDK 25.
  Symja (LGPL-3.0) is optional and not in the published dependency graph.

### Changed since earlier source builds

These matter only if you built Tlaloc from source before this version.

- **Group id `io.tlaloc` → `io.github.pedronahum`.** `io.tlaloc` could not be
  verified on Maven Central. Package names are unchanged (`io.tlaloc.*`), and so is
  the compiler plugin id used in `-P plugin:io.tlaloc.plugin:<option>`.
- **Version `0.0.1-SNAPSHOT` → `0.1.0-alpha01`.**
- **Every artifact id carries a `tlaloc-` prefix**: `tlaloc-core`, `tlaloc-ir`,
  `tlaloc-autograd`, `tlaloc-nn`, `tlaloc-stablehlo`, `tlaloc-maestro`,
  `tlaloc-runtime-pjrt`, `tlaloc-runtime-iree`, `tlaloc-runtime-cuda`,
  `tlaloc-kptx`, `tlaloc-compiler-plugin` (JVM artifacts `tlaloc-core-jvm` and so
  on), plus the new `tlaloc-gradle-plugin` and `tlaloc-bom`.
- `tlaloc-ir`, `tlaloc-autograd`, `tlaloc-stablehlo`, `tlaloc-maestro`,
  `tlaloc-runtime-iree` and `tlaloc-runtime-pjrt` expose the Tlaloc modules their
  public signatures use as `api` dependencies: `tlaloc-autograd` alone is enough to
  write `grad { }` over a `DTensor`.
- **Every compile-time refusal is an error by default.** An unlowerable
  `grad { }` lambda, and every call the IR phase leaves as written, stops the
  build at the call site. `strictLowering = false` turns these into warnings; the
  call then throws `IllegalStateException` when it runs.
- **A working build is silent.** The lowered-IR dumps are INFO messages behind
  `dumpLoweredIr`, so a correct program compiles under `-Werror`.
- `grad(::f)` and other arguments that are not a lambda written at the call site
  are refused by name.
- `dumpGradSource` refuses a value other than `true` or `false`.
- The runtime message for a `grad { }` that was not rewritten lists its three
  possible causes.
- Error messages no longer cite internal work-item numbers.
- `PjrtSession` can be used from several threads. `executeOn` serialises calls on
  one executable, and `close()` waits for calls in flight. A `PjrtBuffer`,
  `PjrtLoadedExecutable` or `PjrtClient` used after its client is closed, and a
  closed `PjrtBuffer` passed to `executeOn` or `execute`, throw instead of touching
  freed memory. Closing twice, or after the client, does nothing.
- `IreeModule` is `AutoCloseable`; closing it deletes its compiled VMFB.
  `runOnIree` and `IreeRuntime.invoke` delete their temporary files.
- The PJRT CUDA plugin is searched for under `$VIRTUAL_ENV`, `~/.local/venvs/*`,
  `~/.venv`, `~/venv`, `~/.local`, `/usr/local` and `/usr`, from the JVM and from
  Python alike; IREE tools under `$TLALOC_IREE_BIN`, `$VIRTUAL_ENV/bin`,
  `~/.local/venvs/*/bin` and `PATH`. A failed search lists every place it looked.
- In the Python serving runtime (`harness/python`, not published to Maven),
  `PjrtApi.load(plugin_path=None, platform="cuda")` takes a `platform` and, with no
  path and no `TLALOC_PJRT_PLUGIN_PATH`, searches for a plugin; when none is found
  it raises `FileNotFoundError` with the search report, where it used to raise
  `ValueError`.
- `CosineDecay` holds its final rate past `decaySteps`, where PyTorch's
  `CosineAnnealingLR` rises again.

### Added since earlier source builds

- The Gradle plugin and the BOM (see *Contents*).
- `KotlinVersionGuard`: an unsupported Kotlin version is a compile error naming
  the version found and the supported range; `unsafeAllowUnsupportedKotlin` turns
  it into a warning.
- "Tlaloc internal error" diagnostics: an unexpected exception inside the plugin,
  or a lowered call the IR phase did not find, is reported at the call site with
  the issue-tracker address instead of crashing the compiler or compiling green.
- Model checkpoints, learning-rate schedules and gradient clipping in `:nn`; a
  safetensors writer in `:core`.
- `LICENSE` (Apache-2.0), complete Maven Central POM metadata, sources and javadoc
  jars for every publication, and signing.
- `./gradlew releaseToCentralPortal` and `.github/workflows/release.yml`.
- CI on x86_64 Linux, aarch64 Linux and arm64 macOS, a JDK 21 lane for the library
  modules, and a probe against the next Kotlin release.
- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) sections on running on a GPU,
  configuration (every environment variable and system property Tlaloc reads) and
  troubleshooting.

### Removed

- `io.tlaloc.maestro.MaestroDescriptor` and `io.tlaloc.maestro.StubExecutor`. The
  first-class Maestro step type that replaces them lives in the vendored Maestro
  build under `third-party/maestro`, which is not published: a Maven Central user
  has no replacement.
- The `main` entry points in `tlaloc-maestro`; the exporters run through
  `./gradlew :maestro:exportServingArtifact` and
  `:maestro:exportLlamaServingArtifact`.
- The unused `timeoutSeconds` parameter of `runOnPjrt`.

### Fixed

- Two `grad { }` calls at the same character offsets in two different files could
  compile each other's gradient, and in a shared Kotlin daemon one compilation
  could clear another's pending gradients.
- An f64 `grad { }` body with a literal constant crashed the compiler with a
  `ClassCastException`, and an f64 constant close to 1 or 0 could be folded as if
  it were exactly 1 or 0.
- A `grad { t: Tracer<…> -> … }` call (the plugin-free tape overload) drew a
  spurious lowering warning.
- The Python serving runtime did not search for a PJRT plugin, so `serve.py`
  reported no plugin on a machine whose JVM lane ran on CUDA.
- Both PJRT bindings check the plugin's `PJRT_Api` size and API version before
  reading a function pointer and refuse an incompatible plugin by name.
- A malformed `TLALOC_PJRT_MEMORY_FRACTION`, `TLALOC_PJRT_PREALLOCATE`,
  `TLALOC_PJRT_NODE_ID` or `TLALOC_PJRT_NUM_NODES` is refused with the variable's
  name and value.
- A `PjrtSession` whose client creation fails releases the memory it allocated.
- Symbolic simplification no longer swallows JVM errors such as
  `OutOfMemoryError`.
- The KPTX kernel caches and the kernel-resolver registry are safe to use from
  several threads.

[0.1.0-alpha01]: https://github.com/pedronahum/tlaloc/releases/tag/v0.1.0-alpha01
[0.1.0-alpha02]: https://github.com/pedronahum/tlaloc/releases/tag/v0.1.0-alpha02
