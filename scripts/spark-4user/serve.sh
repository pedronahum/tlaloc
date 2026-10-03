#!/bin/bash
# Start vLLM for the 4-user long-context measurements and wait until it serves.
#   serve.sh <hf-repo> <label> [extra vllm flags...]
# Writes $OUT/serve-<label>.log. Port 8100. VENV defaults to ~/.local/venvs/vllm (vLLM 0.29.0).
#
# Best configuration measured on the GB10 (docs/work-log/spark-4user-serving.md):
#   serve.sh nvidia/Qwen3.6-35B-A3B-NVFP4 moe --kv-cache-dtype fp8 \
#     --enable-prefix-caching --mamba-cache-mode align --max-num-batched-tokens 16384 \
#     --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
set -e
REPO=$1; LABEL=$2; shift 2
OUT=${OUT:-$(pwd)/out}; VENV=${VENV:-$HOME/.local/venvs/vllm}; mkdir -p "$OUT"
# The HPC SDK nvcc that may come first on PATH has no libcudart to link against;
# FlashInfer JIT-builds kernels on first start and needs a full toolkit.
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}; export PATH=$CUDA_HOME/bin:$PATH
# Cap parallel kernel compiles: at the default, cicc was OOM-killed while the model
# held 60% of the unified memory.
export MAX_JOBS=${MAX_JOBS:-3}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_MARLIN_USE_ATOMIC_ADD=1
export VLLM_PLUGINS=""   # stock CUDA platform, not vllm-tlaloc
"$(dirname "$0")/stop.sh"
L="$OUT/serve-$LABEL.log"; rm -f "$L"
# 0.60 of the unified memory leaves ~48 GiB to the OS; 4 users at 131K need far less.
nohup "$VENV/bin/vllm" serve "$REPO" --served-model-name m --port 8100 --host 127.0.0.1 \
  --max-model-len 131072 --max-num-seqs 4 --gpu-memory-utilization 0.60 \
  --language-model-only --enable-chunked-prefill --enable-prompt-tokens-details \
  "$@" > "$L" 2>&1 &
sleep 5
until grep -qE "Application startup complete|EngineCore failed" "$L"; do sleep 3; done
grep -E "Application startup complete|EngineCore failed|GPU KV cache size|Model loading took|Error:" "$L" | cut -c1-250 | tail -6
