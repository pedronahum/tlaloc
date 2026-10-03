#!/bin/bash
# Stop the vLLM server. Matches the python entry point, so it never matches a shell
# whose command line merely mentions "vllm serve".
VENV=${VENV:-$HOME/.local/venvs/vllm}
pat="^$VENV/bin/python $VENV/bin/vllm"
for p in $(pgrep -f "$pat serve"); do kill "$p"; done
for i in $(seq 60); do pgrep -f "$pat" >/dev/null || exit 0; sleep 1; done
pkill -9 -f "$pat"
