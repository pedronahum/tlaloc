#!/bin/bash
# The scenarios behind the tables in docs/work-log/spark-4user-serving.md, against a running server.
#   suite.sh <label> <tokenizer-repo> <corpus.txt>
OUT=${OUT:-$(pwd)/out}; VENV=${VENV:-$HOME/.local/venvs/vllm}; mkdir -p "$OUT"
run() { "$VENV/bin/python" "$(dirname "$0")/users.py" --model m --tokenizer "$2" --corpus "$3" \
          --out "$OUT/results.jsonl" --label "$1-$4" "${@:5}" 2>&1 | grep -E "^(==|turn|Trace|\w+Error)"; }
run "$@" warm     --users 1 --ctx 1000   --turns 1 --max-tokens 64 >/dev/null
run "$@" short-4u --users 4 --ctx 2000   --turns 2 --max-tokens 512
run "$@" 100k-1u  --users 1 --ctx 100000 --turns 2 --max-tokens 512
run "$@" 100k-4u  --users 4 --ctx 100000 --turns 3 --max-tokens 512
run "$@" stagger  --users 4 --ctx 100000 --turns 4 --max-tokens 512 --stagger 30 --think 10
grep -E "SpecDecoding metrics" "$OUT/serve-$1.log" 2>/dev/null | tail -1 | grep -oE "Mean acceptance length: [0-9.]+"
