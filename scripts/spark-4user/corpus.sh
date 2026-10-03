#!/bin/bash
# Documents for the simulated users: this repository's Markdown and Kotlin (about 14 MB).
#   corpus.sh > corpus.txt
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" && git ls-files '*.md' '*.kt' | xargs cat
