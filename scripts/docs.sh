#!/usr/bin/env bash
# The documentation site (mkdocs.yml).
#
#   scripts/docs.sh serve    preview at http://127.0.0.1:8100, rebuilt on save
#   scripts/docs.sh build    build/site, failing on any warning (what CI runs)
#
# The tools are installed from docs/requirements.txt into $TLALOC_DOCS_VENV
# (default ~/.local/venvs/tlaloc-docs) the first time, and again when that file changes.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
venv="${TLALOC_DOCS_VENV:-$HOME/.local/venvs/tlaloc-docs}"
req="$root/docs/requirements.txt"
stamp="$venv/.requirements"
if [[ ! -x "$venv/bin/mkdocs" ]] || ! cmp -s "$req" "$stamp"; then
  python3 -m venv "$venv"
  "$venv/bin/pip" install --quiet -r "$req"
  cp "$req" "$stamp"
fi
cd "$root"
case "${1:-build}" in
  serve) exec "$venv/bin/mkdocs" serve --dev-addr 127.0.0.1:8100 ;;
  build) exec "$venv/bin/mkdocs" build --strict ;;
  *) echo "usage: scripts/docs.sh [build|serve]" >&2; exit 2 ;;
esac
