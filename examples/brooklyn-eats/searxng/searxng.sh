#!/usr/bin/env bash
# Starts or stops the self-hosted SearXNG behind the web_search tool.
#   searxng/searxng.sh start   # container brooklyn-eats-searxng on 127.0.0.1:$SEARXNG_PORT (default 8890)
#   searxng/searxng.sh stop
#   searxng/searxng.sh status  # prints the URL when it answers, exits 1 otherwise
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
# 2026.11.9-57ff753ad, pinned by digest (also in manifest.json)
image="searxng/searxng@sha256:287af61f1a13e86d02974fcde91a3413a9945788de38b61ffbc491d81a19bcc2"
name="brooklyn-eats-searxng"
port="${SEARXNG_PORT:-8890}"
url="http://127.0.0.1:$port"

answers() { curl -s -m 5 -o /dev/null -w '%{http_code}' "$url/search?q=brooklyn&format=json" 2>/dev/null | grep -q '^200$'; }

case "${1:-}" in
start)
    command -v docker >/dev/null || { echo "docker not found" >&2; exit 1; }
    if answers; then echo "$url"; exit 0; fi
    docker rm -f "$name" >/dev/null 2>&1 || true
    # settings.yml is mounted read-only; the secret comes from the environment, fresh per start.
    secret="$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
    docker run -d --name "$name" -p "127.0.0.1:$port:8080" -e "SEARXNG_SECRET=$secret" \
        -v "$here/settings.yml:/etc/searxng/settings.yml:ro" "$image" >/dev/null
    for _ in $(seq 1 60); do
        if answers; then echo "$url"; exit 0; fi
        timeout 1 tail -f /dev/null 2>/dev/null || true
    done
    echo "searxng did not answer on $url within 60 s" >&2
    docker logs --tail 20 "$name" >&2 || true
    exit 1
    ;;
stop)
    docker rm -f "$name" >/dev/null 2>&1 || true
    ;;
status)
    if answers; then echo "$url"; else exit 1; fi
    ;;
*)
    echo "usage: $0 start|stop|status" >&2
    exit 2
    ;;
esac
