#!/usr/bin/env bash
# Serves a pinned checkpoint through Triton (tlaloc backend) and tlaloc-endpoint's OpenAI API, from
# this Tlaloc checkout, so that no stage needs the serving runbook.
#
#   serving/server.sh export [config]   # serving artifact + Triton model repository (skipped when current)
#   serving/server.sh start  [config]   # export if needed, Triton, then the endpoint; waits until both answer
#   serving/server.sh stop   [config]   # endpoint, then Triton; prints the peak memory seen while serving
#   serving/server.sh status [config]   # prints the endpoint URL when it answers, exits 1 otherwise
#
# config is a file serving/<config>.env (default twil-bf16). Outputs go to build/serving/<config>/
# (BROOKLYN_SERVING moves build/serving): artifact/, repository/, triton.log, endpoint.log, mem.log,
# export.stamp; compiled executables are kept in build/serving/pjrt-cache. The Triton backend is
# built from triton/ when its sources changed since the last build (build/serving/backend.stamp),
# and the PJRT plugin is fetched by triton/fetch_pjrt_plugin.sh (pinned by SHA-256).
#
# Memory guard: while Triton runs, machine memory is sampled every 2 s (mem.log);
# when MemAvailable falls below $BROOKLYN_MEM_GUARD_GIB GiB (default 8; 0 turns it off), the guard
# kills the endpoint and the Triton container and writes the reason to mem-guard.log, so a run that
# would exhaust unified memory fails instead of rebooting the machine.
#
# Ports (127.0.0.1 only): Triton HTTP/gRPC/metrics $BROOKLYN_TRITON_PORT (8110) and the next two;
# the endpoint $BROOKLYN_ENDPOINT_PORT (8120). 8000-8002 are left alone.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
ex="$(cd "$here/.." && pwd)"
root="$(cd "$ex/../.." && pwd)"
cmd="${1:-}"
config="${2:-twil-bf16}"
envfile="$here/$config.env"
[ -f "$envfile" ] || { echo "no serving config $envfile" >&2; exit 2; }

declare -A C
while IFS= read -r line; do
    case "$line" in ''|'#'*) continue ;; esac
    C["${line%%=*}"]="${line#*=}"
done < "$envfile"

serving="${BROOKLYN_SERVING:-$ex/build/serving}"
out="$serving/$config"
artifact="$out/artifact"
repository="$out/repository"
cache="$serving/pjrt-cache"
tport="${BROOKLYN_TRITON_PORT:-8110}"
eport="${BROOKLYN_ENDPOINT_PORT:-8120}"
container="brooklyn-eats-triton"
app="$ex/build/install/tlaloc-brooklyn-eats/bin/tlaloc-brooklyn-eats"
url="http://127.0.0.1:$eport/v1"
mkdir -p "$out"
# The app reads manifest.json from its working directory.
cd "$ex"

ensure_app() {
    [ -x "$app" ] || "$root/gradlew" -q -p "$ex" installDist
}
ckpt() { ensure_app; "$app" models path "${C[MODEL_KEY]}"; }
# A draft model's checkpoint (DRAFT_KEY), when the config has one.
draft_ckpt() { if [ -n "${C[DRAFT_KEY]:-}" ]; then ensure_app; "$app" models path "${C[DRAFT_KEY]}"; fi; }

# What the export depends on: the settings, the checkpoint and the last commit of the modules
# that write the artifact. A working tree with changes there always re-exports.
export_key() {
    local mods=(maestro ir stablehlo core autograd nn tokenizer)
    local dirty
    dirty="$(git -C "$root" status --porcelain -- "${mods[@]/#/$root/}" | sha256sum | cut -c1-16)"
    printf '%s\n' "${C[EXPORT_PROPS]}" "${C[TRITON_PROPS]}" "${C[TRITON_MODEL]}" "$(ckpt)" "$(draft_ckpt)" \
        "$(git -C "$root" log -1 --format=%H -- "${mods[@]/#/$root/}")" \
        "$( [ -n "$(git -C "$root" status --porcelain -- "${mods[@]/#/$root/}")" ] && echo "dirty $dirty" || echo clean)"
}

do_export() {
    local key; key="$(export_key)"
    if [ -f "$out/export.stamp" ] && [ "$(cat "$out/export.stamp")" = "$key" ] && [ -d "$repository/${C[TRITON_MODEL]}" ]; then
        echo "serving artifact current: $artifact"
        return 0
    fi
    local s; s="$(ckpt)"
    local repo_id; repo_id="$("$app" models repo "${C[MODEL_KEY]}")"
    local draft=(); local d; d="$(draft_ckpt)"; [ -n "$d" ] && draft=(-PdraftCkptDir="$d")
    rm -rf "$artifact" "$repository" "$out/export.stamp"
    local t0=$SECONDS
    echo "exporting ${C[MODEL_KEY]} ($s) to $artifact"
    # shellcheck disable=SC2086
    "$root/gradlew" -q -p "$root" :maestro:exportHfServingArtifact -PexportHeap=24g -PckptDir="$s" \
        -PoutDir="$artifact" -PmodelName="$repo_id" "${draft[@]}" ${C[EXPORT_PROPS]}
    # shellcheck disable=SC2086
    "$root/gradlew" -q -p "$root" :maestro:exportTritonModel -PartifactDir="$artifact" -PoutDir="$repository" \
        -PmodelName="${C[TRITON_MODEL]}" ${C[TRITON_PROPS]}
    echo "$key" > "$out/export.stamp"
    echo "exported in $((SECONDS - t0)) s"
}

# The backend and its CUDA kernels, built in the Triton container when their sources changed.
ensure_backend() {
    local key so="$root/triton/backends/tlaloc/libtriton_tlaloc.so"
    key="${C[TRITON_IMAGE]} $(git -C "$root" log -1 --format=%H -- triton/backend triton/kernels triton/build_backend.sh) $(git -C "$root" status --porcelain -- triton/backend triton/kernels | sha256sum | cut -c1-16)"
    if [ -f "$so" ] && [ -f "$serving/backend.stamp" ] && [ "$(cat "$serving/backend.stamp")" = "$key" ]; then return 0; fi
    echo "building the Triton backend (triton/build_backend.sh)"
    TRITON_IMAGE="${C[TRITON_IMAGE]}" "$root/triton/build_backend.sh" > "$serving/backend-build.log" 2>&1 || { tail -30 "$serving/backend-build.log" >&2; return 1; }
    echo "$key" > "$serving/backend.stamp"
}

triton_ready() { curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$tport/v2/models/${C[TRITON_MODEL]}/ready" 2>/dev/null | grep -q '^200$'; }
endpoint_ready() { curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$eport/health" 2>/dev/null | grep -q '^200$'; }
container_running() { [ "$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null)" = true ]; }
# Used memory of the machine (MemTotal - MemAvailable), bytes.
used_bytes() { awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{print (t-a)*1024}' /proc/meminfo; }
avail_kib() { awk '/^MemAvailable:/{print $2}' /proc/meminfo; }
guard_gib="${BROOKLYN_MEM_GUARD_GIB:-8}"
# The guard's verdict, when it stopped the server.
guard_report() { if [ -s "$out/mem-guard.log" ]; then echo "memory guard: $(cat "$out/mem-guard.log")" >&2; fi; }

# Samples memory while the container runs; kills the server when MemAvailable < guard_gib GiB.
mem_loop() {
    local limit a
    limit="$(awk -v g="$guard_gib" 'BEGIN{printf "%d", g * 1048576}')"
    while container_running; do
        a="$(avail_kib)"
        echo "$(date +%s) $(used_bytes)" >> "$out/mem.log"
        if [ "$limit" -gt 0 ] && [ "$a" -lt "$limit" ]; then
            echo "$(date -Is) MemAvailable $((a / 1024)) MiB < ${guard_gib} GiB; killed the endpoint and Triton (used $(awk -v u="$(used_bytes)" 'BEGIN{printf "%.1f", u / 2^30}') GiB)" > "$out/mem-guard.log"
            echo "$(date +%s) $(used_bytes) guard" >> "$out/mem.log"
            [ -f "$out/endpoint.pid" ] && kill -KILL "$(cat "$out/endpoint.pid")" 2>/dev/null || true
            docker kill "$container" >/dev/null 2>&1 || true
            break
        fi
        timeout 2 tail -f /dev/null 2>/dev/null || true
    done
}

wait_until() { # seconds, description, command...
    local limit=$1 what=$2; shift 2
    local t0=$SECONDS
    until "$@"; do
        if [ $((SECONDS - t0)) -ge "$limit" ]; then echo "$what: not ready after $limit s" >&2; return 1; fi
        if ! container_running; then echo "$what: Triton exited; last lines of $out/triton.log:" >&2; tail -30 "$out/triton.log" >&2; return 1; fi
        timeout 5 tail -f /dev/null 2>/dev/null || true
    done
}

start() {
    if endpoint_ready && triton_ready; then echo "$url"; return 0; fi
    command -v docker >/dev/null || { echo "docker not found" >&2; return 1; }
    for p in "$tport" "$((tport + 1))" "$((tport + 2))" "$eport"; do
        if (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null; then echo "port $p is in use" >&2; return 1; fi
    done
    do_export
    ensure_backend
    (cd "$root" && ./triton/fetch_pjrt_plugin.sh >/dev/null)
    mkdir -p "$cache"
    docker rm -f "$container" >/dev/null 2>&1 || true
    used_bytes > "$out/mem.base"
    rm -f "$out/mem-guard.log"
    : > "$out/mem.log"
    echo "starting Triton (${C[TRITON_MODEL]}) on 127.0.0.1:$tport-$((tport + 2))"
    local t0=$SECONDS
    TRITON_IMAGE="${C[TRITON_IMAGE]}" CONTAINER_NAME="$container" MODEL_REPOSITORY="$repository" HTTP_PORT="$tport" GRPC_PORT="$((tport + 1))" \
        METRICS_PORT="$((tport + 2))" TRITON_BIND=127.0.0.1 TLALOC_PJRT_MEMORY_FRACTION="${C[MEMORY_FRACTION]}" \
        TLALOC_PJRT_CACHE_DIR="$cache" TLALOC_DETERMINISTIC="${C[DETERMINISTIC]:-0}" \
        "$root/triton/run_server.sh" --detach > "$out/triton.cid"
    (docker logs -f "$container" > "$out/triton.log" 2>&1 &)
    # Memory sampler and guard (see the top of this file).
    (mem_loop > /dev/null 2>&1 &)
    wait_until 7200 "Triton" triton_ready || { stop; return 1; }
    echo "Triton ready in $((SECONDS - t0)) s"
    local s; s="$(ckpt)"
    local t1=$SECONDS
    local seqs=(); [ -n "${C[ENDPOINT_MAX_SEQUENCES]:-}" ] && seqs=(--max-sequences "${C[ENDPOINT_MAX_SEQUENCES]}")
    nohup "$app" endpoint "${seqs[@]}" --model "${C[TRITON_MODEL]}" --served-name "${C[SERVED_NAME]}" --tokenizer "$s" \
        --triton "localhost:$((tport + 1))" --triton-http "localhost:$tport" --host 127.0.0.1 --port "$eport" \
        --sampling --state-file "$out/endpoint-state.json" > "$out/endpoint.log" 2>&1 &
    echo $! > "$out/endpoint.pid"
    until endpoint_ready; do
        if ! kill -0 "$(cat "$out/endpoint.pid")" 2>/dev/null; then echo "the endpoint exited:" >&2; tail -20 "$out/endpoint.log" >&2; stop; return 1; fi
        if [ $((SECONDS - t1)) -ge 300 ]; then echo "the endpoint did not answer within 300 s" >&2; stop; return 1; fi
        timeout 1 tail -f /dev/null 2>/dev/null || true
    done
    echo "endpoint ready: $url (model ${C[SERVED_NAME]})"
}

stop() {
    if [ -f "$out/endpoint.pid" ]; then
        local pid; pid="$(cat "$out/endpoint.pid")"
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
            for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; timeout 1 tail -f /dev/null 2>/dev/null || true; done
            kill -KILL "$pid" 2>/dev/null || true
        fi
        rm -f "$out/endpoint.pid"
    fi
    if docker inspect "$container" >/dev/null 2>&1; then docker stop -t 30 "$container" >/dev/null 2>&1 || docker rm -f "$container" >/dev/null 2>&1 || true; fi
    if [ -s "$out/mem.log" ] && [ -f "$out/mem.base" ]; then
        awk -v base="$(cat "$out/mem.base")" '{ if ($2 > m) m = $2 } END { if (m > 0) printf "serving: machine memory in use peaked at %.1f GiB (%.1f GiB before Triton started)\n", m / 2^30, base / 2^30 }' "$out/mem.log"
    fi
    echo "stopped Triton and the endpoint"
    guard_report
}

case "$cmd" in
export) ensure_app; do_export ;;
start) ensure_app; start ;;
stop) stop ;;
status) if endpoint_ready; then echo "$url"; else guard_report; exit 1; fi ;;
*) echo "usage: $0 export|start|stop|status [config]" >&2; exit 2 ;;
esac
