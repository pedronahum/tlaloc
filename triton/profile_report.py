#!/usr/bin/env python3
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
"""Summarises an Nsight Systems capture of the tlaloc backend (exported to
SQLite), written by triton/profile.sh.

Executions. The capture is cut into executions at each device-to-host copy
(every execution ends by copying its logits back). For each one: the number
of kernels, the kernel time (sum), the device span (first kernel start to the
logits copy end), and the time from the previous execution's end to this
one's first host-to-device copy, which is spent on the host (Triton, the
backend, the client's round trip).

Components. With --xla-dump (the directory XLA_FLAGS=--xla_dump_to wrote), each
kernel is tied to the HLO instruction that launched it: fusion kernels by
name, library kernels (cuBLAS) by their place in the program's thunk
sequence. The instruction is classed from the optimized HLO:

  weights: <name>    reads a weight (an entry parameter the manifest names,
                     digits dropped: qProj, gateProj, ...); the projections,
                     the head and the embedding lookup
  kv write           writes a KV pool (scatter / dynamic-update-slice into a
                     rank-4 pool parameter)
  kv gather          gathers pages out of a KV pool
  attention dot      a dot or cuBLAS call that reads no weight (scores, P.V)
  attention softmax  exponentials / softmax fusions that read no weight
  other              norms, RoPE, residual adds, masks, converts

Per class: kernel time per execution, share, the bytes the instruction's
operands and result occupy (a lower bound on the traffic of a kernel that
reads each operand once), the time those bytes take at --bandwidth (default
273 GB/s, the GB10's nominal rate), the bandwidth implied, and the rate of
its dots in TFLOPS. It also lists the
instructions that convert a weight to another type into a buffer of their
own (a materialized convert), and each dot's operand and result types as
the dot sees them: a bf16 weight converted to f32 inside the same fusion
shows as an f32 operand, and is still read from memory as bf16.

  profile_report.py decode_p8_n16.sqlite --xla-dump xla_dump --manifest artifact/tlaloc-serving.json
"""

import argparse
import glob
import json
import os
import re
import sqlite3
import statistics
import sys
from collections import defaultdict


DT_BYTES = {"f32": 4, "bf16": 2, "f16": 2, "s32": 4, "u32": 4, "s8": 1, "u8": 1, "pred": 1,
            "s64": 8, "u64": 8, "f64": 8, "s16": 2, "u16": 2, "f8e4m3fn": 1, "f8e5m2": 1}
SHAPE = re.compile(r"\b(f32|bf16|f16|s32|u32|s8|u8|pred|s64|u64|f64|s16|u16|f8e4m3fn|f8e5m2)\[([0-9,]*)\]")


def shape_bytes(text):
    total = 0
    for dt, dims in SHAPE.findall(text):
        n = 1
        for d in dims.split(","):
            if d:
                n *= int(d)
        total += n * DT_BYTES[dt]
    return total


def first_shape(text):
    m = SHAPE.search(text)
    return (m.group(1), [int(d) for d in m.group(2).split(",") if d]) if m else (None, [])


class Hlo:
    """One optimized HLO module: instructions of the entry computation (and of
    fused computations), entry parameters, and the thunk order."""

    def __init__(self, path):
        self.path = path
        self.text = open(path).read()
        self.comps = {}  # computation name -> list of instruction lines
        cur = None
        self.entry = None
        for line in self.text.splitlines():
            m = re.match(r"^(ENTRY )?%?([\w.\-]+) .*\{\s*$", line)
            if m and not line.startswith(" "):
                cur = m.group(2)
                self.comps[cur] = []
                if m.group(1):
                    self.entry = cur
                continue
            if cur and line.startswith("  "):
                self.comps[cur].append(line.strip())
        self.instr = {}  # name -> line (all computations)
        for comp, lines in self.comps.items():
            for ln in lines:
                m = re.match(r"(?:ROOT )?%([\w.\-]+) = ", ln)
                if m:
                    self.instr[m.group(1)] = ln
        self.params = {}  # entry param instruction name -> (index, dtype, dims)
        for ln in self.comps.get(self.entry, []):
            m = re.match(r"(?:ROOT )?%([\w.\-]+) = (.*?) parameter\((\d+)\)", ln)
            if m:
                dt, dims = first_shape(m.group(2))
                self.params[m.group(1)] = (int(m.group(3)), dt, dims)
        base = path.split(".sm_")[0]
        # Thunks in launch order: (instruction, kernel name or None, kind).
        # A kernel thunk names its kernel (XLA launches one deduplicated
        # kernel for many identical fusions); a library call has none.
        self.thunks = []
        anns = {}
        meta = base + ".thunk_metadata.txt"
        if os.path.exists(meta):
            for ann, tid in re.findall(r'profile_annotation: "([^"]+)"\s*thunk_id: (\d+)', open(meta).read()):
                anns[int(tid)] = ann
        seqf = base + ".thunk_sequence.txt"
        if os.path.exists(seqf):
            for line in open(seqf):
                m = re.match(r"\s*(\d+): (k\w+)", line)
                if not m:
                    continue
                tid, kind = int(m.group(1)), m.group(2)
                k = re.search(r"kernel=([\w.\-]+), profile_annotation=([\w.\-]+)", line)
                if k:
                    self.thunks.append((k.group(2), k.group(1), kind))
                elif kind not in ("kSequential", "kCommandBuffer", "kWhile", "kConditional"):
                    self.thunks.append((anns.get(tid, f"thunk{tid}"), None, kind))

    def resolve(self, name, depth=0):
        """Follow bitcast / get-tuple-element / copy to an entry parameter."""
        if name in self.params or depth > 8:
            return name
        ln = self.instr.get(name, "")
        m = re.search(r"= .*? (?:bitcast|get-tuple-element|copy|reshape)\(%([\w.\-]+)", ln)
        return self.resolve(m.group(1), depth + 1) if m else name

    def operands(self, name):
        ln = self.instr.get(name, "")
        m = re.search(r"= .*? (?:fusion|custom-call)\(([^)]*)\)", ln)
        if not m:
            return []
        return [o.strip().lstrip("%") for o in m.group(1).split(",") if o.strip()]

    def called(self, name):
        ln = self.instr.get(name, "")
        m = re.search(r"calls=%([\w.\-]+)", ln)
        if not m:
            return []
        out, todo, seen = [], [m.group(1)], set()
        while todo:
            c = todo.pop()
            if c in seen:
                continue
            seen.add(c)
            for l in self.comps.get(c, []):
                out.append(l)
                for sub in re.findall(r"calls=%([\w.\-]+)", l):
                    todo.append(sub)
        return out


def dot_flops(hlo, instr):
    """2 * result elements * contracted size, summed over the dots (or the
    cuBLAS call) an instruction runs."""
    ln = hlo.instr.get(instr, "")
    lines = hlo.called(instr) + [ln]
    local = {}
    for l in lines:
        m = re.match(r"(?:ROOT )?%([\w.\-]+) = ", l)
        if m:
            local[m.group(1)] = l
    total = 0
    for l in lines:
        if re.search(r" dot\(", l):
            m = re.search(r" dot\(%([\w.\-]+),", l)
            cd = re.search(r"lhs_contracting_dims=\{([0-9,]*)\}", l)
            lhs = local.get(m.group(1)) or hlo.instr.get(m.group(1), "") if m else ""
            if not lhs or not cd:
                continue
            _, ldims = first_shape(lhs.split(" = ", 1)[1])
            _, rdims = first_shape(l.split(" = ", 1)[1])
        elif "__cublas" in l and "custom-call(" in l:
            m = re.search(r"custom-call\(%([\w.\-]+),", l)
            cd = re.search(r'"lhs_contracting_dimensions":\[([^\]]*)\]', l)
            lhs = hlo.instr.get(m.group(1), "") if m else ""
            if not lhs or not cd:
                continue
            _, ldims = first_shape(lhs.split(" = ", 1)[1])
            _, rdims = first_shape(l.split(" = ", 1)[1])
        else:
            continue
        k = 1
        for d in re.findall(r"\d+", cd.group(1)):
            if int(d) < len(ldims):
                k *= ldims[int(d)]
        n = 1
        for d in rdims:
            n *= d
        total += 2 * n * k
    return total


def classify(hlo, instr, param_names):
    """(class, detail) of one HLO instruction."""
    ln = hlo.instr.get(instr, "")
    body = hlo.called(instr)
    ops = [hlo.resolve(o) for o in hlo.operands(instr)]
    weights = []
    pools = []
    for o in ops:
        if o in hlo.params:
            idx, dt, dims = hlo.params[o]
            nm = param_names.get(idx, f"arg{idx}")
            role = param_names.get(("role", idx), "")
            if "POOL" in role:
                pools.append(nm)
            elif role in ("", "WEIGHT") and len(dims) >= 1 and idx in param_names:
                if not role or role == "WEIGHT":
                    weights.append(re.sub(r"\d+$", "", nm))
    joined = " ".join(body) + " " + ln
    has = lambda op: re.search(r"\b" + op + r"\(", joined) is not None
    big_weights = [w for w in weights if w not in ("inputNorm", "postAttnNorm", "attnOutNorm", "ffnOutNorm",
                                                   "qNorm", "kNorm", "norm", "finalNorm", "preFfnNorm",
                                                   "postFfnNorm")]
    if big_weights and (has("dot") or has("reduce") or "__cublas" in ln or "gemm" in ln or has("gather")):
        return "weights: " + "+".join(sorted(set(big_weights))), ""
    if pools:
        # The fused computation's root says whether it writes the pool or
        # reads pages out of it (a read may also splice in the new token).
        m = re.search(r"calls=%([\w.\-]+)", ln)
        root = next((l for l in hlo.comps.get(m.group(1), []) if l.startswith("ROOT ")), "") if m else ""
        if re.search(r"\b(scatter|dynamic-update-slice)\(", root):
            return "kv write", ""
        if has("gather") or has("dynamic-slice"):
            return "kv gather", ""
        if has("scatter") or has("dynamic-update-slice"):
            return "kv write", ""
    if "__cublas" in ln or has("dot") or "gemm_fusion" in instr:
        return "attention dot", ""
    if "softmax" in instr or has("exponential"):
        return "attention softmax", ""
    # a running maximum over the scores (the softmax's first pass)
    for comp in re.findall(r"to_apply=%([\w.\-]+)", joined):
        if any(" maximum(" in l for l in hlo.comps.get(comp, [])):
            return "attention softmax", ""
    return "other", ""


def materialized_converts(hlo, param_names):
    """Entry-level fusions whose one operand is a bf16 weight and whose
    result is that weight in f32: a converted copy the program writes to
    memory and reads back (as opposed to a convert fused into the dot)."""
    mats = []
    for ln in hlo.comps.get(hlo.entry, []):
        m = re.match(r"(?:ROOT )?%([\w.\-]+) = (f32\[[0-9,]*\])\S* fusion\(%([\w.\-]+)\)", ln)
        if m and hlo.resolve(m.group(3)) in hlo.params:
            idx, dt, dims = hlo.params[hlo.resolve(m.group(3))]
            body = hlo.called(m.group(1))
            if dt == "bf16" and len(dims) >= 2 and any(" convert(" in b for b in body):
                mats.append({"instruction": m.group(1), "weight": param_names.get(idx, f"arg{idx}"),
                             "bytes": shape_bytes(m.group(2))})
    return mats


def load_manifest_names(manifest, hlo):
    """Entry parameter index -> manifest input name (and role), for the
    entry whose program has the HLO's parameter shapes."""
    if not manifest:
        return {}
    m = json.load(open(manifest))
    sig = sorted(hlo.params.values())
    shapes = [(dt, dims) for _, dt, dims in sig]
    for e in m["entries"]:
        ins = e["inputs"]
        if len(ins) != len(shapes):
            continue
        ok = all(list(i["type"]["dims"]) == list(d) for i, (_, d) in zip(ins, shapes))
        if ok:
            out = {}
            for k, i in enumerate(ins):
                out[k] = i["name"]
                out[("role", k)] = i.get("role", "")
            return out
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sqlite")
    ap.add_argument("--xla-dump", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--bandwidth", type=float, default=273.0, help="GB/s for the floor column")
    args = ap.parse_args()
    bw = args.bandwidth * 1e9
    db = sqlite3.connect(args.sqlite)
    strings = dict(db.execute("select id, value from StringIds"))
    kernels = [(s, e, strings.get(n, "?"), g) for s, e, n, g in db.execute(
        "select start, end, shortName, graphId from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
    copies = list(db.execute("select start, end, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY order by start"))
    d2h = [c for c in copies if c[2] == 2]
    h2d = [c for c in copies if c[2] == 1]
    if not kernels or not d2h:
        print(f"{args.sqlite}: no kernels or no device-to-host copies captured")
        return 1

    # Executions: kernels between consecutive logits copies.
    execs = []
    prev_end = None
    ki = 0
    for c in d2h:
        ks = []
        while ki < len(kernels) and kernels[ki][0] < c[1]:
            ks.append(kernels[ki])
            ki += 1
        if not ks:
            prev_end = c[1]
            continue
        first_h2d = [h for h in h2d if (prev_end is None or h[0] >= prev_end) and h[0] <= ks[0][0]]
        begin = first_h2d[0][0] if first_h2d else ks[0][0]
        busy = sum(e - s for s, e, _, _ in ks)
        last_end = max(e for _, e, _, _ in ks)
        execs.append({"kernels": ks, "kernelMs": busy / 1e6, "spanMs": (c[1] - begin) / 1e6,
                      "hostGapMs": (begin - prev_end) / 1e6 if prev_end else None,
                      "launchMs": (ks[0][0] - begin) / 1e6,
                      "kernelSpanMs": (last_end - ks[0][0]) / 1e6,
                      "toCopyMs": (c[0] - last_end) / 1e6, "copyMs": (c[1] - c[0]) / 1e6,
                      "count": len(ks), "logitsBytes": c[3]})
        prev_end = c[1]

    n = len(execs)
    out = {"sqlite": os.path.basename(args.sqlite), "executions": n,
           "kernelsPerExecution": statistics.median(x["count"] for x in execs),
           "kernelMsPerExecution": round(statistics.median(x["kernelMs"] for x in execs), 3),
           "deviceSpanMsPerExecution": round(statistics.median(x["spanMs"] for x in execs), 3)}
    gaps = [x["hostGapMs"] for x in execs if x["hostGapMs"] is not None]
    out["hostGapMsBetweenExecutions"] = round(statistics.median(gaps), 3) if gaps else None
    out["idleInsideSpanMs"] = round(out["deviceSpanMsPerExecution"] - out["kernelMsPerExecution"], 3)
    med = lambda k: round(statistics.median(x[k] for x in execs), 3)
    out["phases"] = {"inputsToFirstKernelMs": med("launchMs"), "firstToLastKernelMs": med("kernelSpanMs"),
                     "lastKernelToLogitsCopyMs": med("toCopyMs"), "logitsCopyMs": med("copyMs")}
    print(f"{out['sqlite']}: {n} executions; per execution (median): {out['kernelsPerExecution']} kernels, "
          f"{out['kernelMsPerExecution']:.2f} ms of kernels, device span {out['deviceSpanMsPerExecution']:.2f} ms "
          f"(inputs in to logits out; {out['idleInsideSpanMs']:.2f} ms of it with no kernel running), "
          f"host time between executions {out['hostGapMsBetweenExecutions']} ms")
    ph = out["phases"]
    print(f"  phases (median ms): previous logits copy -> first input copy {out['hostGapMsBetweenExecutions']}; "
          f"first input copy -> first kernel {ph['inputsToFirstKernelMs']}; kernels {ph['firstToLastKernelMs']}; "
          f"last kernel -> logits copy {ph['lastKernelToLogitsCopyMs']}; logits copy {ph['logitsCopyMs']}")

    by_name = defaultdict(float)
    count = defaultdict(int)
    for x in execs:
        for s, e, nm, _ in x["kernels"]:
            by_name[nm] += (e - s) / 1e6 / n
            count[nm] += 1
    top = sorted(by_name.items(), key=lambda kv: -kv[1])[: args.top]
    out["topKernels"] = [{"kernel": k, "msPerExecution": round(v, 3), "launchesPerExecution": count[k] / n}
                         for k, v in top]

    if args.xla_dump:
        # The module this execution ran: its integer inputs (tokens,
        # positions, block tables, ...) are the host-to-device copies before
        # the first kernel, so their sizes pick the entry; the kernels' names
        # break ties.
        first = execs[0]["kernels"][0][0]
        sizes = sorted(h[3] for h in h2d if h[0] <= first and h[0] >= first - 50e6)
        names = set(by_name)
        best = None
        for path in glob.glob(os.path.join(args.xla_dump, "*gpu_after_optimizations.txt")):
            base = path.split(".sm_")[0]
            meta = base + ".thunk_metadata.txt"
            if not os.path.exists(meta):
                continue
            ent = None
            with open(path) as fh:
                for line in fh:
                    if line.startswith("ENTRY "):
                        ent = re.match(r"ENTRY [^(]*\((.*)$", line)
                        break
            want = sorted(shape_bytes(f"s32[{d}]") for d in re.findall(r"s32\[([0-9,]*)\]", ent.group(1) if ent else ""))
            pool = list(sizes)
            hits = 0
            for w in want:
                if w in pool:
                    pool.remove(w)
                    hits += 1
            anns = set(a.replace(".", "_") for a in re.findall(r'kernel=([\w.\-]+)', open(base + ".thunk_sequence.txt").read())) \
                if os.path.exists(base + ".thunk_sequence.txt") else set()
            score = len(names & anns) / max(len(names), 1)
            key = (hits - abs(len(want) - len(sizes)), score)
            if best is None or key > best[0]:
                best = (key, path)
        hlo = Hlo(best[1])
        out["hloModule"] = os.path.basename(best[1])
        pnames = load_manifest_names(args.manifest, hlo)
        # kernel -> instruction: fusion kernels by name; others by thunk order.
        seq = hlo.thunks
        cls_cache = {}

        def cls_of(instr):
            if instr not in cls_cache:
                cls_cache[instr] = classify(hlo, instr, pnames)[0]
            return cls_cache[instr]

        per_class = defaultdict(float)
        bytes_class = defaultdict(float)
        instr_time = defaultdict(float)
        unmatched = 0
        for x in execs:
            ks = x["kernels"]
            ti = 0
            lib_taken = 0  # kernels charged to the library thunk at ti
            for s, e, nm, _ in ks:
                instr = None
                j = ti
                while j < len(seq) and j < ti + 256:
                    ann, kname, kind = seq[j]
                    if kname == nm:
                        instr = ann
                        ti = j + 1
                        lib_taken = 0
                        break
                    j += 1
                if instr is None and ti < len(seq) and seq[ti][1] is None:
                    # A library call. Consecutive calls launch one kernel each;
                    # the last one before a fusion kernel keeps any extra
                    # kernels it launches (a split-K reduction, say).
                    if lib_taken >= 1 and ti + 1 < len(seq) and seq[ti + 1][1] is None:
                        ti += 1
                        lib_taken = 0
                    instr = seq[ti][0]
                    lib_taken += 1
                if instr is None:
                    ann = nm.rsplit("_", 1)
                    guess = ann[0] + "." + ann[1] if len(ann) == 2 and ann[1].isdigit() else nm
                    instr = guess if guess in hlo.instr else None
                if instr is None:
                    unmatched += 1
                    per_class["unattributed"] += (e - s) / 1e6 / n
                    continue
                c = cls_of(instr)
                per_class[c] += (e - s) / 1e6 / n
                instr_time[instr] += (e - s) / 1e6 / n

        def value_bytes(name):
            r = hlo.resolve(name)
            if r in hlo.params:
                _, dt, dims = hlo.params[r]
                return shape_bytes("%s[%s]" % (dt, ",".join(map(str, dims))))
            ln = hlo.instr.get(name, "")
            if " = " not in ln:
                return 0
            head = re.split(r" [\w\-]+\(", ln.split(" = ", 1)[1], 1)[0]
            return shape_bytes(re.sub(r"s8\[[0-9]*\]", "", head))  # not a cuBLAS workspace

        def touched(instr):
            """Operands and result, except that an operand an instruction only
            indexes into (gather, dynamic slice, in-place update) counts for
            no more than the result."""
            res = value_bytes(instr)
            body = " ".join(hlo.called(instr)) + " " + hlo.instr.get(instr, "")
            m = re.search(r"calls=%([\w.\-]+)", hlo.instr.get(instr, ""))
            root = next((l for l in hlo.comps.get(m.group(1), []) if l.startswith("ROOT ")), "") if m else ""
            if re.search(r"\b(scatter|dynamic-update-slice)\(", root):
                # an in-place update: it reads and writes the update, not the pool
                return 2 * sum(b for b in (value_bytes(o) for o in hlo.operands(instr)) if b < res / 4)
            indexes = re.search(r"\b(gather|dynamic-slice)\(", body)
            total = res
            for o in hlo.operands(instr):
                b = value_bytes(o)
                total += min(b, res) if indexes and b > 4 * res else b
            return total

        flops_class = defaultdict(float)
        for instr in instr_time:
            bytes_class[cls_of(instr)] += touched(instr)
            flops_class[cls_of(instr)] += dot_flops(hlo, instr)
        total = sum(per_class.values())
        rows = []
        for c, t in sorted(per_class.items(), key=lambda kv: -kv[1]):
            b = bytes_class.get(c, 0.0)
            f = flops_class.get(c, 0.0)
            rows.append({"component": c, "msPerExecution": round(t, 3), "share": round(t / total, 3),
                         "bytes": int(b), "floorMs": round(b / bw * 1e3, 3),
                         "gbPerS": round(b / (t / 1e3) / 1e9, 1) if t > 0 else None,
                         "flops": int(f), "tflops": round(f / (t / 1e3) / 1e12, 1) if t > 0 and f else None})
        out["components"] = rows
        out["unattributedKernels"] = unmatched
        print(f"  HLO module {out['hloModule']}; kernels not tied to an instruction: {unmatched}")
        lost = per_class.get("unattributed", 0.0) / max(total, 1e-9)
        if lost > 0.05:
            print(f"  WARNING: {lost:.0%} of kernel time is not tied to an instruction of this dump; "
                  "it is probably another model's or another entry's")
        print(f"  {'component':32s} {'ms/exec':>9s} {'share':>6s} {'MB touched':>11s} {'floor ms':>9s} {'GB/s':>7s} {'TFLOPS':>7s}")
        for r in rows:
            print(f"  {r['component'][:32]:32s} {r['msPerExecution']:9.3f} {r['share']*100:5.1f}% "
                  f"{r['bytes']/1e6:11.1f} {r['floorMs']:9.3f} {r['gbPerS'] if r['gbPerS'] is not None else '-':>7} "
                  f"{r['tflops'] if r['tflops'] is not None else '-':>7}")
        print(f"  (floor: the bytes at {bw/1e9:.0f} GB/s)")
        # Dots: operand and result types, and weight converts.
        dots = defaultdict(int)
        for ln in hlo.text.splitlines():
            m = re.search(r"= (\w+)\[[^\]]*\][^ ]* dot\(%([\w.\-]+), %([\w.\-]+)\)", ln)
            if m:
                a = hlo.instr.get(m.group(2), "") or ""
                b = hlo.instr.get(m.group(3), "") or ""
                ta = first_shape(a.split(" = ", 1)[1])[0] if " = " in a else "?"
                tb = first_shape(b.split(" = ", 1)[1])[0] if " = " in b else "?"
                prec = re.search(r"operand_precision=\{([^}]*)\}", ln)
                dots[(ta, tb, m.group(1), prec.group(1) if prec else "default")] += 1
            if "__cublas" in ln:
                m2 = re.search(r"custom-call\(%([\w.\-]+), %([\w.\-]+)\)", ln)
                if m2:
                    ta = first_shape(hlo.instr.get(m2.group(1), "= ?").split(" = ", 1)[1])[0]
                    tb = first_shape(hlo.instr.get(m2.group(2), "= ?").split(" = ", 1)[1])[0]
                    res = first_shape(ln.split(" = ", 1)[1])[0]
                    prec = re.search(r'"precision_config":\{"operand_precision":\[([^\]]*)\]', ln)
                    alg = re.search(r'"algorithm":"(\w+)"', ln)
                    dots[("cublas " + ta, tb, res, (prec.group(1) if prec else "") + (" " + alg.group(1) if alg else ""))] += 1
        out["dots"] = [{"lhs": k[0], "rhs": k[1], "result": k[2], "precision": k[3], "count": v}
                       for k, v in sorted(dots.items(), key=lambda kv: -kv[1])]
        print("  dots (lhs x rhs -> result, precision: count):")
        for d in out["dots"][:10]:
            print(f"    {d['lhs']} x {d['rhs']} -> {d['result']}, {d['precision']}: {d['count']}")
        mats = materialized_converts(hlo, pnames)
        out["materializedWeightConverts"] = mats
        print(f"  weights converted into their own f32 buffer: {len(mats)}"
              + (f" ({sum(x['bytes'] for x in mats)/1e6:.0f} MB), e.g. {mats[0]['weight']}" if mats else ""))
        # Largest intermediate buffers (the temporaries a program allocates).
        temps = []
        for ln in hlo.comps.get(hlo.entry, []):
            m = re.match(r"(?:ROOT )?%([\w.\-]+) = (\S+) (fusion|custom-call)\(", ln)
            if m:
                temps.append((shape_bytes(m.group(2)), m.group(1), m.group(2)[:80]))
        temps.sort(reverse=True)
        out["largestResults"] = [{"instruction": t[1], "shape": t[2], "bytes": t[0]} for t in temps[:6]]
        print("  largest instruction results: " + "; ".join(f"{t[1]} {t[2]} {t[0]/2**20:.0f} MiB" for t in temps[:4]))
    print("  top kernels (ms per execution, launches): " +
          "; ".join(f"{k['kernel']} {k['msPerExecution']:.3f} x{k['launchesPerExecution']:.0f}" for k in out["topKernels"][:8]))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
