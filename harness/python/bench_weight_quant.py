"""Time a Muse Glimmer-sized projection with its weights in several formats,
through PJRT (the ctypes binding in tlaloc_pjrt.py) on a CUDA device.

For batch B, one program multiplies x [B, 6656] by W1 [6656, 19968] and the
result by W2 [19968, 6656], 16 times over in a straight line (32 projections),
and the time of one execution divided by 32 is reported with the weight bytes
it read per second. The formats:

  bf16:deq        bf16 weights, input rounded to bf16, dot into f32
  bf16:f32        bf16 weights widened to f32, dot in f32
  i8:deq          int8 weights widened to bf16 in the program, dot into f32,
                  times a per-output-channel scale (what -PweightQuant=int8 emits)
  f8e4m3fn:deq    the same with float8 e4m3 weights
  f8e4m3fn:native8  input and weights both f8e4m3, one dot into f32
  i4:deq          int4 weights widened to bf16
  f4e2m1fn:deq    float4 e2m1 weights widened to bf16
  nvfp4:16        __op$block_scaled_dot of f4e2m1 input and weights with
                  f8e4m3 scales per 16 values (NVFP4); nvfp4:32 with
                  f8e8m0 scales per 32 (MXFP4)

The weights are random (normal, std 0.02) and converted on the device, so the
numbers are times, not accuracies.

  TLALOC_PJRT_MEMORY_FRACTION=0.08 python bench_weight_quant.py [variant,...]

Needs numpy and ml_dtypes, and the plugin at --plugin (default: the one
triton/fetch_pjrt_plugin.sh puts in triton/pjrt/xla_cuda13).
"""

import argparse
import ctypes
import os
import statistics
import sys
import time

import ml_dtypes
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tlaloc_pjrt as tp  # noqa: E402

D, F, N = 6656, 19968, 16
HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.join(HERE, "..", "..", "triton", "pjrt", "xla_cuda13", "xla_cuda_plugin.so")
TYPES = {"bf16": ("bf16", 16), "i8": ("i8", 8), "f8e4m3fn": ("f8E4M3FN", 8), "i4": ("i4", 4),
         "f4e2m1fn": ("f4E2M1FN", 4)}
PJRT_S8, PJRT_F32, PJRT_BF16 = 2, 11, 13


def prep_program(t, rows, cols):
    """bf16 weights in, the same weights in type t out (scaled to use its range)."""
    mt = TYPES[t][0]
    k = {"i8": 60.0, "i4": 4.0, "f8e4m3fn": 20.0, "f4e2m1fn": 20.0}[t]
    ty = f"tensor<{rows}x{cols}x"
    body = [f"%k = stablehlo.constant dense<{k}> : {ty}f32>",
            f"%wf = stablehlo.convert %w : ({ty}bf16>) -> {ty}f32>",
            f"%s = stablehlo.multiply %wf, %k : {ty}f32>"]
    if t in ("i8", "i4"):
        lim = 127.0 if t == "i8" else 7.0
        body += [f"%r = stablehlo.round_nearest_even %s : {ty}f32>",
                 f"%lo = stablehlo.constant dense<{-lim}> : {ty}f32>",
                 f"%hi = stablehlo.constant dense<{lim}> : {ty}f32>",
                 f"%c = stablehlo.clamp %lo, %r, %hi : {ty}f32>",
                 f"%q = stablehlo.convert %c : ({ty}f32>) -> {ty}{mt}>"]
    else:
        body += [f"%q = stablehlo.convert %s : ({ty}f32>) -> {ty}{mt}>"]
    lines = "\n    ".join(body)
    return (f"module @prep {{\n  func.func @main(%w: {ty}bf16>) -> {ty}{mt}> {{\n    {lines}\n"
            f"    return %q : {ty}{mt}>\n  }}\n}}")


def bench_program(t, b, mode):
    mt = TYPES[t][0]
    xs, hs = f"tensor<{b}x{D}xf32>", f"tensor<{b}x{F}xf32>"

    def proj(x, w, win, wout, xt, yt, sc, tag):
        lines = []
        if mode == "f32":
            lines.append(f"%{tag}_wf = stablehlo.convert {w} : (tensor<{win}x{wout}x{mt}>) -> tensor<{win}x{wout}xf32>")
            lines.append(f"%{tag}_y = stablehlo.dot_general {x}, %{tag}_wf, contracting_dims = [1] x [0] : "
                         f"({xt}, tensor<{win}x{wout}xf32>) -> {yt}")
        elif mode == "native8":
            lines.append(f"%{tag}_xq = stablehlo.convert {x} : ({xt}) -> tensor<{b}x{win}x{mt}>")
            lines.append(f"%{tag}_y = stablehlo.dot_general %{tag}_xq, {w}, contracting_dims = [1] x [0] : "
                         f"(tensor<{b}x{win}x{mt}>, tensor<{win}x{wout}x{mt}>) -> {yt}")
        else:
            lines.append(f"%{tag}_xb = stablehlo.convert {x} : ({xt}) -> tensor<{b}x{win}xbf16>")
            wb = w
            if mt != "bf16":
                wb = f"%{tag}_wb"
                lines.append(f"{wb} = stablehlo.convert {w} : (tensor<{win}x{wout}x{mt}>) -> tensor<{win}x{wout}xbf16>")
            lines.append(f"%{tag}_y = stablehlo.dot_general %{tag}_xb, {wb}, contracting_dims = [1] x [0] : "
                         f"(tensor<{b}x{win}xbf16>, tensor<{win}x{wout}xbf16>) -> {yt}")
        out = f"%{tag}_y"
        if mt != "bf16":
            lines.append(f"%{tag}_sb = stablehlo.broadcast_in_dim {sc}, dims = [1] : (tensor<{wout}xf32>) -> {yt}")
            lines.append(f"%{tag}_z = stablehlo.multiply {out}, %{tag}_sb : {yt}")
            out = f"%{tag}_z"
        return lines, out

    lines, cur = [], "%x0"
    for i in range(N):
        l1, o1 = proj(cur, "%w1", D, F, xs, hs, "%s1", f"a{i}")
        l2, o2 = proj(f"%t{i}", "%w2", F, D, hs, xs, "%s2", f"b{i}")
        lines += l1 + [f"%t{i} = stablehlo.tanh {o1} : {hs}"] + l2 + [f"%x{i + 1} = stablehlo.tanh {o2} : {xs}"]
        cur = f"%x{i + 1}"
    body = "\n    ".join(lines)
    return (f"module @bench {{\n  func.func @main(%x0: {xs}, %w1: tensor<{D}x{F}x{mt}>, %w2: tensor<{F}x{D}x{mt}>, "
            f"%s1: tensor<{F}xf32>, %s2: tensor<{D}xf32>) -> {xs} {{\n    {body}\n    return {cur} : {xs}\n  }}\n}}")


def nvfp4_programs(block, b):
    st = "f8E4M3FN" if block == 16 else "f8E8M0FNU"
    kb = D // block
    prep = (f"module @p {{ func.func @main(%w: tensor<{F}x{D}xbf16>) -> (tensor<{F}x{D}xf4E2M1FN>, tensor<{F}x{kb}x{st}>) {{\n"
            f"  %k = stablehlo.constant dense<50.0> : tensor<{F}x{D}xbf16>\n"
            f"  %s = stablehlo.multiply %w, %k : tensor<{F}x{D}xbf16>\n"
            f"  %wq = stablehlo.convert %s : (tensor<{F}x{D}xbf16>) -> tensor<{F}x{D}xf4E2M1FN>\n"
            f"  %sc = stablehlo.constant dense<0.02> : tensor<{F}x{kb}x{st}>\n"
            f"  return %wq, %sc : tensor<{F}x{D}xf4E2M1FN>, tensor<{F}x{kb}x{st}>\n}} }}")
    lines = []
    for i in range(2 * N):
        lines += [f"%c{i} = stablehlo.constant dense<{1.0 + i * 0.01}> : tensor<{b}x{D}xf32>",
                  f"%x{i} = stablehlo.multiply %x, %c{i} : tensor<{b}x{D}xf32>",
                  f"%xq{i} = stablehlo.convert %x{i} : (tensor<{b}x{D}xf32>) -> tensor<{b}x{D}xf4E2M1FN>",
                  f"%y{i} = stablehlo.custom_call @__op$block_scaled_dot(%xq{i}, %wq, %xs, %ws) : "
                  f"(tensor<{b}x{D}xf4E2M1FN>, tensor<{F}x{D}xf4E2M1FN>, tensor<{b}x{kb}x{st}>, "
                  f"tensor<{F}x{kb}x{st}>) -> tensor<{b}x{F}xf32>",
                  f"%a{i} = stablehlo.add " + (f"%a{i - 1}, %y{i}" if i else "%y0, %y0") + f" : tensor<{b}x{F}xf32>"]
    body = "\n  ".join(lines)
    run = (f"module @m {{ func.func @main(%x: tensor<{b}x{D}xf32>, %wq: tensor<{F}x{D}xf4E2M1FN>, "
           f"%ws: tensor<{F}x{kb}x{st}>) -> tensor<{b}x{F}xf32> {{\n"
           f"  %xs = stablehlo.constant dense<1.0> : tensor<{b}x{kb}x{st}>\n  {body}\n"
           f"  return %a{2 * N - 1} : tensor<{b}x{F}xf32>\n}} }}")
    return prep, run


def upload(client, dev, arr, code):
    arr = np.ascontiguousarray(arr)
    return client._buffer_from_host(dev, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes, code, list(arr.shape))


def timed(exe, args, dev, reps):
    for _ in range(3):
        exe.execute(args, dev)[0].close()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        exe.execute(args, dev)[0].close()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) / (2 * N)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variants", nargs="?", default="bf16:deq,bf16:f32,i8:deq,f8e4m3fn:deq,f8e4m3fn:native8,"
                                                     "i4:deq,f4e2m1fn:deq,nvfp4:16,nvfp4:32")
    ap.add_argument("--plugin", default=PLUGIN)
    ap.add_argument("--reps", type=int, default=7)
    args = ap.parse_args()
    api = tp.PjrtApi.load(os.path.abspath(args.plugin))
    with api.create_client("cuda") as client:
        dev = client.addressable_devices()[0]
        rng = np.random.default_rng(0)
        w1 = (rng.standard_normal((D, F), dtype=np.float32) * 0.02).astype(ml_dtypes.bfloat16).view(np.uint16)
        w2 = (rng.standard_normal((F, D), dtype=np.float32) * 0.02).astype(ml_dtypes.bfloat16).view(np.uint16)
        bw1, bw2 = upload(client, dev, w1, PJRT_BF16), upload(client, dev, w2, PJRT_BF16)
        bw1t = upload(client, dev, np.ascontiguousarray(w1.T), PJRT_BF16)
        del w1, w2
        s1 = upload(client, dev, np.full((F,), 1 / 60, np.float32), PJRT_F32)
        s2 = upload(client, dev, np.full((D,), 1 / 60, np.float32), PJRT_F32)
        cache = {"bf16": (bw1, bw2)}
        for v in args.variants.split(","):
            t, mode = v.split(":")
            for b in (1, 4):
                try:
                    x = upload(client, dev, rng.standard_normal((b, D), dtype=np.float32), PJRT_F32)
                    if t == "nvfp4":
                        prep, run = nvfp4_programs(int(mode), b)
                        wq, ws = client.compile(prep).execute([bw1t], dev)
                        per = timed(client.compile(run), [x, wq, ws], dev, args.reps)
                        wbytes = D * F // 2
                    else:
                        if t not in cache:
                            cache[t] = (client.compile(prep_program(t, D, F)).execute([bw1], dev)[0],
                                        client.compile(prep_program(t, F, D)).execute([bw2], dev)[0])
                        q1, q2 = cache[t]
                        per = timed(client.compile(bench_program(t, b, mode)), [x, q1, q2, s1, s2], dev, args.reps)
                        wbytes = D * F * TYPES[t][1] // 8
                    print(f"{v:18s} B={b}  {per * 1e3:7.3f} ms a projection  {wbytes / per / 1e9:6.1f} GB/s "
                          f"of weights ({wbytes / 1e6:.0f} MB)", flush=True)
                except Exception as e:  # noqa: BLE001 - report and go on to the next format
                    print(f"{v:18s} B={b}  FAILED: {str(e).splitlines()[0][:200]}", flush=True)


if __name__ == "__main__":
    main()
