#!/usr/bin/env python3
"""Oracle fixture for CAUSAL_CONV1D and GATED_DELTA_RULE.

Runs transformers' own Qwen3.5 functions (``causal_conv1d_fn``,
``causal_conv1d_update``, ``torch_recurrent_gated_delta_rule`` and, as a
cross-check, ``torch_chunk_gated_delta_rule``) in float32 on the CPU over a
small block of three rows and two calls, and writes inputs and outputs as JSON.
Run it in ``~/.local/venvs/vllm``:

    python gdn_ops_fixture.py > ir/src/jvmTest/resources/io/tlaloc/ir/passes/gdn_ops_fixture.json

Call 1 is a prefill block of T=5 tokens per row:
  row 0: a new sequence, positions 0..4, slot 2;
  row 1: padding (every slot -1);
  row 2: two padding tokens, then positions 7..9 continuing from slot 0.
Call 2 is a decode step (T=1): row 0 at position 5, row 1 padding, row 2 at 10,
reading the pools call 1 wrote.
"""

import json

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as m

B, T, HK, HV, DK, DV, C, K, S = 3, 5, 2, 4, 3, 2, 6, 4, 4


def rnd(*shape, scale=1.0):
    return (torch.randn(*shape, dtype=torch.float32) * scale)


def l2(x):
    return m.l2norm(x, dim=-1, eps=1e-6)


def main():
    torch.manual_seed(7)
    weight = rnd(K, C, scale=0.5)  # [K, C]: weight[j, c] = conv1d.weight[c, 0, j]
    conv_pool = rnd(S, K - 1, C)
    state_pool = rnd(S, HV, DK, DV, scale=0.3)

    slots1 = [[2, 2, 2, 2, 2], [-1] * 5, [-1, -1, 0, 0, 0]]
    pos1 = [[0, 1, 2, 3, 4], [0] * 5, [0, 0, 7, 8, 9]]
    slots2 = [[2], [-1], [0]]
    pos2 = [[5], [0], [10]]

    calls = []
    conv_in, state_in = conv_pool.clone(), state_pool.clone()
    for slots, pos in ((slots1, pos1), (slots2, pos2)):
        t = len(slots[0])
        x = rnd(B, t, C)
        q = l2(rnd(B, t, HK, DK)) * DK ** -0.5
        k = l2(rnd(B, t, HK, DK))
        v = rnd(B, t, HV, DV)
        g = -torch.nn.functional.softplus(rnd(B, t, HV))
        beta = torch.sigmoid(rnd(B, t, HV))

        y = torch.zeros(B, t, C)
        out = torch.zeros(B, t, HV, DV)
        conv_out, state_out = conv_in.clone(), state_in.clone()
        chunk_err = 0.0
        for b in range(B):
            live = [i for i in range(t) if slots[b][i] >= 0]
            if not live:
                continue
            f, slot = live[0], slots[b][live[0]]
            fresh = pos[b][f] == 0
            xs = x[b, f:].T.unsqueeze(0)  # [1, C, n]
            w = weight.T.contiguous()  # [C, K]
            if fresh:
                yb = m.causal_conv1d_fn(xs, w, None, activation=None)
                padded = torch.cat([torch.zeros(1, C, K - 1), xs], dim=-1)
                new_conv = padded[:, :, -(K - 1):]
            else:
                st = conv_in[slot].T.unsqueeze(0).clone()  # [1, C, K-1]
                yb = m.causal_conv1d_update(xs, st, w, None, activation=None)
                new_conv = st
            y[b, f:] = yb[0].T
            conv_out[slot] = new_conv[0].T

            rep = HV // HK
            qb = q[b:b + 1, f:].repeat_interleave(rep, dim=2)
            kb = k[b:b + 1, f:].repeat_interleave(rep, dim=2)
            init = None if fresh else state_in[slot:slot + 1].clone()
            ob, sb = m.torch_recurrent_gated_delta_rule(
                qb * DK ** 0.5, kb, v[b:b + 1, f:], g=g[b:b + 1, f:], beta=beta[b:b + 1, f:],
                initial_state=init, output_final_state=True, use_qk_l2norm_in_kernel=False,
            )
            oc, sc = m.torch_chunk_gated_delta_rule(
                qb * DK ** 0.5, kb, v[b:b + 1, f:], g=g[b:b + 1, f:], beta=beta[b:b + 1, f:],
                initial_state=init, output_final_state=True, use_qk_l2norm_in_kernel=False,
            )
            chunk_err = max(chunk_err, (ob - oc).abs().max().item(), (sb - sc).abs().max().item())
            out[b, f:] = ob[0]
            state_out[slot] = sb[0]

        calls.append({
            "T": t, "tokenSlots": slots, "positions": pos,
            "x": x.flatten().tolist(), "q": q.flatten().tolist(), "k": k.flatten().tolist(),
            "v": v.flatten().tolist(), "g": g.flatten().tolist(), "beta": beta.flatten().tolist(),
            "y": y.flatten().tolist(), "out": out.flatten().tolist(),
            "convPool": conv_out.flatten().tolist(), "statePool": state_out.flatten().tolist(),
            "recurrentVsChunk": chunk_err,
        })
        conv_in, state_in = conv_out, state_out

    json.dump({
        "dims": {"B": B, "HK": HK, "HV": HV, "DK": DK, "DV": DV, "C": C, "K": K, "S": S},
        "weight": weight.flatten().tolist(),
        "convPool": conv_pool.flatten().tolist(),
        "statePool": state_pool.flatten().tolist(),
        "calls": calls,
    }, fp=__import__("sys").stdout)


if __name__ == "__main__":
    main()
