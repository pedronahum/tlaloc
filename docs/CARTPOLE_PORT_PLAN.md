# CartPole Benchmark Port — Plan

**Status:** Implementation **2/3 phases complete** (§0.4.181 amendment, refreshed §0.4.191); Phase 3 gated on **rectangular MATMUL** in synthesis (square-MATMUL surface closed §0.4.187 + §0.4.189).

**Ship state** (updated 2026-04-27):

| Phase | Plan estimate | Actual | Closing entry |
|---|---|---|---|
| Phase 0a-1 — scalar sin/cos | 1 firing | 1 firing | §0.4.166 |
| Phase 0a-2 — scalar abs | 1 firing | 1 firing | §0.4.167 |
| Phase 0b — max/sign as IF chains | 1 firing | not needed (Phase 1 used direct IF) | n/a |
| Phase 0c — plugin MATMUL (square) | deferred | **3 firings (slices a/b/c)** | §0.4.185 + §0.4.186 + §0.4.187 |
| Phase 0c-followup — DTensor → Float bridge + irMatmul/irTranspose | not in original plan | **2 firings** | §0.4.188 + §0.4.189 |
| Phase 0c-rectangular — per-operand IrType tracking | not in original plan | **NOT DONE** (multi-session) | pending |
| Phase 1 — physics-only port | 1 firing | 6 firings (first attempt §0.4.168 hit downstream gate; closure via §0.4.169–§0.4.175 diagnostic + structural arc) | §0.4.175 |
| Phase 2 — B=3 loop with state passing | 2 firings | 1 firing (regression test only; no new code) | §0.4.178 |
| Phase 3 — NN + outer training loop | 4-5 firings | **NOT DONE** (gated on Phase 0c-rectangular) | pending |
| **CartPole-specific total (closed)** | **8-10 firings (Phases 0a + 1 + 2)** | **11 firings (165 + 166 + 167 + 168 + 175 + 178 + 185 + 186 + 187 + 188 + 189)** | |

Plus **8 firings of cross-cutting platform work** (§0.4.169–§0.4.176) that closed Phase 2 #1 (Plugin IR-side synthesis closure) for the scalar-arithmetic surface — discovered through the CartPole Phase 1 attempt (§0.4.168). Combined: **19 firings actually shipped** for the closed pieces vs. 10-12 originally planned. The plan didn't budget either the diagnostic + structural arc OR the Phase 0c slice fan-out.

**Phase 0b never landed** because Phase 1's source uses the direct `if (maxArg > 0.0f) maxArg else 0.0f` IF expression — no `max` / `sign` extension needed. Phase 0b stays in the plan as a future addition if a different CartPole-style port surfaces the need.

**Phase 0c is now SQUARE-MATRIX-CLOSED**: §0.4.185 wired Rank2/3 param recognition (FIR side); §0.4.186 widened `DxirToIrSynthesis` to accept rank-1/2/3 F32 + lowered rank-N F32 const through `broadcastLike`; §0.4.187 added `:core.ops.matmul` to `BINARY_OP_MAP`. §0.4.188 then landed the DTensor → Float bridge (via `:core.ops.toFloat` + `:core.ops.sum`) so lambda bodies can return Float computed from tensor intermediates. §0.4.189 added `irTranspose` + `irMatmul` synthesis arms, and verified the first end-to-end MATMUL gradient: `grad { a -> (a matmul a).sum().toFloat() }` on A=[[1,2],[3,4]] produces [[7,11],[9,13]] within `1e-3` tolerance.

**Phase 3 is now gated on rectangular MATMUL**, NOT square. CartPole's NN forward (`a = sign(tanh(relu(relu(X·W1)W2)W3) - ε)`) uses rectangular weights (X is batch×4, W1 is 4×8, W2 is 8×4, W3 is 4×1). The current synthesis surface threads ONE shape parameter through `tensorIrType` — works for `Rank2<R, R>` square shapes but not for `Rank2<R, K> matmul Rank2<K, C>` where R, K, C are distinct. Per the §0.4.190 register, the rectangular-MATMUL widening is a multi-session item: extend `SynthesisContext` to carry per-operand IrTypes (likely a `Map<DxirNode.id, IrType>`); rewrite `irMatmul` / `irTranspose` to use per-operand types instead of the single `tensorIrType`.

The historical planning content below is preserved verbatim for reference.

---

**Target benchmark:** Deep-reinforcement-learning training of a cart-pole control system,
per the OOPSLA 2021 paper §3 (`docs/papers/coarsening-autodiff.txt:247-360`) and §7.2.
The paper reports **1.22×–4.42× end-to-end speedup from coarsening on this benchmark**
(see lines 1117–1124 of the paper text), driven by a combination of numerical-stability
benefits + symbolic differentiation collapsing chains of state-update operations.

**Why CartPole is non-trivial:** the paper names CartPole's characteristic feature as
"a combination of matrix-based Deep Neural Network and scalar-based environment
simulations". The forward pass through one time step combines:
- A small Neural Network (`a = sign(tanh(relu(relu(X·W1)W2)W3) - e)`) — tensor MATMULs.
- Physics state update with **`sin` / `cos` / `abs` / `sign` / `max`** — scalars not
  yet exposed through the K2 plugin (similar gap to scalar `exp` / `log` pre-§0.4.158).
- A nested loop over B time steps inside an outer training loop.

This is the **first benchmark where the K2 plugin's BINARY_OP_MAP / UNARY_OP_MAP
needs substantial widening** (HMC needed only `exp` / `log` from this list). The
plan below treats those plumbing pieces as Phase 0 prerequisites.

## Reference forward pass for one time step (paper Eq. 3.1 + Figure 2c)

Per-time-step computation (with constants pre-folded per the paper's Equation 3.1 worked
example):

```
at = sign(tanh(relu(relu(Xt·W1)W2)W3) - ε)        # action ∈ {-1, +1}
rt = 9·at + 0.045·xt,3²·sin(xt,2)
qt = (9.8·sin(xt,2) - rt·cos(xt,2)) / (0.65 - 0.4·cos²(xt,2))
pt = rt - 0.045·qt·cos(xt,2)
xt+1,0 = xt,0 + 0.02·xt,1
xt+1,1 = xt,1 + 0.02·pt
xt+1,2 = xt,2 + 0.02·xt,3
xt+1,3 = xt,3 + 0.02·qt
lt+1 = (0.5 - max(0, (2.4 - |xt+1,0|) · (0.21 - |xt+1,2|)))²
```

Total loss: `L = lt+1 + lt+2 + lt+3` over B=3 time steps.

The differentiation target is **`dL/dat`** (the action's gradient w.r.t. the loss),
which the paper unrolls in §3 as a 29-operation expression.

## Tlaloc gap analysis (as of §0.4.164)

Per-primitive support map for the CartPole forward pass:

| Sub-expression | Op kind | Status | Notes |
|---|---|---|---|
| `relu(...)` | `RELU` | ✅ Shipped | §0.4.7 (scalar + tensor, both via UNARY_OP_MAP) |
| `tanh(...)` | `TANH` | ✅ Shipped | tensor `:core.ops.tanh` exists; **scalar `Float.tanh()` plugin lowering missing** |
| `sin(...)` | NEW | ⬜ Missing | new `OpKind.SIN`, `SinRule` (`d/dx sin = cos`), scalar `:core.sin` extension, plugin map entry |
| `cos(...)` | NEW | ⬜ Missing | new `OpKind.COS`, `CosRule` (`d/dx cos = -sin`), scalar `:core.cos` extension, plugin map entry |
| `\|x\|` | `ABS` | 🟡 Partial | `OpKind.ABS` exists in dxir; **`AbsRule` missing** (per §0.4.51's `rejectsUnsupportedOp` test which uses ABS as the canonical unregistered op); scalar `:core.abs` extension missing; plugin map entry missing |
| `sign(x)` | NEW or via IF | ⬜ Missing | could express as `if (x > 0) 1.0f else (if (x < 0) -1.0f else 0.0f)` (zero gradient via IF; sign is non-differentiable mathematically). Or new `OpKind.SIGN` + custom rule. |
| `max(a, b)` (binary) | NEW or via IF | ⬜ Missing | could express as `if (a > b) a else b`. Or new `OpKind.MAX_BINARY`. The reduction `OpKind.MAX` over a tensor exists but is different. |
| `x²` | `MUL(x, x)` | ✅ Shipped | scalar arithmetic |
| `MATMUL` (rank-2) | `MATMUL` | 🟡 Partial | tensor + IR + emitter all support it (§0.4.135 / §0.4.137 / §0.4.138); **K2 plugin's `BINARY_OP_MAP` doesn't recognise `matmul(...)` calls** (the same gap §0.4.158 flagged for HMC). |
| Nested loop over B time steps | nested `WHILE` | 🟡 Partial | §0.4.161 covers structural region-recursive C5; §0.4.163 fixes FIR `collectMutatedTargets` but **HMC Phase 3 nested-loop end-to-end is checkpointed** pending IR-side synthesis closure. Same blocker. |

**Five new plumbing items emerge**:

1. **Scalar `sin` / `cos`** — direct precedent: §0.4.158's scalar `exp` / `log` arc.
2. **Scalar `abs`** — `OpKind.ABS` exists but no `AbsRule` and no plugin lowering.
3. **Binary `max(a, b)`** — express via IF (no new op needed) OR add `OpKind.MAX_BINARY`.
4. **`sign(x)`** — express via two-IF chain (since sign is non-differentiable) OR add `OpKind.SIGN`.
5. **Plugin MATMUL recognition** — required for the neural-net half. Same work item as the §0.4.158 / §0.4.164 register's "MATMUL through K2 plugin" entry.

Plus the existing **HMC Phase 3 nested-loop blocker** (Plugin IR-side synthesis closure, §17 step 6) is shared.

## Three-phase migration

The Plan structure mirrors `docs/HMC_PORT_PLAN.md`'s three-phase approach with a
**Phase 0** added for the plumbing prerequisites. Total estimate: ~10-12 firings
(2-3 Phase 0 + 1 Phase 1 + 2 Phase 2 + 4-5 Phase 3).

### Phase 0 — plumbing prerequisites

- **Phase 0a (1-2 firings):** scalar `sin` / `cos` + `AbsRule` + plugin map entries.
  Direct port of §0.4.158's pattern. New OpKinds (SIN, COS) added to dxir. New VJP
  rules. New `:core/DScalar.kt` extensions. New plugin UNARY_OP_MAP entries. New
  `DxirToIrSynthesis.irSin` / `irCos` / `irAbs` arms (mirror `irExp` / `irLog`).
  Test scaffold mirrors `ScalarExpLogTest.kt`.

- **Phase 0b (1 firing):** binary `max(a, b)` and `sign(x)` lowered as `if/else`
  chains in the FIR side. No new OpKinds needed. The lowering recognises
  `kotlin.math.max(a, b)` (or `:core.max(a, b)`) and produces an IF expression.
  The `sign` extension wraps `if (x > 0) 1.0f else (if (x < 0) -1.0f else 0.0f)`.
  Both are non-differentiable, so VJP rules emit zero contributions (mirror how
  STEP is handled — it's in the registry but produces no gradient flow).

- **Phase 0c (deferred to a later phase OR shared with HMC):** plugin MATMUL
  recognition. This is the §0.4.164 register's "MATMUL through K2 plugin" item.
  Phase 1 of CartPole sidesteps it by hard-coding the neural-net output `at` as
  a Float input to the differentiable physics computation; Phase 2 brings
  MATMUL into the loop.

### Phase 1 — physics-only port at one time step

**Deliverable:** a compiler-plugin test that ports the per-time-step physics
computation (everything from `rt` to `lt+1`) given a hard-coded action `at` (skips
the neural-net half). Differentiates the loss `lt+1` w.r.t. `at` and one of the
state components. Verifies against finite-differencing.

**Source shape:**
```kotlin
val g = grad { packed: DTensor<Rank1<Sym>, F32> ->
    val at = packed[0]    // action (the differentiation target)
    val x0 = packed[1]    // xt,0 (cart position)
    val x1 = packed[2]    // xt,1 (cart velocity)
    val x2 = packed[3]    // xt,2 (pole angle)
    val x3 = packed[4]    // xt,3 (pole angular velocity)
    val rt = 9.0f * at + 0.045f * x3 * x3 * x2.sin()
    val cosX2 = x2.cos()
    val qt = (9.8f * x2.sin() - rt * cosX2) / (0.65f - 0.4f * cosX2 * cosX2)
    val pt = rt - 0.045f * qt * cosX2
    val xn0 = x0 + 0.02f * x1
    val xn2 = x2 + 0.02f * x3
    val maxArg = (2.4f - xn0.abs()) * (0.21f - xn2.abs())
    val clipped = if (maxArg > 0.0f) maxArg else 0.0f
    val term = 0.5f - clipped
    term * term  // lt+1
}
```

**Tlaloc support required:** Phase 0a (scalar sin/cos/abs) and Phase 0b
(binary max via IF). All other primitives are scalar arithmetic + GATHER + IF.

**Acceptance:** numerical agreement with finite-difference at one (at, X)
configuration, 1e-3 absolute / 5e-3 relative tolerance.

**Estimated:** 1 firing.

### Phase 2 — loop form over B time steps

**Deliverable:** rewrite Phase 1's port to loop over B=3 time steps with
state carried as `var` accumulators. The state update produces a new state each
iteration; the loss accumulates.

**New surface required:** none beyond Phase 0 + Phase 1 — loop form follows
HookeanSpring's N=10 chain pattern (§0.4.47). C5 (concrete B=3) unrolls.

**Acceptance:** same gradient as Phase 1's first-time-step at the matching
configuration; finite-difference cross-check at the multi-step β.

**Estimated:** 2 firings.

### Phase 3 — neural-net portion + outer training loop

**Deliverable:** complete CartPole forward pass including the neural net
(`a = sign(tanh(relu(relu(X·W1)W2)W3) - ε)`) and the outer `while (loss > threshold)`.
Differentiates loss w.r.t. NN weights W1/W2/W3.

**New surface required:** Phase 0c (plugin MATMUL) + the IR-side synthesis closure
(shared with HMC Phase 3 nested-loop). Both are documented as Phase-2 blockers
in the §0.4.164 register.

**Acceptance:** gradient agrees with finite-differencing on a small NN
(W1: 4×8, W2: 8×4, W3: 4×1, batch B=3 time steps).

**Estimated:** 4-5 firings.

## Phase 0a first-slice — concrete next firing

The immediate next firing should land Phase 0a's first slice — scalar `sin` / `cos`
plumbing, mirroring §0.4.158's exact pattern:

1. Add `Float.sin()` / `Double.sin()` / `FloatScalar.sin()` / `DoubleScalar.sin()` /
   `DScalar.sin()` extensions in `:core/DScalar.kt`.
2. Same for `cos`.
3. Add `OpKind.SIN`, `OpKind.COS` to `:ir/OpKind.kt`.
4. Add `SinRule` (returns `cos(x) * upstream` for x.id), `CosRule` (returns
   `-sin(x) * upstream`) in `:ir/Vjp.kt`.
5. Add interpreter arms in `DxirInterpreter` for SIN, COS.
6. Add emitter arms in `:stablehlo/Emitter` for SIN, COS (lowers to
   `stablehlo.sine` / `stablehlo.cosine`).
7. Add UNARY_OP_MAP entries `io.tlaloc.core.sin` → `OpKind.SIN`, `io.tlaloc.core.cos`
   → `OpKind.COS` in `FirLambdaToDxirLowering`.
8. Add `irSin` / `irCos` arms in `DxirToIrSynthesis` (mirror `irExp` / `irLog`).
9. Add `ScalarSinCosTest.kt` with 4-6 tests covering single-call, composed, and a
   physics-pattern term.

**Estimated:** 1 firing for the full Phase 0a-1 slice (sin + cos together — they're
mutual gradients so landing both at once avoids double-touching the same files).

## Out of scope

- The CartPole training harness's outer `while (loss > threshold)` loop with
  symbolic threshold — the optimisation termination criterion. Treated as
  out-of-scope for the benchmark port; the benchmark exercises one-step training
  iteration's gradient.
- Reinforcement-learning policy (the action-selection process beyond `sign(NN(X) - ε)`)
  — defer.
- Trajectory recording / replay buffers.
- The actual neural-network weight update (`w = w - η * dw`) — Tlaloc differentiates;
  the SGD step is just scalar arithmetic that doesn't need the AD path.
- HMC / QWOP / head-to-head harness — separate plans.

## Why this plan is structured differently from HMC's

HMC Phase 1 had **all primitives shipped** at the time of `docs/HMC_PORT_PLAN.md`'s
landing (§0.4.157) — the gap analysis named only two 🟡 blockers (per-element mask,
WHILE-in-WHILE), both with mitigations. CartPole has **five new plumbing items** before
Phase 1 can even attempt a port. Phase 0 makes those explicit so the next firings
know what's needed to land them in the right order, rather than discovering them
mid-implementation as HMC did (the §0.4.158 / §0.4.162 / §0.4.163 surprises).
