package io.tlaloc.nn

import io.tlaloc.autograd.Tracer
import io.tlaloc.autograd.plus
import io.tlaloc.autograd.times
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.Shape
import io.tlaloc.core.ops.plus
import io.tlaloc.core.ops.times

// Fixed-step ODE integration, y' = f(t, y). The Tracer overloads record the unrolled
// integrator on the capture tape, so a captured function that integrates is
// differentiated by DxirReverseTransform / DxirForwardTransform like any other: with
// respect to the initial state and to every tensor `f` reads from the tape (its
// parameters). The DTensor overloads compute the same steps on the host.
//
// Inside `grad { }` a lambda cannot call these (the plugin lowers only known
// operations); write the loop in the lambda instead, as
// examples/differentiable-physics does. The step count must be a constant there.

/**
 * Integrates `y' = f(t, y)` from `y(t0) = y0` over [steps] steps of size [dt] with
 * the classical fourth-order Runge–Kutta method, and returns `y(t0 + steps·dt)`:
 *
 *     k1 = f(t, y)
 *     k2 = f(t + dt/2, y + dt/2·k1)
 *     k3 = f(t + dt/2, y + dt/2·k2)
 *     k4 = f(t + dt, y + dt·k3)
 *     y ← y + dt/6·(k1 + 2·k2 + 2·k3 + k4)
 *
 * The global error is O(dt⁴). The captured graph grows by four evaluations of [f]
 * per step, and its gradient is backpropagation through every step (the memory of
 * the reverse pass is proportional to [steps]; there is no adjoint-ODE solve).
 */
fun <S : Shape> rk4(
    y0: Tracer<S>,
    t0: Float,
    dt: Float,
    steps: Int,
    f: (t: Float, y: Tracer<S>) -> Tracer<S>,
): Tracer<S> = rk4Trajectory(y0, t0, dt, steps, f).last()

/** [rk4], returning the states at every step: `steps + 1` values, `y0` first. */
fun <S : Shape> rk4Trajectory(
    y0: Tracer<S>,
    t0: Float,
    dt: Float,
    steps: Int,
    f: (t: Float, y: Tracer<S>) -> Tracer<S>,
): List<Tracer<S>> {
    require(steps >= 0) { "rk4: steps must be ≥ 0; got $steps" }
    val out = ArrayList<Tracer<S>>(steps + 1)
    var y = y0
    out += y
    for (i in 0 until steps) {
        val t = t0 + i * dt
        val k1 = f(t, y)
        val k2 = f(t + dt / 2, y + k1 * (dt / 2))
        val k3 = f(t + dt / 2, y + k2 * (dt / 2))
        val k4 = f(t + dt, y + k3 * dt)
        y = y + (k1 + k2 * 2f + k3 * 2f + k4) * (dt / 6)
        out += y
    }
    return out
}

/** [rk4] on host tensors: the same steps, evaluated directly. */
fun <S : Shape> rk4(
    y0: DTensor<S, F32>,
    t0: Float,
    dt: Float,
    steps: Int,
    f: (t: Float, y: DTensor<S, F32>) -> DTensor<S, F32>,
): DTensor<S, F32> {
    require(steps >= 0) { "rk4: steps must be ≥ 0; got $steps" }
    var y = y0
    for (i in 0 until steps) {
        val t = t0 + i * dt
        val k1 = f(t, y)
        val k2 = f(t + dt / 2, y + k1 * (dt / 2))
        val k3 = f(t + dt / 2, y + k2 * (dt / 2))
        val k4 = f(t + dt, y + k3 * dt)
        y = y + (k1 + k2 * 2f + k3 * 2f + k4) * (dt / 6)
    }
    return y
}
