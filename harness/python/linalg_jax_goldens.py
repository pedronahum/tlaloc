"""JAX reference values for Tlaloc's linear-algebra ops.

Prints the forward values and gradients that `LinalgJaxParityTest` (in :ir)
pins, as Kotlin `doubleArrayOf(...)` literals. Float64 throughout.

    python harness/python/linalg_jax_goldens.py

Generated with jax 0.10.0 on the CPU. The matrices are the ones the Kotlin
tests use; the losses are the same functions the tests differentiate.
"""

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp  # noqa: E402
from jax.scipy.linalg import solve_triangular  # noqa: E402

N = 4
# Slightly non-symmetric: both Tlaloc and jnp.linalg.cholesky factor (A + Aᵀ)/2.
SPD = jnp.array([
    [6.2, 1.1, -0.7, 0.4],
    [1.3, 5.1, 0.9, -1.2],
    [-0.5, 0.9, 4.8, 0.6],
    [0.4, -1.0, 0.6, 5.5],
])
# Lower triangle well conditioned; the upper triangle is junk the solve must ignore.
TRI = jnp.array([
    [2.0, 9.0, -7.0, 5.0],
    [0.6, 1.7, 8.0, -3.0],
    [-0.4, 0.3, 2.4, 4.0],
    [0.9, -0.8, 0.5, 1.9],
])
# Nonsymmetric; the first column's largest entry is in row 2, so LU pivots.
GEN = jnp.array([
    [0.5, 2.0, -1.0, 0.3],
    [1.2, -0.4, 0.8, 2.2],
    [-3.0, 0.7, 1.5, -0.2],
    [0.9, 1.1, -0.6, 1.4],
])
RHS = jnp.array([[0.7, -1.2], [0.4, 2.1], [-0.3, 0.8], [1.5, -0.6]])


def cube_sum(x):
    return jnp.sum(x * x * x)


def chol(a):
    return jnp.linalg.cholesky(a, symmetrize_input=True)


def spd_solve(a, b):
    return jax.scipy.linalg.cho_solve((chol(a), True), b)


def log_det(a):
    return 2.0 * jnp.sum(jnp.log(jnp.diag(chol(a))))


def inv(a):
    return spd_solve(a, jnp.eye(a.shape[0]))


def emit(name, x):
    flat = ", ".join(repr(float(v)) for v in jnp.ravel(x))
    print(f"    val {name} = doubleArrayOf({flat})")


def main():
    emit("choleskyValue", chol(SPD))
    emit("choleskyGrad", jax.grad(lambda a: cube_sum(chol(a)))(SPD))

    for lower in (True, False):
        for trans in (False, True):
            a = TRI if lower else TRI.T
            tag = f"{'Lower' if lower else 'Upper'}{'T' if trans else 'N'}"

            def loss(a, b, lower=lower, trans=trans):
                return cube_sum(solve_triangular(a, b, trans=1 if trans else 0, lower=lower))

            ga, gb = jax.grad(loss, argnums=(0, 1))(a, RHS)
            emit(f"solve{tag}Value", solve_triangular(a, RHS, trans=1 if trans else 0, lower=lower))
            emit(f"solve{tag}GradA", ga)
            emit(f"solve{tag}GradB", gb)

    emit("solveSpdValue", spd_solve(SPD, RHS))
    ga, gb = jax.grad(lambda a, b: cube_sum(spd_solve(a, b)), argnums=(0, 1))(SPD, RHS)
    emit("solveSpdGradA", ga)
    emit("solveSpdGradB", gb)

    emit("logDetValue", jnp.array([log_det(SPD)]))
    emit("logDetGrad", jax.grad(log_det)(SPD))
    emit("logDetHessian", jax.hessian(log_det)(SPD).reshape(N * N, N * N))

    emit("invValue", inv(SPD))
    emit("invGrad", jax.grad(lambda a: cube_sum(inv(a)))(SPD))

    for trans in (False, True):
        tag = "T" if trans else "N"

        def gsolve(a, b, trans=trans):
            return jnp.linalg.solve(a.T if trans else a, b)

        emit(f"solve{tag}Value", gsolve(GEN, RHS))
        ga, gb = jax.grad(lambda a, b: cube_sum(gsolve(a, b)), argnums=(0, 1))(GEN, RHS)
        emit(f"solve{tag}GradA", ga)
        emit(f"solve{tag}GradB", gb)

    emit("detValue", jnp.array([jnp.linalg.det(GEN)]))
    emit("detGrad", jax.grad(jnp.linalg.det)(GEN))
    emit("detHessian", jax.hessian(jnp.linalg.det)(GEN).reshape(N * N, N * N))


if __name__ == "__main__":
    main()
