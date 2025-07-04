# ------------------------------------------------------------
# 0. Imports and helpers
# ------------------------------------------------------------
import time
import jax
import jax.numpy as jnp
from jax import random, jit
from jax.scipy import linalg
jax.config.update("jax_enable_x64", False)   # float32 for speed / memory
jax.config.update("jax_default_matmul_precision", "float32")   # elevate matmul precision to float32 (on GPUs it defaults to mixed precision)

def walltime(f, *args, warm=1, runs=5):
    """Rough wall-clock timing helper."""
    # one warm-up (compilation) run
    f(*args).block_until_ready() if hasattr(f(*args), "block_until_ready") else None
    for _ in range(warm):
        f(*args).block_until_ready()
    # timed runs
    t0 = time.time()
    for _ in range(runs):
        out = f(*args).block_until_ready()
    return (time.time() - t0) / runs, out  # mean per-run time

# ------------------------------------------------------------
# 1. Synthetic batched data
# ------------------------------------------------------------
key = random.PRNGKey(0)

batch_shape = (256, 8)   # leading batch dims  (e.g. 2048 matrices total)
dim         = 32         # matrix dimension

A      = random.normal(key, batch_shape + (dim, dim))
Lambda = A @ jnp.swapaxes(A, -1, -2) + 1e-3 * jnp.eye(dim)   # SPD precision

_, key = random.split(key)
eta    = random.normal(key, batch_shape + (dim,))            # η = Λ μ

# ------------------------------------------------------------
# 2. Method 1 – inverse-then-multiply
# ------------------------------------------------------------
@jit
def mu_via_inverse(Lambda, eta):
    # Σ from Λ  (two triangular solves under the hood)
    Sigma = linalg.cho_solve(linalg.cho_factor(Lambda, lower=True),
                             jnp.broadcast_to(jnp.eye(dim), Lambda.shape))
    mu    = Sigma @ eta[...,None]  # μ = Σ η
    return mu.squeeze(-1)  # remove last dimension

# ------------------------------------------------------------
# 3. Method 2 – single triangular solve
# ------------------------------------------------------------
@jit
def mu_via_triangular(Lambda, eta):
    L = linalg.cholesky(Lambda, lower=True)        # Λ = L Lᵀ
    # solve Lᵀ μ = L⁻¹ η   ⇒   μ = (Lᵀ)⁻¹ (L⁻¹ η)
    y  = jax.lax.linalg.triangular_solve(L, eta,left_side=True,
                                         lower=True, transpose_a=False)
    mu = jax.lax.linalg.triangular_solve(L, y,left_side=True,
                                         lower=True, transpose_a=True)
    return mu

# ------------------------------------------------------------
# 4. Run benchmark
# ------------------------------------------------------------
t_inv,  mu1 = walltime(mu_via_inverse,   Lambda, eta, runs=100)
t_tri,  mu2 = walltime(mu_via_triangular, Lambda, eta, runs=100)

err = jnp.max(jnp.abs(mu1 - mu2))

print(f"Inverse-then-multiply   : {t_inv*1e3:8.2f} ms per run")
print(f"Single triangular solve : {t_tri*1e3:8.2f} ms per run")
print(f"max |μ₁ − μ₂|           : {err:.3e}")
