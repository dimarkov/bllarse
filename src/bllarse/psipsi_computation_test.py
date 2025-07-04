# psipsi_computation_test.py
import time
import jax, jax.numpy as jnp
from jax import random, jit
from jax.scipy import linalg
import numpy as np

jax.config.update("jax_enable_x64", False)      # float32
jax.config.update("jax_default_matmul_precision", "float32")

# ----------------------------
# hyper-parameters
# ----------------------------
d            = 500           # dimension of β
n_samples    = 512          # number of data points  (N)
n_classes    = 10           # stick-breaking classes (C)
runs         = 30           # timing repeats

key          = random.PRNGKey(500)
key, kA, kη, kx = random.split(key, 4)

# ----------------------------
# synthetic posterior
# ----------------------------
A   = random.normal(kA, (n_classes, d, d))
Lam = A @ A.mT + 1e-2 * jnp.eye(d)     # precision Λ
L   = linalg.cholesky(Lam, lower=True)                    # its Cholesky

eta  = random.normal(kη, (n_classes, d))                  # η = Λ μ
# μ via 2 triangular solves (fast – O(d²)) --------------------------
def get_mu(L, η):
    y  = jax.lax.linalg.triangular_solve(L, η,
                                         left_side=True, lower=True)
    μ  = jax.lax.linalg.triangular_solve(L, y,
                                         left_side=True, lower=True,
                                         transpose_a=True)
    return μ
mu = get_mu(L, eta)                                       # (C, d)

# dense Σ (reference)  ---------------------------------------------
Sigma = linalg.cho_solve((L, True),
                         jnp.broadcast_to(jnp.eye(d), Lam.shape))  # (C, d, d)

# ----------------------------
x = random.normal(kx, (n_samples, d))                     # (N, d)

# ----------------------------
# φφ – dense path  (reference)
# ----------------------------
@jit
def psipsi_dense(Sigma, mu, x):
    betabeta = Sigma + mu[..., None] * mu[..., None, :]     # (C, d, d)
    # Frobenius inner product  (broadcast over classes)
    # More efficient: compute per-sample, not NxN:
    x_expanded = x[...,None]
    xxT = x_expanded @ x_expanded.mT            # (N, d, d)
    result = (jnp.expand_dims(xxT,-3) * betabeta).sum((-2, -1)) # shape (N, C)
    return result

@jit
def psipsi_dense_diag(Sigma, mu, x):
    """
    Same quantity but without allocating xxᵀ (N×d×d).
    Works via quadratic forms Σx and dot-products μ·x.
    Returns (N, C)
    """
    # quadratic term  xᵀ Σ x  for every (n,c)
    Sigma_x = jnp.einsum("cij,nj->cni", Sigma, x)   # Σ_c  @  x_n   → (C, N, d) 
    quad    = (x[None, :, :] * Sigma_x).sum(-1).T              # (N, C)

    # outer-product term  (μᵀ x)²
    outer   = (x @ mu.T) ** 2                                  # (N, C)

    return quad + outer                              # (N,C)

# ----------------------------
# φφ – Cholesky shortcut (no Σ at all, O(d²))
# ----------------------------
@jit
def psipsi_chol_left_batching(L, mu, x):
    """
    Same φφ but using only the Cholesky of Λ.
    Returns (N, C).
    """
    shape_to_use = jnp.broadcast_shapes(jnp.expand_dims(x[...,None], -3).shape, L.shape)
    rhs = jnp.broadcast_to(jnp.expand_dims(x, -2), shape_to_use[:-1])  # (N, ..., C, d)
    y     = jax.lax.linalg.triangular_solve(
              jnp.broadcast_to(L, shape_to_use), rhs, left_side=True, lower=True)             # (N, ..., C, d)
    quad  = (y ** 2).sum(-1)                                # (N, C)
    outer = (x @ mu.T) ** 2                                   # (N, C)
    return quad + outer

@jit
def psipsi_chol_right_batching(L, mu, x):
    """
    Same φφ but using only the Cholesky of Λ.
    Returns (N, C).
    """

    # x.shape = (..., d)
    # write a permutation command that puts the last dimension (d) of x to the front and the rest to the back
    # x.shape = (..., d)  # where ... is any number of dimensions --> (d,  ...)
    x_transposed = jnp.moveaxis(x, -1, 0)                     # (d, ...)
    rhs   = jnp.broadcast_to(x_transposed, (L.shape[-3],) + x_transposed.shape)  # (C, d, ...)
    y     = jax.lax.linalg.triangular_solve(
              L, rhs, left_side=True, lower=True)             # (C, d, ...)
    quad  = jnp.moveaxis((y ** 2).sum(1), 0, -1)                                 # (..., C)
    # x.shape = (..., d)
    # mu.shape = (...., C, d)
    # mu.T.shape = (...., d, C)
    outer = (x[...,None] * mu.mT).sum(-2) ** 2  
    return quad + outer                                       # (..., C)

# ----------------------------
# timing helper
# ----------------------------
def timeit(fn, *args, runs=30):
    fn(*args).block_until_ready()        # compile / warm-up
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args).block_until_ready()
        ts.append(time.perf_counter() - t0)
    return np.array(ts)

# ----------------------------
# run benchmark
# ----------------------------
t_dense = timeit(psipsi_dense, Sigma, mu, x, runs=runs)
t_dense_diag = timeit(psipsi_dense_diag, Sigma, mu, x, runs=runs)
t_chol_left  = timeit(psipsi_chol_left_batching,       L,     mu, x, runs=runs)
t_chol_right  = timeit(psipsi_chol_right_batching,       L,     mu, x, runs=runs)

psi_dense = psipsi_dense(Sigma, mu, x)
psi_dense_diag = psipsi_dense_diag(Sigma, mu, x)
psi_chol_left   = psipsi_chol_left_batching(L, mu, x)
psi_chol_right  = psipsi_chol_right_batching(L, mu, x)

err_psi_dense_psi_chol = jnp.max(jnp.abs(psi_dense - psi_chol_left)).item()
err_psi_dense_psi_diag = jnp.max(jnp.abs(psi_dense - psi_dense_diag)).item()
err_psi_diag_psi_chol = jnp.max(jnp.abs(psi_dense_diag - psi_chol_left)).item()
err_psi_chol_left_right = jnp.max(jnp.abs(psi_chol_left - psi_chol_right)).item()


print(f"[dense]   mean {t_dense.mean()*1e3:.2f} ms   IQR [{np.percentile(t_dense,25)*1e3:.2f}, "
      f"{np.percentile(t_dense,75)*1e3:.2f}] ms")
print(f"[chol left]   mean {t_chol_left.mean()*1e3:.2f} ms   IQR [{np.percentile(t_chol_left,25)*1e3:.2f}, "
      f"{np.percentile(t_chol_left,75)*1e3:.2f}] ms")
print(f"[chol right]   mean {t_chol_right.mean()*1e3:.2f} ms   IQR [{np.percentile(t_chol_right,25)*1e3:.2f}, "
      f"{np.percentile(t_chol_right,75)*1e3:.2f}] ms")
print(f"max |φφ_dense − φφ_chol| = {err_psi_dense_psi_chol:.3e}")
print(f"max |φφ_dense − φφ_dense_diag| = {err_psi_dense_psi_diag:.3e}")
print(f"max |φφ_dense_diag − φφ_chol| = {err_psi_diag_psi_chol:.3e}")
print(f"max |φφ_chol_left − φφ_chol_right| = {err_psi_chol_left_right:.3e}")