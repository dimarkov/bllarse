# psipsi_scaling_benchmark.py
import time, math, itertools, collections
import jax, jax.numpy as jnp
from jax import random, jit
from jax.scipy import linalg
import numpy as np
import matplotlib.pyplot as plt

# ------------ configurable -------------------------------------------------
USE_FP64       = False          # flip to True if you want fp64
RNG_SEED       = 0
RUNS_PER_DIM   = 30
DIMS           = [2**k for k in range(1, 11)]   # 2,4,…,1024
N_SAMPLES      = 256
N_CLASSES      = 10
JITTER         = 1e-2           # keeps κ moderate
# ----------------------------------------------------------------------------

jax.config.update("jax_enable_x64", USE_FP64)
jax.config.update("jax_default_matmul_precision", "float32")

key0 = random.PRNGKey(RNG_SEED)

# ---------- helper: build posterior + data for a given d -------------------
def make_problem(key, d):
    key, kA, kη, kx = random.split(key, 4)
    A   = random.normal(kA, (N_CLASSES, d, d))
    Lam = A @ jnp.swapaxes(A, -1, -2) + JITTER * jnp.eye(d)
    L   = linalg.cholesky(Lam, lower=True)

    η   = random.normal(kη, (N_CLASSES, d))

    # μ via two triangular solves
    def get_mu(L, η):
        y = jax.lax.linalg.triangular_solve(L, η, left_side=True, lower=True)
        return jax.lax.linalg.triangular_solve(L, y,
                                               left_side=True, lower=True,
                                               transpose_a=True)
    μ   = get_mu(L, η)

    Σ   = linalg.cho_solve((L, True),
                           jnp.broadcast_to(jnp.eye(d), Lam.shape))

    x   = random.normal(kx, (N_SAMPLES, d))
    return L, Σ, μ, x


# ----------------------------
# φφ – dense path  (reference)
# ----------------------------
@jit
def psipsi_dense(Sigma, mu, x):
    betabeta = Sigma + mu[..., None] * mu[..., None, :]     # (*batch_shape, C, d, d)
    # Frobenius inner product  (broadcast over classes)
    # More efficient: compute per-sample, not NxN:
    x_expanded = x[...,None]
    xxT = x_expanded @ x_expanded.mT            # (N, *batch_shape, d, d)
    result = (jnp.expand_dims(xxT,-3) * betabeta).sum((-2, -1)) # shape (N, *batch_shape, C)
    return result

@jit
def psipsi_dense_diag(Sigma, mu, x):
    """
    Same quantity but without allocating xxᵀ (N×d×d).
    Works via quadratic forms Σx and dot-products μ·x.
    Returns (N, C)
    """
    # quadratic term  xᵀ Σ x  for every (n,c)
    Sigma_x = (Sigma @ jnp.expand_dims(x, (-3, -1))).squeeze(-1)  # Σ_c  @  x_n   → (N, *batch_shape, C, d) 
    quad    = (jnp.expand_dims(x, -2) * Sigma_x).sum(-1)             # (N, *batch_shape, C)

    # outer-product term  (μᵀ x)²
    outer   = (x[...,None] * mu.mT).sum(-2) ** 2                                  # (N, *batch_shape, C)

    return quad + outer                              # (N,*batch_shape, C)

# ----------------------------
# φφ – Cholesky shortcut (no Σ at all, O(d²))
# ----------------------------
@jit
def psipsi_chol_left_batching(L, mu, x):
    """
    Same φφ but using only the Cholesky of Λ.
    Returns (N, C).
    """
    shape_to_use = jnp.broadcast_shapes(jnp.expand_dims(x, (-3, -1)).shape, L.shape)
    rhs = jnp.broadcast_to(jnp.expand_dims(x, -2), shape_to_use[:-1])  # (N, *batch_shape, C, d)
    y     = jax.lax.linalg.triangular_solve(
              jnp.broadcast_to(L, shape_to_use), rhs, left_side=True, lower=True)             # (N,*batch_shape, C, d)
    quad  = (y ** 2).sum(-1)                                # (N, *batch_shape, C)
    outer   = (x[...,None] * mu.mT).sum(-2) ** 2                                     # (N,*batch_shape, C)
    return quad + outer

@jit
def psipsi_chol_right_batching(L, mu, x):
    """
    Same φφ but using only the Cholesky of Λ.
    Returns (N, C).
    """

    # we have to move just the sample dimensions (N) to the back and treat those as columns in the matrix that we do the triangular solve over, any batch dimensions that are shared with L, we CANNOT treat as columns in X
    # desired x.shape = (*batch_shape, C, d, N)  # where *batch_shape is any number of batch dimensions
    # L.shape = (*batch_shape, C, d, d)

    # current x.shape = (N, *batch_shape, d)  # where *batch_shape is any number of batch dimensions
    # we want to move the first dimension (N) to the back, so that we can treat it as a column in the matrix that we do the triangular solve over
    # first add the C dimension to the front for broadcast compatbility with L
    # x.shape = (N, *batch_shape, 1, d)  # where *batch_shape is any number of batch dimensions
    x_reshaped = jnp.moveaxis(jnp.expand_dims(x, -2), 0, -1)                    # (*batch_shape, 1, d, N)

    rhs   = jnp.broadcast_to(x_reshaped, L.shape[:-1] + (x_reshaped.shape[-1],))  # (*batch_shape, C, d, N)
    y     = jax.lax.linalg.triangular_solve(
              L, rhs, left_side=True, lower=True)             # (*batch_shape, C, d, N)
    quad  = jnp.moveaxis((y ** 2).sum(-2), -1, 0)                                 # (N, *batch_shape, C)
    outer = (x[...,None] * mu.mT).sum(-2) ** 2  
    return quad + outer                                       # (N, *batch_shape,C)

# ---------- timing helper ---------------------------------------------------
def walltime(fn, *args, repeat=30):
    fn(*args).block_until_ready()
    t = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args).block_until_ready()
        t.append(time.perf_counter() - t0)
    return np.array(t)

# ---------- run benchmark across dimensions ---------------------------------
Timing = collections.namedtuple("Timing", "mean iqr")
Error  = collections.namedtuple("Error",  "mean iqr max")

results = {}
key = key0
for d in DIMS:
    L, Σ, μ, x = make_problem(key, d)

    t_dense      = walltime(psipsi_dense,      Σ, μ, x, repeat=RUNS_PER_DIM)
    t_dense_diag = walltime(psipsi_dense_diag, Σ, μ, x, repeat=RUNS_PER_DIM)
    t_chol_left       = walltime(psipsi_chol_left_batching,       L, μ, x, repeat=RUNS_PER_DIM)
    t_chol_right  = walltime(psipsi_chol_right_batching,       L, μ, x, repeat=RUNS_PER_DIM)  

    φ_dense      = psipsi_dense(Σ, μ, x)
    φ_dense_diag = psipsi_dense_diag(Σ, μ, x)
    φ_chol_left       = psipsi_chol_left_batching(L, μ, x)
    φ_chol_right  = psipsi_chol_right_batching(L, μ, x)

    def err_stats(ref, other):
        diff = np.abs(np.asarray(ref - other)).ravel()
        return Error(diff.mean(),
                     np.percentile(diff, [25, 75]),
                     diff.max())

    results[d] = {
        "dense":      Timing(t_dense.mean()*1e3,      np.percentile(t_dense,      [25, 75])*1e3),
        "dense_diag": Timing(t_dense_diag.mean()*1e3, np.percentile(t_dense_diag, [25, 75])*1e3),
        "chol_left":       Timing(t_chol_left.mean()*1e3,       np.percentile(t_chol_left,       [25, 75])*1e3),
        "chol_right": Timing(t_chol_right.mean()*1e3, np.percentile(t_chol_right, [25, 75])*1e3),
        "err_dense_to_chol":   err_stats(φ_dense, φ_chol_left),
        "err_diag":   err_stats(φ_dense, φ_dense_diag),
        "err_chol_left_to_right": err_stats(φ_chol_left, φ_chol_right),
    }

# ---------- plotting --------------------------------------------------------
dims = np.array(DIMS)

def plot_runtimes():
    plt.figure()
    for tag, label, color in [("dense", "dense xxᵀ", "C0"),
                              ("dense_diag", "dense diag", "C1"),
                              ("chol_left", "Cholesky (left)", "C2"),
                              ("chol_right", "Cholesky (right)", "C3")]:
        means = [results[d][tag].mean for d in dims]
        iqrs  = np.array([results[d][tag].iqr for d in dims])
        plt.plot(dims, means, marker="o", label=label, color=color)
        plt.fill_between(dims, iqrs[:,0], iqrs[:,1], alpha=0.2, color=color)

    plt.xscale("log", base=2) 
    # plt.yscale("log")
    plt.xlabel("dimension $d$"); plt.ylabel("runtime per call [ms]")
    plt.title("φφ runtimes vs dimension"); plt.grid(True, ls="--", lw=0.5)
    plt.legend()
    plt.savefig('psipsi_runtime_vs_dim.pdf', bbox_inches='tight')

def plot_errors():
    plt.figure()
    for tag, label, color in [("err_dense_to_chol",  "|dense − chol|",  "C2"),
                              ("err_diag",  "|dense − diag|", "C1"),
                              ("err_chol_left_to_right", "|chol left − chol right|", "C3")]:
        means = [results[d][tag].mean for d in dims]
        iqrs  = np.array([results[d][tag].iqr for d in dims])
        maxs  = [results[d][tag].max  for d in dims]
        plt.plot(dims, means, marker="o", label=f"mean {label}", color=color)
        plt.fill_between(dims, iqrs[:,0], iqrs[:,1],
                         alpha=0.15, color=color, linewidth=0)
        plt.scatter(dims, maxs, marker="x", color=color, label=f"max {label}")

    plt.xscale("log", base=2)
    # plt.yscale("log")
    plt.xlabel("dimension $d$"); plt.ylabel("absolute error")
    plt.title("φφ error vs dimension"); plt.grid(True, ls="--", lw=0.5)
    plt.legend()
    plt.savefig('psipsi_error_vs_dim.pdf', bbox_inches='tight')

plot_runtimes()
plot_errors()
# plt.show()
