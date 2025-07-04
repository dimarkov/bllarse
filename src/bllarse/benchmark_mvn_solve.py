# benchmark_mvn_solve.py
import time
import math
import jax, jax.numpy as jnp
from jax import random, jit
from jax.scipy import linalg
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", False)   # float32 for speed / memory
jax.config.update("jax_default_matmul_precision", "float32")   # elevate matmul precision to float32 (on GPUs it defaults to mixed precision)

key0 = random.PRNGKey(0)

# --------------------------------------------------------------
# utilities
# --------------------------------------------------------------
def batch_for_dim(d):
    """Return a reasonable (M,N) batch shape to keep memory bounded."""
    # if d <= 50:   return (256, 8)
    # if d <= 100:  return (128, 4)
    # if d <= 200:  return (32,  4)
    # return (8, 1)                    # for dim = 1000
    return (16, 8)

@jit
def mu_via_inverse(Lam, eta):
    Sigma = linalg.cho_solve(linalg.cho_factor(Lam, lower=True),
                             jnp.broadcast_to(jnp.eye(Lam.shape[-1]), Lam.shape))
    return (Sigma @ eta[..., None]).squeeze(-1)

@jit
def mu_via_triangular(Lam, eta):
    L  = linalg.cholesky(Lam, lower=True)
    y  = jax.lax.linalg.triangular_solve(L, eta,
                                         left_side=True, lower=True)
    mu = jax.lax.linalg.triangular_solve(L, y,
                                         left_side=True, lower=True,
                                         transpose_a=True)
    return mu

def walltime(fn, *args, runs=30):
    fn(*args).block_until_ready()                    # compile & warm-up
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args).block_until_ready()
        times.append(time.perf_counter() - t0)
    return np.array(times)

# --------------------------------------------------------------
# main experiment
# --------------------------------------------------------------
dims        = [5, 10, 25, 50, 100, 250, 500, 1000,1500, 2000]
runs_per_d  = 30

mean_inv,   iqr_inv   = [], []
mean_tri,   iqr_tri   = [], []
mean_err,   iqr_err   = [], []

for d in dims:
    batch_shape  = batch_for_dim(d)
    key, kA, kη  = random.split(key0, 3)
    A            = random.normal(kA, batch_shape + (d, d))
    Lam          = A @ jnp.swapaxes(A, -1, -2) + 1e-3 * jnp.eye(d)
    eta          = random.normal(kη, batch_shape + (d,))

    # runtimes
    t_inv = walltime(mu_via_inverse,   Lam, eta, runs=runs_per_d)
    t_tri = walltime(mu_via_triangular, Lam, eta, runs=runs_per_d)

    # one error eval is enough – it doesn't vary across runs
    mu1 = mu_via_inverse(  Lam, eta)
    mu2 = mu_via_triangular(Lam, eta)
    # Flatten every absolute element-wise difference in the whole batch
    errs_1d = jnp.abs(mu1 - mu2).reshape(-1, d).mean(axis=-1)          # shape = (batch_size,)
    err_mean = errs_1d.mean().item()                    # mean error over batches
    q25, q75 = np.percentile(np.asarray(errs_1d), [25, 75])

    # store stats
    mean_inv.append(t_inv.mean()*1e3)                       # ms
    iqr_inv.append([np.percentile(t_inv,25)*1e3,
                    np.percentile(t_inv,75)*1e3])
    mean_tri.append(t_tri.mean()*1e3)
    iqr_tri.append([np.percentile(t_tri,25)*1e3,
                    np.percentile(t_tri,75)*1e3])

    mean_err.append(err_mean)
    iqr_err.append([q25, q75])  # deterministic – no spread

# --------------------------------------------------------------
# plotting
# --------------------------------------------------------------
dims_np = np.array(dims)

# (1) runtime
plt.figure()
plt.plot(dims_np, mean_inv, marker='o', label='inverse + mat-vec')
plt.fill_between(dims_np,
                 [q[0] for q in iqr_inv],
                 [q[1] for q in iqr_inv],
                 alpha=0.2)
plt.plot(dims_np, mean_tri, marker='o', label='two triangular solves')
plt.fill_between(dims_np,
                 [q[0] for q in iqr_tri],
                 [q[1] for q in iqr_tri],
                 alpha=0.2)
plt.xlabel("dimension $d$")
plt.ylabel("runtime per call [ms]")
plt.title("MVN moment retrieval runtimes")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xscale("log")
# plt.yscale("log")   # optional: comment out for linear y-axis
plt.savefig('mvn_solve_runtime_vs_dim.pdf', bbox_inches='tight')

# (2) error distribution
plt.figure()
plt.plot(dims_np, mean_err, marker='o')
plt.fill_between(dims_np,
                 [q[0] for q in iqr_err],
                 [q[1] for q in iqr_err],
                 alpha=0.2)
plt.xlabel("dimension $d$")
plt.ylabel(r"element-wise $|\mu_{\mathrm{inv}}-\mu_{\mathrm{tri}}|$")
plt.title("Error distribution vs. dimension")
# plt.yscale("log")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.savefig('mvn_solve_error_vs_dim.pdf', bbox_inches='tight')