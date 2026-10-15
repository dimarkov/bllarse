# VBLL Parameterization Notes

## Comparison: Linear Layer vs VBLL vs IBProbit

### Standard Linear Layer
A simple linear layer has:
- **Weight matrix**: `D_in × D_out`
- **Bias**: `D_out`
- **Total parameters**: `D_in × D_out + D_out`

For example (`D_in=512`, `D_out=10`):
- Parameters: `512 × 10 + 10 = 5,130`

---

### VBLL (Variational Bayesian Last Layer)

VBLL maintains uncertainty over the weights using a variational distribution. The parameterization significantly affects memory usage.

#### Dense Parameterization (`parameterization='dense'`)
| Parameter | Shape | Count |
|-----------|-------|-------|
| `W_mean` | (D_out, D_in) | 5,120 |
| `W_logdiag` | (D_out, D_in) | 5,120 |
| `W_offdiag` | **(D_out, D_in, D_in)** | **2,621,440** |
| `noise_mean` | (D_out,) | 10 |
| `noise_logdiag` | (D_out,) | 10 |
| **Total** | | **2,631,700** |

The `W_offdiag` stores a full covariance factor per output class: `10 × 512 × 512`.

#### Lowrank Parameterization (`parameterization='lowrank'`, `cov_rank=r`)
| Parameter | Shape | Count (r=2) |
|-----------|-------|-------------|
| `W_mean` | (D_out, D_in) | 5,120 |
| `W_logdiag` | (D_out, D_in) | 5,120 |
| `W_offdiag` | (D_out, D_in, **r**) | 10,240 |
| `noise_*` | (D_out,) each | 20 |
| **Total** | | **20,500** |

**Memory savings**: 128× fewer parameters with rank=2.

#### Diagonal Parameterization (`parameterization='diagonal'`)
Most memory-efficient option - only stores diagonal variance, no off-diagonal covariance.

| Parameter | Shape | Count |
|-----------|-------|-------|
| `W_mean` | (D_out, D_in) | 5,120 |
| `W_logdiag` | (D_out, D_in) | 5,120 |
| `noise_*` | (D_out,) each | 20 |
| **Total** | | **~10,260** |

---

### Discriminative vs Generative VBLL

VBLL offers two classification head types:

| Type | Class | Description |
|------|-------|-------------|
| **Discriminative** | `vbll.DiscClassification` | Models P(y|x) directly. Standard for classification. |
| **Generative** | `vbll.GenClassification` | Models P(x|y) for each class. Better OOD detection in some cases. |

Both support the same parameterization options and return OOD scores when `return_ood=True`.

---

### IBProbit (`src/bllarse/losses/bayesian.py`)

IBProbit keeps one covariance shared across classes:
- **Posterior mean** `mu`: `(d, D_out)`
- **Posterior covariance** `Sigma`: `(d, d)`, stored as a full matrix
- **Process-noise Inverse-Wishart scale** `Psi`: `(d, d)`
- a handful of scalars

Here `d = D_in + 1` with the bias. For `D_in=512`, `D_out=10`:
- `mu`: `513 × 10 = 5,130`; `Sigma` and `Psi`: `513 × 513 = 263,169` each
- **Total**: ~531K -- smaller than VBLL dense (2.6M), larger than VBLL lowrank (20K) or diagonal (10K)

Because the covariance is shared, its size does not grow with the number of classes, whereas every VBLL parameterization scales linearly in `D_out`.

#### IBProbit vs VBLL Key Differences:
| Aspect | IBProbit | VBLL |
|--------|----------|------|
| Covariance scope | Single shared covariance | Per-class covariance |
| Parameterization | Full `(d, d)` matrix | Diagonal, lowrank or dense |
| Memory (D=512, C=10) | ~531K | 2.6M (dense) / 20K (lowrank) / 10K (diagonal) |
| Posterior fit | Recursive CAVI updates, not gradient-trained | Gradient-based ELBO |

---

## Recommendations

1. The probing and fine-tuning scripts default to the **diagonal** parameterization; lowrank is a memory dial, since head parameters scale as `D × C × (2 + cov_rank)`
2. Use **lowrank** with a small rank when off-diagonal covariance is wanted and dense does not fit
3. For many classes, IBProbit's shared covariance can be more memory-efficient than VBLL dense
4. VBLL lowrank with small rank is more memory-efficient than IBProbit for moderate embedding dimensions
