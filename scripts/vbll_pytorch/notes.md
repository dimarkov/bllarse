# VBLL Parameterization Notes

## Comparison: Linear Layer vs VBLL vs IBProbit

### Standard Linear Layer
A simple linear layer has:
- **Weight matrix**: `D_in × D_out`
- **Bias**: `D_out`
- **Total parameters**: `D_in × D_out + D_out`

For our case (`D_in=512`, `D_out=10`):
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

### IBProbit (from blrax)

IBProbit uses a simpler parameterization:
- **Mean array**: Same shape as linear layer weights `D_in × D_out`
- **Covariance matrix**: Single matrix of size `D × (D - 1) / 2`
  - Only stores the **lower triangular part** of the Cholesky decomposition
  - `D` = dimension of output from MLP (embedding dimension, e.g., 512)

For `D=512`:
- Covariance parameters: `512 × 511 / 2 = 130,816`
- Much smaller than VBLL dense (2.6M) but larger than VBLL lowrank (10K)

#### IBProbit vs VBLL Key Differences:
| Aspect | IBProbit | VBLL |
|--------|----------|------|
| Covariance scope | Single shared covariance | Per-class covariance |
| Parameterization | Cholesky lower triangular | Dense or lowrank |
| Memory (D=512, C=10) | ~130K | 2.6M (dense) / 20K (lowrank) |
| Uncertainty type | Feature-space | Weight-space |

---

## Recommendations

1. **Use lowrank parameterization** for VBLL when memory is constrained
2. **rank=2** is a good default balance of expressiveness and efficiency
3. For very high-dimensional embeddings, IBProbit's shared covariance may be more memory-efficient than VBLL dense
4. VBLL lowrank with small rank can be more memory-efficient than IBProbit for moderate embedding dimensions
