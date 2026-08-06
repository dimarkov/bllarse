"""Framework-independent fine-tuning protocol.

The JAX arm (scripts/finetune/finetune_jax.py) and the PyTorch VBLL arm both
import from here, so the two cannot disagree about step counts, effective batch
sizes, the eval schedule or the order the training data arrives in.
"""
