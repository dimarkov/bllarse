"""Framework-independent fine-tuning protocol.

The JAX arm (scripts/finetune/finetune_jax.py) and the PyTorch VBLL arm both
import from here, so the two cannot disagree about step counts, effective batch
sizes, the eval schedule or the order the training data arrives in.
"""


def equimo_vision_models():
    """equimo's model package, whichever layout is installed.

    equimo 1.1.1 moved `equimo.models` -> `equimo.vision.models`, alongside new
    audio/language/tabular siblings. The DINOv3 factory functions are unchanged
    in name and signature; only the path moved.

    Centralised because getting it wrong is invisible until it runs on a
    machine with the other version, and the failure is misleading: the
    ImportError from a missing SUBMODULE is indistinguishable from the package
    being absent, so it surfaces as "equimo not installed" on a machine where
    equimo is installed and importable. That cost a job here once via
    vit_classification and again via a benchmark script that reimplemented the
    same two-line try/except. Import this instead of writing a third copy.
    """
    try:
        import equimo.vision.models as models  # equimo >= 1.1.1
    except ImportError:
        import equimo.models as models  # equimo <= 1.1.0
    return models
