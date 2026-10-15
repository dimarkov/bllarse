"""The VBLL classification head shared by the probing and fine-tuning arms."""

import vbll


def build_vbll_head(
    input_dim: int,
    num_classes: int,
    *,
    regularization_weight: float = 1.0,
    prior_scale: float = 1.0,
    wishart_scale: float = 0.1,
    parameterization: str = "diagonal",
    cov_rank: int = 2,
    vbll_type: str = "discriminative",
):
    """A bare VBLL classification head, with no backbone.

    Linear probing on cached features needs only this -- the features were
    already extracted by the JAX side, so there is no PyTorch backbone in the
    probing path at all and the two arms consume byte-identical inputs. The
    DINOv3 fine-tuning arm (scripts/finetune/finetune_vbll.py) builds its head
    here too, on top of its own trunk.

    `cov_rank` is accepted only by DiscClassification; GenClassification
    rejects it and supports diagonal only, so a generative head with any other
    parameterization raises. `cov_rank` is inert unless
    parameterization == "lowrank".
    """
    if vbll_type == "generative":
        if parameterization != "diagonal":
            raise ValueError(
                f"GenClassification supports parameterization='diagonal' only, "
                f"got {parameterization!r}"
            )
        return vbll.GenClassification(
            in_features=input_dim,
            out_features=num_classes,
            regularization_weight=regularization_weight,
            parameterization=parameterization,
            prior_scale=prior_scale,
            wishart_scale=wishart_scale,
            return_ood=True,
        )
    return vbll.DiscClassification(
        in_features=input_dim,
        out_features=num_classes,
        regularization_weight=regularization_weight,
        parameterization=parameterization,
        prior_scale=prior_scale,
        wishart_scale=wishart_scale,
        return_ood=True,
        cov_rank=cov_rank,
    )
