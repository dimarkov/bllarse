import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "alternatives"))

import vit_classification as vc


def test_svhn_is_registered():
    assert "svhn" in vc.DATASET_CONFIGS
    cfg = vc.DATASET_CONFIGS["svhn"]
    assert cfg["num_classes"] == 10
    assert {"hf_path", "label_key", "img_key", "test_split"} <= set(cfg)


def test_svhn_resolution_matches_cifar():
    """OOD features must be extracted at the same resolution as the ID features
    or they do not live in the same feature space."""
    assert vc.get_img_size("dinov3_huge", "svhn") == vc.get_img_size("dinov3_huge", "cifar10")
    assert vc.get_img_size("dinov3_huge", "svhn") == 224


def test_fig2_id_and_ood_datasets_all_share_a_resolution():
    sizes = {vc.get_img_size("dinov3_large", d) for d in ("cifar10", "cifar100", "svhn")}
    assert len(sizes) == 1
