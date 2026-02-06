"""Data loading utilities for VBLL experiments."""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Normalization constants matching the original JAX script
MEAN_DICT = {
    "cifar10": (0.49139968, 0.48215841, 0.44653091),
    "cifar100": (0.49139968, 0.48215841, 0.44653091),
}

STD_DICT = {
    "cifar10": (0.24703233, 0.24348505, 0.26158768),
    "cifar100": (0.24703233, 0.24348505, 0.26158768),
}


def get_transforms(dataset_name: str, augment: bool = True, img_size: int = 64):
    """Get train and test transforms for a dataset.
    
    Args:
        dataset_name: Name of dataset (cifar10, cifar100)
        augment: Whether to apply data augmentation for training
        img_size: Target image size (scaling_mlps uses 64x64)
    
    Returns:
        Tuple of (train_transform, test_transform)
    """
    mean = MEAN_DICT[dataset_name]
    std = STD_DICT[dataset_name]
    
    # Test transform: resize and normalize only
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = test_transform
    
    return train_transform, test_transform


def get_dataloaders(
    dataset_name: str,
    batch_size: int = 64,
    augment: bool = True,
    img_size: int = 64,
    num_workers: int = 4,
    data_root: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """Get train and test dataloaders for a dataset.
    
    Args:
        dataset_name: Name of dataset (cifar10, cifar100)
        batch_size: Batch size for training
        augment: Whether to apply data augmentation
        img_size: Target image size
        num_workers: Number of data loading workers
        data_root: Root directory for dataset downloads
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_transform, test_transform = get_transforms(dataset_name, augment, img_size)
    
    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=test_transform
        )
    elif dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=data_root, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR100(
            root=data_root, train=False, download=True, transform=test_transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, test_loader


def get_num_classes(dataset_name: str) -> int:
    """Get number of classes for a dataset."""
    if dataset_name == "cifar10":
        return 10
    elif dataset_name == "cifar100":
        return 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
