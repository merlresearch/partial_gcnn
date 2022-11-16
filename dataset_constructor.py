# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 David W. Romero & Robert-Jan Bruintjes
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
#
# Code adapted from https://github.com/rjbruin/flexconv -- MIT License


# typing
from typing import Dict, Tuple

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, random_split

# validation splits
# datasets
from datasets import (  # 2D Datasets
    CIFAR10,
    CIFAR10_VALIDATION_SPLIT,
    CIFAR100,
    CIFAR100_VALIDATION_SPLIT,
    MNIST6_180,
    MNIST6_M,
    STL10,
    PCam,
    RotatedMNIST,
    rotMNIST_VALIDATION_SPLIT,
)


def construct_dataset(
    cfg: OmegaConf,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create Dataset instances for each partition (train, val, test) of a given dataset.
    :return: Tuple (training_set, validation_set, test_set)
    """

    dataset = {
        "rotMNIST": RotatedMNIST,
        "MNIST6-180": MNIST6_180,
        "MNIST6-M": MNIST6_M,
        "CIFAR10": CIFAR10,
        "CIFAR100": CIFAR100,
        "STL10": STL10,
        "PCam": PCam,
    }[cfg.dataset]

    training_set = dataset(
        partition="train",
        augment=cfg.augment,
        rot_interval=cfg.dataset_params.rot_interval,
    )
    test_set = dataset(
        partition="test",
        augment="None",
        rot_interval=cfg.dataset_params.rot_interval,
    )
    # validation dataset
    if cfg.dataset in ["PCam"]:
        validation_set = dataset(
            partition="valid",
            augment="None",
            rot_interval=cfg.dataset_params.rot_interval,
        )
    elif cfg.dataset in ["rotMNIST", "CIFAR10", "CIFAR100"]:
        training_set, validation_set = random_split(
            training_set,
            eval(f"{cfg.dataset}_VALIDATION_SPLIT"),
            generator=torch.Generator().manual_seed(42),
        )
    else:
        validation_set = None
    return training_set, validation_set, test_set


def construct_dataloaders(
    cfg: OmegaConf,
) -> Dict[str, DataLoader]:
    """
    Construct DataLoaders for the selected dataset
    :return dict("train": train_loader, "validation": val_loader , "test": test_loader)
    """
    training_set, validation_set, test_set = construct_dataset(cfg)

    num_workers = cfg.no_workers
    num_workers = num_workers * torch.cuda.device_count()
    training_loader = DataLoader(
        training_set,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    if validation_set is not None:
        val_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )
    else:
        val_loader = test_loader

    dataloaders = {
        "train": training_loader,
        "validation": val_loader,
        "test": test_loader,
    }

    return dataloaders
