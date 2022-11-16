# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 David W. Romero & Robert-Jan Bruintjes
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
#
# Code adapted from https://github.com/rjbruin/flexconv -- MIT License


# This file contains all global variables of the repository

IMG_DATASETS = [
    "MNIST",
    "rotMNIST",
    "MNIST6-180",
    "MNIST6-M",
    "CIFAR10",
    "CIFAR100",
    "STL10",
    "PCam",
]

DATASET_SIZES = {
    "rotMNIST": 10000,
    "MNIST6-180": 11836,
    "MNIST6-M": 11836,
    "CIFAR10": 45000,
    "CIFAR100": 45000,
    # "STL10": 5000, TODO update size with validation set
    "PCam": 262144,
}
