# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 David W. Romero & Robert-Jan Bruintjes
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
#
# Code adapted from https://github.com/rjbruin/flexconv -- MIT License

import os
import os.path

from hydra import utils
from torchvision import datasets, transforms


class STL10(datasets.STL10):
    def __init__(
        self,
        partition: str,
        augment: str,
        **kwargs,
    ):
        assert partition in ["train", "test"]

        if "root" in kwargs:
            root = kwargs["root"]
        else:
            root = utils.get_original_cwd()
            root = os.path.join(root, "data")

        transform = []
        if augment == "resnet":
            transform.extend(augmentations_resnet())

        transform.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
            ]
        )

        transform = transforms.Compose(transform)

        super().__init__(root=root, split=partition, transform=transform, download=True)


def augmentations_resnet():
    augmentations = [
        transforms.RandomCrop(96, padding=12),
        transforms.RandomHorizontalFlip(),
    ]
    return augmentations
