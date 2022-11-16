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

VALIDATION_SPLIT = [45000, 5000]


class CIFAR100(datasets.CIFAR100):
    def __init__(
        self,
        partition: str,
        augment: str,
        **kwargs,
    ):
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
                transforms.Normalize((0.49137, 0.48235, 0.44667), (0.24706, 0.24353, 0.26157)),
            ],
        )

        transform = transforms.Compose(transform)

        if partition == "train":
            train = True
        elif partition == "test":
            train = False
        else:
            raise NotImplementedError("The dataset partition {} does not exist".format(partition))

        super().__init__(root=root, train=train, transform=transform, download=True)


def augmentations_resnet():
    """
    Following "A branching and merging convolutional network with homogeneous filter capsules"
    - Biearly et al., 2020 - https://arxiv.org/abs/2001.09136
    """
    augmentations = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
    ]
    return augmentations
