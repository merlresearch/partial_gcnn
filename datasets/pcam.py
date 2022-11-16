# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 David W. Romero & Robert-Jan Bruintjes
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
#
# Code adapted from https://github.com/rjbruin/flexconv -- MIT License

import os
import os.path
import pathlib

# typing
from typing import Callable, Optional

from hydra import utils
from torchvision import transforms
from torchvision.datasets import ImageFolder


class PCam_Base(ImageFolder):
    """
    PCam dataset.
    Download the dataset from https://drive.google.com/file/d/1THSEUCO3zg74NKf_eb3ysKiiq2182iMH/view
    and put it in a dataset called PCam.

    For more information, please refer to the README.md of the repository.
    """

    def __init__(
        self,
        root: str,
        partition: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        assert partition in ["train", "valid", "test"]

        root = pathlib.Path(root) / "PCam"
        path = root / partition

        if download or not (root.exists() and path.exists()):
            raise FileNotFoundError("Please download the PCam dataset. How to download it can be found in 'README.md'")

        super().__init__(
            root=str(path),
            transform=transform,
            target_transform=target_transform,
        )


class PCam(PCam_Base):
    def __init__(
        self,
        partition,
        augment,
        **kwargs,
    ):
        if "root" in kwargs.keys():
            root = kwargs["root"]
        else:
            root = utils.get_original_cwd()
            root = os.path.join(root, "data")

        data_mean = (0.701, 0.538, 0.692)
        data_stddev = (0.235, 0.277, 0.213)

        transform = []
        if augment != "None":
            raise NotImplementedError("No augmentations are implemented for this dataset.")

        transform.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(data_mean, data_stddev),
            ],
        )

        transform = transforms.Compose(transform)

        super().__init__(
            root=root,
            partition=partition,
            transform=transform,
            download=False,
        )
