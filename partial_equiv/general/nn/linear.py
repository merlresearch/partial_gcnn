# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 David W. Romero & Robert-Jan Bruintjes
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
#
# Code adapted from https://github.com/rjbruin/flexconv -- MIT License


import torch
import torch.nn as nn


def Linear1d(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = True,
) -> torch.nn.Module:
    """
    Implements a Linear Layer in terms of a pointwise convolution.
    """
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


def Linear2d(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = True,
) -> torch.nn.Module:
    """
    Implements a Linear Layer in terms of a pointwise convolution.
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


def Linear3d(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = True,
) -> torch.nn.Module:
    """
    Implements a Linear Layer in terms of a pointwise convolution.
    """
    return nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)
