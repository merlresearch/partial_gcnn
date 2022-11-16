# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


# torch
import torch

from partial_equiv.general.nn.misc import FunctionAsModule


def Swish():
    """x * sigmoid(x)"""
    return FunctionAsModule(lambda x: x * torch.sigmoid(x))


def Sine():
    return FunctionAsModule(lambda x: torch.sin(x))
