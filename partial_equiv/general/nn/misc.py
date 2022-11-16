# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch


class FunctionAsModule(torch.nn.Module):
    def __init__(self, func):
        """
        Creates a torch.nn.Module that applies the function func.
        :param func: lambda function
        """
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def Multiply(
    omega_0: float,
):
    """
    out = omega_0 * x
    """
    return FunctionAsModule(lambda x: omega_0 * x)
