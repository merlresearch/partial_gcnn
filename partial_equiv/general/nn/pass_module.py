# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# typing
from typing import Tuple

import torch


class ApplyFirstElem(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
    ):
        """
        Wrapper used to apply an input torch.nn.Module along the first element of an array.
        :param module:
        """
        super().__init__()

        self.module = module

    def forward(self, input_tuple: Tuple[torch.Tensor, torch.Tensor]):
        """
        :param input_tuple: Tuple[ x, g_elems]
        :return: self.module.forward(x), g_elems
        """
        assert len(input_tuple) == 2
        return self.module.forward(input_tuple[0]), input_tuple[1]
