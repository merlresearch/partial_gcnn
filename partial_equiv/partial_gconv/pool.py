# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch


class MaxPoolRn(torch.nn.Module):
    def __init__(
        self,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        input_size = x.size()
        out = x.view(input_size[0], input_size[1] * input_size[2], input_size[3], input_size[4])
        out = torch.max_pool2d(out, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        out = out.view(input_size[0], input_size[1], input_size[2], out.size()[2], out.size()[3])
        return out
