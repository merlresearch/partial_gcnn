# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 David W. Romero & Robert-Jan Bruintjes
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
#
# Code adapted from https://github.com/rjbruin/flexconv -- MIT License

from functools import partial

import torch

import partial_equiv.general.nn.activation_functions


class MLPBase(torch.nn.Module):
    def __init__(
        self,
        dim_input_space: int,
        out_channels: int,
        hidden_channels: int,
        no_layers: int,
        bias: bool,
        Linear: torch.nn.Module,
        Norm: torch.nn.Module,
        Activation: torch.nn.Module,
    ):

        super().__init__()

        # Save params in self
        self.dim_input_space = dim_input_space
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.no_layers = no_layers

        # 1st layer:
        kernel_net = [
            Linear(dim_input_space, hidden_channels, bias=bias),
            Norm(hidden_channels),
            Activation(),
        ]
        # Hidden layers:
        for _ in range(no_layers - 2):
            kernel_net.extend(
                [
                    Linear(hidden_channels, hidden_channels, bias=bias),
                    Norm(hidden_channels),
                    Activation(),
                ]
            )
        # Last layer:
        kernel_net.extend(
            [
                Linear(hidden_channels, out_channels, bias=bias),
            ]
        )
        self.kernel_net = torch.nn.Sequential(*kernel_net)

    def forward(self, x):
        # If dim_linear >3 , we cannot use a convolution operation. In this case, we must use a Linear layer instead.
        # However, since transposes are expensive, we try to avoid using t his as much as possible.
        if self.dim_linear > 3:
            x_shape = x.shape
            # Put in_channels dimension at last and compress all other dimensions to one [batch_size, -1, in_channels]
            out = x.view(x_shape[0], x_shape[1], -1).transpose(1, 2)
            # Pass through the network
            out = self.kernel_net(out)
            # Restore shape
            out = out.transpose(1, 2).view(x_shape[0], -1, *x_shape[2:])
        else:
            out = self.kernel_net(x)
        return out


class MLP(MLPBase):
    def __init__(
        self,
        dim_linear: int,
        dim_input_space: int,
        out_channels: int,
        hidden_channels: int,
        no_layers: int,
        activation: str,
        norm_type: str,
        bias: bool,
    ):
        self.dim_linear = dim_linear
        # Get Linear Class
        if dim_linear <= 3:
            Linear = getattr(torch.nn, f"Conv{dim_linear}d")
            Linear = partial(Linear, kernel_size=1)
        else:
            Linear = torch.nn.Linear

        # Get Activation class
        if activation == "Swish":
            Activation = partial_equiv.general.nn.Swish
        else:
            Activation = getattr(torch.nn, activation)

        # Get Norm class
        if norm_type == "BatchNorm":
            Norm = getattr(torch.nn, f"{norm_type}{dim_linear}d")
        else:
            Norm = getattr(torch.nn, f"{norm_type}")

        super().__init__(
            dim_input_space=dim_input_space,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            no_layers=no_layers,
            bias=bias,
            Linear=Linear,
            Norm=Norm,
            Activation=Activation,
        )
