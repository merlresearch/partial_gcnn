# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 David W. Romero & Robert-Jan Bruintjes
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
#
# Code adapted from https://github.com/rjbruin/flexconv -- MIT License

import numpy as np
import torch

import partial_equiv.general as gral

from .mfn import MFNBase, gaussian_window


class MAGNet(MFNBase):
    def __init__(
        self,
        dim_linear: int,
        dim_input_space: int,
        hidden_channels: int,
        out_channels: int,
        no_layers: int,
        steerable: bool,
        input_scale: float = 256.0,
        weight_scale: float = 1.0,
        alpha: float = 6.0,
        beta: float = 1.0,
        bias: bool = True,
        init_spatial_value: float = 1.0,
    ):

        Linear = getattr(gral.nn, f"Linear{dim_linear}d")

        super().__init__(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            no_layers=no_layers,
            weight_scale=weight_scale,
            bias=bias,
            Linear=Linear,
        )
        self.filters = torch.nn.ModuleList(
            [
                MAGNetLayer(
                    dim_input_space=dim_input_space,
                    dim_linear=dim_linear,
                    hidden_channels=hidden_channels,
                    Linear=Linear,
                    steerable=steerable,
                    input_scale=input_scale / np.sqrt(no_layers + 1),
                    alpha=alpha / (layer + 1),
                    beta=beta,
                    bias=bias,
                    init_spatial_value=init_spatial_value,
                )
                for layer in range(no_layers + 1)
            ]
        )


class MAGNetLayer(torch.nn.Module):
    """
    Anisotropic Gabor Filters
    """

    def __init__(
        self,
        dim_input_space: int,
        dim_linear: int,
        hidden_channels: int,
        Linear: torch.nn.Module,
        steerable: bool,
        input_scale: float,
        alpha: float,
        beta: float,
        bias: bool,
        init_spatial_value: float,
    ):
        super().__init__()

        self.dim_linear = dim_linear
        self.dim_input_space = dim_input_space
        self.input_scale = input_scale

        self.linear = Linear(dim_input_space, hidden_channels, bias=bias)

        mu = init_spatial_value * (2 * torch.rand(hidden_channels, dim_input_space) - 1)
        self.mu = torch.nn.Parameter(mu)

        self.gamma = torch.nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((hidden_channels, dim_input_space))
        )

        self.linear.weight.data *= input_scale * self.gamma.view(*self.gamma.shape, *((1,) * self.dim_linear))
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-np.pi, np.pi)

        # If steerable, create thetas
        self.steerable = steerable  # TODO! Implement for 3d
        if self.steerable:

            if dim_input_space > 2:
                raise NotImplementedError(f"steerable only implemented for 2D. Current: {dim_input_space}")

            self.theta = torch.nn.Parameter(
                torch.rand(
                    hidden_channels,
                )
            )

        return

    def forward(self, x):
        if self.steerable:
            gauss_window = rotated_gaussian_window(
                x,
                self.gamma.view(1, *self.gamma.shape, *((1,) * self.dim_linear)),
                self.theta,
                self.mu.view(1, *self.mu.shape, *((1,) * self.dim_linear)),
            )
        else:
            gauss_window = gaussian_window(
                x,
                self.gamma.view(1, *self.gamma.shape, *((1,) * self.dim_linear)),
                self.mu.view(1, *self.mu.shape, *((1,) * self.dim_linear)),
            )
        return gauss_window.view(x.shape[0], -1, *x.shape[2:]) * torch.sin(self.linear(x))


def rotation_matrix(theta):
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    return torch.stack([cos, sin, -sin, cos], dim=-1).view(-1, 2, 2)


def rotate(theta, input):
    # theta.shape = [Out, 1]
    # input.shape = [B, Channels, 2, X, Y]
    return torch.einsum("coi, bcixy -> bcoxy", rotation_matrix(theta), input)


def rotated_gaussian_window(x, gamma, theta, mu):
    return torch.exp(-0.5 * ((gamma * rotate(2 * np.pi * theta, x.unsqueeze(1) - mu)) ** 2).sum(2))
