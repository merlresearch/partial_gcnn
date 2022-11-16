# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 David W. Romero & Robert-Jan Bruintjes
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
#
# Code adapted from https://github.com/rjbruin/flexconv -- MIT License

from math import sqrt

import torch
from torch.nn.utils import weight_norm as w_norm

# project
import partial_equiv.general as gral


class SIRENBase(torch.nn.Module):
    def __init__(
        self,
        dim_input_space: int,
        out_channels: int,
        hidden_channels: int,
        no_layers: int,
        weight_norm: bool,
        bias: bool,
        omega_0: float,
        learn_omega_0: bool,
        Linear_hidden: torch.nn.Module,
        Linear_out: torch.nn.Module,
    ):

        super().__init__()

        # Save params in self
        self.dim_input_space = dim_input_space
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.no_layers = no_layers

        ActivationFunction = gral.nn.Sine

        # 1st layer:
        kernel_net = [
            Linear_hidden(dim_input_space, hidden_channels, omega_0, learn_omega_0, bias),
            ActivationFunction(),
        ]
        # Hidden layers:
        for _ in range(no_layers - 2):
            kernel_net.extend(
                [
                    Linear_hidden(
                        hidden_channels,
                        hidden_channels,
                        omega_0,
                        learn_omega_0,
                        bias,
                    ),
                    ActivationFunction(),
                ]
            )
        # Last layer:
        kernel_net.extend(
            [
                Linear_out(hidden_channels, out_channels, bias=bias),
            ]
        )
        self.kernel_net = torch.nn.Sequential(*kernel_net)

        # initialize the kernel function
        self.initialize(omega_0=omega_0)

        # Weight_norm
        if weight_norm:
            for (i, module) in enumerate(self.kernel_net):
                if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
                    # All Conv layers are subclasses of torch.nn.Conv
                    self.kernel_net[i] = w_norm(module)

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

    def initialize(self, omega_0):

        net_layer = 1
        for (i, m) in enumerate(self.modules()):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)):
                # First layer
                if net_layer == 1:
                    w_std = 1 / m.weight.shape[1]
                    m.weight.data.uniform_(-w_std, w_std)
                else:
                    w_std = sqrt(6.0 / m.weight.shape[1]) / omega_0
                    m.weight.data.uniform_(
                        -w_std,
                        w_std,
                    )
                net_layer += 1
                # Important! Bias is not defined in original SIREN implementation
                if m.bias is not None:
                    m.bias.data.zero_()


#############################################
#       SIREN as in Romero et al., 2021
##############################################
class SIREN(SIRENBase):
    """SIREN model.
    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool):
        final_activation (torch.nn.Module): Activation function.
    """

    def __init__(
        self,
        dim_linear: int,
        dim_input_space: int,
        out_channels: int,
        hidden_channels: int,
        no_layers: int,
        weight_norm: bool,
        bias: bool,
        omega_0: float,
        learn_omega_0: bool,
    ):

        # Get class of multiplied Linear Layers
        if dim_linear <= 3:
            Linear_hidden = globals()[f"SIRENLayer{dim_linear}d"]
            Linear_out = getattr(gral.nn, f"Linear{dim_linear}d")
        else:
            # There are no native implementations of ConvNd layers, with N > 3. In this case, we must define
            # Linear layers and perform permutations to achieve an equivalent point-wise conv in high dimensions.
            Linear_hidden = globals()["SIRENLayerNd"]
            Linear_out = torch.nn.Linear

        super().__init__(
            dim_input_space=dim_input_space,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            weight_norm=weight_norm,
            no_layers=no_layers,
            bias=bias,
            omega_0=omega_0,
            learn_omega_0=learn_omega_0,
            Linear_hidden=Linear_hidden,
            Linear_out=Linear_out,
        )
        self.dim_linear = dim_linear


class SIRENLayer1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        omega_0: float,
        learn_omega_0: bool,
        bias: bool,
    ):
        """
        Implements a Linear Layer of the form y = omega_0 * [W x + b] as in Sitzmann et al., 2020, Romero et al., 2021,
        where x is 1 dimensional.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=bias,
        )

        # omega_0
        if learn_omega_0:
            self.omega_0 = torch.nn.Parameter(torch.Tensor(1))
            with torch.no_grad():
                self.omega_0.fill_(omega_0)
        else:
            tensor_omega_0 = torch.zeros(1)
            tensor_omega_0.fill_(omega_0)
            self.register_buffer("omega_0", tensor_omega_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.omega_0 * torch.nn.functional.conv1d(x, self.weight, self.bias, stride=1, padding=0)


class SIRENLayer2d(torch.nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        omega_0: float,
        learn_omega_0: bool,
        bias: bool,
    ):
        """
        Implements a Linear Layer of the form y = omega_0 * W x + b, where x is 2 dimensional
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=bias,
        )

        # omega_0
        if learn_omega_0:
            self.omega_0 = torch.nn.Parameter(torch.Tensor(1))
            with torch.no_grad():
                self.omega_0.fill_(omega_0)
        else:
            tensor_omega_0 = torch.zeros(1)
            tensor_omega_0.fill_(omega_0)
            self.register_buffer("omega_0", tensor_omega_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.omega_0 * torch.nn.functional.conv2d(x, self.weight, self.bias, stride=1, padding=0)


class SIRENLayer3d(torch.nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        omega_0: float,
        learn_omega_0: bool,
        bias: bool,
    ):
        """
        Implements a Linear Layer of the form y = omega_0 * W x + b, where x is 2 dimensional
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=bias,
        )

        # omega_0
        if learn_omega_0:
            self.omega_0 = torch.nn.Parameter(torch.Tensor(1))
            with torch.no_grad():
                self.omega_0.fill_(omega_0)
        else:
            tensor_omega_0 = torch.zeros(1)
            tensor_omega_0.fill_(omega_0)
            self.register_buffer("omega_0", tensor_omega_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.omega_0 * torch.nn.functional.conv3d(
            x,
            self.weight,
            self.bias,
            stride=1,
            padding=0,
        )


class SIRENLayerNd(torch.nn.Linear):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        omega_0: float,
        learn_omega_0: bool,
        bias: bool,
    ):
        """
        Implements a Linear Layer of the form y = omega_0 * W x + b, where x is >3 dimensional
        """
        super().__init__(
            in_features=in_channels,
            out_features=out_channels,
            bias=bias,
        )

        # omega_0
        if learn_omega_0:
            self.omega_0 = torch.nn.Parameter(torch.Tensor(1))
            with torch.no_grad():
                self.omega_0.fill_(omega_0)
        else:
            tensor_omega_0 = torch.zeros(1)
            tensor_omega_0.fill_(omega_0)
            self.register_buffer("omega_0", tensor_omega_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.omega_0 * torch.nn.functional.linear(x, self.weight, self.bias)
