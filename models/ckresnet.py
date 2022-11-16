# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 David W. Romero & Robert-Jan Bruintjes
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
#
# Code adapted from https://github.com/rjbruin/flexconv -- MIT License


import copy
from functools import partial

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf

import partial_equiv.general as gral
import partial_equiv.groups as groups

# project
import partial_equiv.partial_gconv as partial_gconv
from partial_equiv.general.nn import ApplyFirstElem

# typing
from partial_equiv.groups import Group, SamplingMethods


class CKResBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group: Group,
        base_group_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
        NormType: torch.nn.Module,
        dropout: float,
        pool: bool,
    ):
        super().__init__()

        # Define convolutional layers
        Conv = partial(
            partial_gconv.GroupConv,
            base_group_config=base_group_config,
            kernel_config=kernel_config,
            conv_config=conv_config,
        )
        self.gconv1 = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            group=copy.deepcopy(group),
        )
        self.gconv2 = Conv(
            in_channels=out_channels,
            out_channels=out_channels,
            group=copy.deepcopy(group),
        )
        # Norm layers:
        self.norm_out = ApplyFirstElem(NormType(out_channels))
        # Dropout layer
        self.dp = ApplyFirstElem(torch.nn.Dropout(dropout))
        # Activation
        self.activ = ApplyFirstElem(torch.nn.ReLU())
        # Pool
        if pool:
            pool = partial_gconv.pool.MaxPoolRn(
                kernel_size=2,
                stride=2,
                padding=0,
            )
        else:
            pool = torch.nn.Identity()
        self.pool = ApplyFirstElem(pool)

        # Shortcut connection
        shortcut = []
        if (in_channels != out_channels) or base_group_config.sample_per_layer or conv_config.partial_equiv:
            # Make the width of the network smaller
            kernel_config_shortcut = copy.deepcopy(kernel_config)
            kernel_config_shortcut.no_hidden = kernel_config_shortcut.no_hidden // 2
            # Create the shortcut
            shortcut.append(
                partial_gconv.PointwiseGroupConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    group=group,
                    base_group_config=base_group_config,
                    kernel_config=kernel_config_shortcut,
                    conv_config=conv_config,
                )
            )
            self.shortcut_is_pointwise = True
        else:
            self.shortcut_is_pointwise = False
        self.shortcut = torch.nn.Sequential(*shortcut)

    def forward(self, input_tuple):
        x, input_g_elems = input_tuple
        # Following Sosnovik et al. 2020, dropout placed after first ReLU.
        out, g_elems = self.gconv1([x, input_g_elems])
        out = torch.nn.functional.layer_norm(out, out.shape[-3:])  # InstanceNorm
        out, g_elems = self.dp(self.activ([out, g_elems]))

        out, g_elems = self.gconv2([out, g_elems])
        out = torch.nn.functional.layer_norm(out, out.shape[-3:])  # InstanceNorm
        out, g_elems = self.activ([out, g_elems])

        # Shortcut
        if self.shortcut_is_pointwise:
            shortcut, g_elems = self.shortcut([x, input_g_elems, g_elems])
        else:
            shortcut, g_elems = self.shortcut([x, input_g_elems])
        out = out + shortcut

        out, g_elems = self.activ(self.pool(self.norm_out([out, g_elems])))
        return out, g_elems


class CKResNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_group: Group,
        net_config: OmegaConf,
        base_group_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
        **kwargs,
    ):
        super().__init__()

        # Unpack arguments from net_config
        hidden_channels = net_config.no_hidden
        norm = net_config.norm
        no_blocks = net_config.no_blocks
        dropout = net_config.dropout
        pool_blocks = net_config.pool_blocks
        block_width_factors = net_config.block_width_factors
        last_conv_is_T2 = net_config.last_conv_T2
        learnable_final_pooling = net_config.learnable_final_pooling
        final_spatial_dim = net_config.final_spatial_dim

        # Params in self
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.last_conv_is_T2 = last_conv_is_T2
        self.partial_equiv = conv_config.partial_equiv
        self.learnable_final_pooling = learnable_final_pooling
        self.final_spatial_dim = torch.tensor(final_spatial_dim)

        # Define type of normalization layer to use
        if norm == "BatchNorm":
            NormType = getattr(torch.nn, f"BatchNorm{base_group.dimension}d")
        elif norm == "LayerNorm":
            NormType = gral.nn.LayerNorm
        else:
            raise NotImplementedError(f"No norm type {norm} found.")

        # Activation layer
        self.activ = ApplyFirstElem(torch.nn.ReLU())

        # Lifting
        self.lift_conv = partial_gconv.LiftingConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            group=copy.deepcopy(base_group),
            base_group_config=base_group_config,
            kernel_config=kernel_config,
            conv_config=conv_config,
        )
        # Lifting normalization layer
        self.lift_norm = ApplyFirstElem(NormType(hidden_channels))

        # Pooling after lifting
        if 0 in pool_blocks:
            pool = partial_gconv.pool.MaxPoolRn(
                kernel_size=2,
                stride=2,
                padding=0,
            )
        else:
            pool = torch.nn.Identity()
        self.lift_pool = ApplyFirstElem(pool)

        # Define blocks
        # Create vector of width_factors:
        # If value is zero, then all values are one
        if block_width_factors[0] == 0.0:
            width_factors = (1,) * no_blocks
        else:
            width_factors = [
                (factor,) * n_blcks for factor, n_blcks in gral.utils.pairwise_iterable(block_width_factors)
            ]
            width_factors = [factor for factor_tuple in width_factors for factor in factor_tuple]

        if len(width_factors) != no_blocks:
            raise ValueError("The size of the width_factors does not matched the number of blocks in the network.")

        blocks = []
        for i in range(no_blocks):
            print(f"Block {i+1}/{no_blocks}")

            if i == 0:
                input_ch = hidden_channels
                hidden_ch = int(hidden_channels * width_factors[i])
            else:
                input_ch = int(hidden_channels * width_factors[i - 1])
                hidden_ch = int(hidden_channels * width_factors[i])

            blocks.append(
                CKResBlock(
                    in_channels=input_ch,
                    out_channels=hidden_ch,
                    group=base_group,
                    base_group_config=base_group_config,
                    kernel_config=kernel_config,
                    conv_config=conv_config,
                    dropout=dropout,
                    NormType=NormType,
                    pool=(i + 1) in pool_blocks,
                )
            )
        self.blocks = torch.nn.Sequential(*blocks)

        # Last layer
        # calculate output channels of blocks
        if block_width_factors[0] == 0.0:
            final_no_hidden = hidden_channels
        else:
            final_no_hidden = int(hidden_channels * block_width_factors[-2])
        # Construct layer
        last_lyr_conv_config = copy.deepcopy(conv_config)
        last_lyr_base_group_config = copy.deepcopy(base_group_config)

        last_lyr_conv_config.padding = "valid"

        if self.last_conv_is_T2:
            # Set partial equivariance to False
            last_lyr_conv_config.partial_equiv = False

            # Change the configs of the last layer
            last_lyr_base_group_config.no_samples = 1
            last_lyr_base_group_config.sample_per_layer = False
            last_lyr_base_group_config.sampling_method = SamplingMethods.DETERMINISTIC

        self.last_gconv = partial_gconv.GroupConv(
            in_channels=final_no_hidden,
            out_channels=final_no_hidden,
            group=base_group,
            base_group_config=last_lyr_base_group_config,
            kernel_config=kernel_config,
            conv_config=last_lyr_conv_config,
        )
        # Last g_conv normalization layer
        self.last_gconv_norm = ApplyFirstElem(NormType(final_no_hidden))

        # Create learnable pooling layer, if required:
        if self.learnable_final_pooling:
            if conv_config.partial_equiv:
                raise ValueError(f"learnable final pooling can only be used without partial equivariance.")

            self.learnable_pooling = torch.nn.Linear(
                in_features=last_lyr_base_group_config.no_samples * torch.prod(self.final_spatial_dim).item(),
                out_features=1,
                bias=True,
            )

        # Last Layer
        LastLinearType = getattr(gral.nn, f"Linear{base_group.dimension_Rd}d")
        # create
        self.out_layer = LastLinearType(in_channels=final_no_hidden, out_channels=out_channels)

    def forward(self, x):
        # Lifting
        out, g_samples = self.activ(self.lift_pool(self.lift_norm(self.lift_conv(x))))

        # Group blocks
        out, g_samples = self.blocks([out, g_samples])

        # Last g_conv
        if self.last_conv_is_T2:
            out = torch.mean(out, dim=-3, keepdim=True)
            g_samples = torch.zeros_like(g_samples[:, :1], device=g_samples.device)

        out, g_samples = self.activ(self.last_gconv_norm(self.last_gconv([out, g_samples])))

        # global pooling
        if self.learnable_final_pooling:
            out_shape = out.shape
            out = self.learnable_pooling(out.view(*out_shape[:-3], -1))
            out = out.view(*out_shape[:-3], *((1,) * (len(out_shape) - 2)))
        else:
            out = torch.amax(out, dim=(-3, -2, -1), keepdim=True)

        # Final layer
        out = self.out_layer(out.squeeze(-3))
        return out.view(-1, self.out_channels)


class AugerinoCKResNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_group: Group,
        net_config: OmegaConf,
        base_group_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
        **kwargs,
    ):
        super().__init__()

        assert net_config.last_conv_T2 is False
        assert net_config.learnable_final_pooling is False
        assert conv_config.partial_equiv is False
        assert base_group_config.sampling_method is SamplingMethods.RANDOM
        assert base_group_config.sample_per_batch_element is False
        assert base_group_config.sample_per_layer is False

        # The group is used to sample transformations
        self.augment_sampler = copy.deepcopy(base_group)
        self.sampler_type = base_group_config.name
        self.group_no_samples = base_group_config.no_samples
        self.group_sampling_method = base_group_config.sampling_method
        # Probabilities
        probs = self.augment_sampler.construct_probability_variables(
            self.group_sampling_method,
            self.group_no_samples,
        )
        self.probs = torch.nn.Parameter(probs)

        # Modify the config files of the model, to make it a CNN
        base_group_config_modif = copy.deepcopy(base_group_config)
        base_group_config_modif.no_samples = 1
        base_group_config_modif.sampling_method = SamplingMethods.DETERMINISTIC
        # The group must be SE2, deterministic, with a single sample.
        base_group = groups.SE2(
            gumbel_init_temperature=self.augment_sampler.gumbel_init_temperature,
            gumbel_end_temperature=self.augment_sampler.gumbel_end_temperature,
            gumbel_no_iterations=self.augment_sampler.gumbel_no_iterations,
        )

        # Create base network
        self.net = CKResNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_group=base_group,
            net_config=net_config,
            base_group_config=base_group_config_modif,
            kernel_config=kernel_config,
            conv_config=conv_config,
        )

    def forward(self, x):
        # Sample elements
        g_elements = self.augment_sampler.sample_from_stabilizer(
            no_samples=1,
            no_elements=self.group_no_samples,
            method=self.group_sampling_method,
            device=x.device,
            partial_equivariance=True,
            probs=self.probs,
        )
        if len(g_elements.shape) == 2:
            g_elements = g_elements.unsqueeze(-1)

        # Use them to augment input
        x_modif = torch.stack(
            [self.transformation(x, element) for element in g_elements[0]],
            dim=0,
        )
        x_modif = x_modif.view(-1, *x_modif.shape[2:])

        # pass through the network
        out_modif = self.net(x_modif)
        out_modif = out_modif.view(g_elements.shape[1], -1, *out_modif.shape[1:])

        # Get final result: Average of responses
        return out_modif.mean(dim=0)

    @staticmethod
    def transformation(x, element):
        x_modif = rot_img(x, element[0], dtype=x.dtype)
        if element.shape[0] == 2 and element[-1].item() == -1:
            x_modif = TF.hflip(x_modif)
        return x_modif


def get_rot_mat(theta):
    R = torch.zeros(2, 3, device=theta.device, dtype=theta.dtype)
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    R[0, 0] = cos
    R[0, 1] = -sin
    R[1, 0] = sin
    R[1, 1] = cos
    return R


def rot_img(x, theta, dtype):
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, x.size(), align_corners=True).type(dtype)
    x = F.grid_sample(x, grid)
    return x
