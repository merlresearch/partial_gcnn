# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy

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

from .ckresnet import rot_img


class ConvNormNonlin(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group: Group,
        base_group_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
        NormType: torch.nn.Module,
    ):
        super().__init__()

        # Conv layer
        self.conv = partial_gconv.GroupConv(
            in_channels=in_channels,
            out_channels=out_channels,
            group=copy.deepcopy(group),
            base_group_config=base_group_config,
            kernel_config=kernel_config,
            conv_config=conv_config,
        )

        # Normalization layer
        self.norm = ApplyFirstElem(NormType(out_channels))

        # Activation
        self.activ = ApplyFirstElem(torch.nn.ReLU())

    def forward(self, input_tuple):
        return self.activ(self.norm(self.conv(input_tuple)))


class GCNN(torch.nn.Module):
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
        dropout_blocks = net_config.dropout_blocks
        pool_blocks = net_config.pool_blocks
        block_width_factors = net_config.block_width_factors

        # Params in self
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # Define type of normalization layer to use
        if norm == "BatchNorm":
            NormType = getattr(torch.nn, f"BatchNorm{base_group.dimension}d")
        elif norm == "LayerNorm":
            NormType = gral.nn.LayerNorm
        else:
            raise NotImplementedError(f"No norm type {norm} found.")

        # Lifting
        self.lift_conv = partial_gconv.LiftingConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            group=copy.deepcopy(base_group),
            base_group_config=base_group_config,
            kernel_config=kernel_config,
            conv_config=conv_config,
        )
        self.lift_norm = ApplyFirstElem(NormType(hidden_channels))
        self.lift_nonlinear = ApplyFirstElem(torch.nn.ReLU())

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
            print(f"Block {i + 1}/{no_blocks}")

            if i == 0:
                input_ch = hidden_channels
                hidden_ch = int(hidden_channels * width_factors[i])
            else:
                input_ch = int(hidden_channels * width_factors[i - 1])
                hidden_ch = int(hidden_channels * width_factors[i])

            blocks.append(
                ConvNormNonlin(
                    in_channels=input_ch,
                    out_channels=hidden_ch,
                    group=base_group,
                    base_group_config=base_group_config,
                    kernel_config=kernel_config,
                    conv_config=conv_config,
                    NormType=NormType,
                )
            )

            # Pool layer
            if (i + 1) in pool_blocks:
                blocks.append(
                    ApplyFirstElem(
                        partial_gconv.pool.MaxPoolRn(
                            kernel_size=2,
                            stride=2,
                            padding=0,
                        )
                    )
                )

            # Pool layer
            if (i + 1) in dropout_blocks:
                blocks.append(ApplyFirstElem(torch.nn.Dropout2d(dropout)))

        self.blocks = torch.nn.Sequential(*blocks)

        # Last layer
        if block_width_factors[0] == 0.0:
            final_no_hidden = hidden_channels
        else:
            final_no_hidden = int(hidden_channels * block_width_factors[-2])
        self.last_layer = torch.nn.Linear(in_features=final_no_hidden, out_features=out_channels)

    def forward(self, x):
        out, g_samples = self.lift_nonlinear(self.lift_norm(self.lift_conv(x)))
        out, g_samples = self.blocks([out, g_samples])
        out = torch.mean(out, dim=(-1, -2, -3))
        out = self.last_layer(out)
        return out


class AugerinoGCNN(torch.nn.Module):
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
        self.net = GCNN(
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
