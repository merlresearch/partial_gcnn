# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 David W. Romero & Robert-Jan Bruintjes
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
#
# Code adapted from https://github.com/rjbruin/flexconv -- MIT License


import math

import torch

# logger
import wandb
from omegaconf import OmegaConf

import models
import partial_equiv.general as gral

# project
import partial_equiv.groups as groups
from globals import DATASET_SIZES


def construct_model(
    cfg: OmegaConf,
) -> torch.nn.Module:
    """
    :param cfg: A config file specifying the parameters of the model.
    :return: An instance of the model specified in the config (torch.nn.Module)
    """

    # Define in_channels
    if cfg.dataset in ["MNIST", "rotMNIST", "MNIST6-180", "MNIST6-M"]:
        in_channels = 1
    elif cfg.dataset in ["CIFAR10", "CIFAR100", "STL10", "PCam"]:
        in_channels = 3
    else:
        raise NotImplementedError(f"Not in_channels for dataset {cfg.dataset} found.")

    # Define output_channels
    if cfg.dataset in [
        "MNIST6-180",
        "MNIST6-M",
        "PCam",
    ]:
        out_channels = 2
    elif cfg.dataset in [
        "MNIST",
        "rotMNIST",
        "CIFAR10",
        "STL10",
    ]:
        out_channels = 10
    elif cfg.dataset in [
        "CIFAR100",
    ]:
        out_channels = 100
    else:
        raise NotImplementedError(f"Not out_channels for dataset {cfg.dataset} found.")

    # Define sampling strategy
    cfg.base_group.sampling_method = {
        "random": groups.SamplingMethods.RANDOM,
        "deterministic": groups.SamplingMethods.DETERMINISTIC,
    }[cfg.base_group.sampling_method]

    # Print the defined parameters.
    print(
        f"Automatic Parameters:\n dataset = {cfg.dataset}, net_type = {cfg.net.type}, in_channels = {in_channels},"
        f"out_channels = {out_channels}, sampling_method = {cfg.base_group.sampling_method}"
    )

    # Define group
    base_group = construct_group(cfg)

    # Construct model
    model_name = getattr(models, cfg.net.type)
    model = model_name(
        in_channels=in_channels,
        out_channels=out_channels,
        base_group=base_group,
        net_config=cfg.net,
        base_group_config=cfg.base_group,
        kernel_config=cfg.kernel,
        conv_config=cfg.conv,
    )

    # print number parameters
    # no_params = gral.utils.num_params(model)
    # print("Number of parameters:", no_params)
    # wandb.run.summary["no_params"] = no_params

    # Create DataParallel for multi-GPU support
    model = torch.nn.DataParallel(model)

    return model


def construct_group(
    cfg: OmegaConf,
):
    gumbel_no_iterations = math.ceil(DATASET_SIZES[cfg.dataset] / float(cfg.train.batch_size))  # Iter per epoch
    gumbel_no_iterations = cfg.train.epochs * gumbel_no_iterations

    base_group = getattr(groups, cfg.base_group.name)(
        gumbel_init_temperature=cfg.base_group.gumbel_init_temp,
        gumbel_end_temperature=cfg.base_group.gumbel_end_temp,
        gumbel_no_iterations=gumbel_no_iterations,
    )

    return base_group
