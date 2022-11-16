# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 David W. Romero & Robert-Jan Bruintjes
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
#
# Code adapted from https://github.com/rjbruin/flexconv -- MIT License


import math

import torch

# project
import partial_equiv.general as gral
import partial_equiv.partial_gconv.conv
from globals import DATASET_SIZES


def construct_optimizer(model, cfg):
    """
    Constructs the optimizer to be used during training.
    :param model: model to optimize,
    :param cfg: config dict.
    :return: optimizer
    """

    # Unpack parameters
    optim_type = cfg.train.optimizer
    lr = cfg.train.lr
    lr_probs = cfg.train.lr_probs
    lr_omega0 = cfg.train.lr_omega0
    momentum = cfg.train.optimizer_params.momentum
    nesterov = cfg.train.optimizer_params.nesterov

    # Check value of prob_lr and replace if undefined.
    if lr_probs == 0.0:
        lr_probs = lr
    if lr_omega0 == 0.0:
        lr_omega0 = lr

    # Divide params in probs, omega_0s and others
    all_parameters = set(model.parameters())
    # probs
    probs = []
    for m in model.modules():
        if isinstance(m, partial_equiv.partial_gconv.GroupConv):
            probs += list(
                map(
                    lambda x: x[1],
                    list(filter(lambda kv: "probs" in kv[0], m.named_parameters())),
                )
            )
    probs = set(probs)
    other_params = all_parameters - probs
    # omega_0
    omega_0s = []
    for m in model.modules():
        if isinstance(
            m,
            (
                partial_equiv.ck.siren.SIRENLayer1d,
                partial_equiv.ck.siren.SIRENLayer2d,
                partial_equiv.ck.siren.SIRENLayer3d,
                partial_equiv.ck.siren.SIRENLayerNd,
            ),
        ):
            omega_0s += list(
                map(
                    lambda x: x[1],
                    list(filter(lambda kv: "omega_0" in kv[0], m.named_parameters())),
                )
            )
    omega_0s = set(omega_0s)
    other_params = other_params - omega_0s
    # The parameters must be given as a list
    probs = list(probs)
    omega_0s = list(omega_0s)
    other_params = list(other_params)

    # Construct optimizer
    if optim_type == "SGD":
        optimizer = torch.optim.SGD(
            [
                {"params": other_params},
                {"params": probs, "lr": lr_probs},
                {"params": omega_0s, "lr": lr_omega0},
            ],
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
        )
    elif optim_type == "Adam":
        optimizer = torch.optim.Adam(
            [
                {"params": other_params},
                {"params": probs, "lr": lr_probs},
                {"params": omega_0s, "lr": lr_omega0},
            ],
            lr=lr,
        )
    else:
        raise NotImplementedError(f"Optimizer {optim_type} not implemented.")

    return optimizer


def construct_scheduler(optimizer, cfg):
    """
    Constructs a learning rate scheduler
    :param optimizer: the optimizer to be used.
    :param cfg: config dict.
    :return: scheduler
    """

    if cfg.train.scheduler == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.train.scheduler_params.decay_steps,
            gamma=1.0 / cfg.train.scheduler_params.decay_factor,
        )
    elif cfg.train.scheduler == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=1.0 / cfg.train.scheduler_params.decay_factor,
            patience=cfg.train.scheduler_params.patience,
            verbose=True,
        )
    elif cfg.train.scheduler == "cosine":
        size_dataset = DATASET_SIZES[cfg.dataset]
        if cfg.train.scheduler_params.warmup_epochs != -1:
            T_max = (cfg.train.epochs - cfg.train.scheduler_params.warmup_epochs) * math.ceil(
                size_dataset / float(cfg.train.batch_size)
            )  # - warmup epochs
        else:
            T_max = cfg.train.epochs * math.ceil(size_dataset / float(cfg.train.batch_size))

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=1e-6,
        )
    else:
        lr_scheduler = None
        print(f"WARNING! No scheduler will be used. Input value = {cfg.train.scheduler}")

    if cfg.train.scheduler_params.warmup_epochs != -1 and lr_scheduler is not None:
        if cfg.train.scheduler != "cosine":
            raise NotImplementedError(
                f"Warmup lr is currently only implemented for cosine schedulers. Current: {cfg.train.scheduler}"
            )

        size_dataset = DATASET_SIZES[cfg.dataset]

        lr_scheduler = gral.lr_scheduler.LinearWarmUp_LRScheduler(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            warmup_iterations=cfg.train.scheduler_params.warmup_epochs
            * math.ceil(size_dataset / float(cfg.train.batch_size)),
        )

    return lr_scheduler
