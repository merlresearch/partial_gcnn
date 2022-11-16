# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 David W. Romero & Robert-Jan Bruintjes
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
#
# Code adapted from https://github.com/rjbruin/flexconv -- MIT License


import torch


class LinearWarmUp_LRScheduler(torch.nn.Module):
    def __init__(
        self,
        optimizer,
        lr_scheduler,
        warmup_iterations,
    ):
        """
        Creates a learning rate scheduler, which warms up the learning rate from 0 to lr of the optimizer,
        in a total of warmup_iterations.
        """
        if not isinstance(lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            raise NotImplementedError(
                f"LinearWarmUp currently works with CosineAnnealingLR only. Current: {lr_scheduler.__class__}."
            )

        super().__init__()

        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda iter: iter / warmup_iterations,
        )
        self.lr_scheduler = lr_scheduler
        self.warmup_iterations = warmup_iterations
        self.iteration = 0

    def step(self):
        if self.iteration <= self.warmup_iterations:
            self.warmup_scheduler.step()
        else:
            self.lr_scheduler.step()
        self.iteration += 1
