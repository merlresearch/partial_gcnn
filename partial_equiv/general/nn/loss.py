# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch

from models import CKResNet
from partial_equiv.partial_gconv.conv import GroupConv, LiftingConv, PointwiseGroupConv


class LnLoss(torch.nn.Module):
    def __init__(
        self,
        weight_loss: float,
        norm_type: int,
    ):
        """
        Computes the Ln loss on the CKConv kernels in a CKCNN.
        :param weight_loss: Specifies the weight with which the loss will be summed to the total loss.
        :param norm_type: Type of norm, e.g., 1 = L1 loss, 2 = L2 loss, ...
        """
        super().__init__()
        self.weight_loss = weight_loss
        self.norm_type = norm_type

    def forward(
        self,
        model: torch.nn.Module,
    ):
        loss = 0.0
        # Go through modules that are instances of CKConvs
        for m in model.modules():
            if isinstance(m, (LiftingConv, GroupConv, PointwiseGroupConv)):
                loss += m.conv_kernel.norm(self.norm_type)

                if m.bias is not None:
                    loss += m.bias.norm(self.norm_type)
        loss = self.weight_loss * loss
        return loss


class MonotonicPartialEquivarianceLoss(torch.nn.Module):
    def __init__(
        self,
        weight_loss: float,
    ):
        """
        Computes the Ln loss on the learned subset values for a partial equivariant network.
        It implicitly penalizes networks whose learned subsets increase during traning.
        """
        super().__init__()
        self.weight_loss = weight_loss

    def forward(
        self,
        model: torch.nn.Module,
    ):
        if not isinstance(model.module, CKResNet):
            raise NotImplementedError(f"Model of type {model.__class__.__name__} not a CKResNet.")

        # Only calculated with partial_equivariant models
        if not model.module.partial_equiv:
            return 0.0

        learned_equivariances = []
        for m in model.modules():
            if isinstance(m, (GroupConv)):
                if m.probs is not None and (m.probs.nelement() != 0):
                    learned_equivariances.append(m.probs)
        # Take the difference between the next element, and the previous.
        # Then check if it's larger than 0. If that;'s the case, then
        # the network is increasing, and thus, must be penalized.
        differences = torch.relu(
            torch.stack(
                [y - x for (x, y) in zip(learned_equivariances[:-1], learned_equivariances[1:])],
                dim=0,
            )
        )
        loss = self.weight_loss * differences.sum()
        return loss
