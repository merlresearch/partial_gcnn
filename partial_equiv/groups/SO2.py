# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2021 David W. Romero & Jean-Baptiste Cordonnier
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
#
# Code adapted from https://github.com/dwromero/g_selfatt-- MIT License


# built-in
import math

# typing
from typing import Optional

import numpy as np

# torch
import torch

# project
from .group import Group, SamplingMethods


class SE2(Group):
    def __init__(
        self,
        gumbel_init_temperature: Optional[float] = None,
        gumbel_end_temperature: Optional[float] = None,
        gumbel_no_iterations: Optional[int] = None,
    ):

        dimension = 3
        dimension_Rd = 2
        dimension_stabilizer = 1
        super().__init__(
            dimension=dimension,
            dimension_Rd=dimension_Rd,
            dimension_stabilizer=dimension_stabilizer,
        )

        # Define variables for sampling
        self.gumbel_init_temperature = gumbel_init_temperature
        self.gumbel_end_temperature = gumbel_end_temperature
        self.gumbel_no_iterations = gumbel_no_iterations
        self.register_buffer("gumbel_iter_counter", torch.zeros(1))

        self.sigmoid_temp = 2.0

    def product(self, g1, g2):
        """Computes the group product of two group elements.
        :param g1:
        :param g2:
        :return [g1.shape[0], g1.shape[1], g2.shape[2] ]
        """
        assert g1.shape[0] == g2.shape[0]
        return torch.remainder(g1.unsqueeze(-1) + g2.unsqueeze(-2), 2 * np.pi)

    def inv(self, g):
        """
        :param g:
        :return:
        """
        return -g

    def exponential_map(self, lie_element):
        """Exponential map from algebra to group
        :param lie_element: a Lie algebra element from the rotation group
        :return:
        """
        return torch.remainder(lie_element, 2 * np.pi)

    def logarithmic_map(self, g):
        """Logarithmic map from group to algebra
        :param g: a group element from the rotation group
        """
        return g

    def left_action_on_Rd(self, g, x):
        """Transform an Rd input meshgrid by a vector of group elements g
        :param g: a vector of group elements from the rotation group
        :param x: a meshgrid with relative positions on Rd,
            expected format: [2, dim_x, dim_y]
        """
        return torch.einsum("noi,ixy->noxy", self.matrix_form(g), x.double()).type(torch.float32)

    def left_action_on_H(self, g, h):
        """
        @param g: element of rotation group
        @param x: element of subgroup H
        @return:
        """
        return self.product(g, h)

    def matrix_form(self, g):
        """Represents abstract group elements in their matricial form.
        :param g: a vector of group elements from the rotation group
        """
        g = g.squeeze(-1)
        R = torch.zeros(*g.shape, 2, 2, device=g.device, dtype=g.dtype)
        cos = g.cos()
        sin = g.sin()

        """
        A rotation matrix is defined as:
            cos \theta, - sin \theta
            sin \theta, cos \theta

        This matrix assumes that the input are [x, y] coordinates. Nevertheless, in PyTorch,
        tensors are defined as [batch_dim, no_channels, y_coords, x_coords]. As a result,
        in order to compute the corresponding rotation, we must transpose it. That is, the used
        rotation matrix is defined as:
            cos \theta, sin \theta
            - sin \theta, cos \theta
        """

        R[..., 0, 0] = cos
        R[..., 1, 0] = -sin
        R[..., 0, 1] = sin
        R[..., 1, 1] = cos

        return R

    def determinant(self, m):
        return 1.0

    def construct_probability_variables(self, method, no_elements):
        if method == SamplingMethods.RANDOM:
            # The theta value to learn (it is multiplied by 2 pi when sampling).
            probs = torch.ones(1)

        if method == SamplingMethods.DETERMINISTIC:
            # The dimension is no_elements - 1, because we always sample the identity.
            probs = torch.ones(no_elements - 1)

        return probs

    def sample_from_stabilizer(
        self,
        no_samples,
        no_elements,
        method,
        device,
        partial_equivariance,
        probs,
    ):
        """Sample a set of group elements. These elements are subsequently used to transform
        the ckconv kernel grids. We sample a grid uniformly on the Lie algebra, which we map to the group with the
        exponential map.
        :param no_samples: number of independent samples to take. Only if random.
        :param no_elements: number of group elements to sample
        :param method: sampling method
        """
        if method == SamplingMethods.DETERMINISTIC:

            uniform_grid = torch.linspace(
                0,
                2 * math.pi * float(no_elements - 1) / float(no_elements),
                no_elements,
                dtype=torch.float64,
                device=device,
            )

            if partial_equivariance:

                # Get samples for each rotation
                prob_rotations = (
                    torch.distributions.RelaxedBernoulli(
                        temperature=self.get_current_gumbel_temperature(),
                        logits=torch.sigmoid(self.sigmoid_temp * probs),
                    )
                    .rsample([1])
                    .squeeze()
                )

                sample_rotation = (prob_rotations > 0.5).float()

                sample_rotation = sample_rotation - prob_rotations.detach() + prob_rotations

                # Concatenate a 1.0 at the beginning (the probability of the identity)
                sample_rotation = torch.cat([torch.ones(1, device=device), sample_rotation])
                uniform_grid = sample_rotation * uniform_grid

                mask = torch.nonzero(uniform_grid[1:] == 0.0).squeeze() + 1
                uniform_grid = tensor_delete(uniform_grid, mask)

            g_elems = self.exponential_map(uniform_grid.unsqueeze(0))

        elif method == SamplingMethods.RANDOM:

            if not partial_equivariance:

                uniform_grid = torch.linspace(
                    0,
                    2 * math.pi * float(no_elements - 1) / float(no_elements),
                    no_elements,
                    dtype=torch.float64,
                    device=device,
                )

                delta = torch.rand(no_samples, 1, device=device) * (2 * math.pi) / float(no_elements)

                g_elems = self.exponential_map(uniform_grid.unsqueeze(0) + delta)

            elif partial_equivariance:

                # Based on the current theta value, we reduce the number of samples to use.
                # no_samples corresponds to the entire circle, and thus, from that, we can derive the new number of samples.

                no_elements = min(math.ceil(no_elements * abs(probs.item())), no_elements)

                # In the case that the group is reduced to its identity, we have only one sample.
                if no_elements == 0:
                    no_elements = 1

                # It can be that the probs is smaller than one, in which case the manifold direction is simply inverted.
                # This is not a problem, but we need to take the absolute value of no_elements to form the lin-space.
                no_elements = abs(no_elements)

                uniform_grid = torch.linspace(
                    0,
                    2 * math.pi * probs.item() * float(no_elements - 1) / float(no_elements),
                    no_elements,
                    dtype=torch.float64,
                    device=device,
                )

                delta = torch.rand(no_samples, 1, device=device) * (2 * math.pi * probs) / float(no_elements)

                g_elems = self.exponential_map((uniform_grid - math.pi * probs).unsqueeze(0) + delta)

        # Perturb the entire grid by a random rotation
        return g_elems

    def normalize_g_distance(self, g):
        """Normalize values of group elements to range between -1 and 1 for CKNet
        :param g:
        :return:
        """
        return (1 / np.pi) * g - 1.0

    def get_current_gumbel_temperature(self):
        current_temperature = self.gumbel_init_temperature - self.gumbel_iter_counter / float(
            self.gumbel_no_iterations
        ) * (self.gumbel_init_temperature - self.gumbel_end_temperature)
        if self.training:
            self.gumbel_iter_counter += 1
        return current_temperature


def tensor_delete(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]
