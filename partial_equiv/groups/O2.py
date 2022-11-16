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


class E2(Group):
    def __init__(
        self,
        gumbel_init_temperature: Optional[float] = None,
        gumbel_end_temperature: Optional[float] = None,
        gumbel_no_iterations: Optional[int] = None,
    ):
        """
        The optional parameters gumbel_* are only required if partial_equivariance is used.

        :param gumbel_init_temperature:
        :param gumbel_end_temperature:
        :param gumbel_iterations:

        E2 is a three dimensional group, where the mirroring and rotation components are "merged" into one dimension.

        Here, for easier parameterization of the probability distributions of the group in components rotation, and
        mirroring, we parameterize the E2 group as a group with 4 elements as follows:

        [rotation, mirroring, shift_y, shift_x]

        However, to avoid computational complexities related to working with Conv4d, we use the proper dimension when
        computing the convolution operation.
        """

        dimension = 3
        dimension_Rd = 2
        dimension_stabilizer = 2

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
        :return
        """

        """
        This can be implemented as a matrix multiplication between matrix_form(g1) * matrix_form(g2).
        However, we want this implementation to be fast. Consequently we implement it in a more efficient form.

        In our code, we implement the group 02 as a mirroring element m followed by a rotation r applied by left multiplication:
         g1 = r1 m1.
         The mirroring element m can take values (1, -1), where -1 depicts a mirroring along the y axis.

        An important property that is crucial here is the group rule that a mirroring followed by a rotation is equivalent to
        an inverse rotation first, followed by the same mirroring.

        Consequently, for two elements g1, g2, we have that:

         r1 m1 * r2 m2 = r1 ( m1 * r2) m2
                       = r1 ( r2 ^ {m1}) * m1 * m2
                       = (r1 r2^{m1}) (m1 m2)

        In other words, we have that the result is given by an element rm defined as:
        r = r1 + m1 * r2
        m = m1 * m2
        """
        assert g1.shape[0] == g2.shape[0]

        rot = torch.remainder(
            g2[..., 0].unsqueeze(-2) + (g2[..., 1].unsqueeze(-2) * g1[..., 0].unsqueeze(-1)),
            2 * np.pi,
        )
        mirror = g1[..., 1].unsqueeze(-1) * g2[..., 1].unsqueeze(-2)

        return torch.stack([rot, mirror], dim=-1)

    def inv(self, g):
        """
        :param g:
        :return:
        """
        """
        The inverse of this group works as follows:
        If m = -1, i.e., mirroring, then (-1, r)^{-1} = (-1, r)
        If m = 1, i.e., no mirroring, then the rotation must be inverted. I.e., (m, r)^{-1} = (1, -r)
        """

        g_out = g.clone()
        g_out[..., 0] = -1.0 * g[..., 1] * g[..., 0]
        return g_out

    def exponential_map(self, lie_element):
        """Exponential map from algebra to group
        :param lie_element: a Lie algebra element from the rotation group
        :return:
        """
        lie_element[..., 0] = torch.remainder(lie_element[..., 0], 2 * np.pi)
        return lie_element

    def logarithmic_map(self, g):
        """Logarithmic map from group to algebra
        :param g: a group element from the rotation group
        """
        return g

    def left_action_on_Rd(self, g, x):
        """Transform an Rd input meshgrid by a vector of group elements g
        :param g: a vector of group elements from the rotation group
        :param x: a meshgrid with relative positions on Rd,
            expected format: [2, dim_y, dim_x]
        """
        return torch.einsum("noi,iyx->noyx", self.matrix_form(g), x.double()).type(torch.float32)

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
        # Elements are defined as [rotation, mirroring]
        rot, mirror = g[..., 0], g[..., 1]

        R = torch.zeros(*rot.shape, 2, 2, device=rot.device, dtype=rot.dtype)
        cos = rot.cos()
        sin = rot.sin()

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

        Furthermore, a mirroring along the y axis is applied. In other words a multiplication with a matrix
            1 0
            0 m
        where m is either -1 for mirroring, and 1 for not mirroring. The final matrix is then:
            cos \theta,       sin \theta
            m * - sin \theta, m * cos \theta
        """

        R[..., 0, 0] = cos
        R[..., 1, 0] = -sin * mirror
        R[..., 0, 1] = sin
        R[..., 1, 1] = cos * mirror

        return R

    def determinant(self, m):
        return 1.0

    def construct_probability_variables(self, method, no_elements):
        if method == SamplingMethods.RANDOM:
            # First value is the theta value, second value is the mirroring probability.
            probs = torch.ones(
                2,
            )

        if method == SamplingMethods.DETERMINISTIC:
            # The first no_elements // 2 -1 values are the probabilities of sampling a rotation.
            # The last element is the probability of sampling mirroring.
            probs = torch.ones(no_elements // 2)

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

        # Each rotation sample will be paired with a pair or reflexions [1, -1].
        # Hence, we need to sample no_elements // 2 rotations to get a total of no_samples.
        no_elements_rotation = no_elements // 2

        if method == SamplingMethods.DETERMINISTIC:

            if not partial_equivariance:

                # Rotation
                uniform_grid_rot = torch.linspace(
                    0,
                    2 * math.pi * float(no_elements_rotation - 1) / float(no_elements_rotation),
                    no_elements_rotation,
                    dtype=torch.float64,
                    device=device,
                )

                # Reflexion
                grid_reflexion = torch.cat(
                    [
                        torch.ones(1, no_elements_rotation, device=device),
                        -1.0 * torch.ones(1, no_elements_rotation, device=device),
                    ],
                    dim=-1,
                ).repeat(no_samples, 1)

                # Concatenation
                grid = torch.stack([uniform_grid_rot.repeat(1, 2), grid_reflexion], dim=-1)

                g_elems = self.exponential_map(grid)

            else:
                # Partial equivariance

                unnormalized_prob_rotation = probs[:-1]
                unnormalized_prob_reflexion = probs[-1]

                # Rotation
                uniform_grid_rot = torch.linspace(
                    0,
                    2 * math.pi * float(no_elements_rotation - 1) / float(no_elements_rotation),
                    no_elements_rotation,
                    dtype=torch.float64,
                    device=device,
                )

                if unnormalized_prob_rotation.numel() == 0:
                    # If there's only one element, no need to compute probabilities on rotations
                    pass
                else:
                    # Get samples for each rotation
                    prob_rotations = (
                        torch.distributions.RelaxedBernoulli(
                            temperature=self.get_current_gumbel_temperature(),
                            # logits=unnormalized_prob_rotation,
                            probs=torch.sigmoid(self.sigmoid_temp * unnormalized_prob_rotation),
                        )
                        .rsample([1])
                        .squeeze()
                    )

                    sample_rotation = (prob_rotations > 0.5).float()

                    sample_rotation = sample_rotation - prob_rotations.detach() + prob_rotations

                    # Concatenate a 1.0 at the beginning (the probability of the identity)
                    sample_rotation = torch.cat([torch.ones(1, device=device), sample_rotation])
                    uniform_grid_rot = sample_rotation * uniform_grid_rot

                    mask = torch.nonzero(uniform_grid_rot[1:] == 0.0).squeeze() + 1
                    uniform_grid_rot = tensor_delete(uniform_grid_rot, mask)

                    # Modify no_elements_rotation based on the eliminated samples
                    no_elements_rotation = uniform_grid_rot.shape[0]

                # Reflexions
                prob_reflexion = torch.distributions.RelaxedBernoulli(
                    temperature=self.get_current_gumbel_temperature(),
                    # logits=unnormalized_prob_reflexion,
                    probs=torch.sigmoid(self.sigmoid_temp * unnormalized_prob_reflexion),
                ).rsample(unnormalized_prob_reflexion.shape)

                if prob_reflexion > 0.5:
                    sample_reflexion = torch.ones(prob_reflexion.shape, device=device)
                else:
                    sample_reflexion = torch.zeros(prob_reflexion.shape, device=device)

                sample_reflexion = sample_reflexion - prob_reflexion.detach() + prob_reflexion

                grid_reflexed = sample_reflexion * -1.0 * torch.ones(1, no_elements_rotation, device=device)

                reflect = -1.0 in grid_reflexed
                if reflect:
                    grid_reflexion = torch.cat(
                        [
                            torch.ones(1, no_elements_rotation, device=device),
                            grid_reflexed,
                        ],
                        dim=-1,
                    )
                else:
                    grid_reflexion = torch.ones(1, no_elements_rotation, device=device)

                grid = torch.stack(
                    [uniform_grid_rot.repeat(1, 1 + int(reflect)), grid_reflexion],
                    dim=-1,
                )
                g_elems = self.exponential_map(grid)

        elif method == SamplingMethods.RANDOM:

            if not partial_equivariance:

                # Rotation
                uniform_grid_rot = torch.linspace(
                    0,
                    2 * math.pi * float(no_elements_rotation - 1) / float(no_elements_rotation),
                    no_elements_rotation,
                    dtype=torch.float64,
                    device=device,
                )
                delta = torch.rand(no_samples, 1, device=device) * (2 * math.pi) / float(no_elements_rotation)
                uniform_grid_rot = uniform_grid_rot.unsqueeze(0) + delta

                # Reflexion
                grid_reflexion = torch.cat(
                    [
                        torch.ones(1, no_elements_rotation, device=device),
                        -1.0 * torch.ones(1, no_elements_rotation, device=device),
                    ],
                    dim=-1,
                ).repeat(no_samples, 1)

                grid = torch.stack([uniform_grid_rot.repeat(1, 2), grid_reflexion], dim=-1)

                g_elems = self.exponential_map(grid)

            elif partial_equivariance:

                # Probs has two values, one depicting theta, and one depicting the probability of sampling mirroring.
                theta = probs[0]
                unnormalized_prob_reflexion = probs[1]

                no_elements_rotation = min(math.ceil(no_elements_rotation * theta.item()), no_elements_rotation)
                # In the case that the group is reduced to its identity, we have only one sample.
                if no_elements_rotation == 0:
                    no_elements_rotation = 1

                # It can be that the probs is smaller than one, in which case the manifold direction is simply inverted.
                # This is not a problem, but we need to take the absolute value of no_elements to form the lin-space.
                no_elements_rotation = abs(no_elements_rotation)

                # Rotation
                uniform_grid_rot = torch.linspace(
                    0,
                    2 * math.pi * theta.item() * float(no_elements_rotation - 1) / float(no_elements_rotation),
                    no_elements_rotation,
                    dtype=torch.float64,
                    device=device,
                )
                delta = torch.rand(no_samples, 1, device=device) * (2 * math.pi * theta) / float(no_elements)
                uniform_grid_rot = (uniform_grid_rot - math.pi * theta).unsqueeze(0) + delta

                # Reflexion
                prob_reflexion = torch.distributions.RelaxedBernoulli(
                    temperature=self.get_current_gumbel_temperature(),
                    # logits=unnormalized_prob_reflexion,
                    probs=torch.sigmoid(self.sigmoid_temp * unnormalized_prob_reflexion),
                ).rsample(unnormalized_prob_reflexion.shape)

                if prob_reflexion > 0.5:
                    sample_reflexion = torch.ones(prob_reflexion.shape, device=device)
                else:
                    sample_reflexion = torch.zeros(prob_reflexion.shape, device=device)

                sample_reflexion = sample_reflexion - prob_reflexion.detach() + prob_reflexion

                grid_reflexed = sample_reflexion * -1.0 * torch.ones(1, no_elements_rotation, device=device)

                reflect = -1.0 in grid_reflexed
                if reflect:
                    grid_reflexion = torch.cat(
                        [
                            torch.ones(1, no_elements_rotation, device=device),
                            grid_reflexed,
                        ],
                        dim=-1,
                    )
                else:
                    grid_reflexion = torch.ones(1, no_elements_rotation, device=device)

                grid = torch.stack(
                    [uniform_grid_rot.repeat(1, 1 + int(reflect)), grid_reflexion],
                    dim=-1,
                )
                g_elems = self.exponential_map(grid)

        return g_elems

    def normalize_g_distance(self, g):
        """Normalize values of group elements to range between -1 and 1 for CKNet
        :param g:
        :return:
        """
        g_out = g.clone()
        g_out[..., 0] = (1 / np.pi) * g[..., 0] - 1.0
        return g_out

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
