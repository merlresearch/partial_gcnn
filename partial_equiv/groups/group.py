# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# built-in
from enum import Enum, auto

# torch
import torch


class SamplingMethods(Enum):
    RANDOM = auto()
    DETERMINISTIC = auto()


class Group(torch.nn.Module):
    def __init__(self, dimension, dimension_Rd, dimension_stabilizer):
        """Implements a Lie group.
        @param dimension: Dimensionality of the lie group (number of dimensions in the basis of the algebra).
        @param identity: Identity element of the group.
        """
        super().__init__()
        self.dimension = dimension
        self.dimension_Rd = dimension_Rd
        self.dimension_stabilizer = dimension_stabilizer

    def product(self, g1, g2):
        """Defines group product on two group elements.
        :param g1: Group element 1
        :param g2: Group element 2
        """
        raise NotImplementedError()

    def inv(self, g):
        """Defines inverse for group element.
        :param g: A group element.
        """
        raise NotImplementedError()

    def logarithmic_map(self, g):
        """Defines logarithmic map from lie group to algebra.
        :param g: A Lie group element.
        """
        raise NotImplementedError()

    def exponential_map(self, h):
        """Defines exponential map from lie algebra to group.
        :param h: A Lie algebra element.
        """
        raise NotImplementedError()

    def determinant(self, m):
        """Calculates the determinant of a representation of a given group element.
        :param m: matrix representation of a group element.
        """
        raise NotImplementedError()

    def left_action_on_Rd(self, g, x):
        """Group action of an element from the subgroup H on a vector in Rd.
        :param g: Group element.
        :param x: Vector in Rd.
        """
        raise NotImplementedError()

    def left_action_on_H(self, g1, g2):
        """Group action of an element from the subgroup H on an element in H.
        :param g1: Group element
        :param g2: Other group element
        """
        raise NotImplementedError()

    def matrix_form(self, g):
        """
        Returns the matrix representation of an abstract group element g.

        :param g:  Group element
        :return: matrix representation: e.g., rotation matrix of an angle theta for a rotation group element theta
        """
        raise NotImplementedError()

    def normalize_g_distance(self, g):
        """Normalize values of group elements to range between -1 and 1 for CKNet
        :param g: group element
        """
        raise NotImplementedError()

    def sample_from_stabilizer(
        self,
        no_samples: int,
        no_elements: int,
        method: SamplingMethods,
        device: str,
        partial_equivariance: bool,
        probs: torch.Tensor,
    ):
        """
        :param no_samples: Number of independent samples to take
        :param num_elements: Number of group elements to sample from the group
        :param method: Method of sampling used
        :return:
        """
        raise NotImplementedError()
