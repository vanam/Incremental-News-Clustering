from typing import Tuple, List

import numpy as np

from clustering_system.clustering.gmm.GaussianMixtureABC import PriorABC, GaussianMixtureABC


class GaussianCustomerAssignment:

    def __init__(self, c: list, z: list, prior: PriorABC, mixture: GaussianMixtureABC, probability_threshold: float):
        self.z = z
        self.c = c
        self.prior = prior
        self.mixture = mixture
        self.probability_threshold = probability_threshold

    def add_assignment(self, i: int, c: int):
        """
        Add new customer assignment c_i.

        :param i: document index
        :param c: new customer assignment (document index to which document i points to)
        """
        raise NotImplementedError

    def remove_assignment(self, i: int) -> None:
        """
        Remove customer assignment c_i.

        :param i: document index
        """
        raise NotImplementedError

    def get_assignment_probabilitites(self, i: int) -> np.ndarray:
        """
        Get assignment probabilities. Probabilities under the threshold are not returned.
        Always at least one probability (self assignment) is returned.

        :param i: document index
        :return: list of tuples (document index, assignment probability)
        """
        # return np.array([(i*1.1, 1.0)])
        raise NotImplementedError

    def likelihood(self) -> float:
        """
        :return: Return average log likelihood of data.
        """
        raise NotImplementedError
