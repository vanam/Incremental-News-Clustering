from abc import ABC, abstractmethod
from collections import Counter
from typing import Tuple, List

import numpy as np


class PriorABC(ABC):
    """Abstract mixture prior class"""
    pass


class NormalInverseWishartPrior(PriorABC):
    """The fully conjugate prior for the MVN"""

    def __init__(self, m_0: np.ndarray, k_0: float, S_0: np.ndarray, v_0: int):
        """
        For more information see:
        - https://www.rdocumentation.org/packages/Boom/versions/0.7/topics/normal.inverse.wishart.prior
        - Gelman, Carlin, Stern, Rubin (2003), "Bayesian Data Analysis", [72-73], Chapman and Hall.

        :param m_0: The guessed mean of the prior distribution.
        :param k_0: The number of observations worth of weight assigned to mean.guess.
        :param s_0: A prior estimate of variance
        :param v_0: The number of observations worth of weight assigned to a prior estimate of variance.
        """
        self.m_0 = m_0
        self.k_0 = k_0
        D = len(m_0)
        assert v_0 >= D, "v_0 must be larger or equal to dimension of data"
        self.S_0 = S_0
        self.v_0 = v_0


class GaussianMixtureABC(ABC):
    """Abstract class for Gaussian mixtures"""

    def __init__(self, D: int, prior: PriorABC):
        """
        :param D: the dimension
        :param prior: the set of prior parameters
        """
        self.D = D
        self.prior = prior

        self.X = np.empty((0, D), float)
        self.z = np.empty((0, 1), int)  # -1 - unassigned, <0, K) assigned
        self.N_k = Counter()

        self._cache_N = 100
        self._cache(self._cache_N)

    def new_vector(self, vector: np.ndarray, z: int):
        """
        Add new vector to the mixture.

        :param vector: A vector
        :param z: A vector assignment
        """
        i = len(self.z)
        self.X = np.vstack((self.X, np.array([vector])))
        self.z = np.append(self.z, -1)
        self._cache_i(vector)

        self.update_z(i, z)

        # Enlarge cache if necessary
        if len(self.X) > self._cache_N:
            self._cache_N = int(self._cache_N * 1.5)
            self._cache(self._cache_N)

    @abstractmethod
    def update_z(self, i: int, z: int):
        """
        Update cluster assignment for vector i.
        """
        pass

    @abstractmethod
    def _cache(self, N: int):
        """
        Cache expensive calculation for N samples.
        """
        pass

    @abstractmethod
    def merge(self, k: int, l: int):
        """
        Merge two clusters.
        """
        pass

    @abstractmethod
    def split(self, k: int, l: int, members: list):
        """
        Move members from cluster k to cluster l.
        """
        pass

    @abstractmethod
    def _cache_i(self, vector: np.ndarray):
        """
        Cache expensive calculation for next sample i.
        """
        pass

    @property
    @abstractmethod
    def likelihood(self) -> float:
        """
        :return: Return average log likelihood of data.
        """
        pass

    @property
    @abstractmethod
    def number_of_parameters(self) -> int:
        """
        :return: Return number of mixture model parameters.
        """
        pass

    @property
    @abstractmethod
    def parameters(self) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Get parameters of the Gaussian components.

        :return: (number of components, component weights, means, covariance matrices, data for each component)
        """
        pass

    @abstractmethod
    def get_marginal_likelihood(self, members: frozenset) -> float:
        """
        Compute marginal log likelihood p(X)

        :param members: set of vector indices
        :return: log likelihood
        """
        pass

    @abstractmethod
    def get_posterior_predictive(self, i: int, cluster_numbers: List[int]) -> np.ndarray:
        """
        Return the log posterior predictive probability of `X[i]` under component `k` for each component.

        :param i: vector index
        :param cluster_numbers: list of component numbers
        :return: np.ndarray of K floats where K is number of components
        """
        pass

    @abstractmethod
    def get_prior_predictive(self, i: int) -> float:
        """
        Return the probability of `X[i]` under the prior alone.

        :param i: vector id
        :return: log prior
        """
        pass
