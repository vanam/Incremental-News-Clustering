from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np


class PriorABC(ABC):
    pass


class NormalInverseWishartPrior(PriorABC):

    def __init__(self, m_0: np.ndarray, k_0: float, S_0: np.ndarray, v_0: float):
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

    def __init__(self, D: int, prior: PriorABC):
        self.D = D
        self.prior = prior

        self.X = np.empty((0, D), float)
        self.z = np.empty((0, 1), int)  # -1 - unassigned, <0, K) assigned

    def add(self, vector: np.ndarray, z: int):
        # print("add before")
        # print(self.X)
        # print(self.z)
        self.X = np.vstack((self.X, np.array([vector])))
        self.z = np.append(self.z, z)
        # print("add aftrer")
        # print(self.X)
        # print(self.z)

    @property
    @abstractmethod
    def likelihood(self) -> float:
        """
        :return: Return average log likelihood of data.
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

    # def get_posterior_predictive(self, i: int) -> np.ndarray:
    #     """
    #     Return the log posterior predictive probability of `X[i]` under component `k` for each component.
    #
    #     :param i: document index
    #     :return: np.ndarray of K floats where K is number of components
    #     """
    #     pass
    #
    # def get_prior_predictive(self, i: int) -> float:
    #     """
    #     Return the probability of `X[i]` under the prior alone.
    #
    #     :param i: document id
    #     :return:
    #     """
    #     pass
