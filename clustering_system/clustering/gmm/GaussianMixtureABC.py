from abc import ABC

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

    def __init__(self, prior: PriorABC):
        self.prior = prior

    def add_assignment(self, i: int, k: int):
        """
        Add document i to a component k.

        :param i: document index
        :param k: component index
        """
        pass

    def remove_assignment(self, i: int):
        """
        Remove document i from its component.

        :param i: document index
        """
        pass

    def get_posterior_predictive(self, i: int) -> np.ndarray:
        """
        Return the log posterior predictive probability of `X[i]` under component `k` for each component.

        :param i: document index
        :return: np.ndarray of K floats where K is number of components
        """
        pass

    def get_prior_predictive(self, i: int) -> float:
        """
        Return the probability of `X[i]` under the prior alone.

        :param i: document id
        :return:
        """
        pass
