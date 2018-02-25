import numpy as np

from clustering_system.clustering.gmm.GaussianMixtureABC import GaussianMixtureABC


class FullGaussianMixture(GaussianMixtureABC):

    def add_assignment(self, i: int, k: int):
        """
        Add document i to a component k.

        :param i: document index
        :param k: component index
        """
        raise NotImplementedError

    def remove_assignment(self, i: int):
        """
        Remove document i from its component.

        :param i: document index
        """
        raise NotImplementedError

    def get_posterior_predictive(self, i: int) -> np.ndarray:
        """
        Return the log posterior predictive probability of `X[i]` under component `k` for each component.

        :param i: document index
        :return: np.ndarray of K floats where K is number of components
        """
        raise NotImplementedError

    def get_prior_predictive(self, i: int) -> float:
        """
        Return the probability of `X[i]` under the prior alone.

        :param i: document id
        :return:
        """
        raise NotImplementedError
