import math
from typing import Tuple, List

import numpy as np
from scipy.special import multigammaln, gammaln
from scipy.stats import multivariate_normal

from clustering_system.clustering.mixture.GaussianMixtureABC import GaussianMixtureABC


class FullGaussianMixture(GaussianMixtureABC):

    @property
    def likelihood(self) -> float:
        """
        :return: Return average log likelihood of data.
        """
        k, alpha, mean, covariance, _ = self.parameters

        rv = [None] * k
        for i in range(k):
            rv[i] = multivariate_normal(mean[i], covariance[i])

        likelihood = sum(np.log([sum([alpha[j] * rv[j].pdf(d) for j in range(k)]) for d in self.X]))

        return likelihood / len(self.X)

    @property
    def number_of_parameters(self) -> int:
        cluster_numbers = np.unique(self.z)
        K = len(cluster_numbers)

        mean_parameters = self.D * K
        cov_parameters = K * self.D * (self.D + 1) / 2

        return int(mean_parameters + cov_parameters + K - 1)

    @property
    def parameters(self) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Get parameters of the Gaussian components.

        :return: (number of components, component weights, means, covariance matrices, data for each component)
        """
        cluster_numbers = np.unique(self.z)
        K = len(cluster_numbers)

        alpha = np.empty(K, dtype=float)
        mean = np.empty((K, self.D), dtype=float)
        covariance = np.empty((K, self.D, self.D), dtype=float)
        X = []

        for i, cn in enumerate(cluster_numbers):
            a, m, c = self._map(cn)
            alpha[i] = a
            mean[i] = m
            covariance[i] = c
            X.append(self.X[self.z == cn])

        return K, alpha, mean, covariance, X

    def get_marginal_likelihood(self, members: frozenset) -> float:
        """
        Compute marginal log likelihood p(X)

        :param members: set of document indices
        :return: log likelihood
        """
        D = self.D
        N = len(members)

        v_0 = self.prior.v_0
        v_n = v_0 + N

        k_0 = self.prior.k_0
        k_n = k_0 + N

        X = self.X[list(members)]

        m_0 = self.prior.m_0
        m_n = (k_0 * m_0 + np.sum(X, axis=0)) / k_n

        logdet_0 = np.linalg.slogdet(self.prior.S_0)[1]
        S = np.sum([np.outer(_, _) for _ in X], axis=0)
        logdet_n = np.linalg.slogdet(self.prior.S_0 + S + k_0 * np.outer(m_0, m_0) - k_n * np.outer(m_n, m_n))[1]

        return \
            - N * D / 2 * math.log(math.pi) \
            + multigammaln(v_n / 2, D) + \
            + v_0 / 2 * logdet_0 + \
            - multigammaln(v_0 / 2, D) \
            - v_n / 2 * logdet_n \
            + D / 2 * (math.log(k_0) - math.log(k_n))

    def _map(self, c: int):
        """
        Return MAP estimate of the mean vector and covariance matrix of `k`.

        See (4.215) in Murphy, p. 134.
        The Dx1 mean vector and DxD covariance
        matrix is returned.

        :param c: cluster number
        """
        N = len(self.X)
        X_k = self.X[self.z == c]
        N_k = len(X_k)

        k_N = self.prior.k_0 + N_k
        v_N = self.prior.v_0 + N_k
        D = len(self.prior.m_0)
        m_N = (self.prior.k_0 * self.prior.m_0 + np.sum(X_k, axis=0)) / k_N

        # (4.214) in Murphy, p. 134
        S = np.sum([np.outer(_, _) for _ in X_k], axis=0)
        S_N = self.prior.S_0 + S + self.prior.k_0 * np.outer(self.prior.m_0, self.prior.m_0) - k_N * np.outer(m_N, m_N)

        sigma = S_N / (v_N + D + 2)
        return N_k / N, m_N, sigma

    def get_posterior_predictive(self, i: int, cluster_numbers: List[int]) -> np.ndarray:
        """
        Return the log posterior predictive probability of `X[i]` under component `k` for each component.

        :param i: document index
        :return: np.ndarray of K floats where K is number of components
        """
        K = len(cluster_numbers)

        probabilities = np.empty(K, float)
        for j, cn in enumerate(cluster_numbers):
            probabilities[j] = self._get_posterior_predictive_k(i, cn)

        return probabilities

    def _get_posterior_predictive_k(self, i: int, c: int) -> np.ndarray:

        k_N = self.prior.k_0 + self.N_k[c]
        v_N = self.prior.v_0 + self.N_k[c]

        X_k = self.X[self.z == c]
        m_N = (self.prior.k_0 * self.prior.m_0 + np.sum(X_k, axis=0)) / k_N
        mu = m_N
        v = v_N - self.D + 1

        # (4.214) in Murphy, p. 134
        S = np.sum([np.outer(_, _) for _ in X_k], axis=0)
        S_N = self.prior.S_0 + S + self.prior.k_0 * np.outer(self.prior.m_0, self.prior.m_0) - k_N * np.outer(m_N, m_N)

        cov = (k_N + 1.) / (k_N * (v_N - self.D + 1.)) * S_N
        log_det_cov_k = np.linalg.slogdet(cov)[1]
        inv_cov_k = np.linalg.inv(cov)

        return self._multivariate_students_t(i, mu, log_det_cov_k, inv_cov_k, v)

    def get_prior_predictive(self, i: int) -> float:
        """
        Return the probability of `X[i]` under the prior alone.

        :param i: document id
        :return:
        """
        mu = self.prior.m_0
        v = self.prior.v_0 - self.D + 1

        # Special case of _log_posterior_predictive with no data
        # (4.214) in Murphy, p. 134
        S_N = self.prior.S_0

        cov = (self.prior.k_0 + 1) / (self.prior.k_0 * (self.prior.v_0 - self.D + 1)) * S_N
        logdet_cov = np.linalg.slogdet(cov)[1]
        inv_cov = np.linalg.inv(cov)

        return self._multivariate_students_t(i, mu, logdet_cov, inv_cov, v)

    def _multivariate_students_t(self, i, mu, logdet_cov, inv_cov, v):
        delta = self.X[i] - mu
        return (
                + gammaln((v + self.D) / 2.)
                - gammaln(v / 2.)
                - self.D / 2. * math.log(v)
                - self.D / 2. * math.log(np.pi)
                - 0.5 * logdet_cov
                - (v + self.D) / 2. * math.log(1 + 1. / v * np.dot(np.dot(delta, inv_cov), delta))
        )