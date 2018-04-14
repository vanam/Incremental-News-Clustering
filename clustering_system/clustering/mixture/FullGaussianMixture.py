import math
from collections import defaultdict
from functools import lru_cache
from typing import Tuple, List

import numpy as np
from scipy.special import gammaln
from scipy.stats import multivariate_normal

from clustering_system.clustering.mixture.GaussianMixtureABC import GaussianMixtureABC, PriorABC


class FullGaussianMixture(GaussianMixtureABC):

    def __init__(self, D: int, prior: PriorABC):
        super().__init__(D, prior)

        self.m_n = defaultdict(lambda: self.prior.k_0 * self.prior.m_0)
        self.S_n_outer = defaultdict(lambda: self.prior.S_0 + self.prior.k_0 * self._prior_outer_m_0)
        self.logdet_covars = defaultdict(int)
        self.inv_covars = defaultdict(lambda: np.zeros((self.D, self.D), np.float))

        # Cache outer product of X[i]
        self._outer = np.zeros((0, self.D, self.D), np.float)

    def update_z(self, i: int, z: int):
        old_z = self.z[i]

        if old_z == z:  # Nothing to change
            return

        if z == -1:  # Remove cluster assignment
            self.m_n[old_z] -= self.X[i]
            self.S_n_outer[old_z] -= self._outer[i]
            self.N_k[old_z] -= 1

        else:  # Set cluster assignment
            self.m_n[z] += self.X[i]
            self.S_n_outer[z] += self._outer[i]
            self.N_k[z] += 1

        self.z[i] = z
        self._update_logdet_covar_and_inv_covar(z)

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

    # def get_marginal_likelihood(self, c: int) -> float:
    #     """
    #     Compute marginal log likelihood p(X)
    #
    #     :param c: cluster number
    #     :return: log likelihood
    #     """
    #     N = len(self.X)
    #     N_k = self.N_k[c]
    #     v_0 = self.prior.v_0
    #
    #     k_N = self.prior.k_0 + N_k
    #     v_N = self.prior.v_0 + N_k
    #     D = len(self.prior.m_0)
    #     m_N = self.m_n[c] / k_N
    #
    #     logdet_n = np.linalg.slogdet(self.S_n_outer[c] - k_N * np.outer(m_N, m_N))[1]
    #
    #     i = np.arange(1, D + 1, dtype=np.int)
    #
    #     return \
    #         - N * D / 2 * self._log_pi \
    #         + v_0 / 2 * self.logdet_0 + \
    #         - v_N / 2 * logdet_n \
    #         + D / 2 * (self._log_k_0 - math.log(k_N)) \
    #         + np.sum(
    #             self._gammaln_by_2[v_N + 1 - i] -
    #             self._gammaln_by_2[v_0 + 1 - i]
    #         )
    #
    # def get_marginal_likelihood_combined(self, c: int, d: int) -> float:
    #     """
    #     Compute marginal log likelihood p(X)
    #
    #     :param c: cluster number
    #     :param d: cluster number
    #     :return: log likelihood
    #     """
    #     N = len(self.X)
    #     N_k = self.N_k[c] + self.N_k[d]
    #     v_0 = self.prior.v_0
    #
    #     k_N = self.prior.k_0 + N_k
    #     v_N = self.prior.v_0 + N_k
    #     D = len(self.prior.m_0)
    #     m_N = (self.m_n[c] + self.m_n[d]) / k_N
    #
    #     logdet_n = np.linalg.slogdet(self.S_n_outer[c] + self.S_n_outer[d] - k_N * np.outer(m_N, m_N))[1]
    #
    #     i = np.arange(1, D + 1, dtype=np.int)
    #
    #     return \
    #         - N * D / 2 * self._log_pi \
    #         + v_0 / 2 * self.logdet_0 + \
    #         - v_N / 2 * logdet_n \
    #         + D / 2 * (self._log_k_0 - math.log(k_N)) \
    #         + np.sum(
    #             self._gammaln_by_2[v_N + 1 - i] -
    #             self._gammaln_by_2[v_0 + 1 - i]
    #         )

    @lru_cache(maxsize=512)
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

        S = np.sum([self._outer[i] for i in members], axis=0)
        logdet_n = np.linalg.slogdet(self.prior.S_0 + S + k_0 * self._prior_outer_m_0 - k_n * np.outer(m_n, m_n))[1]

        i = np.arange(1, D + 1, dtype=np.int)

        return \
            - N * D / 2 * self._log_pi \
            + v_0 / 2 * self.logdet_0 + \
            - v_n / 2 * logdet_n \
            + D / 2 * (self._log_k_0 - math.log(k_n)) \
            + np.sum(
                self._gammaln_by_2[v_n + 1 - i] -
                self._gammaln_by_2[v_0 + 1 - i]
            )

    def _map(self, c: int):
        """
        Return MAP estimate of the mean vector and covariance matrix of `k`.

        See (4.215) in Murphy, p. 134.
        The Dx1 mean vector and DxD covariance
        matrix is returned.

        :param c: cluster number
        """
        N = len(self.X)
        N_k = self.N_k[c]

        k_N = self.prior.k_0 + N_k
        v_N = self.prior.v_0 + N_k
        D = len(self.prior.m_0)
        m_N = self.m_n[c] / k_N

        # (4.214) in Murphy, p. 134
        S_N = self.prior.S_0 + self.S_n_outer[c] + self.prior.k_0 * self._prior_outer_m_0 - k_N * np.outer(m_N, m_N)

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

    def _get_posterior_predictive_k(self, i: int, c: int) -> float:
        k_N = self.prior.k_0 + self.N_k[c]
        v_N = self.prior.v_0 + self.N_k[c]
        m_N = self.m_n[c] / k_N
        mu = m_N
        v = v_N - self.D + 1
        return self._multivariate_students_t(i, mu, self.logdet_covars[c], self.inv_covars[c], v)

    @lru_cache(maxsize=512)
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
                + self._gammaln_by_2[v + self.D]
                - self._gammaln_by_2[v]
                - self.D / 2. * self._log_v[v]
                - self.D / 2. * self._log_pi
                - 0.5 * logdet_cov
                - (v + self.D) / 2. * math.log(1 + 1. / v * np.dot(np.dot(delta, inv_cov), delta))
        )

    def _cache(self, N: int):
        self._log_pi = math.log(math.pi)
        self._log_k_0 = math.log(self.prior.k_0)

        n = np.concatenate([[1], np.arange(1, self.prior.v_0 + N + 2)])  # first element is dummy for indexing
        self._log_v = np.log(n)
        self._gammaln_by_2 = gammaln(n / 2)
        self._prior_outer_m_0 = np.outer(self.prior.m_0, self.prior.m_0)
        self.logdet_0 = np.linalg.slogdet(self.prior.S_0)[1]

    def _cache_i(self, vector: np.ndarray):
        self._outer = np.vstack((self._outer, [np.outer(vector, vector)]))

    def _update_logdet_covar_and_inv_covar(self, k: int):
        k_N = self.prior.k_0 + self.N_k[k]
        v_N = self.prior.v_0 + self.N_k[k]
        m_N = self.m_n[k] / k_N
        covar = (k_N + 1.) / (k_N * (v_N - self.D + 1.)) * (self.S_n_outer[k] - k_N * np.outer(m_N, m_N))
        self.logdet_covars[k] = np.linalg.slogdet(covar)[1]
        self.inv_covars[k] = np.linalg.inv(covar)

    def merge(self, k: int, l: int):
        # Destoy cluster l
        self.N_k[l] = 0
        self.m_n.pop(l, None)
        self.S_n_outer.pop(l, None)

        for i in np.where(self.z == l)[0]:
            # Set new cluster assignment
            self.z[i] = k
            self.m_n[k] += self.X[i]
            self.S_n_outer[k] += self._outer[i]
            self.N_k[k] += 1

        self._update_logdet_covar_and_inv_covar(k)
        self._update_logdet_covar_and_inv_covar(l)

    def split(self, k: int, l: int, members: list):
        if k == l:
            print("%d = %d" % (k, l))

        for i in members:
            # Remove old cluster assignment
            self.m_n[k] -= self.X[i]
            self.S_n_outer[k] -= self._outer[i]
            self.N_k[k] -= 1


            if self.N_k[k] < 0:
                raise ValueError

            # Set new cluster assignment
            self.z[i] = l
            self.m_n[l] += self.X[i]
            self.S_n_outer[l] += self._outer[i]
            self.N_k[l] += 1

            self._update_logdet_covar_and_inv_covar(k)
            self._update_logdet_covar_and_inv_covar(l)
