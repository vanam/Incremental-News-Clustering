import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_almost_equal, assert_array_almost_equal
from scipy.special import multigammaln
from scipy.special.cython_special import gammaln

from clustering_system.clustering.mixture.GaussianMixtureABC import NormalInverseWishartPrior, PriorABC
from clustering_system.clustering.mixture.FullGaussianMixture import FullGaussianMixture


class TestFullGaussianMixture:

    D = 2

    _m_0 = np.array([1, 2])
    _k_0 = 0.05
    _S_0 = np.array([[2, 1], [1, 3]])
    _v_0 = D + 2

    prior = NormalInverseWishartPrior(_m_0, _k_0, _S_0, _v_0)

    _d_0 = np.array([0, 1])
    _d_1 = np.array([2, 3])
    _d_2 = np.array([3, 4])
    _d_3 = np.array([5, 6.6])
    _d_4 = np.array([-1.1, -2.2])

    _z = [1, 2, 1, 1, 2]

    @staticmethod
    def _init_mixture(D: int, prior: PriorABC):
        mixture = FullGaussianMixture(D, prior)
        mixture.new_vector(TestFullGaussianMixture._d_0, -1)
        mixture.new_vector(TestFullGaussianMixture._d_1, TestFullGaussianMixture._z[0])  # Wrong assignment
        mixture.new_vector(TestFullGaussianMixture._d_2, TestFullGaussianMixture._z[2])
        mixture.new_vector(TestFullGaussianMixture._d_3, TestFullGaussianMixture._z[3])
        mixture.new_vector(TestFullGaussianMixture._d_4, TestFullGaussianMixture._z[4])

        assert mixture.N_k[1] == 3
        assert mixture.N_k[2] == 1

        mixture.update_z(0, TestFullGaussianMixture._z[0])
        # Fix wrong assignment
        mixture.update_z(1, -1)
        mixture.update_z(1, TestFullGaussianMixture._z[1])

        assert mixture.N_k[1] == 3
        assert mixture.N_k[2] == 2

        return mixture

    def test_data(self):
        mixture = self._init_mixture(self.D, self.prior)

        data = [
            self._d_0,
            self._d_1,
            self._d_2,
            self._d_3,
            self._d_4
        ]

        for i in range(5):
            assert_array_equal(mixture.X[i], data[i])
            assert self._z[i] == mixture.z[i]

    @pytest.mark.parametrize("members", [frozenset(), frozenset([0]), frozenset([0, 1, 3, 4])])
    def test_marginal_likelihood(self, members: frozenset):
        mixture = self._init_mixture(self.D, self.prior)

        D = self.D
        N = len(members)

        v_0 = self.prior.v_0
        v_n = v_0 + N

        k_0 = self.prior.k_0
        k_n = k_0 + N

        X = mixture.X[list(members)]

        m_0 = self.prior.m_0
        m_n = (k_0 * m_0 + np.sum(X, axis=0)) / k_n

        logdet_0 = np.linalg.slogdet(self.prior.S_0)[1]
        S = np.sum([np.outer(_, _) for _ in X], axis=0)
        logdet_n = np.linalg.slogdet(self.prior.S_0 + S + k_0 * np.outer(m_0, m_0) - k_n * np.outer(m_n, m_n))[1]

        marginal_likelihood = \
            - N * D / 2 * math.log(math.pi) \
            + multigammaln(v_n / 2, D) + \
            + v_0 / 2 * logdet_0 + \
            - multigammaln(v_0 / 2, D) \
            - v_n / 2 * logdet_n \
            + D / 2 * (math.log(k_0) - math.log(k_n))

        assert_almost_equal(mixture.get_marginal_likelihood(members), marginal_likelihood)

    @pytest.mark.parametrize("i, c", zip(range(5), [1, 2, 1, 1, 2]))
    def test_posterior_predictive(self, i: int, c: int):
        mixture = self._init_mixture(self.D, self.prior)

        k_N = self.prior.k_0 + mixture.N_k[c]
        v_N = self.prior.v_0 + mixture.N_k[c]

        X_k = mixture.X[mixture.z == c]
        m_N = (self.prior.k_0 * self.prior.m_0 + np.sum(X_k, axis=0)) / k_N
        mu = m_N
        v = v_N - self.D + 1

        # (4.214) in Murphy, p. 134
        S = np.sum([np.outer(_, _) for _ in X_k], axis=0)
        S_N = self.prior.S_0 + S + self.prior.k_0 * np.outer(self.prior.m_0, self.prior.m_0) - k_N * np.outer(m_N, m_N)

        cov = (k_N + 1.) / (k_N * (v_N - self.D + 1.)) * S_N
        log_det_cov_k = np.linalg.slogdet(cov)[1]
        inv_cov_k = np.linalg.inv(cov)

        posterior_predictive = mixture._multivariate_students_t(i, mu, log_det_cov_k, inv_cov_k, v)

        assert_almost_equal(mixture._get_posterior_predictive_k(i, c), posterior_predictive)

    @pytest.mark.parametrize("i", range(5))
    def test_prior_predictive(self, i: int):
        mixture = self._init_mixture(self.D, self.prior)

        v = self.prior.v_0 - self.D + 1

        # Special case of _log_posterior_predictive with no data
        # (4.214) in Murphy, p. 134
        cov = (self.prior.k_0 + 1) / (self.prior.k_0 * (self.prior.v_0 - self.D + 1)) * self.prior.S_0
        logdet_cov = np.linalg.slogdet(cov)[1]
        inv_cov = np.linalg.inv(cov)

        prior_predictive = mixture._multivariate_students_t(i, self.prior.m_0, logdet_cov, inv_cov, v)

        assert_almost_equal(mixture.get_prior_predictive(i), prior_predictive)

    @pytest.mark.parametrize("i", range(5))
    def test_multivariate_students_t(self, i: int):
        mixture = self._init_mixture(self.D, self.prior)

        mu = mixture.prior.m_0
        v = mixture.prior.v_0 + i

        cov = np.array([[5.5, -1.1], [-1.1, 6.6]])
        logdet_cov = np.linalg.slogdet(cov)[1]
        inv_cov = np.linalg.inv(cov)

        delta = mixture.X[i] - mu
        stud = (
                + gammaln((v + self.D) / 2.)
                - gammaln(v / 2.)
                - self.D / 2. * math.log(v)
                - self.D / 2. * math.log(np.pi)
                - 0.5 * logdet_cov
                - (v + self.D) / 2. * math.log(1 + 1. / v * np.dot(np.dot(delta, inv_cov), delta))
        )

        assert_almost_equal(mixture._multivariate_students_t(i, mu, logdet_cov, inv_cov, v), stud)

    @pytest.mark.parametrize("c", range(1, 3))
    def test_map(self, c: int):
        mixture = self._init_mixture(self.D, self.prior)

        N = len(mixture.X)
        X_k = mixture.X[mixture.z == c]
        N_k = len(X_k)

        k_N = self.prior.k_0 + N_k
        v_N = self.prior.v_0 + N_k
        D = len(self.prior.m_0)
        m_N = (self.prior.k_0 * self.prior.m_0 + np.sum(X_k, axis=0)) / k_N

        # (4.214) in Murphy, p. 134
        S = np.sum([np.outer(_, _) for _ in X_k], axis=0)
        S_N = self.prior.S_0 + S + self.prior.k_0 * np.outer(self.prior.m_0, self.prior.m_0) - k_N * np.outer(m_N, m_N)

        sigma = S_N / (v_N + D + 2)

        pi_k, m_k, sigma_k = mixture._map(c)

        assert_almost_equal(pi_k, N_k / N)
        assert_array_almost_equal(m_k, m_N)
        assert_array_almost_equal(sigma_k, sigma)

    def test_merge_split(self):
        mixture = self._init_mixture(self.D, self.prior)

        posterior_predictive = [mixture.get_posterior_predictive(i, [1, 2]) for i in range(5)]

        mixture.merge(1, 2)
        mixture.split(1, 3, [1, 4])

        assert_array_almost_equal([mixture.get_posterior_predictive(i, [1, 3]) for i in range(5)], posterior_predictive)
