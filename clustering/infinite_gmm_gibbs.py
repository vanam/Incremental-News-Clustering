import argparse
import math
import random

import numpy as np
import scipy.misc
from scipy.special import gammaln

import clustering.utils
from clustering.gaussianMixture import GaussianMixture
from clustering.plot import plot_gaussian_mixture, plot_show
from clustering.readData import read_data, extract_points


class Cache:
    def __init__(self, N, prior):
        """
        :type N: int
        :type prior: HyperParameters

        :param N:
        :param prior:
        """
        n = np.concatenate([[1], np.arange(1, prior.v_0 + N + 2)])  # first element added for indexing
        self.log_v = np.log(n)
        self.gammaln_divided_by_2 = gammaln(n / 2.)


class HyperParameters:
    def __init__(self, alpha, m_0, k_0, s_0, v_0):
        """
        :type alpha: float
        :type m_0: np.ndarray
        :type k_0: float
        :type s_0: np.ndarray
        :type v_0: float
        :param alpha: Parameter for Dirichlet prior on the mixing weights pi
        :param m_0: Prior mean for mu
        :param k_0: How strongly we believe the m_0
        :param s_0: Proportional to prior mean for sigma
        :param v_0: How strongly we believe the s_0
        """
        self.alpha = alpha
        self.m_0 = m_0
        self.k_0 = k_0
        D = len(m_0)
        assert v_0 >= D, "v_0 must be larger or equal to dimension of data"
        self.s_0 = s_0
        self.v_0 = v_0


def gibbs(data, k, tries, prior):
    """
    Collapsed Gibbs sampling
    Inspired by https://github.com/kamperh/bayes_gmm/

    :type data: list
    :type k: int
    :type tries: int
    :type prior: HyperParameters

    :param data:
    :param k:
    :param tries:
    :param prior:
    :return:
    """
    # Number of samples
    n = len(data)

    cache = Cache(n, prior)

    # Randomly initiate a priori probabilities (a.k.a. weights)
    pi = np.random.dirichlet(np.full(k, prior.alpha))

    # Randomly initiate cluster indicator for each sample
    # Format: [..., [0, 0, 1, 0], ...]
    z = np.random.multinomial(1, pi, n)

    # Transform z to one dimension in range [1, K]
    z = np.argmax(z, axis=1) + 1

    # Repeat Gibbs sampling iterations
    for t in range(tries):
        print("Iteration #%03d k = %d" % (t+1, k))

        # For each data sample (in random order)
        for i in random.sample(range(n), n):
            # Unset sample from old cluster
            z[i] = 0  # Cluster zero = unassigned

            # Calculate how many samples are in the each cluster
            z_n = np.bincount(z, minlength=k + 1)

            # Find empty cluster(s)
            empty_clusters = [i for i in range(1, k + 1) if z_n[i] == 0]

            # If z_n contains zero, re-label observation with highest cluster number
            for j in reversed(empty_clusters):
                # If not last cluster is empty
                if j != k:
                    # Swap last cluster assignments with the zero-one
                    z_n[j] = z_n[k]
                    z = [j if z_j == k else z_j for z_j in z]

                # Decrement number of clusters
                k -= 1

            # Keep track of probability for each cluster
            log_z_probability = np.zeros(k + 1, np.float)

            # For each cluster
            for c in range(1, k + 1):
                # Calculate P(z_i=c|z_{-i}, alpha) - (25.35) in Murphy, p. 886 (ignoring constant denominator)
                probability = np.log(z_n[c])

                # Calculate P(x_i|x_{-i}, z_i=c, z_{-i}, beta) = P(x_i|D_{-i, c})
                # (25.33) in Murphy, p. 886
                # The posterior predictive distribution
                probability += _log_posterior_predictive(data, z, i, c, prior, z_n, cache)

                # Store the probability
                log_z_probability[c] = probability

            # Consider new cluster k*
            # (25.35) in Murphy, p. 886 (ignoring constant denominator) * (25.38) in Murphy, p. 886
            log_z_probability[0] = np.log(prior.alpha) + _log_prior_predictive(data, i, prior, cache)

            # Convert log probabilities to probabilities
            log_z_probability = np.exp(log_z_probability - scipy.misc.logsumexp(log_z_probability))

            # Sample the new cluster assignment for `data[i]`
            c = clustering.utils.draw(log_z_probability)

            # Shall we create new cluster?
            if c == 0:
                k += 1
                c = k

            # Set new sample to a cluster
            z[i] = c

    # Calculate how many samples are in the each cluster
    z_n = np.bincount(z, minlength=k + 1)

    weights = np.zeros(k)
    means = []
    covariances = []

    # For each cluster
    for c in range(k):
        w_c, mu_c, S_c = _map(data, c, prior, z, z_n)
        weights[c] = w_c
        means.append(mu_c)
        covariances.append(S_c)

    # Recreate one-hot cluster assignments
    z2 = []
    for d in range(len(data)):
        z_i = np.zeros(k)
        z_i[z[d] - 1] = 1
        z2.append(z_i)

    return GaussianMixture(data, k, weights, means, covariances, z2)


def _log_prior_predictive(data, i, prior, cache):
    """
    Return the probability of `data[i]` under the prior alone.

    :type i: int
    :type prior: HyperParameters

    :param i:
    :param prior:
    """
    D = len(prior.m_0)
    mu = prior.m_0
    v = prior.v_0 - D + 1

    # Special case of _log_posterior_predictive with no data
    # (4.214) in Murphy, p. 134
    S_N = prior.s_0

    cov = (prior.k_0 + 1) / (prior.k_0 * (prior.v_0 - D + 1)) * S_N
    logdet_cov = np.linalg.slogdet(cov)[1]
    inv_cov = np.linalg.inv(cov)

    return _multivariate_students_t(data, i, mu, logdet_cov, inv_cov, v, cache)


def _log_posterior_predictive(data, z, i, c, prior, z_n, cache):
    """
    Return the log posterior predictive probability of `data[i]` under
    component `c`.

    (4.222) in Murphy, p. 135

    :type data: list
    :type z: np.ndarray
    :type i: int
    :type c: int
    :type prior: HyperParameters
    :type z_n: np.ndarray
    :type cache: Cache

    :param i:
    :param c:
    :param prior:
    :param z_n:
    :param cache:
    """
    k_N = prior.k_0 + z_n[c]
    v_N = prior.v_0 + z_n[c]

    data_k = [data[i] for i in range(len(data)) if z[i] == c]
    m_N = (prior.k_0 * prior.m_0 + np.sum(data_k, axis=0)) / k_N
    mu = m_N
    D = len(prior.m_0)
    v = v_N - D + 1

    # (4.214) in Murphy, p. 134
    S = np.sum([np.outer(_, _) for _ in data_k], axis=0)
    S_N = prior.s_0 + S + prior.k_0 * np.outer(prior.m_0, prior.m_0) - k_N * np.outer(m_N, m_N)

    cov = (k_N + 1.)/(k_N * (v_N - D + 1.)) * S_N
    log_det_cov_k = np.linalg.slogdet(cov)[1]
    inv_cov_k = np.linalg.inv(cov)

    return _multivariate_students_t(data, i, mu, log_det_cov_k, inv_cov_k, v, cache)


def _multivariate_students_t(data, i, mu, logdet_covar, inv_covar, v, cache):
    """
    Return the value of the log multivariate Student's t PDF at `X[i]`.
    (2.71) in Murphy, p. 46
    :type cache: Cache
    """
    # Dimension
    D = len(mu)

    delta = data[i] - mu
    return (
        + cache.gammaln_divided_by_2[v + D]
        - cache.gammaln_divided_by_2[v]
        - D / 2. * cache.log_v[v]
        - D / 2. * math.log(np.pi)
        - 0.5 * logdet_covar
        - (v + D) / 2. * math.log(1 + 1. / v * np.dot(np.dot(delta, inv_covar), delta))
    )


def _map(data, c, prior, z, z_n):
    """
    Return MAP estimate of the mean vector and covariance matrix of `k`.

    See (4.215) in Murphy, p. 134.
    The Dx1 mean vector and DxD covariance
    matrix is returned.

    :type data: list
    :type c: int
    :type prior: HyperParameters
    :type z: np.ndarray
    :type z_n: np.ndarray
    """
    k_N = prior.k_0 + z_n[c]
    v_N = prior.v_0 + z_n[c]
    D = len(prior.m_0)
    data_k = [data[i] for i in range(len(data)) if z[i] == c]
    m_N = (prior.k_0 * prior.m_0 + np.sum(data_k, axis=0)) / k_N

    # (4.214) in Murphy, p. 134
    S = np.sum([np.outer(_, _) for _ in data_k], axis=0)
    S_N = prior.s_0 + S + prior.k_0 * np.outer(prior.m_0, prior.m_0) - k_N * np.outer(m_N, m_N)

    sigma = S_N / (v_N + D + 2)
    return len(data_k) / len(data), m_N, sigma

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run infinite (CRP) GMM clustering on data.')

    parser.add_argument('file', metavar='file', type=argparse.FileType('rb'),
                        help='a data file')
    parser.add_argument('-k', dest='k', type=int,
                        help='the initial number of clusters (default: 2)')
    parser.add_argument('-t, --tries', dest='tries', type=int,
                        help='the number of tries (default: 20)')
    parser.add_argument('-a, --alpha', dest='alpha', type=float,
                        help='alpha (default: 3.0)')
    parser.add_argument('-b, --believe', dest='k_0', type=float,
                        help='believe (default: 0.01)')
    parser.add_argument('-p, --plot', dest='plot', action='store_true',
                        help='plot found clusters')
    parser.add_argument('-s, --shuffle', dest='shuffle', action='store_true',
                        help='randomly shuffle data points')
    parser.set_defaults(k=2, tries=20, alpha=1.0, k_0=0.05, plot=False, shuffle=False)

    args = parser.parse_args()

    data = extract_points(read_data(args.file))

    # Shuffle points if desired
    if args.shuffle:
        np.random.shuffle(data)

    # Calculate clusters
    D = np.shape(data)[1]

    alpha = args.alpha
    m_0 = np.zeros(D)
    k_0 = args.k_0
    v_0 = D + 3
    S_0 = v_0 * np.eye(D)

    prior = HyperParameters(alpha, m_0, k_0, S_0, v_0)
    mixture = gibbs(data, args.k, args.tries, prior)
    print(mixture)

    # Plot mixture if desired
    if args.plot:
        plot_gaussian_mixture(mixture)
        plot_show()
