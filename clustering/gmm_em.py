import argparse

import numpy as np
from scipy.stats import multivariate_normal

from clustering.gaussianMixture import GaussianMixture
from clustering.plot import plot_gaussian_mixture, plot_show
from clustering.readData import read_data, extract_points


def _em(data, k, threshold=1e-5):
    # Randomly assign points to their initial clusters
    initial_cluster_data = np.array_split(data, k)

    # Initial parameters estimate
    n = len(data)

    # [k] - k weights that specify how likely each Gaussian is to be chosen
    weights = [len(d) / float(n) for d in initial_cluster_data]

    # [k][d] - k means
    means = [np.mean(d, axis=0) for d in initial_cluster_data]

    # [k][d][d] - k covariance matrices [d][d]
    covariances = [np.cov(d, rowvar=False) for d in initial_cluster_data]

    # Initialize log-likelihood
    new_likelihood = _log_likelihood(data, k, weights, means, covariances)
    old_likelihood = 2 * new_likelihood

    # Iterate while log-likelihood changes are significant
    while abs(old_likelihood - new_likelihood) > threshold:
        # Expectation step
        # For each point: P(c|x_i) = does it look like point belongs to cluster c?
        # [n][k]
        membership_weights = _expectation(data, k, weights, means, covariances)

        # Maximisation step
        # Adjust cluster means and covariances to fit the points assigned to them
        weights, means, covariances = _maximisation(data, k, membership_weights)

        # Keep track of changes in log-likelihood function
        old_likelihood = new_likelihood
        new_likelihood = _log_likelihood(data, k, weights, means, covariances)

    return new_likelihood, GaussianMixture(data, k, weights, means, covariances, membership_weights)


def em(data, k, tries=1, threshold=1e-5):
    if int(k) < 1:
        raise ValueError('the number of clusters must be at least 1.')

    if int(k) > len(data):
        raise ValueError('not enough data.')

    if int(tries) < 1:
        raise ValueError('the number of tries must be at least 1.')

    if float(threshold) <= 0.0:
        raise ValueError('threshold must be greater than zero.')

    # Initialize best likelihood value to a large value
    best_likelihood = np.inf
    best_mixture = None

    # Run em algorithm several times
    for i in range(tries):
        # Run em
        likelihood, mixture = _em(data, k, threshold=threshold)

        # Accept best result
        if likelihood < best_likelihood:
            best_likelihood = likelihood
            best_mixture = mixture

    return best_likelihood, best_mixture


def _pdf(x, mean, cov):
    """Gaussian probability density function"""
    return multivariate_normal.pdf(x, mean, cov)


def _expectation(data, k, weights, means, covariances):
    n = len(data)
    membership_weights = np.zeros([n, k], dtype=float)

    # For each observation
    for i in range(n):
        # Calculate total membership weight
        sum_membership_weights = 0.0

        # For each Gaussian compute membership weight
        for j in range(k):
            membership_weights[i][j] = weights[j] * _pdf(data[i], means[j], covariances[j])
            sum_membership_weights += membership_weights[i][j]

        # For each Gaussian make membership weights proportional
        membership_weights[i] = [x / sum_membership_weights for x in membership_weights[i]]

    return membership_weights


def _maximisation(data, k, membership_weights):
    # Calculate total membership weight of the jth Gaussian
    total_membership_weights = np.sum(membership_weights, axis=0)
    total_membership = np.sum(total_membership_weights)

    weights = [total_membership_weights[i] / total_membership for i in range(k)]

    means = []
    for i in range(k):
        mean = np.average(data, axis=0, weights=membership_weights.T[i])
        means.append(mean)

    covariances = []
    for i in range(k):
        covariance = np.cov(data, rowvar=False, aweights=membership_weights.T[i])
        covariances.append(covariance)

    return weights, means, covariances


def _log_likelihood(data, k, weights, means, covariances):
    likelihood = 0.0

    for d in data:
        likelihood += np.log(sum([weights[j] * _pdf(d, means[j], covariances[j]) for j in range(k)]))

    return likelihood / len(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GMM clustering on data.')

    parser.add_argument('file', metavar='file', type=argparse.FileType('rb'),
                        help='a data file')
    parser.add_argument('-k', dest='k', type=int,
                        help='the number of clusters (default: 2)')
    parser.add_argument('-t, --tries', dest='tries', type=int,
                        help='the number of tries (default: 20)')
    parser.add_argument('--threshold', dest='threshold', type=float,
                        help='threshold (default: 1e-5)')
    parser.add_argument('-p, --plot', dest='plot', action='store_true',
                        help='plot found clusters')
    parser.add_argument('-s, --shuffle', dest='shuffle', action='store_true',
                        help='randomly shuffle data points')
    parser.set_defaults(k=2, tries=20, threshold=1e-5, plot=False, shuffle=False)

    args = parser.parse_args()

    data = extract_points(read_data(args.file))

    # Shuffle points if desired
    if args.shuffle:
        np.random.shuffle(data)

    # Calculate clusters
    likelihood, mixture = em(data, args.k, args.tries, args.threshold)
    print(mixture)
    print(likelihood)

    # Plot mixture if desired
    if args.plot:
        plot_gaussian_mixture(mixture, title='EM \'%s\'' % args.file.name)
        plot_show()
