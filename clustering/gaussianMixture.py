from operator import itemgetter

import numpy as np
from scipy.stats import multivariate_normal

import clustering.cluster as cluster


class GaussianMixture:
    def __init__(self, data, k, weights, means, covariances, membership_weights):
        self.membership_weights = membership_weights
        self.data = data
        self.k = k
        self.weights = weights
        self.means = means
        self.covariances = covariances

    def get_clusters(self):
        cluster_data = [[] for _ in range(self.k)]

        for i in range(len(self.data)):
            # Find cluster index of most probable cluster
            cluster_index = max(enumerate(self.membership_weights[i]), key=itemgetter(1))[0]

            # Assign observation to cluster
            cluster_data[cluster_index].append(self.data[i])

        clusters = []
        for j in range(self.k):
            clusters.append(cluster.Cluster(cluster_data[j]))

        return clusters

    def pdf(self, s):
        density = 0.0

        # For each Gaussian compute membership weight
        for j in range(self.k):
            density += self.weights[j] * multivariate_normal.pdf(s, self.means[j], self.covariances[j])

        return density

    def score_samples(self, samples):
        scores = []

        for s in samples:
            scores.append(self.pdf(s))

        return np.array(scores)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string = 'GaussianMixture {\n'
        string += '  k                  = ' + str(self.k) + ',\n'
        string += '  weights            = ' + str(self.data) + ',\n'
        string += '  means              = ' + str(self.means) + ',\n'
        string += '  covariances        = ' + str(self.covariances) + ',\n'
        string += '  data               = ' + str(self.data) + ',\n'
        string += '  membership_weights = ' + str(self.membership_weights) + '\n'
        string += '}'
        return string
