import math

import numpy as np
import scipy.misc

from clustering_system.clustering.ClusteringABC import CovarianceType
from clustering_system.clustering.GibbsClusteringABC import GibbsClusteringABC
from clustering_system.clustering.mixture.GaussianMixtureABC import PriorABC
from clustering_system.utils import draw
from clustering_system.visualization.LikelihoodVisualizer import LikelihoodVisualizer


class CrpClustering(GibbsClusteringABC):

    def __init__(self, D: int, alpha: float, prior: PriorABC, n_iterations: int,
                 visualizer: LikelihoodVisualizer = None,
                 covariance_type: CovarianceType = CovarianceType.full):
        super().__init__(D, alpha, prior, n_iterations, visualizer=visualizer, covariance_type=covariance_type)

    def _sample_document(self, i: int):
        # Remove component assignment for a document i
        self._remove_document(i)

        # Calculate component assignment probabilities for each component
        probabilities = self.mixture.get_posterior_predictive(i, cluster_numbers=np.unique(self.mixture.z)) + self._get_mixture_probability()

        # Calculate component assignment probabilities for new component
        probabilities = np.append(
            self.mixture.get_prior_predictive(i) + self._get_new_cluster_mixture_probability(),
            probabilities
        )

        # Convert log probabilities to probabilities (softmax)
        probabilities = np.exp(probabilities - scipy.misc.logsumexp(probabilities))

        # Sample new component assignment
        z = draw(probabilities)

        # Add document to new component
        self._add_document(i, z)

    def _get_mixture_probability(self) -> np.ndarray:
        """
        Return the log mixture probability under component `k` for each component.

        :return: np.ndarray of K floats where K is number of non-empty components sorted by cluster number
        """
        cluster_numbers = np.unique(self.mixture.z)
        K = len(cluster_numbers)

        probabilities = np.empty(K, float)
        for i, cn in enumerate(cluster_numbers):
            probabilities[i] = self.mixture.N_k[cn]

        return probabilities

    def _get_new_cluster_mixture_probability(self) -> float:
        """
        Return the log mixture probability for new component.

        :return: log probability
        """
        return math.log(self.alpha)
