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

        # Cache
        self.log_alpha = math.log(self.alpha)

    def _sample_document(self, i: int):
        # Remove component assignment for a document i
        self._remove_document(i)

        cluster_numbers = np.unique(self.mixture.z)
        cluster_numbers = cluster_numbers[cluster_numbers != -1]  # Do not consider unassigned items

        # Calculate component assignment probabilities for each component
        probabilities = self.mixture.get_posterior_predictive(i, cluster_numbers) + self._get_mixture_probability(cluster_numbers)

        # Calculate component assignment probabilities for new component
        probabilities = np.append(
            self.mixture.get_prior_predictive(i) + self._get_new_cluster_mixture_probability(),
            probabilities
        )

        # Convert log probabilities to probabilities (softmax)
        probabilities = np.exp(probabilities - scipy.misc.logsumexp(probabilities))

        # Sample new component assignment
        z_i = draw(probabilities)
        z = cluster_numbers[z_i] if z_i < len(cluster_numbers) else self._get_new_cluster_number()

        # Add document to new component
        self._add_document(i, z)

    def _get_mixture_probability(self, cluster_numbers: np.ndarray) -> np.ndarray:
        """
        Return the log mixture probability under component `k` for each component.

        :return: np.ndarray of K floats where K is number of non-empty components sorted by cluster number
        """
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
        return self.log_alpha
