import numpy as np
import scipy.misc

from clustering_system.clustering.ClusteringABC import CovarianceType
from clustering_system.clustering.GibbsClusteringABC import GibbsClusteringABC
from clustering_system.clustering.mixture.GaussianMixtureABC import PriorABC
from clustering_system.utils import draw
from clustering_system.visualization.LikelihoodVisualizer import LikelihoodVisualizer


class BgmmClustering(GibbsClusteringABC):
    """Clustering based on the Bayesian Gaussian Mixture model"""

    def __init__(self, K: int, D: int, alpha: float, prior: PriorABC, n_iterations: int,
                 visualizer: LikelihoodVisualizer = None,
                 covariance_type: CovarianceType = CovarianceType.full):
        """
        :param K: Maximum number of clusters
        :param D: The length of a feature vector
        :param alpha: Hyperparameter
        :param prior: Prior
        :param n_iterations: The number of iterations to perform each update
        :param visualizer: Likelihood visualizer
        :param covariance_type: Covariance type
        """
        super().__init__(D, alpha, prior, n_iterations, K_max=K, visualizer=visualizer, covariance_type=covariance_type)

    def _sample_document(self, i: int):
        """
        Sample document i

        :param i: document id
        """
        # Remove component assignment for a document i
        self._remove_document(i)

        # Calculate component assignment probabilities for each component
        probabilities = self.mixture.get_posterior_predictive(i, list(range(self.K_max))) + np.log(self._get_mixture_probability())

        # Convert log probabilities to probabilities (softmax)
        probabilities = np.exp(probabilities - scipy.misc.logsumexp(probabilities))

        # Sample new component assignment
        k = draw(probabilities)

        # Add document to new component
        self._add_document(i, k)

    def _get_mixture_probability(self) -> np.ndarray:
        """
        Return the mixture probability under component `k` for each component.

        :return: np.ndarray of K floats where K is number of components sorted by cluster number
        """
        probabilities = np.empty(self.K_max, float)
        for i in range(self.K_max):
            probabilities[i] = self.mixture.N_k[i] + self.alpha / self.K_max

        return probabilities
