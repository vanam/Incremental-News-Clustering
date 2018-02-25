import numpy as np
import scipy.misc

from clustering_system.clustering.ClusteringABC import CovarianceType
from clustering_system.clustering.GibbsClusteringABC import GibbsClusteringABC
from clustering_system.clustering.gmm.FullGaussianMixture import FullGaussianMixture
from clustering_system.clustering.gmm.GaussianMixtureABC import PriorABC
from clustering_system.utils import draw


class CollapsedGibbsClustering(GibbsClusteringABC):

    def __init__(self, K: int, D: int, alpha: float, prior: PriorABC, n_iterations: int, covariance_type: CovarianceType):
        super().__init__(K, D, alpha, prior, n_iterations)

        if covariance_type == CovarianceType.full:
            self.mixture = FullGaussianMixture(prior)
        else:
            raise NotImplementedError("Unsupported covariance type %s." % covariance_type)

    def _sample_document(self, i: int):
        # Remove component assignment for a document i
        self.mixture.remove_assignment(i)

        # Calculate component assignment probabilities for each component
        probabilities = self.mixture.get_posterior_predictive(i) + self._get_mixture_probability()

        # Convert log probabilities to probabilities (softmax)
        probabilities = np.exp(probabilities - scipy.misc.logsumexp(probabilities))

        # Sample new component assignment
        k = draw(probabilities)

        # Add document to new component
        self.mixture.add_assignment(i, k)

    def __iter__(self):
        raise NotImplementedError

    def _get_mixture_probability(self) -> np.ndarray:
        """
        Return the log mixture probability under component `k` for each component.

        :return: np.ndarray of K floats where K is number of components
        """
        raise NotImplementedError
