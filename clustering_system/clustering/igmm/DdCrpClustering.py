import numpy as np
import scipy.misc

from clustering_system.clustering.ClusteringABC import CovarianceType
from clustering_system.clustering.GibbsClusteringABC import GibbsClusteringABC
from clustering_system.clustering.gmm.FullGaussianMixture import FullGaussianMixture
from clustering_system.clustering.gmm.GaussianMixtureABC import PriorABC
from clustering_system.clustering.igmm.GaussianCustomerAssignment import GaussianCustomerAssignment
from clustering_system.utils import draw_indexed
from clustering_system.visualization.LikelihoodVisualizer import LikelihoodVisualizer


class DdCrpClustering(GibbsClusteringABC):

    def __init__(self, K: int, D: int, alpha: float, prior: PriorABC, n_iterations: int, probability_threshold: float = 0.001, visualizer: LikelihoodVisualizer = None, covariance_type: CovarianceType = CovarianceType.full):
        super().__init__(K, D, alpha, prior, n_iterations, visualizer=visualizer)

        if covariance_type == CovarianceType.full:
            self.mixture = FullGaussianMixture(prior)
        else:
            raise NotImplementedError("Unsupported covariance type %s." % covariance_type)

        self.customer_assignment = GaussianCustomerAssignment(prior, self.mixture, probability_threshold)

    def add_documents(self, ids, vectors: np.ndarray):
        pass

    def likelihood(self) -> float:
        return self.customer_assignment.likelihood()

    def _sample_document(self, i: int):
        # Remove customer assignment for a document i
        self.customer_assignment.remove_assignment(i)

        # Calculate customer assignment probabilities for each document (including self)
        probabilities = self.customer_assignment.get_assignment_probabilitites(i)

        # Convert log probabilities to probabilities (softmax)
        probabilities = np.exp(probabilities - scipy.misc.logsumexp(probabilities))

        # Sample new customer assignment
        c = draw_indexed(probabilities)

        # Link document to new customer
        self.customer_assignment.add_assignment(i, c)

    def __iter__(self):
        raise NotImplementedError
