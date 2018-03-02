import random
from abc import abstractmethod
from heapq import heappop
from queue import PriorityQueue

import numpy as np

from clustering_system.clustering.ClusteringABC import ClusteringABC, CovarianceType
from clustering_system.clustering.gmm.FullGaussianMixture import FullGaussianMixture

from clustering_system.clustering.gmm.GaussianMixtureABC import PriorABC
from clustering_system.visualization.LikelihoodVisualizer import LikelihoodVisualizer


class GibbsClusteringABC(ClusteringABC):

    def __init__(self, D: int, alpha: float, prior: PriorABC, n_iterations: int, K_max: int = None,
                 visualizer: LikelihoodVisualizer = None, covariance_type: CovarianceType = CovarianceType.full):
        super().__init__(D)
        self.K_max = K_max
        self.alpha = alpha
        self.prior = prior
        self.n_iterations = n_iterations
        self.visualizer = visualizer

        # Each document has its id, feature vector, table assignment
        self.ids = []
        if covariance_type == CovarianceType.full:
            self.mixture = FullGaussianMixture(D, prior)
        else:
            raise NotImplementedError("Unsupported covariance type %s." % covariance_type)

        # Maintain cluster numbers as concise as possible
        self.counter = 0
        self.reusable_numbers = PriorityQueue()

    def add_documents(self, vectors: np.ndarray, metadata: np.ndarray):
        raise NotImplementedError
        # for md, vector in zip(metadata, vectors):
        #     # Add document at the end of arrays
        #     self.ids.append(md[0])
        #     self.X = np.vstack((self.X, np.array([vector])))
        #     self.z = -1  # customer is unassigned to a table

    @abstractmethod
    def _sample_document(self, i: int):
        pass

    def update(self):
        # TODO clear cache if necessary

        # Repeat Gibbs sampling iterations
        for _ in range(self.n_iterations):

            # For each document (in random order)
            for i in random.sample(range(self.N), self.N):
                self._sample_document(i)

            # Keep track of likelihood
            if self.visualizer is not None:
                self.visualizer.add(self.likelihood, self.N)

        # TODO update cluster components if necessary

    def _get_new_cluster_number(self):
        if not self.reusable_numbers.empty():
            z = self.reusable_numbers.get_nowait()
        else:
            z = self.counter
            self.counter += 1

        return z
