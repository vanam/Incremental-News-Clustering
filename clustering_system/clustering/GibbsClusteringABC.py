import random
from abc import abstractmethod
from queue import PriorityQueue
from typing import Tuple, List

import numpy as np

from clustering_system.clustering.ClusteringABC import ClusteringABC, CovarianceType
from clustering_system.clustering.mixture.FullGaussianMixture import FullGaussianMixture
from clustering_system.clustering.mixture.GaussianMixtureABC import PriorABC
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

    @property
    def likelihood(self) -> float:
        return self.mixture.likelihood

    @property
    def parameters(self) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        return self.mixture.parameters

    def add_documents(self, vectors: np.ndarray, metadata: np.ndarray):
        for md, vector in zip(metadata, vectors):
            doc_id, timestamp, *_ = md

            # Add document at the end of arrays
            self.ids.append(doc_id)
            self.mixture.add(vector, -1)  # New customer waits outside of the restaurant
            self.N += 1                   # Increment number of documents (customers)

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

    def __iter__(self):
        """
        For each document return (doc_id, cluster_id)
        """
        for doc_id, cluster_id in zip(self.ids, self.mixture.z):
            yield doc_id, cluster_id

    def _get_new_cluster_number(self):
        if not self.reusable_numbers.empty():
            z = self.reusable_numbers.get_nowait()
        else:
            z = self.counter
            self.counter += 1

        return z

    def _remove_document(self, i: int):
        """
        Remove document from component z[i].

        :param i: document index
        """
        z = self.mixture.z[i]
        if z != -1:
            self.mixture.z[i] = -1
            self.mixture.N_k[z] -= 1

            if self.mixture.N_k[z] == 0:
                self.K -= 1

    def _add_document(self, i: int, z: int):
        """
        Add document to component z[i] = z.

        :param i: document id
        :param z: component id
        """
        self.mixture.z[i] = z
        self.mixture.N_k[z] += 1

        if self.mixture.N_k[z] == 1:
            self.K += 1

    @abstractmethod
    def _sample_document(self, i: int):
        pass
