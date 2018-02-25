import random
from abc import abstractmethod

import numpy as np

from clustering_system.clustering.ClusteringABC import ClusteringABC

from clustering_system.clustering.gmm.GaussianMixtureABC import PriorABC
from clustering_system.visualization.LikelihoodVisualizer import LikelihoodVisualizer


class GibbsClusteringABC(ClusteringABC):

    def __init__(self, K: int, D: int, alpha: float, prior: PriorABC, n_iterations: int, visualizer: LikelihoodVisualizer = None):
        super().__init__(K, D)
        self.alpha = alpha
        self.prior = prior
        self.n_iterations = n_iterations
        self.visualizer = visualizer

        # Each document has its id, feature vector
        self.ids = np.empty(0, str)
        self.X = np.empty((0, D), float)

    def add_documents(self, ids, vectors: np.ndarray):
        for doc_id, vector in zip(ids, vectors):
            # Add document at the end of arrays
            self.ids = np.append(self.ids, doc_id)
            self.X = np.vstack((self.X, np.array([vector])))

    @abstractmethod
    def _sample_document(self, i: int):
        pass

    def update(self):
        # Repeat Gibbs sampling iterations
        for _ in range(self.n_iterations):

            # For each document (in random order)
            for i in random.sample(range(self.N), self.N):
                self._sample_document(i)

            # Keep track of likelihood
            if self.visualizer is not None:
                self.visualizer.add(self.likelihood(), self.N)

        # TODO update cluster components if necessary
