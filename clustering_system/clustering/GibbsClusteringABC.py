import random
from abc import abstractmethod

import numpy as np

from clustering_system.clustering.ClusteringABC import ClusteringABC

from clustering_system.clustering.gmm.GaussianMixtureABC import PriorABC
from clustering_system.visualization.LikelihoodVisualizer import LikelihoodVisualizer


class GibbsClusteringABC(ClusteringABC):

    def __init__(self, D: int, alpha: float, prior: PriorABC, n_iterations: int, K_max: int = None, visualizer: LikelihoodVisualizer = None):
        super().__init__(D)
        self.K_max = K_max
        self.alpha = alpha
        self.prior = prior
        self.n_iterations = n_iterations
        self.visualizer = visualizer

        # Each document has its id, feature vector, table assignment
        self.ids = []
        self.X = np.empty((0, D), float)
        self.z = []  # -1 - unassigned, <0, K) assigned

    def add_documents(self, vectors: np.ndarray, metadata: np.ndarray):
        for md, vector in zip(metadata, vectors):
            # Add document at the end of arrays
            self.ids.append(md[0])
            self.X = np.vstack((self.X, np.array([vector])))
            self.z = -1  # customer is unassigned to a table

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
                self.visualizer.add(self.likelihood, self.N)

        # TODO update cluster components if necessary
