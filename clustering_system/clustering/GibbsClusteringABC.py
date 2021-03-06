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
    """Abstract class for clustering using Gibbs sampling"""

    def __init__(self, D: int, alpha: float, prior: PriorABC, n_iterations: int, K_max: int = None,
                 visualizer: LikelihoodVisualizer = None, covariance_type: CovarianceType = CovarianceType.full):
        """
        :param D: The length of a feature vector
        :param alpha: Hyperparameter
        :param prior: Prior
        :param n_iterations: The number of iterations to perform each update
        :param K_max: The maximum number of clusters
        :param visualizer: Likelihood visualizer
        :param covariance_type: Covariance type
        """
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
        """
        Calculate average log likelihood of data
        L(theta | x) = f(x | theta)
        """
        return self.mixture.likelihood

    @property
    def parameters(self) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        :return: Return parameters of a clustering model.
        """
        return self.mixture.parameters

    def _number_of_parameters(self) -> int:
        """
        :return: The number of model parameters.
        """
        return self.mixture.number_of_parameters

    def add_documents(self, vectors: np.ndarray, metadata: np.ndarray):
        """
        Add documents represented by a list of vectors.

        :param vectors: A list of vectors
        :param metadata: A list of metadata
        """
        for md, vector in zip(metadata, vectors):
            doc_id, timestamp, *_ = md

            # Add document at the end of arrays
            self.ids.append(doc_id)
            clusters = range(0, self.K_max) if self.K < self.K_max else np.unique(self.mixture.z)
            self.mixture.new_vector(vector, random.choice(clusters))  # New customer is assigned to random table
            self.N += 1                                               # Increment number of documents (customers)

        self.K = len(np.unique(self.mixture.z))

    def update(self):
        """
        Update clustering after adding/removing documents
        """
        # Repeat Gibbs sampling iterations
        for _ in range(self.n_iterations):

            # For each document (in random order)
            for i in random.sample(range(self.N), self.N):
                self._sample_document(i)

            # Keep track of likelihood
            if self.visualizer is not None:
                self.visualizer.add(self.likelihood, self.N)

    def __iter__(self):
        """
        For each document return (doc_id, cluster_id)
        """
        for doc_id, cluster_id in zip(self.ids, self.mixture.z):
            yield doc_id, cluster_id

    def _get_new_cluster_number(self):
        """
        Get new cluster number

        :return New cluster number
        """
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
            self.mixture.update_z(i, -1)

            if self.mixture.N_k[z] == 0:
                self.reusable_numbers.put_nowait(z)
                self.K -= 1

    def _add_document(self, i: int, z: int):
        """
        Add document to component z[i] = z.

        :param i: document id
        :param z: component id
        """
        self.mixture.update_z(i, z)

        if self.mixture.N_k[z] == 1:
            self.K += 1

    @abstractmethod
    def _sample_document(self, i: int):
        """
        Sample document i

        :param i: document id
        """
        pass
