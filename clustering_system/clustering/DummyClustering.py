import random

import numpy as np

from clustering_system.clustering.ClusteringABC import ClusteringABC


class DummyClustering(ClusteringABC):
    """Randomly cluster observations."""

    def __init__(self, K: int, D: int, error: float = 0.2):
        """
        Dummy clustering which assigns new document to random cluster
        :param K: The number of clusters
        :param D: The length of a feature vector
        :param error: The number of clusters error
        """
        super().__init__(D)
        e = int(K * error)

        self.K = K + random.randint(-e, e)
        self.ids = np.empty(0, str)
        self.X = np.empty((0, D), float)
        self.z = np.empty(0, int)

    def _number_of_parameters(self) -> int:
        """
        Return the number of parameters.

        :return: Return one
        """
        return 1

    @property
    def parameters(self):
        """
        Return model parameters.

        :return: None
        """
        return None

    @property
    def likelihood(self) -> float:
        """
        :return: Return random likelihood
        """
        return random.random() * 50

    def add_documents(self, vectors: np.ndarray, metadata: np.ndarray):
        """
        Add documents represented by a list of vectors.
        """
        for md, vector in zip(metadata, vectors):
            self._add_document(md[0], vector)
            self.N += 1

    def _add_document(self, doc_id, vector: np.ndarray):
        """
        Add document represented by a vector.
        """
        self.X = np.vstack((self.X, np.array([vector])))
        self.z = np.append(self.z, random.randint(0, self.K - 1))
        self.ids = np.append(self.ids, doc_id)

    def update(self):
        """
        There is nothing to update
        :return:
        """
        pass

    def __iter__(self):
        """
        Iterate over clusters.
        """
        for doc_id, cluster_id in zip(self.ids, self.z):
            yield doc_id, cluster_id

    def __str__(self):
        """
        String representation
        """
        string = 'DummyClustering {\n'
        string += "  number of documents         = %d,\n" % np.size(self.z)
        string += "  number of clusters          = %d,\n" % self.K
        string += "  number of docs in a cluster = %s,\n" % np.bincount(self.z, minlength=3)
        string += '}'

        return string
