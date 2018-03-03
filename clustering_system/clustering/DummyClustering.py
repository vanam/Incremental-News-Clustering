import random

import numpy as np

from clustering_system.clustering.ClusteringABC import ClusteringABC


class DummyClustering(ClusteringABC):

    def __init__(self, K: int, D: int):
        """
        Dummy clustering which assigns new document to random cluster
        :param K: number of clusters
        :param D: dimension of a document vector
        """
        super().__init__(D)
        self.K = K
        self.ids = np.empty(0, str)
        self.X = np.empty((0, D), float)
        self.z = np.empty(0, int)

    def _number_of_parameters(self) -> int:
        return 1

    @property
    def parameters(self):
        return None

    @property
    def likelihood(self) -> float:
        return random.random() * 50

    def add_documents(self, vectors: np.ndarray, metadata: np.ndarray):
        for md, vector in zip(metadata, vectors):
            self.add_document(md[0], vector)

    def add_document(self, doc_id, vector: np.ndarray):
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

    # def remove_document(self, vector: np.ndarray):
    #     """
    #     Remove document represented by a vector.
    #
    #     Note: All occurences of the document are deleted
    #     """
    #     vector_position = np.all(self.X != np.array(vector), axis=1)
    #
    #     self.X = self.X[vector_position]
    #     self.z = self.z[vector_position]

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
