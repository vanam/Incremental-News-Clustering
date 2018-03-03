import random

import numpy as np

from clustering.infinite_gmm_gibbs_ddcrp import logistic_decay
from clustering_system.clustering.gmm.GaussianMixtureABC import NormalInverseWishartPrior
from clustering_system.clustering.igmm.DdCrpClustering import DdCrpClustering


class TestDdCrpClustering:
    documents = [
        [0, 0],
        [-2, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 3],
        [5, 3],
        [6, 3],
        [-1, 1]
    ]

    metadata = [
        ("1", 1, "-"),
        ("2", 1, "-"),
        ("3", 1, "-"),
        ("4", 1, "-"),
        ("5", 1, "-"),
        ("6", 1, "-"),
        ("7", 1, "-"),
        ("8", 1, "-"),
        ("9", 1, "-")
    ]

    @staticmethod
    def _get_clustering(n: int):
        size = 2
        v_0 = size + 1

        prior = NormalInverseWishartPrior(
            np.array([0, 0]),
            0.01,
            v_0 * np.eye(size),
            v_0
        )

        # Decay function
        a = 1  # 1 day

        def f(d: float):
            return logistic_decay(d, a)

        # return DdCrpClustering(size, 0.01, prior, 70, f)
        return DdCrpClustering(size, 0.01, prior, n, f)

    def test_number_of_components(self):
        random.seed(0)

        clustering = self._get_clustering(70)
        clustering.add_documents(self.documents, self.metadata)
        clustering.update()

        unique_cluster_numbers = np.unique(clustering.z)

        assert clustering.K == len(unique_cluster_numbers)

    def test_number_of_components_2(self):
        random.seed(0)

        clustering = self._get_clustering(700)
        # clustering = self._get_clustering(500)
        # clustering = self._get_clustering(410)
        # clustering = self._get_clustering(1500)
        clustering.add_documents(self.documents, self.metadata)
        clustering.update()

        unique_cluster_numbers = np.unique(clustering.z)

        assert clustering.K == len(unique_cluster_numbers)