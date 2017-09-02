import pytest
import scipy.spatial.distance

import clustering.cluster


class TestCluster:
    data = [[0, 0], [-2, 0], [1, 1], [-1, 1]]

    def test_constructor_exception(self):
        with pytest.raises(ValueError):
            clustering.cluster.Cluster([])

    def test_centroid(self):
        c = clustering.cluster.Cluster(self.data)
        assert c.centroid == [-0.5, 0.5]

    def test_variability(self):
        c = clustering.cluster.Cluster(self.data)
        metric = getattr(scipy.spatial.distance, c.metric)

        centroid = [-0.5, 0.5]
        variability = 0
        for d in self.data:
            variability += (metric(centroid, d)) ** 2

        assert c.variability() == variability

    def test_dissimilarity_exception(self):
        with pytest.raises(TypeError):
            c = clustering.cluster.Cluster(self.data)
            clustering.cluster.dissimilarity([c, 2, 3])

    def test_dissimilarity(self):
        c = clustering.cluster.Cluster(self.data)
        clusters = [c, c, c]

        dissimilarity = 0
        for c in clusters:
            dissimilarity += c.variability()

        assert clustering.cluster.dissimilarity(clusters) == dissimilarity
