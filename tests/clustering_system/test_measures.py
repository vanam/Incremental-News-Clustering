import numpy as np
import scipy.spatial.distance
import scipy.stats
import sklearn.metrics
from numpy.testing import assert_almost_equal

import clustering_system.evaluator.measures as cm


class TestEntropy:

    def test_entropy_empty_set(self):
        labels = []
        assert cm.entropy(labels) == scipy.stats.entropy([])

    def test_entropy_one_item(self):
        labels = [50]
        assert cm.entropy(labels) == scipy.stats.entropy([1])

    def test_entropy_two_item(self):
        labels = [1, 2]
        assert cm.entropy(labels) == scipy.stats.entropy([0.5, 0.5])

    def test_entropy_multiple_items(self):
        labels = [1, 2, 3, 1]
        assert cm.entropy(labels) == scipy.stats.entropy([0.5, 0.25, 0.25])


class TestMutualInformation:

    def test_normalized_mutual_information(self):
        classes = np.array([
            11, 11, 11,
            11, 11, 22,

            11, 22, 22,
            22, 22, 33,

            11, 11,
            33, 33, 33,
        ])

        clusters = np.array([
            1, 1, 1,
            1, 1, 1,

            2, 2, 2,
            2, 2, 2,

            3, 3,
            3, 3, 3,
        ])

        assert_almost_equal(cm.normalized_mutual_information(clusters, classes), sklearn.metrics.normalized_mutual_info_score(classes, clusters), decimal=10)


class TestFMeasure:

    def test_a(self):
        classes = np.array([
            11, 11, 11,
            11, 11, 22,

            11, 22, 22,
            22, 22, 33,

            11, 11,
            33, 33, 33,
        ])

        clusters = np.array([
            1, 1, 1,
            1, 1, 1,

            2, 2, 2,
            2, 2, 2,

            3, 3,
            3, 3, 3,
        ])

        assert_almost_equal(cm.precision(clusters, classes), 0.5, decimal=10)
        assert_almost_equal(cm.recall(clusters, classes), 0.454545, decimal=6)
        assert_almost_equal(cm.f1_measure(clusters, classes), 0.476, decimal=3)


class TestVMeasure:
    """
    Based on examples found in https://www.researchgate.net/publication/221012656_V-Measure_A_Conditional_Entropy-Based_External_Cluster_Evaluation_Measure
    """

    def test_a(self):
        clusters = np.array([
            1, 1, 1, 1, 1,
            2, 2, 2, 2, 2,
            3, 3, 3, 3, 3
        ])

        classes = np.array([
            1, 1, 1, 2, 3,
            3, 3, 3, 1, 2,
            2, 2, 2, 1, 3
        ])

        assert round(cm.v_measure(clusters, classes), 2) == 0.14
        assert_almost_equal(cm.v_measure(clusters, classes), sklearn.metrics.v_measure_score(classes, clusters), decimal=10)

    def test_b(self):
        clusters = np.array([
            1, 1, 1, 1, 1,
            2, 2, 2, 2, 2,
            3, 3, 3, 3, 3
        ])

        classes = np.array([
            1, 1, 1, 2, 2,
            3, 3, 3, 1, 1,
            2, 2, 2, 3, 3
        ])

        assert round(cm.v_measure(clusters, classes), 2) == 0.39
        assert_almost_equal(cm.v_measure(clusters, classes), sklearn.metrics.v_measure_score(classes, clusters), decimal=10)

    def test_c(self):
        clusters = np.array([
            1, 1, 1, 1, 1,
            2, 2, 2, 2, 2,
            3, 3, 3, 3, 3,
            4, 4,
            5, 5,
            6, 6
        ])

        classes = np.array([
            1, 1, 1, 2, 2,
            3, 3, 3, 1, 1,
            2, 2, 2, 3, 3,
            1, 2,
            3, 1,
            2, 3
        ])

        assert round(cm.v_measure(clusters, classes), 2) == 0.30
        assert_almost_equal(cm.v_measure(clusters, classes), sklearn.metrics.v_measure_score(classes, clusters), decimal=10)

    def test_d(self):
        clusters = np.array([
            1, 1, 1, 1, 1,
            2, 2, 2, 2, 2,
            3, 3, 3, 3, 3,
            4, 5,
            6, 7,
            8, 9
        ])

        classes = np.array([
            1, 1, 1, 2, 2,
            3, 3, 3, 1, 1,
            2, 2, 2, 3, 3,
            1, 2,
            3, 1,
            2, 3
        ])

        assert round(cm.v_measure(clusters, classes), 2) == 0.41
        assert_almost_equal(cm.v_measure(clusters, classes), sklearn.metrics.v_measure_score(classes, clusters), decimal=10)
