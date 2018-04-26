import numpy as np
import scipy.stats
import sklearn.metrics
from numpy.testing import assert_almost_equal, assert_array_almost_equal

import clustering_system.evaluator.measures as cm
from clustering_system.evaluator.SupervisedEvaluation import SupervisedEvaluation


class TestSupervisedEvaluation:

    def test_evaluation(self):
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

        aic = 1.1
        bic = 2.2
        likelihood = 3.3
        K = 3
        C = 3
        N = 17
        cluster_entropy = scipy.stats.entropy([6/17, 6/17, 5/17])
        class_entropy = scipy.stats.entropy([8/17, 5/17, 4/17])

        purity = 12/17
        purity2 = np.array([[5/6, 6], [4/6, 6], [3/5, 5]])
        rand_index = 23/34
        precision = 0.5
        recall = 20/44
        f1_measure = 10/21
        homogeneity = sklearn.metrics.homogeneity_score(classes, clusters)
        completeness = sklearn.metrics.completeness_score(classes, clusters)
        v_measure = sklearn.metrics.v_measure_score(classes, clusters)
        nv_measure = v_measure
        mutual_information = cm.mutual_information(clusters, classes)
        normalized_mutual_information = sklearn.metrics.normalized_mutual_info_score(classes, clusters)
        normalized_mutual_information2 = cm.normalized_mutual_information2(clusters, classes)

        evaluation = SupervisedEvaluation(clusters, classes, aic, bic, likelihood)

        assert aic == evaluation.aic
        assert bic == evaluation.bic
        assert likelihood == evaluation.likelihood
        assert N == evaluation.N
        assert K == evaluation.K
        assert C == evaluation.C
        assert_almost_equal(evaluation.cluster_entropy, cluster_entropy)
        assert_almost_equal(evaluation.class_entropy, class_entropy)
        assert_almost_equal(evaluation.purity, purity)
        assert_array_almost_equal(evaluation.purity2, purity2)
        assert_almost_equal(evaluation.rand_index, rand_index)
        assert_almost_equal(evaluation.precision, precision)
        assert_almost_equal(evaluation.recall, recall)
        assert_almost_equal(evaluation.f1_measure, f1_measure)
        assert_almost_equal(evaluation.homogeneity, homogeneity)
        assert_almost_equal(evaluation.completeness, completeness, )
        assert_almost_equal(evaluation.v_measure, v_measure)
        assert_almost_equal(evaluation.nv_measure, nv_measure)
        assert_almost_equal(evaluation.mutual_information, mutual_information)
        assert_almost_equal(evaluation.normalized_mutual_information, normalized_mutual_information)
        assert_almost_equal(evaluation.normalized_mutual_information2, normalized_mutual_information2)
        assert len(evaluation.get_attribute_names()) == 16 + 5  # Assert the number of attributes
