import numpy as np
import scipy.stats
from numpy.testing import assert_almost_equal

from clustering_system.evaluator.UnsupervisedEvaluation import UnsupervisedEvaluation


class TestUnsupervisedEvaluation:

    def test_evaluation(self):
        clusters = np.array([1, 2, 3, 1])

        aic = 1.1
        bic = 2.2
        likelihood = 3.3
        K = 3
        entropy = scipy.stats.entropy([2/4, 1/4, 1/4])

        evaluation = UnsupervisedEvaluation(clusters, aic, bic, likelihood)

        assert aic == evaluation.aic
        assert bic == evaluation.bic
        assert likelihood == evaluation.likelihood
        assert K == evaluation.K
        assert_almost_equal(evaluation.cluster_entropy, entropy)
        assert len(evaluation.get_attribute_names()) == 5  # Assert the number of attributes
