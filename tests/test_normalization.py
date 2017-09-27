import numpy as np

import clustering.normalization


class TestNormalization:
    data = [[3, 2], [1, 4]]

    def test_mean_normalization(self):
        normalized = clustering.normalization.mean_normalization(self.data)

        assert np.array_equal(normalized, [[1, -1], [-1, 1]])

    def test_max_scaling(self):
        scaled = clustering.normalization.max_scaling(self.data)

        assert np.array_equal(scaled, [[1, 0.5], [1/3, 1]])

    def test_std_scaling(self):
        scaled = clustering.normalization.std_scaling(self.data)

        assert np.array_equal(scaled, self.data)
