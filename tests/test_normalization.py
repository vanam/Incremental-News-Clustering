import numpy as np

import clustering.normalization


class TestNormalization:
    data = [[3, 2], [1, 4]]

    def test_mean_normalization(self):
        normalized, mean = clustering.normalization.mean_normalization(self.data)

        assert np.array_equal(normalized, [[1, -1], [-1, 1]])
        assert np.array_equal(mean, [2, 3])

    def test_mean_denormalization(self):
        data = clustering.normalization.mean_denormalization([[1, -1], [-1, 1]], [2, 3])

        assert np.array_equal(data, self.data)

    def test_max_scaling(self):
        scaled, mx = clustering.normalization.max_scaling(self.data)

        assert np.array_equal(scaled, [[1, 0.5], [1/3, 1]])
        assert np.array_equal(mx, [3, 4])

    def test_max_descaling(self):
        data = clustering.normalization.max_descaling([[1, 0.5], [1/3, 1]], [3, 4])

        assert np.array_equal(data, self.data)

    def test_std_scaling(self):
        scaled, std = clustering.normalization.std_scaling(self.data)

        assert np.array_equal(scaled, self.data)
        assert np.array_equal(std, [1, 1])

    def test_std_descaling(self):
        data = clustering.normalization.std_descaling(self.data, [1, 1])

        assert np.array_equal(data, self.data)
