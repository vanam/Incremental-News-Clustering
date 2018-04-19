import numpy as np

from clustering_system.model.Random import Random


class TestRandom:

    def test_model(self):
        size = 10

        model = Random(size)
        vector = model["anything"]

        assert isinstance(vector, np.ndarray)
        assert size == len(vector)
