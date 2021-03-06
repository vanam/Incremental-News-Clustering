import os

import numpy as np

from clustering_system.corpus.BowNewsCorpus import BowNewsCorpus
from clustering_system.model.Lsa import Lsa


class TestLsa:

    def test_model(self):
        size = 10

        # Current directory
        dir_path = os.path.dirname(os.path.realpath(__file__))

        corpus = BowNewsCorpus(input=os.path.join(dir_path, "..", "data", "genuine"), language="en")

        model = Lsa(corpus.dictionary, corpus, size)
        it = iter(corpus)
        vector = model[next(it)]

        assert isinstance(vector, np.ndarray)
        assert size == len(vector)
