import os
from typing import List

import numpy as np
from gensim.corpora import Dictionary

from clustering_system.corpus.LineNewsCorpus import LineNewsCorpus


def idx2text(dictionary: Dictionary, idx: List):
    return [dictionary[i] for i in idx]


class TestLineNewsCorpus:

    def test_create(self):
        # Current directory
        dir_path = os.path.dirname(os.path.realpath(__file__))

        corpus = LineNewsCorpus(input=os.path.join(dir_path, "..", "data", "genuine"), language="en")

        i = 0
        for _ in corpus:
            i += 1

        # Assert number of times it went through the loop
        assert i == 2

    def test_preprocess(self):
        # Current directory
        dir_path = os.path.dirname(os.path.realpath(__file__))

        corpus = LineNewsCorpus(input=os.path.join(dir_path, "..", "data", "genuine"), language="en")
        dictionary = corpus.dictionary

        texts = []
        for _ in corpus:
            texts.append(idx2text(dictionary, _))

        # Assert number of times it went through the loop
        np.testing.assert_array_equal([['human', 'human', 'steal', 'job'], ['human', 'human', 'steal', 'dog', 'cat']], texts)
