from tempfile import TemporaryFile, mktemp

import numpy as np
import pytest

from clustering_system.corpus.LineCorpus import LineCorpus


class TestLineCorpus:
    CORPUS = [[1, 3, 2], [0, -1]]
    CORPUS_TEXTS = [["hello", "word", "world"], ["!"]]
    CORPUS_WITH_META = [([1, 3, 2], (1, "d1")), ([0, -1], (1, "d1"))]

    dictionary = {
        0: "!",
        1: "hello",
        2: "world",
        3: "word",
    }

    def test_serialize(self):
        file = mktemp()
        print(self.dictionary[1])
        LineCorpus.serialize(file, self.CORPUS, self.dictionary)
        corpus = LineCorpus(file)

        assert np.array_equal(self.CORPUS_TEXTS, list(corpus))

    def test_serialize_with_metadata(self):
        file = mktemp()

        LineCorpus.serialize(file, self.CORPUS_WITH_META, self.dictionary, metadata=True)
        corpus = LineCorpus(file)

        assert np.array_equal(self.CORPUS_TEXTS, list(corpus))
