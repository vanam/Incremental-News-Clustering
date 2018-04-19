import os
import tempfile

import numpy as np

from clustering_system.corpus.LineCorpus import LineCorpus
from clustering_system.corpus.LineNewsCorpus import LineNewsCorpus


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
        file = tempfile.mktemp()
        LineCorpus.serialize(file, self.CORPUS, self.dictionary)
        corpus = LineCorpus(file)

        assert np.array_equal(self.CORPUS_TEXTS, list(corpus))

    def test_serialize_with_metadata(self):
        file = tempfile.mktemp()

        LineCorpus.serialize(file, self.CORPUS_WITH_META, self.dictionary, metadata=True)
        corpus = LineCorpus(file)

        assert np.array_equal(self.CORPUS_TEXTS, list(corpus))

    def test_serialize_load(self):
        # Current directory
        dir_path = os.path.dirname(os.path.realpath(__file__))

        corpus = LineNewsCorpus(input=os.path.join(dir_path, "..", "data", "genuine"), language="en")

        temp_corpus_file = tempfile.NamedTemporaryFile(delete=False)

        # Serialize pre-processed corpus to temp file
        LineCorpus.serialize(temp_corpus_file, corpus, corpus.dictionary)

        loaded_corpus = LineCorpus(temp_corpus_file.name)
        docs = []
        for d in loaded_corpus:
            docs.append(d)

        # Remove temp file
        os.remove(temp_corpus_file.name)

        np.testing.assert_array_equal([['human', 'human', 'steal', 'job'], ['human', 'human', 'steal', 'dog', 'cat']], docs)
