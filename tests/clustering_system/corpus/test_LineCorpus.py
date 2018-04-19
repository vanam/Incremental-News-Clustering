import os
import tempfile

import numpy as np

from clustering_system.corpus.LineCorpus import LineCorpus
from clustering_system.corpus.LineNewsCorpus import LineNewsCorpus


class TestLineCorpus:

    @staticmethod
    def _get_corpus():
        # Current directory
        dir_path = os.path.dirname(os.path.realpath(__file__))

        corpus = LineNewsCorpus(input=os.path.join(dir_path, "data"), language="en")

        return corpus

    def test_serialize_load(self):
        corpus = self._get_corpus()

        temp_corpus_file = tempfile.NamedTemporaryFile(delete=False)

        # Serialize pre-processed corpus to temp files
        LineCorpus.serialize(temp_corpus_file, corpus, corpus.dictionary)

        loaded_corpus = LineCorpus(temp_corpus_file.name)
        docs = []
        for d in loaded_corpus:
            docs.append(d)

        np.testing.assert_array_equal([['human', 'human', 'steal', 'job'], ['human', 'human', 'steal', 'dog', 'cat']], docs)

        # Remove temp file
        os.remove(temp_corpus_file.name)
