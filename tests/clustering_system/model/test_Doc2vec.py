import os
import tempfile

import numpy as np

from clustering_system.corpus.LineCorpus import LineCorpus
from clustering_system.corpus.LineNewsCorpus import LineNewsCorpus
from clustering_system.model.Doc2vec import Doc2vec


class TestDoc2vec:

    def test_model(self):
        size = 10

        # Current directory
        dir_path = os.path.dirname(os.path.realpath(__file__))

        corpus = LineNewsCorpus(input=os.path.join(dir_path, "..", "data", "genuine"), language="en")

        temp_corpus_file = tempfile.NamedTemporaryFile(delete=False)

        # Serialize pre-processed corpus to temp file
        LineCorpus.serialize(temp_corpus_file, corpus, corpus.dictionary)

        loaded_corpus = LineCorpus(temp_corpus_file.name)
        # assert list(loaded_corpus) == []

        model = Doc2vec(loaded_corpus, size, min_count=1)
        it = iter(loaded_corpus)
        vector = model[next(it)]

        # Remove temp file
        os.remove(temp_corpus_file.name)

        assert isinstance(vector, np.ndarray)
        assert size == len(vector)
