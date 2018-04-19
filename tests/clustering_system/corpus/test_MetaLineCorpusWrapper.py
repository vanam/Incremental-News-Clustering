import os

from clustering_system.corpus.LineCorpus import LineCorpus
from clustering_system.corpus.LineNewsCorpus import LineNewsCorpus
from clustering_system.corpus.MetaLineCorpusWrapper import MetaLineCorpusWrapper


class TestMetaLineCorpusWrapper:

    def test_load(self):
        # Current directory
        dir_path = os.path.dirname(os.path.realpath(__file__))
        corpus_file = os.path.join(dir_path, "data", "data.line")
        # dictionary_file = os.path.join(dir_path, "data", "dictionary.dict")
        #
        # corpus = LineNewsCorpus(input=os.path.join(dir_path, "..", "data", "genuine"), language="en", metadata=True)
        # corpus.dictionary.save(dictionary_file)
        #
        # # Serialize pre-processed corpus to temp files
        # LineCorpus.serialize(corpus_file, corpus, corpus.dictionary, metadata=True)

        loaded_corpus = MetaLineCorpusWrapper(corpus_file)

        i = 0
        for _, __ in loaded_corpus:
            i += 1

        # Assert number of times it went through the loop
        assert i == 2
