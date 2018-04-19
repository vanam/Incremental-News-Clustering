import os

from gensim.corpora import MmCorpus

from clustering_system.corpus.BowNewsCorpus import BowNewsCorpus
from clustering_system.corpus.MetaMmCorpusWrapper import MetaMmCorpusWrapper


class TestMetaMmCorpusWrapper:

    def test_load(self):
        # Current directory
        dir_path = os.path.dirname(os.path.realpath(__file__))
        corpus_file = os.path.join(dir_path, "data", "data.mm")

        # corpus = BowNewsCorpus(input=os.path.join(dir_path, "..", "data", "genuine"), language="en", metadata=True)
        #
        # # Serialize pre-processed corpus to temp files
        # MmCorpus.serialize(corpus_file, corpus, metadata=True)

        loaded_corpus = MetaMmCorpusWrapper(corpus_file)

        i = 0
        for _, __ in loaded_corpus:
            i += 1

        # Assert corpus type
        assert isinstance(loaded_corpus.corpus, MmCorpus)

        # Assert number of times it went through the loop
        assert i == 2
