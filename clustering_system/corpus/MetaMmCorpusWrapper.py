from gensim.corpora import MmCorpus
from gensim.utils import unpickle


class MetaMmCorpusWrapper:

    def __init__(self, filename):
        self.corpus = MmCorpus(filename)
        self.metadata = unpickle(filename + ".metadata.cpickle")

    def __iter__(self):
        for i, doc in enumerate(self.corpus):
            yield doc, self.metadata[i]
