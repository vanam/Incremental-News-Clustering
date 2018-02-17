from gensim.utils import unpickle

from clustering_system.corpus.LineCorpus import LineCorpus


class MetaLineCorpusWrapper:

    def __init__(self, filename):
        self.corpus = LineCorpus(filename)
        self.metadata = unpickle(filename + ".metadata.cpickle")

    def __iter__(self):
        for i, doc in enumerate(self.corpus):
            yield doc, self.metadata[i]
