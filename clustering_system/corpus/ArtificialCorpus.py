import numpy as np
from gensim.interfaces import CorpusABC


class ArtificialCorpus(CorpusABC):

    def __init__(self, input=None, metadata=False):
        self.input = input
        self.metadata = metadata
        self.data = np.load(input)
        self.length = None

    def __iter__(self):
        data = self.getstream()
        for doc, metadata in data:
            if self.metadata:
                yield doc, metadata
            else:
                yield doc

    def __len__(self):
        if self.length is None:
            # cache the corpus length
            self.length = sum(1 for _ in self.getstream())
        return self.length

    def getstream(self):
        i = 0
        for c, d in enumerate(self.data):
            for point in d:
                yield point, (i, 1, "This artificial document represents class %d" % c)  # (docId, timestamp, title)
                i += 1
