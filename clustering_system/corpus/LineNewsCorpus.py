from clustering_system.corpus.NewsCorpusABC import NewsCorpusABC


class LineNewsCorpus(NewsCorpusABC):

    def _encode(self, text):
        return self.dictionary.doc2idx(text)
