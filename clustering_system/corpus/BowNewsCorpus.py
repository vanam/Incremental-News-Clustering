from clustering_system.corpus.NewsCorpusABC import NewsCorpusABC


class BowNewsCorpus(NewsCorpusABC):
    """Bag-of-Words corpus class for news articles"""

    def _encode(self, text):
        """Encode text tokens as tuples (id, frequency) in arbitrary order."""
        return self.dictionary.doc2bow(text)

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, metadata=False):
        # Intentionally not implemented since it was not needed
        raise NotImplementedError
