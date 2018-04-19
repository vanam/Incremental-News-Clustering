from clustering_system.corpus.NewsCorpusABC import NewsCorpusABC


class LineNewsCorpus(NewsCorpusABC):
    """Line-of-Words corpus class for news articles"""

    def _encode(self, text):
        """Encode text tokens as list of token ids preserving order of words."""
        return self.dictionary.doc2idx(text)

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, metadata=False):
        # Intentionally not implemented since it was not needed
        raise NotImplementedError
