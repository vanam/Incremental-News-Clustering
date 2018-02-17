from clustering_system.corpus.NewsCorpusABC import NewsCorpusABC


class BowNewsCorpus(NewsCorpusABC):

    def _encode(self, text):
        return self.dictionary.doc2bow(text)


