import os

from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LsiModel

from clustering_system.model.ModelABC import ModelABC


class Lsi(ModelABC):

    def __init__(self, corpus, dictionary: Dictionary, temp_directory, size: int = 200, decay: float = 1.0):
        super().__init__(size)

        # Check if we have already trained the Tfidf model
        tfidf_filename = self._get_tfidf_filename(temp_directory)

        if os.path.exists(tfidf_filename):
            self.tfidf = TfidfModel.load(tfidf_filename)
        else:
            self.tfidf = TfidfModel(dictionary=dictionary)

        # Process the corpus
        corpus_tfidf = self.tfidf[corpus]

        # Check if we have already trained the Lsi model
        lsi_filename = self._get_lsi_filename(temp_directory)

        if os.path.exists(lsi_filename):
            self.lsi = LsiModel.load(lsi_filename)
        else:
            self.lsi = LsiModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=size, onepass=True, decay=decay)

    @staticmethod
    def _get_tfidf_filename(directory):
        return os.path.join(directory, 'model.tfidf')

    @staticmethod
    def _get_lsi_filename(directory):
        return os.path.join(directory, 'model.lsi')

    def update(self, documents):
        """
        Update LSI model
        :param documents:
        """
        self.lsi.add_documents(documents)

    def save(self, directory):
        self.tfidf.save(self._get_tfidf_filename(directory))
        self.lsi.save(self._get_lsi_filename(directory))

    def _get_vector_representation(self, items):
        return self.lsi[self.tfidf[items]]
