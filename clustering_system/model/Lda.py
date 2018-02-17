import os

from gensim.corpora import Dictionary
from gensim.models import LdaModel

from clustering_system.model.ModelABC import ModelABC


class Lda(ModelABC):

    def __init__(self, corpus, dictionary: Dictionary, temp_directory, size: int = 100, decay=0.5):
        super().__init__(size)

        # Check if we have already trained the Lda model
        lda_filename = self._get_lda_filename(temp_directory)

        if os.path.exists(lda_filename):
            self.lda = LdaModel.load(lda_filename)
        else:
            self.lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=size, passes=1, decay=decay, minimum_probability=0.0)

    @staticmethod
    def _get_lda_filename(directory):
        return os.path.join(directory, 'model.lda')

    def update(self, documents):
        self.lda.update(documents)

    def save(self, directory):
        self.lda.save(self._get_lda_filename(directory))

    def _get_vector_representation(self, items):
        return self.lda[items]
