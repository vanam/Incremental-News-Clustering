import os

from gensim.corpora import Dictionary
from gensim.models import LdaModel

from clustering_system.model.ModelABC import ModelABC


class Lda(ModelABC):

    def __init__(self, dictionary: Dictionary, corpus=None, size: int = 100, decay=0.5, lda_filename: str = None):
        super().__init__(size)

        # Check if we have already trained the Lda model
        if lda_filename is not None and os.path.exists(lda_filename):
            self.lda = LdaModel.load(lda_filename)
        else:
            if corpus is None:
                raise ValueError("Corpus must be provided to train LDA")

            self.lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=size, passes=1, decay=decay, minimum_probability=0.0)

    def update(self, documents):
        self.lda.update(documents)

    def save(self, filename: str):
        self.lda.save(filename)

    def _get_vector_representation(self, items):
        return self.lda[items]
