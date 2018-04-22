import logging
import os

from gensim.corpora import Dictionary
from gensim.models import LdaModel

from clustering_system.model.ModelABC import ModelABC


class Lda(ModelABC):
    """Represent news articles as vectors using Latent Dirichlet Allocation."""

    def __init__(self, dictionary: Dictionary, corpus=None, size: int = 100, decay=0.5, lda_filename: str = None):
        """
        :param dictionary: A dictionary
        :param corpus: A corpus for training
        :param size: The length of feature vector
        :param decay: The decay parameter
        :param lda_filename: File name of a previously trained model
        """
        super().__init__(size)

        # Check if we have already trained the Lda model
        if lda_filename is not None and os.path.exists(lda_filename):
            self.lda = LdaModel.load(lda_filename)
            logging.info("LDA model loaded")
        else:
            if corpus is None:
                raise ValueError("Corpus must be provided to train LDA")

            self.lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=size, passes=1, decay=decay, minimum_probability=0.0)

    def update(self, documents):
        """
        Update model using documents.

        :param documents: The new documents used for update
        """
        self.lda.update(documents)

    def save(self, filename: str):
        """
        Save model to a file.

        :param filename: A model file name
        """
        self.lda.save(filename)

    def _get_vector_representation(self, items):
        """
        Represent documents as vectors.

        :param items: A list of documents
        :return: A list of feature vectors.
        """
        return self.lda[items]
