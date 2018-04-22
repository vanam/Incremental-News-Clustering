import logging
import os

from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LsiModel

from clustering_system.model.ModelABC import ModelABC


class Lsa(ModelABC):
    """Represent news articles as vectors using Latent Semantic Indexing."""

    def __init__(self, dictionary: Dictionary, corpus=None, size: int = 200, decay: float = 1.0,
                 lsa_filename: str = None, tfidf_filename: str = None):
        """
        :param dictionary: A dictionary
        :param corpus: A corpus for training
        :param size: The length of feature vector
        :param decay: The decay parameter
        :param lsa_filename: File name of a previously trained model
        :param tfidf_filename: File name of a previously trained TF-IDF model
        """
        super().__init__(size)

        # Check if we have already trained the Tfidf model
        if tfidf_filename is not None and os.path.exists(tfidf_filename):
            self.tfidf = TfidfModel.load(tfidf_filename)
        else:
            self.tfidf = TfidfModel(dictionary=dictionary)

        # Check if we have already trained the Lsa model
        if lsa_filename is not None and os.path.exists(lsa_filename):
            self.lsa = LsiModel.load(lsa_filename)
            logging.info("LSA model loaded")
        else:
            if corpus is None:
                raise ValueError("Corpus must be provided to train LSI")

            # Process the corpus
            corpus_tfidf = self.tfidf[corpus]

            self.lsa = LsiModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=size, onepass=True, decay=decay)

    def update(self, documents):
        """
        Update model using documents.

        :param documents: The new documents used for update
        """
        self.lsa.add_documents(documents)

    def save(self, filename: str):
        """
        Save model to a file.

        :param filename: A model file name
        """
        self.lsa.save(filename)
        self.tfidf.save(filename + '.tfidf')

    def _get_vector_representation(self, items):
        """
        Represent documents as vectors.

        :param items: A list of documents
        :return: A list of feature vectors.
        """
        return self.lsa[self.tfidf[items]]
