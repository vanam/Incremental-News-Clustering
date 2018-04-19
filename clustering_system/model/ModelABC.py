from abc import ABC, abstractmethod

from gensim import utils
from gensim.matutils import sparse2full


class ModelABC(ABC):
    """Abstract model for vector representation for news articles"""

    def __init__(self, size):
        """
        :param size: The length of feature vector
        """
        self.size = size

    @abstractmethod
    def update(self, documents):
        """
        Update model using documents.

        :param documents: The new documents used for update
        """
        pass

    @abstractmethod
    def save(self, filename: str):
        """
        Save model to a file.

        :param filename: A model file name
        """
        pass

    @abstractmethod
    def _get_vector_representation(self, items):
        """
        Represent documents as vectors.

        :param items: A list of documents
        :return: A list of feature vectors.
        """
        pass

    def __getitem__(self, items):
        is_corpus, items = utils.is_corpus(items)

        if not is_corpus:
            v = self._get_vector_representation(items)
            return sparse2full(v, self.size)
        else:
            return list(map(lambda v: sparse2full(v, self.size), self._get_vector_representation(items)))
