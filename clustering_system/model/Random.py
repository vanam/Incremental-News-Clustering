import numpy as np

from gensim import utils
from clustering_system.model.ModelABC import ModelABC


class Random(ModelABC):
    """Represent news articles as random vectors."""

    def update(self, documents):
        """
        Update model using documents.
        Does not support updating.

        :param documents: The new documents used for update
        """
        # There is nothing to update.
        pass

    def save(self, filename: str):
        """
        Save model to a file.
        Does not support saving.

        :param filename: A model file name
        """
        # There is nothing to save
        pass

    def _get_vector_representation(self, items):
        raise NotImplementedError("This method is not implemented intentionally.")

    def __getitem__(self, items):
        """
        Return random vector(s).
        :param items:
        :return:
        """
        is_corpus, items = utils.is_corpus(items)

        if not is_corpus:
            return np.random.random(self.size)
        else:
            return list(map(lambda v: np.random.random(self.size), items))
