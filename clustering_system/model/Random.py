import numpy as np

from gensim import utils
from clustering_system.model.ModelABC import ModelABC


class Random(ModelABC):

    def update(self, documents):
        """
        There is nothing to update.
        :param documents:
        """
        pass

    def save(self, directory):
        """
        There is nothing to save
        :param directory:
        :return:
        """
        pass

    def _get_vector_representation(self, items):
        raise NotImplementedError("This method is not implemented intentionally.")

    def __getitem__(self, items):
        """
        Return item without any change.
        :param items:
        :return:
        """
        is_corpus, items = utils.is_corpus(items)

        if not is_corpus:
            return np.random.random(self.size)
        else:
            return list(map(lambda v: np.random.random(self.size), items))
