from abc import ABC, abstractmethod

from gensim import utils
from gensim.matutils import sparse2full


class ModelABC(ABC):

    def __init__(self, size):
        self.size = size

    @abstractmethod
    def update(self, documents):
        pass

    @abstractmethod
    def save(self, directory):
        pass

    @abstractmethod
    def _get_vector_representation(self, items):
        pass

    def __getitem__(self, items):
        is_corpus, items = utils.is_corpus(items)

        if not is_corpus:
            v = self._get_vector_representation(items)
            return sparse2full(v, self.size)
        else:
            return list(map(lambda v: sparse2full(v, self.size), self._get_vector_representation(items)))
