from abc import ABC, abstractmethod
from typing import Iterable

from gensim.matutils import sparse2full


class IModel(ABC):

    def __init__(self, size):
        self.size = size

    @abstractmethod
    def update(self, documents):
        pass

    @abstractmethod
    def save(self, directory):
        pass

    @abstractmethod
    def _get_vector_representation(self, items: Iterable) -> Iterable:
        pass

    def __getitem__(self, items: Iterable):
        return list(map(lambda v: sparse2full(v, self.size), self._get_vector_representation(items)))

