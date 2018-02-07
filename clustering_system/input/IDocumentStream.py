from abc import ABC, abstractmethod
from typing import Callable


class IDocumentStream(ABC):

    def __init__(self):
        self._filters = []

    @abstractmethod
    def __iter__(self):
        """
        Return documents one by one.
        """
        pass

    def _process(self, doc):
        for f in self._filters:
            doc = f(doc)
        return doc

    def add_filter(self, new_filter: Callable):
        if callable(new_filter):
            self._filters.append(new_filter)

