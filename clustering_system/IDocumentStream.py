from abc import ABC, abstractmethod


class IDocumentStream(ABC):

    @abstractmethod
    def __iter__(self):
        """
        Return documents one by one.
        """
        pass
