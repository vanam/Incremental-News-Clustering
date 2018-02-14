from abc import ABC, abstractmethod


class IModel(ABC):

    # @abstractmethod
    # def get_vector(self):
    #     pass

    @abstractmethod
    def update(self, documents):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

