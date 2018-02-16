from typing import Iterable

from clustering_system.model.IModel import IModel


class Identity(IModel):

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

    def _get_vector_representation(self, items: Iterable) -> Iterable:
        raise NotImplementedError("This method is not implemented intentionally.")

    def __getitem__(self, items):
        """
        Return item without any change.
        :param items:
        :return:
        """
        return items
