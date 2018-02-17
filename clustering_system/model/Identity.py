from clustering_system.model.ModelABC import ModelABC


class Identity(ModelABC):

    def __init__(self):
        super().__init__(-1)

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
        return items
