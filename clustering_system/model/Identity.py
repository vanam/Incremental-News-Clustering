from clustering_system.model.IModel import IModel


class Identity(IModel):

    def update(self, documents):
        """
        There is nothing to update.
        :param documents:
        """
        pass

    def __getitem__(self, item):
        """
        Return item without any change.
        :param item:
        :return:
        """
        return item