from abc import ABC, abstractmethod

import numpy as np


class IClustering(ABC):

    def __init__(self):
        pass

    # @abstractmethod
    # def log_likelihood(self) -> float:
    #     """
    #     Calculate log likelihood of data
    #     L(theta | x) = f(x | theta)
    #     """
    #     pass

    @abstractmethod
    def add_documents(self, ids, vectors: np.ndarray):
        """
        Add documents represented by a list of vectors.
        """
        pass

    @abstractmethod
    def update(self):
        """
        Update clustering after adding/removing documents
        """
        pass

    @abstractmethod
    def __iter__(self):
        """
        Iterate over documents to see how they were clustered
        """
        pass
