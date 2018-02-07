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
    def add_document(self, vector: np.ndarray):
        """
        Add document represented by a vector.
        """
        pass

    @abstractmethod
    def remove_document(self, vector: np.ndarray):
        """
        Remove document represented by a vector.
        """
        pass

    @abstractmethod
    def __iter__(self):
        """
        Iterate over clusters.
        """
        pass
