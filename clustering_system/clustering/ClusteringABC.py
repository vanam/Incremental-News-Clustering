from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class CovarianceType(Enum):
    full = 0       # each component has its own general covariance matrix
    tied = 1       # all components share the same general covariance matrix
    diag = 2       # each component has its own diagonal covariance matrix
    spherical = 3  # each component has its own single variance


class ClusteringABC(ABC):

    def __init__(self, D: int):
        self.D = D  # Length of a feature vector

        self.N = 0  # Number of documents
        self.K = 0  # Current number of documents

    @abstractmethod
    def add_documents(self, vectors: np.ndarray, metadata: np.ndarray):
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

    # @abstractmethod
    # def aic(self, X: np.ndarray) -> float:
    #     """
    #     Akaike information criterion for the current model on the input X.
    #
    #     :param X:
    #     :return:
    #     """
    #     pass
    #
    # @abstractmethod
    # def bic(self, X: np.ndarray) -> float:
    #     """
    #     Bayesian information criterion for the current model on the input X.
    #
    #     :param X:
    #     :return:
    #     """
    #     pass

    @property
    @abstractmethod
    def likelihood(self) -> float:
        """
        Calculate average log likelihood of data
        L(theta | x) = f(x | theta)
        """
        pass

    @abstractmethod
    def __iter__(self):
        """
        For each document return (doc_id, cluster_id)
        """
        pass
