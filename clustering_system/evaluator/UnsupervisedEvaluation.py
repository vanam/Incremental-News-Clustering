import numpy as np

from clustering_system.evaluator.measures import entropy


class UnsupervisedEvaluation:
    """A class containing unsupervised clustering metrics."""

    def __init__(self, clusters: np.ndarray, aic: float, bic: float, likelihood: float):
        """
        :param clusters: The cluster assignments
        :param aic: The Akaike information criterion
        :param bic: The Bayesian information criterion
        :param likelihood: The log likelihood
        """
        self.aic = aic
        self.bic = bic
        self.likelihood = likelihood
        self.K = len(np.unique(clusters))
        self.cluster_entropy = entropy(clusters)

    @staticmethod
    def get_attribute_names():
        """
        Return class attribute names.

        :return: A list of tuples (attribute name, attribute)
        """
        return [
            ('AIC', 'aic'),
            ('BIC', 'bic'),
            ('likelihood', 'likelihood'),
            ('number of clusters', 'K'),
            ('entropy (clusters)', 'cluster_entropy')
        ]

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        string = 'UnsupervisedEvaluation {\n'
        string += "  AIC                = %f,\n" % self.aic
        string += "  BIC                = %f,\n" % self.bic
        string += "  likelihood         = %f,\n" % self.likelihood
        string += "  number of clusters = %d,\n" % self.K
        string += "  entropy (clusters) = %f \n" % self.cluster_entropy
        string += '}'
        return string
