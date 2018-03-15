import numpy as np

from clustering_system.evaluator.measures import dissimilarity, entropy


class UnsupervisedEvaluation:

    def __init__(self, clusters: np.ndarray, aic: float, bic: float, likelihood: float):
        self.aic = aic
        self.bic = bic
        self.likelihood = likelihood
        self.K = len(np.unique(clusters))
        self.cluster_entropy = entropy(clusters)

    @staticmethod
    def get_attribute_names():
        return (
            ('AIC', 'aic'),
            ('BIC', 'bic'),
            ('likelihood', 'likelihood'),
            ('number of clusters', 'K'),
            ('entropy (clusters)', 'cluster_entropy')
         )

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
