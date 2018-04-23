import numpy as np

from clustering_system.evaluator.UnsupervisedEvaluation import UnsupervisedEvaluation
from clustering_system.evaluator.measures import purity, purity2, rand_index, entropy, homogeneity, completeness, \
    v_measure, mutual_information, normalized_mutual_information, normalized_mutual_information2, f1_measure, recall, \
    precision


class SupervisedEvaluation(UnsupervisedEvaluation):
    """A class containing supervised clustering metrics."""

    def __init__(self, clusters: np.ndarray, classes: np.ndarray, aic: float, bic: float, likelihood: float):
        """
        :param clusters: The cluster assignments
        :param classes: The class assignments
        :param aic: The Akaike information criterion
        :param bic: The Bayesian information criterion
        :param likelihood: The log likelihood
        """
        super().__init__(clusters, aic, bic, likelihood)

        self.N = len(classes)
        self.C = len(np.unique(classes))
        self.purity = purity(clusters, classes)
        self.purity2 = purity2(clusters, classes)
        self.rand_index = rand_index(clusters, classes)
        self.class_entropy = entropy(classes)
        self.precision = precision(clusters, classes)
        self.recall = recall(clusters, classes)
        self.f1_measure = f1_measure(clusters, classes)
        self.homogeneity = homogeneity(clusters, classes)
        self.completeness = completeness(clusters, classes)
        self.v_measure = v_measure(clusters, classes)
        self.mutual_information = mutual_information(clusters, classes)
        self.normalized_mutual_information = normalized_mutual_information(clusters, classes)
        self.normalized_mutual_information2 = normalized_mutual_information2(clusters, classes)

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
            ('number of observations', 'N'),
            ('number of classes', 'C'),
            ('number of clusters', 'K'),
            ('purity', 'purity'),
            ('purity 2', 'purity2'),
            ('rand index', 'rand_index'),
            ('entropy (clusters)', 'cluster_entropy'),
            ('entropy (classes)', 'class_entropy'),
            ('precision', 'precision'),
            ('recall', 'recall'),
            ('F-measure', 'f1_measure'),
            ('homogeneity', 'homogeneity'),
            ('completeness', 'completeness'),
            ('V-Measure', 'v_measure'),
            ('mutual information', 'mutual_information'),
            ('normalized mutual information', 'normalized_mutual_information'),
            ('normalized mutual information 2', 'normalized_mutual_information2')
        ]

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        string = 'SupervisedEvaluation {\n'
        string += "  AIC                             = %f,\n" % self.aic
        string += "  BIC                             = %f,\n" % self.bic
        string += "  likelihood                      = %f,\n" % self.likelihood
        string += "  number of observations          = %d,\n" % self.N
        string += "  number of classes               = %d,\n" % self.C
        string += "  number of clusters              = %d,\n" % self.K
        string += "  purity                          = %f,\n" % self.purity
        string += "  purity 2                        = %s,\n" % self.purity2
        string += "  rand index                      = %f,\n" % self.rand_index
        string += "  entropy (clusters)              = %f,\n" % self.cluster_entropy
        string += "  entropy (classes)               = %f,\n" % self.class_entropy
        string += "  homogeneity                     = %f,\n" % self.homogeneity
        string += "  completeness                    = %f,\n" % self.completeness
        string += "  V-Measure                       = %f,\n" % self.v_measure
        string += "  mutual information              = %f,\n" % self.mutual_information
        string += "  normalized mutual information   = %f,\n" % self.normalized_mutual_information
        string += "  normalized mutual information 2 = %f \n" % self.normalized_mutual_information2
        string += '}'
        return string
