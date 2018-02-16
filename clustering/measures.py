from collections import Counter

import numpy as np
# import pandas as pd
from pandas import crosstab
from scipy.spatial.distance import cdist
from scipy.special import comb
from scipy.stats import multivariate_normal
from scipy.stats._multivariate import multivariate_normal_gen


class UnsupervisedEvaluation:

    def __init__(self, X: np.ndarray, clusters: np.ndarray, likelihood: float):
        self.likelihood = likelihood
        self.K = len(np.unique(clusters))
        self.dissimilarity = dissimilarity(X, clusters)
        self.cluster_entropy = entropy(clusters)

    @staticmethod
    def get_attribute_names():
        return (
            ('likelihood', 'likelihood'),
            ('dissimilarity', 'dissimilarity'),
            ('number of clusters', 'K'),
            ('entropy (clusters)', 'cluster_entropy')
         )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string = 'Evaluation {\n'
        string += "  likelihood         = %f,\n" % self.likelihood
        string += "  dissimilarity      = %f,\n" % self.dissimilarity
        string += "  number of clusters = %d,\n" % self.K
        string += "  entropy (clusters) = %f \n" % self.cluster_entropy
        string += '}'
        return string


class SupervisedEvaluation(UnsupervisedEvaluation):

    def __init__(self, X: np.ndarray, clusters: np.ndarray, classes: np.ndarray, likelihood: float):
        super().__init__(X, clusters, likelihood)

        self.N = len(classes)
        self.C = len(np.unique(classes))
        self.purity = purity(clusters, classes)
        self.purity2 = purity2(clusters, classes)
        self.rand_index = rand_index(clusters, classes)
        self.class_entropy = entropy(classes)
        self.homogeneity = homogeneity(clusters, classes)
        self.completeness = completeness(clusters, classes)
        self.v_measure = v_measure(clusters, classes)
        self.mutual_information = mutual_information(clusters, classes)
        self.normalized_mutual_information = normalized_mutual_information(clusters, classes)
        self.normalized_mutual_information2 = normalized_mutual_information2(clusters, classes)

    @staticmethod
    def get_attribute_names():
        return [
            ('likelihood', 'likelihood'),
            ('dissimilarity', 'dissimilarity'),
            ('number of observations', 'N'),
            ('number of classes', 'C'),
            ('number of clusters', 'K'),
            ('purity', 'purity'),
            ('purity 2', 'purity2'),
            ('rand index', 'rand_index'),
            ('entropy (clusters)', 'cluster_entropy'),
            ('entropy (classes)', 'class_entropy'),
            ('homogeneity', 'homogeneity'),
            ('completeness', 'completeness'),
            ('V-Measure', 'v_measure'),
            ('mutual information', 'mutual_information'),
            ('normalized mutual information', 'normalized_mutual_information'),
            ('normalized mutual information 2', 'normalized_mutual_information2')
        ]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string = 'TrainingEvaluation {\n'
        string += "  likelihood                      = %f,\n" % self.likelihood
        string += "  dissimilarity                   = %f,\n" % self.dissimilarity
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


def variability(X: np.ndarray):
    centroid = np.mean(X, axis=0)

    return sum(np.square(cdist(np.array([centroid]), X, metric='euclidean'))[0])


def dissimilarity(X: np.ndarray, clusters: np.ndarray):
    return sum(map(lambda i: variability(X[clusters == i]), np.unique(clusters)))


def purity(clusters, classes):
    """
    The purity ranges between 0 (bad) and 1 (good). However, we can trivially achieve a purity of 1
    by putting each object into its own cluster,
    so this measure does not penalize for the number of clusters.
    """
    # Total number of items
    N = len(clusters)

    # Accumulator
    acc = 0

    # # For each cluster
    # for c in [[classes[i] for i in range(N) if clusters[i] == cluster_label] for cluster_label in np.unique(clusters)]:
    #     # Count class labels
    #     counter = Counter(c)

    # For each cluster
    for cluster_label in np.unique(clusters):
        # Count class labels
        counter = Counter(classes[clusters == cluster_label])

        # Accumulate the most frequent label count
        acc += counter.most_common(1)[0][1]

    return acc / N


def purity2(clusters, classes):
    """
    For each cluster return its purity and size
    """
    result = []

    # For each cluster
    for cluster_label in np.unique(clusters):
        # Cluster items
        c = classes[clusters == cluster_label]

        # Count class labels
        counter = Counter(c)

        # Size of the cluster
        N_i = len(c)

        # the most frequent label count / size of the cluster
        result.append((counter.most_common(1)[0][1] / N_i, N_i))

    return result


def rand_index(clusters, classes):
    """
                            same cluster     |  different clusters
    same class        |  true positives (TP) | false negatives (FN)
    different classes | false positives (FP) |  true negatives (TN)
    """
    # Total number of objects
    N = len(clusters)

    # The number of pairs of objects put in the same cluster, regardless of label
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()

    # The number of pairs of objects put in the same class, regardless of cluster label
    tp_plus_fn = comb(np.bincount(classes), 2).sum()

    # The number of pairs
    c2N = comb(N, 2)

    # The number of pairs of objects put in the same class with the same cluster label
    tp = sum(comb(np.bincount(classes[clusters == cluster_label]), 2).sum() for cluster_label in np.unique(clusters))
    # fp = tp_plus_fp - tp
    # fn = tp_plus_fn - tp
    # tn = c2N - tp - fp - fn
    tn = c2N - tp_plus_fp - tp_plus_fn + tp

    # return (tp + tn) / (tp + fp + fn + tn) = (tp + tn) / nC2
    return (tp + tn) / c2N


def mutual_information(clusters, classes, contingency=None):
    if contingency is None:
        contingency = crosstab(clusters, classes, rownames=['clusters'], colnames=['classes'], margins=True)

    # Number of data points
    N = len(clusters)

    if N == 0:
        return 0.0

    # Find cluster labels
    cluster_labels = np.unique(clusters)

    # Find class labels
    class_labels = np.unique(classes)

    mutual_information = 0.0

    for kl in cluster_labels:
        for cl in class_labels:
            a_ij = contingency[cl][kl]

            if a_ij == 0.0:
                continue

            mutual_information += a_ij / N * (np.log(a_ij * N) - np.log(contingency['All'][kl] * contingency[cl]['All']))

    return mutual_information


def normalized_mutual_information(clusters, classes):
    I = mutual_information(clusters, classes)
    entropy_K, entropy_C = entropy(clusters), entropy(classes)

    nmi = I / np.sqrt(entropy_K * entropy_C)

    return nmi


def normalized_mutual_information2(clusters, classes):
    """
    See (25.12) in Murphy, p. 134.
    """
    I = mutual_information(clusters, classes)
    entropy_K, entropy_C = entropy(clusters), entropy(classes)

    nmi = I / ((entropy_K + entropy_C) / 2)

    return nmi


def homogeneity(clusters, classes):
    return _homogeneity_completeness_v_measure(clusters, classes)[0]


def completeness(clusters, classes):
    return _homogeneity_completeness_v_measure(clusters, classes)[1]


def v_measure(clusters, classes):
    return _homogeneity_completeness_v_measure(clusters, classes)[2]


def _homogeneity_completeness_v_measure(clusters, classes):
    """
    https://www.researchgate.net/publication/221012656_V-Measure_A_Conditional_Entropy-Based_External_Cluster_Evaluation_Measure
    https://github.com/scikit-learn/scikit-learn/blob/f3320a6f/sklearn/metrics/cluster/supervised.py#L217
    """
    # Calculate contingency table
    A = crosstab(clusters, classes, rownames=['clusters'], colnames=['classes'], margins=True)

    # Calculate entropy - H(C) - of class labeling
    # entropy_C = -np.sum([A[l]['All'] / N * np.log(A[l]['All'] / N) for l in class_labels])
    entropy_C = entropy(classes)

    # Calculate entropy - H(K) - of cluster labeling
    # entropy_K = -np.sum([A['All'][l] / N * np.log(A['All'][l] / N) for l in cluster_labels])
    entropy_K = entropy(clusters)

    # # Number of data points
    # N = len(clusters)
    #
    # # Find cluster labels
    # cluster_labels = np.unique(clusters)
    #
    # # Find class labels
    # class_labels = np.unique(classes)
    #
    # # Calculate conditional entropy H(K|C)
    # entropy_C_K = -np.sum([np.sum([A[cl][kl] / N * np.log(A[cl][kl] / A['All'][kl]) for cl in class_labels if A[cl][kl] != 0.0]) for kl in cluster_labels])
    #
    # # Calculate conditional entropy H(C|K)
    # entropy_K_C = -np.sum([np.sum([A[cl][kl] / N * np.log(A[cl][kl] / A[cl]['All']) for kl in cluster_labels if A[cl][kl] != 0.0]) for cl in class_labels])

    # Mutual information
    # I(C; K) = H(C) − H(C|K) = H(K) − H(K|C)
    # I = entropy_C - entropy_C_K
    # I = entropy_K - entropy_K_C
    I = mutual_information(clusters, classes, contingency=A)

    # 1 - H(C|K) / H(C) = I / H(C)
    # homogeneity = 1 - entropy_C_K / entropy_C if entropy_C else 1.0
    homogeneity = I / entropy_C if entropy_C else 1.0
    # 1 - H(K|C) / H(K) = I / H(K)
    # completeness = 1 - entropy_K_C / entropy_K if entropy_K else 1.0
    completeness = I / entropy_K if entropy_K else 1.0

    if homogeneity + completeness == 0.0:
        v_measure = 0.0
    else:
        v_measure = 2 * (homogeneity * completeness) / (homogeneity + completeness)

    return homogeneity, completeness, v_measure


def entropy(labels):
    """
    Calculate entropy using maximum likelihood estimates for label probabilities.
    """
    # Length of an array
    N = len(labels)

    if N == 0:
        return 0.0

    # Convert labels to natural numbers
    label_idx = np.unique(labels, return_inverse=True)[1]

    # Count frequency of labels
    pi = np.bincount(label_idx).astype(np.float64)

    # Keep only non-zero ones
    pi = pi[pi > 0]

    # log(a / b) calculated as log(a) - log(b)
    return -np.sum((pi / N) * (np.log(pi) - np.log(N)))


def log_likelihood(X: np.ndarray, k: int, alpha: list, mean: list, covariance: list):
    """
        Log likelihood for Gaussian Mixture Model

        l(theta) = prod_i=1^N f(x_i; theta)
        ll(theta) = sum_i=1^N f(x_i; theta) = sum_i=1^N log sum_k=1^K alpha_k * f(x_i; theta_k)
    """
    rv = np.empty(k, dtype=multivariate_normal_gen)
    for i in range(k):
        rv[i] = multivariate_normal(mean[i], covariance[i])

    likelihood = sum(np.log([sum([alpha[j] * rv[j].pdf(d) for j in range(k)]) for d in X]))

    return likelihood


def evaluate(clusters, classes):
    print("Evaluation")
    print("==========")
    print("Number of observations:          %d" % len(classes))
    print("Number of classes:               %d" % len(np.unique(classes)))
    print("Number of clusters:              %d" % len(np.unique(clusters)))
    print("Purity:                          %f" % purity(clusters, classes))
    print("Purity2:                         %s" % purity2(clusters, classes))
    print("Rand index:                      %f" % rand_index(clusters, classes))
    print("Entropy (clusters):              %f" % entropy(clusters))
    print("Entropy (classes):               %f" % entropy(classes))
    print("Homogeneity:                     %f" % homogeneity(clusters, classes))
    print("Completeness:                    %f" % completeness(clusters, classes))
    print("V-Measure:                       %f" % v_measure(clusters, classes))
    print("Mutual Information:              %f" % mutual_information(clusters, classes))
    print("Normalized Mutual Information:   %f" % normalized_mutual_information(clusters, classes))
    print("Normalized Mutual Information 2: %f" % normalized_mutual_information2(clusters, classes))


if __name__ == "__main__":
    #  Based on Figure 16.4 of (Manning et al. 2008)
    # clusters = [
    #     [
    #         'A', 'A', 'A',
    #         'A', 'A', 'B'
    #     ],
    #     [
    #         'A', 'B', 'B',
    #         'B', 'B', 'C'
    #     ],
    #     [
    #         'A', 'A',
    #         'C', 'C', 'C'
    #     ]
    # ]
    classes = np.array([
        11, 11, 11,
        11, 11, 22,

        11, 22, 22,
        22, 22, 33,

        11, 11,
        33, 33, 33,
    ])

    clusters = np.array([
        1, 1, 1,
        1, 1, 1,

        2, 2, 2,
        2, 2, 2,

        3, 3,
        3, 3, 3,
    ])

    evaluate(clusters, classes)
