from collections import Counter

import numpy as np
from scipy.special import comb


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

    print("Purity:     %f" % purity(clusters, classes))
    print("Purity2:    %s" % purity2(clusters, classes))
    print("Rand index: %f" % rand_index(clusters, classes))
