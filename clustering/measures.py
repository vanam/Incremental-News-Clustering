from collections import Counter


def purity(clusters):
    """
    The purity ranges between 0 (bad) and 1 (good). However, we can trivially achieve a purity of 1
    by putting each object into its own cluster,
    so this measure does not penalize for the number of clusters.
    """
    # Accumulator
    acc = 0

    # Total number of items
    N = 0

    # For each cluster
    for c in clusters:
        # Add to total number of items
        N += len(c)

        # Count cluster labels
        counter = Counter(c)

        # Accumulate the most frequent label count
        acc += counter.most_common(1)[0][1]

    return acc / N


def purity2(clusters):
    """
    For each cluster return its purity and size
    """
    result = []

    # For each cluster
    for c in clusters:
        # Count cluster labels
        counter = Counter(c)

        N_i = len(c)

        # the most frequent label count /
        result.append((counter.most_common(1)[0][1] / N_i, N_i))

    return result


if __name__ == "__main__":
    #  Based on Figure 16.4 of (Manning et al. 2008)
    clusters = [
        [
            'A', 'A', 'A',
            'A', 'A', 'B'
        ],
        [
            'A', 'B', 'B',
            'B', 'B', 'C'
        ],
        [
            'A', 'A',
            'C', 'C', 'C'
        ]
    ]

    print("Purity:     %f" % purity(clusters))
    print("Purity2:    %s" % purity2(clusters))
