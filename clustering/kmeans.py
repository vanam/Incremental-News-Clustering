import argparse
import random
from operator import itemgetter

import numpy as np
from scipy.spatial.distance import euclidean

import clustering.measures as measures
from clustering.plot import plot_clusters, plot_show
from clustering.readData import read_data, extract_points


def _kmeans(X, k, threshold=1e-5):
    N = len(X)

    # Cluster indicator for each observation
    z = np.empty(N, dtype=int)

    # Pick k points at random in data
    centroids = random.sample(list(X), k)

    # Iterate while distance change is significant
    dist_change = 2*threshold + 1
    while dist_change > threshold:
        # Associate each observation with closest centroid
        for i, d in enumerate(X):
            # Find cluster index to closest centroid
            z[i] = min(enumerate(euclidean(c, d) for c in centroids), key=itemgetter(1))[0]

        # Calculate new centroids and keep track of change between new and old centroids
        dist_change = 0
        new_centroids = np.empty(k, dtype=list)
        for i in range(k):
            data_in_cluster = X[z == i]
            new_centroids[i] = np.mean(data_in_cluster, axis=0)
            dist_change += euclidean(centroids[i], new_centroids[i])

            # Restart k-means if some cluster has no observations
            if len(data_in_cluster) == 0:
                return None

        # Set new centroid
        centroids = new_centroids

    return z


def kmeans(X, k, tries=20, threshold=1e-5):
    if int(k) < 1:
        raise ValueError('the number of clusters must be at least 1.')

    if int(k) > len(X):
        raise ValueError('not enough data.')

    if int(tries) < 1:
        raise ValueError('the number of tries must be at least 1.')

    if float(threshold) <= 0.0:
        raise ValueError('threshold must be greater than zero.')

    # Initialize best dissimilarity value to a large value
    best_dissimilarity = np.inf
    best_clusters = None

    # Run k-means algorithm several times
    for i in range(tries):
        # Run k-means
        clusters = _kmeans(X, k, threshold=threshold)

        if clusters is None:
            continue

        # Compute dissimilarity
        dissimilarity = measures.dissimilarity(X, clusters)

        # Accept best result
        if dissimilarity < best_dissimilarity:
            best_dissimilarity = dissimilarity
            best_clusters = clusters

    return best_clusters

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run k-means clustering on data.')

    parser.add_argument('file', metavar='file', type=argparse.FileType('rb'),
                        help='a data file')
    parser.add_argument('-k', dest='k', type=int,
                        help='the number of clusters (default: 2)')
    parser.add_argument('-t, --tries', dest='tries', type=int,
                        help='the number of tries (default: 20)')
    parser.add_argument('--threshold', dest='threshold', type=float,
                        help='threshold (default: 1e-5)')
    parser.add_argument('-p, --plot', dest='plot', action='store_true',
                        help='plot found clusters')
    parser.add_argument('-s, --shuffle', dest='shuffle', action='store_true',
                        help='randomly shuffle data points')
    parser.set_defaults(k=2, tries=20, threshold=1e-5, plot=False, shuffle=False)

    args = parser.parse_args()

    data = extract_points(read_data(args.file))

    # Shuffle points if desired
    if args.shuffle:
        np.random.shuffle(data)

    # Calculate clusters
    clusters = kmeans(data, args.k, args.tries, args.threshold)
    print(clusters)

    # Plot clusters if desired
    if args.plot:
        plot_clusters(clusters, title='k-means \'%s\'' % args.file.name)
        plot_show()
