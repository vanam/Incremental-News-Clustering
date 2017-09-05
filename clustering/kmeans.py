import argparse
import random
from operator import itemgetter

import numpy as np

import clustering.cluster as cluster
from clustering.plot import plot_clusters
from clustering.readData import read_data, extract_points


def _kmeans(data, k, threshold=1e-5):
    # Pick k points at random in data
    initial_centroids = random.sample(data, k)

    # Create clusters containing only centroids
    clusters = []
    for c in initial_centroids:
        clusters.append(cluster.Cluster([c]))

    # Iterate while distance change is significant
    dist_change = 2*threshold + 1
    while dist_change > threshold:
        # Initialize array for clusters data
        clusters_data = [[] for _ in range(k)]

        # Associate each observation with closest centroid
        for d in data:
            # Find cluster index to closest centroid
            cluster_index = min(enumerate(c.distance_to_centroid(d) for c in clusters), key=itemgetter(1))[0]

            # Assign observation to cluster
            clusters_data[cluster_index].append(d)

        # Keep track of change between new and old centroids
        dist_change = 0
        for i in range(len(clusters)):
            try:
                dist_change += clusters[i].update(clusters_data[i])
            except ValueError:
                # Return nothing if some cluster is empty
                return None

    return clusters


def kmeans(data, k, tries=20, threshold=1e-5):
    if int(k) < 1:
        raise ValueError('the number of clusters must be at least 1.')

    if int(k) > len(data):
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
        clusters = _kmeans(data, k, threshold=threshold)

        # Compute dissimilarity
        dissimilarity = cluster.dissimilarity(clusters)

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
        # plot_clusters(clusters)
        plot_clusters(None)
