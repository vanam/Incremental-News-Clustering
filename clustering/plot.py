import argparse
import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np

from clustering.cluster import Cluster
from clustering.gaussianMixture import GaussianMixture
from clustering.readData import read_data

COLORS = "bgrcmykw"
POINTS = ".,ov^<>12348spP*hH+xXDd|_"

# Calculate all possible point-color combinations
POINT_COLORS = list(map(lambda p: "".join(p), itertools.product(POINTS, COLORS)))


def get_point_color_index():
    n = 0
    while True:
        yield n
        n += 1
        if n is len(POINT_COLORS):
            n = 0

POINT_COLORS_INDEX_GENERATOR = get_point_color_index()


def _plot(data):
    if len(data) is 0:
        return

    dimension = len(data[0])

    if dimension is 1:
        x = np.transpose(data)
        plt.plot(x, 0, POINT_COLORS[next(POINT_COLORS_INDEX_GENERATOR)], alpha=0.5)
    elif dimension is 2:
        x, y = np.transpose(data)
        plt.plot(x, y, POINT_COLORS[next(POINT_COLORS_INDEX_GENERATOR)], alpha=0.5)
    else:
        print("High dimensions are not supported.", file=sys.stderr)


def plot_clusters(clusters):
    if clusters is None:
        print("Nothing to plot.", file=sys.stderr)
        return

    if not all(isinstance(o, Cluster) for o in clusters):
        raise TypeError("Array must contain only Cluster class instances")

    # Plot clusters and collect centroids
    centroids = []
    for c in clusters:
        _plot(c.data)
        centroids.append(c.centroid)

    # Plot centroids
    _plot(centroids)


def plot_show():
    plt.show()


def plot_gaussian_mixture(mixture):
    if not isinstance(mixture, GaussianMixture):
        print(isinstance(mixture, GaussianMixture))
        print(type(mixture))
        raise TypeError("Mixture must be instance of GaussianMixture class")

    # Plot clusters
    plot_clusters(mixture.get_clusters())

    x, y = np.transpose(mixture.data)

    x = np.linspace(min(x), max(x))
    y = np.linspace(min(y), max(y))
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = mixture.score_samples(XX)
    Z = Z.reshape(X.shape)

    CS = plt.contour(X, Y, Z)
    CB = plt.colorbar(CS, shrink=0.8, extend='both')


def plot_data_file(file):
    # Read data file
    data = read_data(file)

    for d in data:
        _plot(d)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot data.')

    parser.add_argument('file', metavar='file', type=argparse.FileType('rb'),
                        help='a data file')

    args = parser.parse_args()
    plot_data_file(args.file)
