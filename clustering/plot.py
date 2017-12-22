import argparse
import itertools
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy import linalg
from scipy.stats import f

from clustering.cluster import Cluster
from clustering.gaussianMixture import GaussianMixture
from clustering.readData import read_data

mpl.rcParams["savefig.directory"] = ""

ALPHA = 0.005

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


def _plot(data, ellipse=False):
    # Number of data points
    n = len(data)

    if n is 0:
        return

    # Dimension
    p = len(data[0])

    if p is 1:
        x = np.transpose(data)
        plt.plot(x, 0, POINT_COLORS[next(POINT_COLORS_INDEX_GENERATOR)], alpha=0.5)
    elif p is 2:
        x, y = np.transpose(data)
        plt.plot(x, y, POINT_COLORS[next(POINT_COLORS_INDEX_GENERATOR)], alpha=0.5)

        if ellipse:
            if n < p:
                print('Not enough points to estimate covariance matrix.', file=sys.stderr)
                return

            # Find mean
            mu = np.mean(data, axis=0)

            # Find covariance matrix
            C = np.cov(data, rowvar=False)  # np.dot(data.T, data) / (data.shape[0] - 1)

            # Find eigenvalues and eigenvectors of covariance matrix
            E, V = linalg.eigh(C)

            # Sort eigenvalues (in descending order)
            keys = np.argsort(E)[::-1]

            # The largest/smallest eigenvalue index
            lg, sm = zip(keys)

            # Get the current Axes instance on the current figure
            a = plt.gca()

            # Since we are using sample covariance = > Hotelling's T-squared distribution
            # (x - mu_hat)' * sigma^-1 * (x - mu_hat) ~ T^2(p, n - 1)
            # Calculate T ^ 2 using F - distribution
            # T ^ 2(p, n) = n * p / (n - p + 1) * F(p, n - p + 1)
            # T ^ 2(p, n - 1) = (n - 1) * p / (n - p) * F(p, n - p)
            F_esti = f.ppf(1 - ALPHA, p, n - p)

            # Ellipse angle
            angle = np.arctan2(V[1][lg], V[0][lg])

            # Plot ellipse
            e = Ellipse(mu, np.sqrt(E[lg]) * F_esti, np.sqrt(E[sm]) * F_esti, np.degrees(angle))
            e.set_clip_box(a.bbox)
            e.set_alpha(0.5)
            a.add_artist(e)

            # Plot
            for i, vec in enumerate(V.T):
                vec *= E[i]

                a.quiver(
                    [mu[0]],
                    [mu[1]],
                    [vec[0]],
                    [vec[1]],
                    angles='xy', scale_units='xy', scale=1, units='dots'
                )
    else:
        print("High dimensions are not supported.", file=sys.stderr)


def plot_clusters(clusters, title=None):
    if clusters is None:
        print("Nothing to plot.", file=sys.stderr)
        return

    if not all(isinstance(o, Cluster) for o in clusters):
        raise TypeError("Array must contain only Cluster class instances")

    # Set figure title
    plt.title(title)

    # Plot clusters and collect centroids
    plt.axis('equal')
    centroids = []
    for c in clusters:
        _plot(c.data, ellipse=True)
        centroids.append(c.centroid)

    # Plot centroids
    _plot(centroids)


def plot_clusters2(X, clusters, title=''):
    if clusters is None:
        print("Nothing to plot.", file=sys.stderr)
        return

    plt.figure()

    # Set figure title
    plt.title(title)

    # Plot clusters and collect centroids
    plt.axis('equal')
    centroids = []
    for c in np.unique(clusters):
        cluster_data = X[clusters == c]
        _plot(cluster_data, ellipse=True)
        centroids.append(np.mean(cluster_data, axis=0))

    # Plot centroids
    _plot(centroids)

    return plt


def plot_show():
    plt.show()


def plot_gaussian_mixture(mixture, title=None):
    if not isinstance(mixture, GaussianMixture):
        print(isinstance(mixture, GaussianMixture))
        print(type(mixture))
        raise TypeError("Mixture must be instance of GaussianMixture class")

    # Plot clusters
    plot_clusters(mixture.get_clusters(), title=title)


def plot_data_file(file, ellipse=False):
    # Read data file
    data = read_data(file)

    plt.title('Original data \'%s\'' % file.name)
    plt.axis('equal')
    centroids = []
    for d in data:
        _plot(d, ellipse=ellipse)
        centroids.append(np.mean(d, axis=0))

    # Plot centroids
    _plot(centroids)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot data.')

    parser.add_argument('file', metavar='file', type=argparse.FileType('rb'),
                        help='a data file')

    parser.add_argument('-e, --ellipse', dest='ellipse', action='store_true',
                        help='plot ellipses')

    args = parser.parse_args()
    plot_data_file(args.file, args.ellipse)
