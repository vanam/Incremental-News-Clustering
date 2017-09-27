import argparse

import numpy as np
from scipy import linalg

from clustering.normalization import std_scaling, mean_normalization
from clustering.plot import plot_show, _plot
from clustering.readData import read_data, extract_points


def pca(data, k=None):
    """
    Principal component analysis using eigenvalues

    Note: Does not pre-process data
    Source: https://stackoverflow.com/a/27933271/3484013
    """
    # Find covariance matrix
    C = np.cov(data, rowvar=False)  # np.dot(data.T, data) / (data.shape[0] - 1)

    # Find eigenvalues and eigenvectors of covariance matrix
    E, V = linalg.eigh(C)

    # Sort eigenvalues and pick k first (in descending order)
    key = np.argsort(E)[::-1][:k]

    # Return eigenvalues and eigenvectors
    return E[key], V[:, key]


def pca_transform(data, V):
    """Apply dimensionality reduction to data"""
    return np.dot(data, V)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run k-means clustering on data.')

    parser.add_argument('file', metavar='file', type=argparse.FileType('rb'),
                        help='a data file')
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

    # Normalize and scale data
    data = std_scaling(mean_normalization(data))

    # Run PCA
    E, V = pca(data, k=1)

    # Plot data if desired
    if args.plot:
        # Plot original (but normalized and scaled) data
        _plot(data)

        # Plot transformed data
        _plot(pca_transform(data, V))

        # Plot reverse transformed data
        _plot(pca_transform(pca_transform(data, V), V.T))

        plot_show()
