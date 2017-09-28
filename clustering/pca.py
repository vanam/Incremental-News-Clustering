import argparse

import numpy as np
from scipy import linalg

from clustering.normalization import std_scaling, mean_normalization, std_descaling, mean_denormalization
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
    normalized_data, mean = mean_normalization(data)
    normalized_data, std = std_scaling(normalized_data)

    # Run PCA
    E, V = pca(normalized_data, k=1)

    # Plot data if desired
    if args.plot:
        # Plot original data
        _plot(data)

        # Plot original (but normalized and scaled) data
        _plot(normalized_data)

        # Plot transformed data
        transformed_data = pca_transform(normalized_data, V)
        _plot(transformed_data)

        # Plot reverse transformed data
        reversed_data = pca_transform(transformed_data, V.T)
        _plot(reversed_data)

        denormalized_data = std_descaling(reversed_data, std)
        denormalized_data = mean_denormalization(denormalized_data, mean)

        # Plot reverse transformed and denormalized data
        _plot(denormalized_data)

        plot_show()
