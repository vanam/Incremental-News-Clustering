import itertools
import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy import linalg
from scipy.stats import f

ALPHA = 0.005

COLORS = "bgrcmykw"
POINTS = "ov^<>12348spP*hH+xXDd|_"

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


class ClusterVisualizer:

    def save(self, filename: str, k: int, mean: np.ndarray, covariance: np.ndarray, X: List[np.ndarray]):
        p = 2

        fig = plt.figure()
        plt.axis('equal')
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")

        # Plot documents
        for i in range(k):
            x, y = np.transpose(X[i])
            plt.plot(x, y, POINT_COLORS[next(POINT_COLORS_INDEX_GENERATOR)], alpha=0.5)

        # Plot ellipse
        # print(X)
        for i in range(k):
            n = len(X[i])
            # print("N_k = %d" % n)

            # if n < p:
            #     logging.error('Not enough points to estimate covariance matrix, %d points given.' % n)
            #     continue
            #
            # # Find mean
            # mu = np.mean(X[i], axis=0)
            #
            # # Find covariance matrix
            # C = np.cov(X[i], rowvar=False)  # np.dot(data.T, data) / (data.shape[0] - 1)

            # Mean, covariance
            mu, C = mean[i], covariance[i]

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
            for j, vec in enumerate(V.T):
                vec *= E[j]

                a.quiver(
                    [mu[0]],
                    [mu[1]],
                    [vec[0]],
                    [vec[1]],
                    angles='xy', scale_units='xy', scale=1, units='dots'
                )

        # Plot centroids
        x, y = np.transpose(mean)
        plt.plot(x, y, POINT_COLORS[next(POINT_COLORS_INDEX_GENERATOR)], alpha=0.5)

        fig.savefig(filename)
        plt.close()
