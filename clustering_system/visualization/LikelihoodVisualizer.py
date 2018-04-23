import matplotlib.pyplot as plt
import numpy as np


class LikelihoodVisualizer:
    """Visualize clustering likelihood during sampling."""

    def __init__(self):
        self.likelihood = []
        self.N = []

    def add(self, likelihood: float, N: int):
        """
        Add likelihood and the number of observations.

        :param likelihood: The log likelihood
        :param N: the number of observations
        """
        self.likelihood.append(likelihood)
        self.N.append(N)

    def save(self, filename: str):
        """
        Save chart containing the likelihood over time.

        :param filename: The filename
        """
        t = len(self.likelihood)

        # x axis is time
        x = np.arange(0, t, 1)

        # y axis contains evaluation metrics

        fig = plt.figure()
        plt.subplot(211)
        plt.plot(x, self.likelihood, alpha=0.5, label="likelihood")
        plt.xlabel("time")
        plt.grid()
        plt.legend()
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(x, self.N, alpha=0.5, label="number of documents")
        plt.xlabel("time")
        plt.grid()
        plt.legend()
        plt.tight_layout()

        fig.savefig(filename)
        plt.close()
