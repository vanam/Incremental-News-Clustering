import matplotlib.pyplot as plt
import numpy as np


class LikelihoodVisualizer:

    def __init__(self):
        self.likelihood = []
        self.N = []

    def add(self, likelihood: float, N: int):
        self.likelihood.append(likelihood)
        self.N.append(N)

    def save(self, filename: str):
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
