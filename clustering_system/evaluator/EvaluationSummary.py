import csv
import os
from collections import defaultdict
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from evaluator.SupervisedEvaluation import SupervisedEvaluation


class EvaluationSummary:

    colors = ['royalblue', 'darkorange', 'forestgreen']

    def __init__(self):
        self.stats = defaultdict(lambda: defaultdict(list))
        self.t = 0
        self.N = 0

    def add(self, filename: str):
        _, attributes = zip(*SupervisedEvaluation.get_attribute_names())

        if not os.path.exists(filename):
            raise ValueError("File '%s' not found" % filename)

        self.N += 1
        i = 0
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)

            # Skip header
            it = iter(reader)
            next(it)

            for row in it:
                for attr, val in zip(attributes, row):
                    self.stats[attr][i].append(val)

                i += 1

            self.t = max(self.t, i)

    def save(self, directory):
        chart_1_file = os.path.join(directory, 'chart_1.png')
        chart_2_file = os.path.join(directory, 'chart_2.png')
        chart_3_file = os.path.join(directory, 'chart_3.png')
        chart_4_file = os.path.join(directory, 'chart_4.png')
        chart_5_file = os.path.join(directory, 'chart_5.png')

        # Generate charts
        self._chart_1(chart_1_file)
        self._chart_2(chart_2_file)
        self._chart_3(chart_3_file)
        self._chart_4(chart_4_file)
        self._chart_5(chart_5_file)

    def _chart_1(self, filename: str):
        # x axis is time
        x = np.arange(0, self.t, 1)

        fig = plt.figure()

        # y axis contains evaluation metrics
        observations = [int(self.stats['N'][t][0]) for t in range(self.t)]

        ax = plt.subplot(211)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.plot(x, observations, alpha=1, label="observations", color=self.colors[0], marker='.')
        plt.xlabel("time")
        plt.grid()
        plt.legend()
        plt.tight_layout()

        ax = plt.subplot(212)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

        for n in range(self.N):
            clusters = [int(self.stats['K'][t][n]) for t in range(self.t)]

            plt.plot(x, clusters, alpha=0.25, color=self.colors[0])

        clusters, classes = zip(*[(median(list(map(int, self.stats['K'][t]))), int(self.stats['C'][t][0])) for t in range(self.t)])

        plt.plot(x, clusters, alpha=1, label="clusters", color=self.colors[0], marker='.')
        plt.plot(x, classes, alpha=1, label="classes", color=self.colors[1], marker='.')

        plt.xlabel("time")
        plt.grid()
        plt.legend()
        plt.tight_layout()

        fig.savefig(filename)
        plt.close()

    def _chart_2(self, filename: str):
        # x axis is time
        x = np.arange(0, self.t, 1)

        fig = plt.figure()
        ax = plt.subplot(311)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

        for n in range(self.N):
            purity, rand_index = zip(*[(float(self.stats['purity'][t][n]), float(self.stats['rand_index'][t][n])) for t in range(self.t)])

            plt.plot(x, purity, alpha=0.25, color=self.colors[0])
            plt.plot(x, rand_index, alpha=0.25, color=self.colors[1])

        purity, rand_index = zip(*[(median(list(map(float, self.stats['purity'][t]))), median(list(map(float, self.stats['rand_index'][t])))) for t in range(self.t)])

        plt.plot(x, purity, alpha=1, label="purity", color=self.colors[0], marker='.')
        plt.plot(x, rand_index, alpha=1, label="rand index", color=self.colors[1], marker='.')

        plt.xlabel("time")
        plt.yticks(np.arange(0, 1.1, 0.2))
        plt.grid()
        plt.legend()
        plt.tight_layout()

        ax = plt.subplot(312)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

        for n in range(self.N):
            precision, recall, f1_measure = zip(*[(float(self.stats['precision'][t][n]), float(self.stats['recall'][t][n]), float(self.stats['f1_measure'][t][n])) for t in range(self.t)])

            plt.plot(x, precision, alpha=0.25, color=self.colors[0])
            plt.plot(x, recall, alpha=0.25, color=self.colors[1])
            plt.plot(x, f1_measure, alpha=0.25, color=self.colors[2])

        precision, recall, f1_measure = zip(*[(median(list(map(float, self.stats['precision'][t]))), median(list(map(float, self.stats['recall'][t]))), median(list(map(float, self.stats['f1_measure'][t])))) for t in range(self.t)])

        plt.plot(x, precision, alpha=1, label="precision", color=self.colors[0], marker='.')
        plt.plot(x, recall, alpha=1, label="recall", color=self.colors[1], marker='.')
        plt.plot(x, f1_measure, alpha=1, label="F1-measure", color=self.colors[2], marker='.')
        plt.xlabel("time")
        plt.yticks(np.arange(0, 1.1, 0.2))
        plt.grid()
        plt.legend()
        plt.tight_layout()

        ax = plt.subplot(313)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

        for n in range(self.N):
            homogeneity, completeness, v_measure = zip(*[(float(self.stats['homogeneity'][t][n]), float(self.stats['completeness'][t][n]), float(self.stats['v_measure'][t][n])) for t in range(self.t)])

            plt.plot(x, homogeneity, alpha=0.25, color=self.colors[0])
            plt.plot(x, completeness, alpha=0.25, color=self.colors[1])
            plt.plot(x, v_measure, alpha=0.25, color=self.colors[2])

        homogeneity, completeness, v_measure = zip(*[(median(list(map(float, self.stats['homogeneity'][t]))), median(list(map(float, self.stats['completeness'][t]))), median(list(map(float, self.stats['v_measure'][t])))) for t in range(self.t)])

        plt.plot(x, homogeneity, alpha=1, label="homogeneity", color=self.colors[0], marker='.')
        plt.plot(x, completeness, alpha=1, label="completeness", color=self.colors[1], marker='.')
        plt.plot(x, v_measure, alpha=1, label="V-measure", color=self.colors[2], marker='.')

        plt.xlabel("time")
        plt.yticks(np.arange(0, 1.1, 0.2))
        plt.grid()
        plt.legend()
        plt.tight_layout()

        fig.savefig(filename)
        plt.close()

    def _chart_3(self, filename: str):
        # x axis is time
        x = np.arange(0, self.t, 1)

        fig = plt.figure()
        ax = plt.subplot(211)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

        for n in range(self.N):
            cluster_entropy = [float(self.stats['cluster_entropy'][t][n]) for t in range(self.t)]

            plt.plot(x, cluster_entropy, alpha=0.25, color=self.colors[0])

        cluster_entropy, class_entropy = zip(*[(median(list(map(float, self.stats['cluster_entropy'][t]))), float(self.stats['class_entropy'][t][0])) for t in range(self.t)])

        plt.plot(x, cluster_entropy, alpha=1, label="entropy (clusters)", color=self.colors[0], marker='.')
        plt.plot(x, class_entropy, alpha=1, label="entropy (classes)", color=self.colors[1], marker='.')
        plt.xlabel("time")
        plt.grid()
        plt.legend()
        plt.tight_layout()

        ax = plt.subplot(212)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

        for n in range(self.N):
            mutual_information, normalized_mutual_information, normalized_mutual_information2 = zip(*[(float(self.stats['mutual_information'][t][n]), float(self.stats['normalized_mutual_information'][t][n]), float(self.stats['normalized_mutual_information2'][t][n])) for t in range(self.t)])

            plt.plot(x, mutual_information, alpha=0.25, color=self.colors[0])
            plt.plot(x, normalized_mutual_information, alpha=0.25, color=self.colors[1])
            plt.plot(x, normalized_mutual_information2, alpha=0.25, color=self.colors[2])

        mutual_information, normalized_mutual_information, normalized_mutual_information2 = zip(*[(median(list(map(float, self.stats['mutual_information'][t]))), median(list(map(float, self.stats['normalized_mutual_information'][t]))), median(list(map(float, self.stats['normalized_mutual_information2'][t])))) for t in range(self.t)])

        plt.plot(x, mutual_information, alpha=1, label="mutual information", color=self.colors[0], marker='.')
        plt.plot(x, normalized_mutual_information, alpha=1, label="normalized mutual information", color=self.colors[1], marker='.')
        plt.plot(x, normalized_mutual_information2, alpha=1, label="normalized mutual information 2", color=self.colors[2], marker='.')
        plt.xlabel("time")
        plt.grid()
        plt.legend()
        plt.tight_layout()

        fig.savefig(filename)
        plt.close()

    def _chart_4(self, filename: str):
        # x axis is time
        x = np.arange(0, self.t, 1)

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

        for n in range(self.N):
            likelihood = [float(self.stats['likelihood'][t][n]) for t in range(self.t)]

            plt.plot(x, likelihood, alpha=0.25, color=self.colors[0])

        likelihood = [median(list(map(float, self.stats['likelihood'][t]))) for t in range(self.t)]

        plt.plot(x, likelihood, alpha=1, label="likelihood", color=self.colors[0], marker='.')
        plt.xlabel("time")
        plt.grid()
        plt.legend()
        plt.tight_layout()

        fig.savefig(filename)
        plt.close()

    def _chart_5(self, filename: str):
        # x axis is time
        x = np.arange(0, self.t, 1)

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

        for n in range(self.N):
            aic, bic = zip(*[(float(self.stats['aic'][t][n]), float(self.stats['bic'][t][n])) for t in range(self.t)])

            plt.plot(x, aic, alpha=0.25, color=self.colors[0])
            plt.plot(x, bic, alpha=0.25, color=self.colors[1])

        aic, bic = zip(*[(median(list(map(float, self.stats['aic'][t]))), median(list(map(float, self.stats['bic'][t])))) for t in range(self.t)])

        plt.plot(x, aic, alpha=1, label="Akaike information criterion", color=self.colors[0], marker='.')
        plt.plot(x, bic, alpha=1, label="Bayesian information criterion", color=self.colors[1], marker='.')

        plt.xlabel("time")
        plt.grid()
        plt.legend()
        plt.tight_layout()

        fig.savefig(filename)
        plt.close()
