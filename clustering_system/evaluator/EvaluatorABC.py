import csv
import logging
import os
from abc import ABC, abstractmethod
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from clustering_system.evaluator.SupervisedEvaluation import SupervisedEvaluation


class EvaluatorABC(ABC):
    """An abstract evaluator class"""

    def __init__(self):
        self.evaluations = {}
        self.classes = {}
        self.clusters = {}

    @abstractmethod
    def _get_clusters_classes(self, time: int, ids: list, clusters: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the true class labels and cluster labels only for ids mentioned in ground truth.

        :param time: The time of evaluation
        :param ids: The list of ids
        :param clusters: The cluster assignments
        :return: (clusters, classes)
        """
        pass

    def evaluate(self, time: int, ids_clusters: list, aic: float, bic: float, likelihood: float):
        """
        Evaluate clustering at given time

        :param time: The time of evaluation
        :param ids_clusters: The list of tuples (doc_id, cluster_id)
        :param aic: The Akaike information criterion
        :param bic: The Bayesian information criterion
        :param likelihood: The log likelihood
        """
        ids, clusters = map(np.array, zip(*ids_clusters))
        clusters, classes = self._get_clusters_classes(time, ids, clusters)

        self.evaluations[time] = SupervisedEvaluation(clusters, classes, aic, bic, likelihood)
        self.classes[time] = classes
        self.clusters[time] = clusters

    def save(self, directory):
        """
        Save evaluation in a directory.

        :param directory: The directory
        """
        if len(self.evaluations) == 0:
            logging.warning("No evaluations to store.")
            return

        csv_file = os.path.join(directory, 'evaluation.csv')
        chart_1_file = os.path.join(directory, 'chart_1.png')
        chart_2_file = os.path.join(directory, 'chart_2.png')
        chart_3_file = os.path.join(directory, 'chart_3.png')
        chart_4_file = os.path.join(directory, 'chart_4.png')
        chart_5_file = os.path.join(directory, 'chart_5.png')
        chart_6_file = os.path.join(directory, 'chart_6.png')

        # Save in text file
        self._export_to_csv(csv_file)

        # Generate charts
        self._chart_1(chart_1_file)
        self._chart_2(chart_2_file)
        self._chart_3(chart_3_file)
        self._chart_4(chart_4_file)
        self._chart_5(chart_5_file)
        self._chart_6(chart_6_file)

    def __iter__(self):
        """
        Iterate over evaluations.

        :return: (time, evaluation)
        """
        for t, e in self.evaluations.items():
            yield t, e

    def _export_to_csv(self, csv_file):
        """
        Export evaluation to CSV file.

        :param csv_file: The filename
        """
        attribute_names, attributes = zip(*SupervisedEvaluation.get_attribute_names())

        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

            # Write header
            writer.writerow(attribute_names)

            # Write evaluation on each row
            for _, e in self:
                writer.writerow([getattr(e, attribute) for attribute in attributes])

    def _chart_1(self, filename: str):
        """
        Save chart containing the number of observations, the number of clusters
        and the number of classes over time.

        :param filename: The filename
        """
        t = len(self.evaluations)

        # x axis is time
        x = np.arange(0, t, 1)

        # y axis contains evaluation metrics
        observations, clusters, classes = zip(*[(e.N, e.K, e.C) for _, e in self])

        fig = plt.figure()
        ax = plt.subplot(211)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.plot(x, observations, alpha=0.5, label="observations")
        plt.xlabel("time")
        plt.grid()
        plt.legend()
        plt.tight_layout()

        ax = plt.subplot(212)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.plot(x, clusters, alpha=0.5, label="clusters")
        plt.plot(x, classes, alpha=0.5, label="classes")
        plt.xlabel("time")
        plt.grid()
        plt.legend()
        plt.tight_layout()

        fig.savefig(filename)
        plt.close()

    def _chart_2(self, filename: str):
        """
        Save chart containing the purity, rand index, precision, recall, F1-measure over time.

        :param filename: The filename
        """
        t = len(self.evaluations)

        # x axis is time
        x = np.arange(0, t, 1)

        # y axis contains evaluation metrics
        purity, rand_index, precision, recall,  f1_measure, homogeneity, completeness, v_measure, nv_measure = zip(*[(e.purity, e.rand_index, e.precision, e.recall, e.f1_measure, e.homogeneity, e.completeness, e.v_measure, e.nv_measure) for _, e in self])

        fig = plt.figure()
        ax = plt.subplot(221)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.plot(x, purity, alpha=0.5, label="purity")
        plt.plot(x, rand_index, alpha=0.5, label="rand index")
        plt.xlabel("time")
        plt.yticks(np.arange(0, 1.1, 0.2))
        plt.grid()
        plt.legend()
        plt.tight_layout()

        ax = plt.subplot(222)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.plot(x, precision, alpha=0.5, label="precision")
        plt.plot(x, recall, alpha=0.5, label="recall")
        plt.plot(x, f1_measure, alpha=0.5, label="F1-measure")
        plt.xlabel("time")
        plt.yticks(np.arange(0, 1.1, 0.2))
        plt.grid()
        plt.legend()
        plt.tight_layout()

        ax = plt.subplot(223)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.plot(x, homogeneity, alpha=0.5, label="homogeneity")
        plt.plot(x, completeness, alpha=0.5, label="completeness")
        plt.plot(x, v_measure, alpha=0.5, label="V-measure")
        plt.xlabel("time")
        plt.yticks(np.arange(0, 1.1, 0.2))
        plt.grid()
        plt.legend()
        plt.tight_layout()

        ax = plt.subplot(224)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.plot(x, nv_measure, alpha=0.5, label="NV-measure")
        plt.xlabel("time")
        plt.yticks(np.arange(0, 1.1, 0.2))
        plt.grid()
        plt.legend()
        plt.tight_layout()

        fig.savefig(filename)
        plt.close()

    def _chart_3(self, filename: str):
        """
        Save chart containing the entropy, mutual information and normalized mutual information over time.

        :param filename: The filename
        """
        t = len(self.evaluations)

        # x axis is time
        x = np.arange(0, t, 1)

        # y axis contains evaluation metrics
        cluster_entropy, class_entropy, mutual_information, normalized_mutual_information, normalized_mutual_information2 = zip(*[(e.cluster_entropy, e.class_entropy, e.mutual_information, e.normalized_mutual_information, e.normalized_mutual_information2) for _, e in self])

        fig = plt.figure()
        ax = plt.subplot(211)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.plot(x, cluster_entropy, alpha=0.5, label="entropy (clusters)")
        plt.plot(x, class_entropy, alpha=0.5, label="entropy (classes)")
        plt.xlabel("time")
        plt.grid()
        plt.legend()
        plt.tight_layout()

        ax = plt.subplot(212)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.plot(x, mutual_information, alpha=0.5, label="mutual information")
        plt.plot(x, normalized_mutual_information, alpha=0.5, label="normalized mutual information")
        plt.plot(x, normalized_mutual_information2, alpha=0.5, label="normalized mutual information 2")
        plt.xlabel("time")
        plt.grid()
        plt.legend()
        plt.tight_layout()

        fig.savefig(filename)
        plt.close()

    def _chart_4(self, filename: str):
        """
        Save chart containing the likelihood over time.

        :param filename: The filename
        """
        t = len(self.evaluations)

        # x axis is time
        x = np.arange(0, t, 1)

        # y axis contains evaluation metrics
        likelihood = [e.likelihood for _, e in self]

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.plot(x, likelihood, alpha=0.5, label="likelihood")
        plt.xlabel("time")
        plt.grid()
        plt.legend()
        plt.tight_layout()

        fig.savefig(filename)
        plt.close()

    def _chart_5(self, filename: str):
        """
        Save chart containing AIC and BIC over time.

        :param filename: The filename
        """
        t = len(self.evaluations)

        # x axis is time
        x = np.arange(0, t, 1)

        # y axis contains evaluation metrics
        aic, bic = zip(*[(e.aic, e.bic) for _, e in self])

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.plot(x, aic, alpha=0.5, label="Akaike information criterion")
        plt.plot(x, bic, alpha=0.5, label="Bayesian information criterion")
        plt.xlabel("time")
        plt.grid()
        plt.legend()
        plt.tight_layout()

        fig.savefig(filename)
        plt.close()

    def _chart_6(self, filename: str):
        """
        Save chart containing cluster size histogram at the end.

        :param filename: The filename
        """
        fig = plt.figure()
        ax = plt.subplot(211)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

        data = self.clusters[len(self.clusters) - 1]
        bins = np.unique(data)
        plt.hist(data, bins=bins, label="Cluster size histogram")

        ax = plt.subplot(212)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

        data = self.classes[len(self.classes) - 1]
        bins = np.unique(data)
        plt.hist(data, bins=bins, label="Classes size histogram")

        fig.savefig(filename)
        plt.close()
