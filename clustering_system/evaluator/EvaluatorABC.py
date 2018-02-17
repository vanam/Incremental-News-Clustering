import csv
import os
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from clustering.measures import SupervisedEvaluation


class EvaluatorABC(ABC):

    def __init__(self, corpora):
        self.corpora = corpora
        self.evaluations = {}

    @abstractmethod
    def _get_classes(self, time: int, ids: list) -> np.ndarray:
        pass

    def evaluate(self, time: int, ids_clusters: list, X: np.array, likelihood: float):
        ids, clusters = map(np.array, zip(*ids_clusters))
        classes = self._get_classes(time, ids)

        self.evaluations[time] = SupervisedEvaluation(X, clusters, classes, likelihood)

    def save(self, directory):
        csv_file = os.path.join(directory, 'evaluation.csv')
        chart_1_file = os.path.join(directory, 'chart_1.png')
        chart_2_file = os.path.join(directory, 'chart_2.png')
        chart_3_file = os.path.join(directory, 'chart_3.png')
        chart_4_file = os.path.join(directory, 'chart_4.png')

        # Save in text file
        self._export_to_csv(csv_file)

        # Generate charts
        self._chart_1(chart_1_file)
        self._chart_2(chart_2_file)
        self._chart_3(chart_3_file)
        self._chart_4(chart_4_file)

    def __iter__(self):
        for t, e in self.evaluations.items():
            yield t, e

    def _export_to_csv(self, csv_file):
        attribute_names, attributes = zip(*SupervisedEvaluation.get_attribute_names())

        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

            # Write header
            writer.writerow(attribute_names)

            # Write evaluation on each row
            for _, e in self:
                writer.writerow([getattr(e, attribute) for attribute in attributes])

    def _chart_1(self, filename: str):
        t = len(self.evaluations)

        # x axis is time
        x = np.arange(0, t, 1)

        # y axis contains evaluation metrics
        observations, clusters, classes = zip(*[(e.N, e.K, e.C) for _, e in self])

        fig = plt.figure()
        plt.subplot(211)
        plt.plot(x, observations, alpha=0.5, label="observations")
        plt.xlabel("time")
        plt.grid()
        plt.legend()
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(x, clusters, alpha=0.5, label="clusters")
        plt.plot(x, classes, alpha=0.5, label="classes")
        plt.xlabel("time")
        plt.grid()
        plt.legend()
        plt.tight_layout()

        fig.savefig(filename)

    def _chart_2(self, filename: str):
        t = len(self.evaluations)

        # x axis is time
        x = np.arange(0, t, 1)

        # y axis contains evaluation metrics
        purity, rand_index, homogeneity, completeness, v_measure = zip(*[(e.purity, e.rand_index, e.homogeneity, e.completeness, e.v_measure) for _, e in self])

        fig = plt.figure()
        plt.subplot(211)
        plt.plot(x, purity, alpha=0.5, label="purity")
        plt.plot(x, rand_index, alpha=0.5, label="rand index")
        plt.xlabel("time")
        plt.yticks(np.arange(0, 1.1, 0.2))
        plt.grid()
        plt.legend()
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(x, homogeneity, alpha=0.5, label="homogeneity")
        plt.plot(x, completeness, alpha=0.5, label="completeness")
        plt.plot(x, v_measure, alpha=0.5, label="V-measure")
        plt.xlabel("time")
        plt.yticks(np.arange(0, 1.1, 0.2))
        plt.grid()
        plt.legend()
        plt.tight_layout()

        fig.savefig(filename)

    def _chart_3(self, filename: str):
        t = len(self.evaluations)

        # x axis is time
        x = np.arange(0, t, 1)

        # y axis contains evaluation metrics
        cluster_entropy, class_entropy, mutual_information, normalized_mutual_information, normalized_mutual_information2 = zip(*[(e.cluster_entropy, e.class_entropy, e.mutual_information, e.normalized_mutual_information, e.normalized_mutual_information2) for _, e in self])

        fig = plt.figure()
        plt.subplot(211)
        plt.plot(x, cluster_entropy, alpha=0.5, label="entropy (clusters)")
        plt.plot(x, class_entropy, alpha=0.5, label="entropy (classes)")
        plt.xlabel("time")
        plt.grid()
        plt.legend()
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(x, mutual_information, alpha=0.5, label="mutual information")
        plt.plot(x, normalized_mutual_information, alpha=0.5, label="normalized mutual information")
        plt.plot(x, normalized_mutual_information2, alpha=0.5, label="normalized mutual information 2")
        plt.xlabel("time")
        plt.grid()
        plt.legend()
        plt.tight_layout()

        fig.savefig(filename)

    def _chart_4(self, filename: str):
        t = len(self.evaluations)

        # x axis is time
        x = np.arange(0, t, 1)

        # y axis contains evaluation metrics
        likelihood, dissimilarity = zip(*[(e.likelihood, e.dissimilarity) for _, e in self])

        fig = plt.figure()
        plt.subplot(211)
        plt.plot(x, likelihood, alpha=0.5, label="likelihood")
        plt.xlabel("time")
        plt.grid()
        plt.legend()
        plt.tight_layout()

        plt.subplot(212)
        plt.plot(x, dissimilarity, alpha=0.5, label="dissimilarity")
        plt.xlabel("time")
        plt.grid()
        plt.legend()
        plt.tight_layout()

        fig.savefig(filename)
