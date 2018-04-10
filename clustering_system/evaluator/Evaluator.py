import csv
import logging
import os
from typing import Tuple

import numpy as np

from clustering_system.evaluator.EvaluatorABC import EvaluatorABC


class Evaluator(EvaluatorABC):

    def __init__(self, filename: str, language=None):
        super().__init__()

        self.truth = {}

        if not os.path.exists(filename):
            raise ValueError("File '%s' not found" % filename)

        i = 0
        cluster_to_id_mapper = {}

        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)

            # Skip header
            it = iter(reader)
            next(it)

            for row in it:
                # Skip documents in different languages
                if language is not None and language != row[1]:
                    continue

                if row[6] not in cluster_to_id_mapper:
                    cluster_to_id_mapper[row[6]] = i
                    i += 1

                # truth[id] = class_id
                self.truth[row[0]] = cluster_to_id_mapper[row[6]]

    def _get_clusters_classes(self, time, ids: list, clusters: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        clu = []
        cla = []
        skipped = 0

        for id, cluster in zip(ids, clusters):
            if id not in self.truth:
                skipped += 1
                continue

            clu.append(cluster)
            cla.append(self.truth[id])

        logging.warning("%d files skipped during evaluation, %d files are being evaluated." % (skipped, len(clu)))

        return np.array(clu), np.array(cla)
