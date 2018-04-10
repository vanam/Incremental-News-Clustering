from typing import Tuple

import numpy as np

from clustering_system.evaluator.EvaluatorABC import EvaluatorABC


class RandomEvaluator(EvaluatorABC):

    def __init__(self, C: int):
        super().__init__()
        self.C = C

    def _get_clusters_classes(self, time, ids: list, clusters: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(ids)
        return clusters, np.random.random_integers(self.C, size=n)
