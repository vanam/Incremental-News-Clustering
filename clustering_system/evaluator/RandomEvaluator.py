import numpy as np

from clustering_system.evaluator.Evaluator import EvaluatorABC


class RandomEvaluator(EvaluatorABC):

    def __init__(self, C, corpora):
        super().__init__(corpora)
        self.C = C

    def _get_classes(self, time, ids: list) -> np.ndarray:
        n = len(ids)
        return np.random.random_integers(self.C, size=n)
