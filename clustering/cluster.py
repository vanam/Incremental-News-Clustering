import numpy as np
from scipy.spatial.distance import cdist


class Cluster:
    def __init__(self, data, distance_metric='euclidean'):
        if len(data) is 0:
            raise ValueError("Cluster must contain data")
        self.data = np.array(data)
        self.metric = distance_metric
        self.centroid = self.compute_centroid()

    def compute_centroid(self):
        return list(map(lambda d: d / self.size(), np.sum(self.data, axis=0)))

    def variability(self):
        return sum(np.square(cdist(np.array([self.centroid]), self.data, metric=self.metric))[0])

    def members(self):
        for d in self.data:
            yield d

    def size(self):
        return len(self.data)

    def __str__(self):
        string = 'Cluster {\n'
        string += '  centroid    = ' + str(self.centroid) + ',\n'
        string += '  size        = ' + str(self.size()) + '\n'
        string += '  variability = ' + str(self.variability()) + '\n'
        string += '}\n'
        return string


def dissimilarity(clusters):
    if not all(isinstance(o, Cluster) for o in clusters):
        raise TypeError("Array must contain only Cluster class instances")

    return sum(map(lambda x: x.variability(), clusters))
