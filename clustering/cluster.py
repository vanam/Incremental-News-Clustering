import numpy as np
from scipy.spatial.distance import cdist


class Cluster:
    def __init__(self, data, metric='euclidean'):
        if len(data) is 0:
            raise ValueError("Cluster must contain data")
        self.data = np.array(data)
        self.metric = metric
        self.centroid = self.compute_centroid()

    def compute_centroid(self):
        return list(map(lambda d: d / self.size(), np.sum(self.data, axis=0)))

    def distance_to_centroid(self, point):
        return cdist(np.array([self.centroid]), np.array([point]), metric=self.metric)[0][0]

    def members(self):
        for d in self.data:
            yield d

    def size(self):
        return len(self.data)

    def update(self, data):
        if len(data) is 0:
            raise ValueError("Cluster must contain data")

        old_centroid = self.centroid
        self.data = np.array(data)
        self.centroid = self.compute_centroid()

        return self.distance_to_centroid(old_centroid)

    def variability(self):
        return sum(np.square(cdist(np.array([self.centroid]), self.data, metric=self.metric))[0])

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string = 'Cluster {\n'
        string += '  centroid    = ' + str(self.centroid) + ',\n'
        string += '  data        = ' + str(self.data) + ',\n'
        string += '  size        = ' + str(self.size()) + '\n'
        string += '  variability = ' + str(self.variability()) + '\n'
        string += '}'
        return string


def dissimilarity(clusters):
    if not all(isinstance(o, Cluster) for o in clusters):
        raise TypeError("Array must contain only Cluster class instances")

    return sum(map(lambda x: x.variability(), clusters))
