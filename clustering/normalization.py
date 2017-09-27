import numpy as np


def mean_normalization(data):
    """Subtract mean from each vector"""
    return data - np.mean(data, axis=0)


def max_scaling(data):
    """Divide each vector compound by its maximum"""
    return data / np.max(data, axis=0)


def std_scaling(data):
    """Divide each vector compound by its standard deviation"""
    return data / np.std(data, axis=0)
