import numpy as np


def mean_normalization(data):
    """Subtract mean from each vector"""
    mean = np.mean(data, axis=0)
    return data - mean, mean


def mean_denormalization(data, mean):
    """Add mean to each vector"""
    return data + np.array(mean)


def max_scaling(data):
    """Divide each vector compound by its maximum"""
    mx = np.max(data, axis=0)
    return data / mx, mx


def max_descaling(data, mx):
    """Multiply each vector compound by its maximum"""
    return data * np.array(mx)


def std_scaling(data):
    """Divide each vector compound by its standard deviation"""
    std = np.std(data, axis=0)
    return data / std, std


def std_descaling(data, std):
    """Multiply each vector compound by its standard deviation"""
    return data * np.array(std)
