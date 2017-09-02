import argparse
import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np

from clustering.readData import read_data

COLORS = "bgrcmykw"
POINTS = ".,ov^<>12348spP*hH+xXDd|_"


def plot(file):
    # Calculate all possible point-color combinations
    point_colors = list(map(lambda p: "".join(p), itertools.product(POINTS, COLORS)))

    # Read data file
    data = read_data(file)

    for i in range(len(data)):
        if len(data[i]) is 0:
            continue

        dimension = len(data[i][0])

        if dimension is 1:
            x = np.transpose(data[i])
            plt.plot(x, 0, point_colors[i], alpha=0.5)
        elif dimension is 2:
            x, y = np.transpose(data[i])
            plt.plot(x, y, point_colors[i], alpha=0.5)
        else:
            print("High dimensions are not supported.", file=sys.stderr)
            continue

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot data.')

    parser.add_argument('file', metavar='file', type=argparse.FileType('rb'),
                        help='a data file')

    args = parser.parse_args()
    plot(args.file)
