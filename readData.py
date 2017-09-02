import argparse

import numpy as np


def read_data(file):
    return np.load(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read binary data.')

    parser.add_argument('file', metavar='file', type=argparse.FileType('rb'),
                        help='a data file')

    args = parser.parse_args()
    print(read_data(args.file))
