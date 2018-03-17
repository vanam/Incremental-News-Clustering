#!/usr/bin/env python3

import argparse
import logging
import os
from collections import Counter
from datetime import date
from time import ctime

import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == "__main__":
    """
    Visualize histogram of articles in a week.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))

    def check_dir(value):
        directory = os.path.join(dir_path, value)

        if not os.path.isdir(directory):
            raise argparse.ArgumentTypeError("%s directory not found" % value)

        return directory

    parser = argparse.ArgumentParser(description='Plot data.')

    parser.add_argument('dir', help='directory', type=check_dir)
    args = parser.parse_args()

    # Skip top level directory
    dir_iter = iter(os.walk(args.dir))
    next(dir_iter)

    for dirpath, subdirs, files in dir_iter:
        logging.info("Visualizing directory '%s'" % dirpath)

        days = []
        for filename in files:
            # Get metadata from filename
            metadata = filename.split("-")
            timestamp = float(metadata[0])
            date = date.fromtimestamp(timestamp)
            # print(ctime(timestamp))
            # print(date.day)
            days.append(date.day)

        print(np.bincount(days, minlength=31))

        counter = Counter(days)
        print(sorted(dict(counter).items(), key=lambda x: x[0]))
        print("")
