#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import date
from time import ctime

import numpy as np

from data.genuine.utils import check_dir

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == "__main__":
    """
    Visualize histogram of articles in a week.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(description='Display histogram of news by days.')
    parser.add_argument('dir', help='directory', type=lambda v: check_dir(dir_path, v))
    parser.add_argument('-s', '--split', dest='split', action='store_true', help='split news in a folder by days')
    args = parser.parse_args()

    # Skip top level directory
    dir_iter = iter(os.walk(args.dir))
    next(dir_iter)

    for dirpath, subdirs, files in dir_iter:
        logging.info("Visualizing directory '%s'" % dirpath)

        docs_by_day = defaultdict(list)
        days = []
        for filename in files:
            # Get metadata from filename
            metadata = filename.split("-")
            timestamp = float(metadata[0])
            date = date.fromtimestamp(timestamp)
            # print(ctime(timestamp))
            # print(date.day)
            days.append(date.day)
            docs_by_day[date.day].append(filename)

        print(np.bincount(days, minlength=31))

        counter = Counter(days)
        days_counts = sorted(dict(counter).items(), key=lambda x: x[0])
        print(days_counts)
        print("")

        if args.split:
            counter = 0

            if len(days_counts) == 0:
                logging.warning("Nothing to split")
                continue

            days, _ = zip(*days_counts)

            if len(days) <= 1:
                logging.warning("Nothing to split")
                continue

            for i, d in enumerate(days):
                print(len(docs_by_day[d]))
                new_dir = os.path.join(dirpath, '{:02d}'.format(i))
                print(new_dir)

                # Create directory
                os.makedirs(new_dir, exist_ok=True)

                for filename in docs_by_day[d]:
                    old_filename = os.path.join(dirpath, filename)
                    new_filename = os.path.join(new_dir, filename)
                    os.rename(old_filename, new_filename)
                    counter += 1

            logging.info("Moved %d files." % counter)
