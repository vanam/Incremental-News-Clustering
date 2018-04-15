#!/usr/bin/env python3

import argparse
import itertools
import logging
import os
import csv
import re

from data.genuine.utils import check_lang

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == "__main__":
    """
    Remove articles which are not in ground truth file.
    """
    parser = argparse.ArgumentParser(description='Retain only data with specified language and GUIDs from list.')
    parser.add_argument('-l', '--lang', help='language not to purge', type=check_lang)
    parser.add_argument('-t, --test', dest='test', action='store_true',
                        help='will not delete files')
    parser.set_defaults(lang='en', test=False)
    args = parser.parse_args()

    processed_filename_pattern = re.compile('[0-9]{10}.[0-9]-[a-z]{2}-[0-9A-Fa-f]{32}-[0-9A-Fa-f]{32}.q.job.xml')

    dir_path = os.path.dirname(os.path.realpath(__file__))

    heldout_dir = os.path.join(dir_path, "heldout")
    test_dir = os.path.join(dir_path, "test")

    filename = os.path.join(dir_path, "2017-10-gold.csv")
    language = args.lang

    if not os.path.exists(filename):
        raise ValueError("File '%s' not found" % filename)

    ids = set()

    # Read valid GUIDs
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)

        # Skip header
        it = iter(reader)
        next(it)

        for row in it:
            # Skip documents in different languages
            if language is not None and language != row[1]:
                continue

            ids.add(row[0])

    counter = 0
    purged = 0
    for dirpath, subdirs, files in itertools.chain(os.walk(heldout_dir), os.walk(test_dir)):
        logging.info("Purging files in directory '%s'" % dirpath)

        for filename in files:
            # Construct old file path
            file_path = os.path.join(dirpath, filename)

            # Check if it is an article
            processed_result = processed_filename_pattern.match(filename)
            if processed_result is None:
                logging.warning("Skipping file '%s'." % file_path)
                continue

            metadata = filename.split("-")
            guid = metadata[2]
            counter += 1

            if guid not in ids:
                # logging.info("Purging '%s'" % os.path.join(dirpath, filename))
                purged += 1

                if not args.test:
                    os.remove(file_path)

    logging.info("%d files were purged, %d files remain." % (purged, counter - purged))


