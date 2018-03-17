#!/usr/bin/env python3

import argparse
import logging
import os
import re

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == "__main__":
    """
    Retain only XML data files in specified language. Other data files will be deleted.
    """
    def check_lang(value):
        lang_patter = re.compile("[a-z]{2}")

        if len(value) != 2 or lang_patter.match(value) is None:
            raise argparse.ArgumentTypeError("%s is an invalid language" % value)

        return value

    parser = argparse.ArgumentParser(description='Plot data.')

    parser.add_argument('lang', help='language to retain', type=check_lang)
    parser.add_argument('-t, --test', dest='test', action='store_true',
                        help='will not delete files')
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))

    filename_pattern = re.compile('[0-9]{10}.[0-9]-[a-z]{2}-[0-9A-Fa-f]{32}.q.job.xml')
    filename_with_lang_pattern = re.compile('[0-9]{10}.[0-9]-%s-[0-9A-Fa-f]{32}.q.job.xml' % args.lang)

    if args.test:
        logging.info("Running in test mode.")

    not_match_count = 0
    match_count = 0
    for dirpath, subdirs, files in os.walk(dir_path):
        logging.info("Retaining files in directory '%s'" % dirpath)

        for filename in files:
            filename_result = filename_pattern.match(filename)
            filename_with_lang_result = filename_with_lang_pattern.match(filename)

            if filename_with_lang_result is not None:
                # Retain file
                match_count += 1
                continue
            elif filename_result is not None:
                # Delete file
                not_match_count += 1

                # Construct file path
                file_path = os.path.join(dirpath, filename)
                # logging.info("Deleting %s" % file_path)

                if not args.test:
                    os.remove(file_path)

    logging.info("%d files with language '%s' were retained." % (match_count, args.lang))
    logging.info("%d files were deleted." % not_match_count)
