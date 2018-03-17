#!/usr/bin/env python3

import logging
import os
import re
from time import strptime, mktime

from xml.dom import minidom

from pyexpat import ExpatError

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == "__main__":
    """
    Read all XML data files and add publication timestamp and language to their names.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))

    processed_filename_pattern = re.compile('[0-9]{10}.[0-9]-[a-z]{2}-[0-9A-Fa-f]{32}.q.job.xml')
    unprocessed_filename_pattern = re.compile('[0-9A-Fa-f]{32}.q.job.xml')

    # Add timestamp to file names
    counter = 0
    for dirpath, subdirs, files in os.walk(dir_path):
        logging.info("Renaming files in directory '%s'" % dirpath)

        for filename in files:
            # Construct old file path
            file_path = os.path.join(dirpath, filename)

            # Check if we already did process file
            unprocessed_result = unprocessed_filename_pattern.match(filename)
            processed_result = processed_filename_pattern.match(filename)
            if unprocessed_result is None and processed_result is None:
                logging.warning("Skipping file '%s'." % file_path)
                continue
            elif unprocessed_result is None:
                continue

            # Parse XML
            try:
                xmldoc = minidom.parse(file_path)
            except ExpatError:
                logging.warning("Could not parse XML, deleting file '%s'." % file_path)
                os.remove(file_path)
                continue

            # Parse publication date
            pub_dates = xmldoc.getElementsByTagName('pubDate')

            if len(pub_dates) == 0:
                logging.warning("Publication date not found in '%s', file skipped." % file_path)
                continue

            # Convert date to integer
            pub_date_raw = pub_dates[0].firstChild.nodeValue
            date_format = "%Y-%m-%d %H:%M:%S %Z"
            try:
                pub_date = strptime(pub_date_raw, date_format)
            except ValueError:
                logging.warning("Publication date has invalid format, '%s' given." % pub_date_raw)
                continue

            # Parse language
            language_raw = xmldoc.getElementsByTagName('iso:language')

            if len(language_raw) == 0:
                logging.warning("Language not found in '%s', file skipped." % file_path)
                continue
            language = language_raw[0].firstChild.nodeValue
            prefix = "%s-%s" % (mktime(pub_date), language)

            # Construct new file path
            new_file_path = os.path.join(dirpath, "%s-%s" % (prefix, filename))

            # Rename files
            # logging.info("Renaming '%s' to '%s" % (file_path, new_file_path))
            os.rename(file_path, new_file_path)
            counter += 1

    logging.info("%d files were renamed." % counter)
