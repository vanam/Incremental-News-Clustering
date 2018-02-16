import logging
import re

import os
from xml.dom import minidom

from gensim.corpora import TextCorpus


def get_docs_in_folder(root, language=None):
    if language is None:
        language = "[a-z]{2}"

    english_filename_pattern = re.compile("[0-9]{10}.[0-9]-%s-[0-9A-Fa-f]{32}.q.job.xml" % language)
    all_files = []
    for dirpath, subdirs, files in os.walk(root):
        for filename in files:
            # Check if we already did process file
            result = english_filename_pattern.match(filename)
            if result is None:
                continue

            # Construct file path
            file_path = os.path.join(dirpath, filename)

            # Store filename and file path
            all_files.append((filename, file_path))

    # Sort by filename
    all_files.sort(key=lambda x: x[0])
    logging.info("Found %d files." % len(all_files))

    return all_files


class NewsCorpus(TextCorpus):
    def __init__(self, input=None, dictionary=None, metadata=False, character_filters=None, tokenizer=None,
                 token_filters=None, language=None):
        self.language = language
        self.documents = get_docs_in_folder(input, language) if input is not None else []

        super().__init__(input, dictionary, metadata, character_filters, tokenizer, token_filters)

    def get_texts(self):
        data = self.getstream()
        for line, metadata in data:
            if self.metadata:
                yield self.preprocess_text(line), metadata
            else:
                yield self.preprocess_text(line)

    def getstream(self):
        for filename, file_path in self.documents:
            # Parse document
            xmldoc = minidom.parse(file_path)

            # Parse title
            title_raw = xmldoc.getElementsByTagName('title')

            if len(title_raw) == 0:
                logging.warning("Title not found in '%s', file skipped." % file_path)
                continue

            # Parse data
            title_raw = xmldoc.getElementsByTagName('title')
            text_raw = xmldoc.getElementsByTagName('emm:text')

            if len(title_raw) == 0 or len(text_raw) == 0:
                logging.warning("Title or text not found in '%s', file skipped." % file_path)
                continue

            title = title_raw[1].firstChild.nodeValue
            text = text_raw[0].firstChild.nodeValue

            # Get metadata from filename
            metadata = filename.split("-")
            metadata = (metadata[2][:-10], metadata[0], title)  # (docId, timestamp, title)

            yield "%s %s" % (title, text), metadata
