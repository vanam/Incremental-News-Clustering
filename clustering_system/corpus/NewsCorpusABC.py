import logging
import os
import re
from abc import abstractmethod
from itertools import chain
from xml.dom import minidom

from gensim.corpora import Dictionary
from gensim.corpora.textcorpus import lower_to_unicode, strip_multiple_whitespaces, remove_short, remove_stopwords
from gensim.interfaces import CorpusABC
from gensim.parsing import strip_punctuation, PorterStemmer
from gensim.utils import deaccent, simple_tokenize


class NewsCorpusABC(CorpusABC):
    """An abstract class representing a news corpus"""

    def __init__(self, input=None, dictionary=None, metadata=False, character_filters=None,
                 tokenizer=None, token_filters=None, language=None, stem=True):
        """
        Args:
            input (str): path to top-level directory to traverse for corpus documents.
            dictionary (Dictionary): if a dictionary is provided, it will not be updated
                with the given corpus on initialization. If none is provided, a new dictionary
                will be built for the given corpus. If no corpus is given, the dictionary will
                remain uninitialized.
            metadata (bool): True to yield metadata with each document, else False (default).
            character_filters (iterable of callable): each will be applied to the text of each
                document in order, and should return a single string with the modified text.
                For Python 2, the original text will not be unicode, so it may be useful to
                convert to unicode as the first character filter. The default character filters
                lowercase, convert to unicode (strict utf8), perform ASCII-folding, then collapse
                multiple whitespaces.
            tokenizer (callable): takes as input the document text, preprocessed by all filters
                in `character_filters`; should return an iterable of tokens (strings).
            token_filters (iterable of callable): each will be applied to the iterable of tokens
                in order, and should return another iterable of tokens. These filters can add,
                remove, or replace tokens, or do nothing at all. The default token filters
                remove tokens less than 3 characters long and remove stopwords using the list
                in `gensim.parsing.preprocessing.STOPWORDS`.
        """
        self.input = input
        self.metadata = metadata
        self.language = language
        self.documents = self.get_news_in_folder(input, language) if input is not None else []

        self.character_filters = character_filters
        if self.character_filters is None:
            self.character_filters = [lower_to_unicode, deaccent, strip_punctuation, strip_multiple_whitespaces]

        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = simple_tokenize

        self.token_filters = token_filters
        if self.token_filters is None:
            self.token_filters = [remove_short, remove_stopwords]

        self.stem = stem
        self.length = None
        self.dictionary = None
        self.init_dictionary(dictionary)

    def init_dictionary(self, dictionary):
        """If `dictionary` is None, initialize to an empty Dictionary, and then if there
        is an `input` for the corpus, add all documents from that `input`. If the
        `dictionary` is already initialized, simply set it as the corpus's `dictionary`.
        """
        self.dictionary = dictionary if dictionary is not None else Dictionary()
        if self.input is not None:
            if dictionary is None:
                logging.info("Initializing dictionary")
                metadata_setting = self.metadata
                self.metadata = False
                self.dictionary.add_documents(self.get_texts())
                self.metadata = metadata_setting
            else:
                logging.info("Input stream provided but dictionary already initialized")
        else:
            logging.warning("No input document stream provided; assuming dictionary will be initialized some other way.")

    @staticmethod
    def get_news_in_folder(root, language=None):
        """Find all news articles in a folder."""
        if language is None:
            language = "[a-z]{2}"

        filename_pattern = re.compile("[0-9]{10}.[0-9]-%s-[0-9A-Fa-f]{32}-[0-9A-Fa-f]{32}.q.job.xml" % language)
        all_files = []

        walk_iter = iter(os.walk(root)) if isinstance(root, str) else chain.from_iterable(
            os.walk(path) for path in root)

        for dirpath, subdirs, files in walk_iter:
            for filename in files:
                # Check if we already did process file
                result = filename_pattern.match(filename)
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

    def get_texts(self):
        """Yield preprocessed documents and its metadata if desired"""
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

            if len(title_raw) != 2:
                logging.warning("Title not found in '%s', file skipped." % file_path)
                continue

            title = title_raw[1].firstChild.nodeValue

            # Parse data
            text_raw = xmldoc.getElementsByTagName('emm:text')

            if len(text_raw) == 0:
                logging.warning("Text not found in '%s', file skipped." % file_path)
                continue

            text = text_raw[0].firstChild.nodeValue

            # Get metadata from filename
            metadata = filename.split("-")
            metadata = (metadata[2], float(metadata[0]), title)  # (docId, timestamp, title)

            yield "%s %s" % (title, text), metadata

    def preprocess_text(self, text):
        """Apply preprocessing to a single text document. This should perform tokenization
        in addition to any other desired preprocessing steps.

        Args:
            text (str): document text read from plain-text file.

        Returns:
            iterable of str: tokens produced from `text` as a result of preprocessing.
        """
        for character_filter in self.character_filters:
            text = character_filter(text)

        tokens = self.tokenizer(text)
        for token_filter in self.token_filters:
            tokens = token_filter(tokens)

        if self.stem:
            p = PorterStemmer()
            tokens = [p.stem(token) for token in tokens]

        return tokens

    def step_through_preprocess(self, text):
        """Yield tuples of functions and their output for each stage of preprocessing.
        This is useful for debugging issues with the corpus preprocessing pipeline.
        """
        for character_filter in self.character_filters:
            text = character_filter(text)
            yield (character_filter, text)

        tokens = self.tokenizer(text)
        yield (self.tokenizer, tokens)

        for token_filter in self.token_filters:
            yield (token_filter, token_filter(tokens))

    @abstractmethod
    def _encode(self, text):
        pass

    def __iter__(self):
        """The function that defines a corpus.

        Iterating over the corpus must yield sparse vectors, one for each document.
        """
        if self.metadata:
            for text, metadata in self.get_texts():
                yield self._encode(text), metadata
        else:
            for text in self.get_texts():
                yield self._encode(text)

    def __len__(self):
        if self.length is None:
            # cache the corpus length
            self.length = sum(1 for _ in self.getstream())
        return self.length
