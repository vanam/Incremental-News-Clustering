import logging
from typing import Iterable


class SinglePassCorpusWrapper:
    """Ensure the corpus is traversed exactly once."""

    def __init__(self, corpus: Iterable):
        """
        :param corpus: An iterable corpus
        """
        self.corpus = corpus
        self.can_iterate = True

    def __iter__(self):
        """Iterate exactly once through the corpus"""
        if self.can_iterate:
            for doc in self.corpus:
                yield doc

            self.can_iterate = False
        else:
            logging.error("Cannot iterate corpus again.")
