import logging


class SinglePassCorpusWrapper:

    def __init__(self, corpus):
        self.corpus = corpus
        self.can_iterate = True

    def __iter__(self):
        if self.can_iterate:
            for doc in self.corpus:
                yield doc

            self.can_iterate = False
        else:
            logging.error("Cannot iterate corpus again.")