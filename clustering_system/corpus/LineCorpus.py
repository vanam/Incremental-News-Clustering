import logging

from gensim import utils
from gensim.interfaces import CorpusABC
from gensim.utils import to_utf8, file_or_filename
from smart_open import smart_open


class LineCorpus(CorpusABC):

    def __init__(self, input):
        self.input = input
        self.length = sum(1 for _ in file_or_filename(self.input))

    def __iter__(self):
        with file_or_filename(self.input) as file:
            for line in file.read().splitlines():
                yield [str(byte_word, 'utf-8') for byte_word in line.split()]

    def __len__(self):
        return self.length

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, progress_cnt=1000, metadata=False):
        logging.info("storing corpus in Line format to %s", fname)

        def word_id2word(word_id):
            try:
                return id2word[word_id]
            except KeyError:
                return ""

        with smart_open(fname, 'wb') as f:
            if metadata:
                docno2metadata = {}

            for docno, doc in enumerate(corpus):
                if metadata:
                    doc, data = doc
                    docno2metadata[docno] = data

                if docno % progress_cnt == 0:
                    logging.info("PROGRESS: saving document #%i", docno)

                fmt = ' '.join(map(word_id2word, doc))

                f.write(to_utf8("%s\n" % fmt))

            if metadata:
                utils.pickle(docno2metadata, fname + '.metadata.cpickle')

    @classmethod
    def serialize(serializer, fname, corpus, id2word, progress_cnt=None, metadata=False):

        kwargs = {'metadata': metadata}
        if progress_cnt is not None:
            kwargs['progress_cnt'] = progress_cnt

        serializer.save_corpus(fname, corpus, id2word, **kwargs)
