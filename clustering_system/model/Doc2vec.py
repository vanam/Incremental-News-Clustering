import itertools
import os

from gensim.models import Doc2Vec as D2v
from gensim.models.doc2vec import TaggedDocument

from clustering_system.model.ModelABC import ModelABC


def is_doc2vec_corpus(obj):
    """Check whether `obj` is a corpus.

    Parameters
    ----------
    obj : object
        Something `iterable of iterable` that contains str.

    Return
    ------
    (bool, object)
        Pair of (is_corpus, `obj`), is_corpus True if `obj` is corpus.

    Warnings
    --------
    An "empty" corpus (empty input sequence) is ambiguous, so in this case
    the result is forcefully defined as (False, `obj`).
    """
    try:
        if hasattr(obj, 'next') or hasattr(obj, '__next__'):
            # the input is an iterator object, meaning once we call next()
            # that element could be gone forever. we must be careful to put
            # whatever we retrieve back again
            doc1 = next(obj)
            obj = itertools.chain([doc1], obj)
        else:
            doc1 = next(iter(obj))  # empty corpus is resolved to False here

        # Document is represented by a list of words, not a string
        if type(doc1) is str:
            return False, obj

        # if obj is not list of strings, it resolves to False here
        string = next(iter(doc1))

        if type(string) is not str:
            return False, obj
    except Exception:
        return False, obj
    return True, obj


class Doc2vec(ModelABC):

    def __init__(self, corpus=None, size: int = 200, epochs: int = 10, d2v_filename: str = None):
        super().__init__(size)

        # Check if we have already trained the doc2vec model
        if d2v_filename is not None and os.path.exists(d2v_filename):
            self.d2v = D2v.load(d2v_filename)
        else:
            if corpus is None:
                raise ValueError("Corpus must be provided to train doc2vec")

            class TaggedLineSentence:
                def __init__(self, documents):
                    self.documents = documents
                    self.len = len(documents)

                def __iter__(self):
                    for uid, line in enumerate(self.documents):
                        yield TaggedDocument(words=line, tags=['%d' % uid])

                def __len__(self):
                    return self.len

            # One sentence is a document
            sentences = TaggedLineSentence(corpus)

            self.d2v = D2v(vector_size=size)
            self.d2v.build_vocab(sentences)
            self.d2v.train(sentences, total_examples=len(sentences), epochs=epochs)

    def update(self, documents):
        """
        Does not support updating.
        :param documents:
        """
        pass

    def save(self, filename: str):
        self.d2v.save(filename)

    def _get_vector_representation(self, items):
        raise NotImplementedError("This method is not implemented intentionally.")

    def __getitem__(self, items):
        is_corpus, items = is_doc2vec_corpus(items)

        if not is_corpus:
            return self.d2v.infer_vector(items)
        else:
            return list(map(self.d2v.infer_vector, items))
