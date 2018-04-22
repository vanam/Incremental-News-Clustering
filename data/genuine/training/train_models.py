#!/usr/bin/env python3

import argparse
import logging
import os
from enum import Enum
from multiprocessing.pool import Pool

import itertools

from gensim.corpora import Dictionary, MmCorpus

from clustering_system.corpus.LineCorpus import LineCorpus
from clustering_system.model.Doc2vec import Doc2vec
from clustering_system.model.Lda import Lda
from clustering_system.model.Lsa import Lsa
from data.genuine.utils import check_uint

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(process)s : %(message)s', level=logging.INFO)

dir_path = os.path.dirname(os.path.realpath(__file__))
dictionary_file = os.path.join(dir_path, 'dictionary.dict')
bow_corpus_file = os.path.join(dir_path, 'training_corpus.mm')
low_corpus_file = os.path.join(dir_path, 'training_corpus.line')


class Model(Enum):
    LSA = 2
    LDA = 3
    doc2vec = 4


def train(model_type: Model, dimension: int):
    if not os.path.exists(dictionary_file):
        logging.warning("Missing dictionary file '%s'" % dictionary_file)
        return

    if not os.path.exists(bow_corpus_file):
        logging.warning("Missing BoW corpus file '%s'" % bow_corpus_file)
        return

    if not os.path.exists(low_corpus_file):
        logging.warning("Missing LoW corpus file '%s'" % low_corpus_file)
        return

    dictionary = Dictionary.load(dictionary_file)
    training_corpus = MmCorpus(bow_corpus_file) if model_type in [Model.LSA, Model.LDA] else LineCorpus(low_corpus_file)

    logging.info("Training model %s with dimension %d..." % (model_type, dimension))
    if model_type == Model.LSA:
        model_file = os.path.join(dir_path, 'model_%d.lsa' % dimension)

        model = Lsa(dictionary, corpus=training_corpus, size=dimension)
    elif model_type == Model.LDA:
        model_file = os.path.join(dir_path, 'model_%d.lda' % dimension)

        model = Lda(dictionary, corpus=training_corpus, size=dimension)
    elif model_type == Model.doc2vec:
        model_file = os.path.join(dir_path, 'model_%d.d2v' % dimension)

        model = Doc2vec(corpus=training_corpus, size=dimension)
    else:
        logging.error("Unknown model type '%s" % model_type)
        return

    logging.info("Saving trained model %s with dimension %d..." % (model_type, dimension))
    model.save(model_file)


if __name__ == "__main__":
    """
    Train LSA, LDA and doc2vec models for selected dimensions.
    """
    parser = argparse.ArgumentParser(description='Train LSA, LDA and doc2vec models.')
    parser.add_argument('dim', nargs='*', help='trained vector dimension', type=check_uint)
    parser.add_argument('-p', '--proc', help='number of processes', type=check_uint)
    parser.set_defaults(dim=[100, 200, 300], proc=4)
    args = parser.parse_args()

    with Pool(processes=args.proc) as pool:
        pool.starmap(train, itertools.product(Model, args.dim))
