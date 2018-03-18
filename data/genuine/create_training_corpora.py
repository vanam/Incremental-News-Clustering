#!/usr/bin/env python3

import argparse
import logging
import os

from gensim.corpora import MmCorpus, Dictionary

from clustering_system.corpus.BowNewsCorpus import BowNewsCorpus
from clustering_system.corpus.LineCorpus import LineCorpus
from clustering_system.corpus.LineNewsCorpus import LineNewsCorpus
from data.genuine.utils import check_lang, check_dir

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == "__main__":
    """
    Create a dictionary, BoW corpus and LoW corpus for training.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dictionary_file = os.path.join(dir_path, 'dictionary.dict')
    bow_corpus_file = os.path.join(dir_path, 'training_corpus.mm')
    low_corpus_file = os.path.join(dir_path, 'training_corpus.line')

    parser = argparse.ArgumentParser(description='Plot data.')
    parser.add_argument('dirs', nargs='+', help='directories containing data', type=lambda v: check_dir(dir_path, v))
    parser.add_argument('-l', '--lang', help='language of data', type=check_lang)
    parser.add_argument('-f, --filter', dest='filter', action='store_true',
                        help='remove unfrequent and too frequent words from dictionary')
    parser.set_defaults(lang='en', filter=True)
    args = parser.parse_args()

    logging.info("Creating training corpora from data in directories: %s" % args.dirs)
    logging.info("Language: %s" % args.lang)

    dictionary = None if not os.path.exists(dictionary_file) else Dictionary.load(dictionary_file)

    # Create BoW corpus and dictionary
    logging.info("Creating BoW corpus...")
    training_corpus = BowNewsCorpus(input=args.dirs, dictionary=dictionary, language=args.lang)

    dictionary = training_corpus.dictionary
    if args.filter:
        logging.info("Filtering dictionary...")
        # https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1756-8765.2010.01108.x
        dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=2000000)
        dictionary.compactify()

    # Serialize pre-processed BoW corpus and dictionary to files
    logging.info("Saving dictionary to '%s'" % dictionary_file)
    dictionary.save(dictionary_file)
    logging.info("Dictionary size: %d MB" % (os.path.getsize(dictionary_file) >> 20))

    logging.info("Saving BoW corpus to '%s'" % bow_corpus_file)
    MmCorpus.serialize(bow_corpus_file, training_corpus, id2word=dictionary)
    logging.info("BoW corpus size: %d MB" % (os.path.getsize(bow_corpus_file) >> 20))

    # Create LoW corpus
    logging.info("Creating LoW corpus...")
    training_corpus = LineNewsCorpus(input=args.dirs, dictionary=dictionary, language=args.lang)
    logging.info("Saving LoW corpus to '%s'" % low_corpus_file)
    LineCorpus.serialize(low_corpus_file, training_corpus, dictionary)
    logging.info("LoW corpus size: %d MB" % (os.path.getsize(low_corpus_file) >> 20))
