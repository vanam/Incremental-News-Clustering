#!/usr/bin/env python3

import logging
import os
from pathlib import Path

from gensim.corpora import Dictionary, MmCorpus
from sklearn.decomposition import IncrementalPCA

from clustering_system.clustering.DummyClustering import DummyClustering
from clustering_system.corpus.ArtificialCorpus import ArtificialCorpus
from clustering_system.corpus.MetaMmCorpusWrapper import MetaMmCorpusWrapper
from clustering_system.corpus.NewsCorpus import NewsCorpus
from clustering_system.corpus.SinglePassCorpusWrapper import SinglePassCorpusWrapper
from clustering_system.model.Identity import Identity
from clustering_system.visualization.Visualizer import Visualizer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == "__main__":
    # Constants
    K = 3     # Number of clusters
    size = 2  # Size of a feature vector

    # Current directory
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Useful directories
    data_dir = os.path.join(dir_path, "..", "data")
    temp_dir = os.path.join(dir_path, "..", "temp")
    temp_corpus_dir = os.path.join(temp_dir, 'corpus')
    temp_visualization_dir = os.path.join(temp_dir, 'visualization')

    # Make sure temp directories exist
    Path(temp_corpus_dir).mkdir(parents=True, exist_ok=True)
    Path(temp_visualization_dir).mkdir(parents=True, exist_ok=True)

    # Paths to data
    training_dir = os.path.join(data_dir, "genuine", "training")
    test_dir = os.path.join(data_dir, "genuine", "test")
    artificial_file = os.path.join(data_dir, "artificial", "06.dat")
    visualization_file = os.path.join(temp_visualization_dir, 'visualization.gexf')

    temp_dictionary_file = os.path.join(temp_corpus_dir, 'dictionary.dict')
    temp_training_corpus_file = os.path.join(temp_corpus_dir, 'training_corpus.mm')
    temp_test_corpus_file = os.path.join(temp_corpus_dir, 'test_corpus.mm')

    ########################
    # Initialization phase #
    ########################

    # Select model

    # Artificial data
    training_corpus = ArtificialCorpus(input=artificial_file, metadata=True)
    test_corpus = SinglePassCorpusWrapper(ArtificialCorpus(input=artificial_file, metadata=True))
    model = Identity()

    # News data
    # model = ...

    # Identity model does not have to be trained
    if not isinstance(model, Identity):
        # Check if we have already pre-processed the corpus
        if not os.path.exists(temp_training_corpus_file):
            # Load and pre-process corpus
            training_corpus = NewsCorpus(input=training_dir, metadata=True, language='en')

            # Serialize pre-processed corpus and dictionary to temp files
            # training_corpus.dictionary.save(temp_dictionary_file)
            MmCorpus.serialize(temp_training_corpus_file, training_corpus, metadata=True)

        # Load corpus and dictionary from temp file
        dictionary = Dictionary.load(temp_dictionary_file)
        training_corpus = MetaMmCorpusWrapper(temp_training_corpus_file)

    # Instead
    # training_corpus = ArtificialCorpus(input=artificial_file, metadata=True)

    # Select clustering method
    clustering = DummyClustering(K, size)

    ##############################
    # Online document clustering #
    ##############################

    test_corpus = ArtificialCorpus(input=artificial_file, metadata=True)

    # You cannot use Identity model on news data
    if not isinstance(model, Identity):
        # Check if we have already pre-processed the corpus
        if not os.path.exists(temp_test_corpus_file):
            # Load and pre-process corpus
            test_corpus = NewsCorpus(input=test_dir, dictionary=dictionary, metadata=True, language='en')

            # Serialize pre-processed corpus and dictionary to temp files
            MmCorpus.serialize(temp_test_corpus_file, test_corpus, metadata=True)

        test_corpus = MetaMmCorpusWrapper(temp_test_corpus_file)

    # Reduce dimension for visualization
    ipca = IncrementalPCA(n_components=2, batch_size=10)
    ipca.fit_transform([vec for vec, metadata in model[training_corpus]])

    # Init visualizer
    visualizer = Visualizer()

    # TODO Iterate over weeks instead of single documents
    for t, docs_metadata in enumerate([[_ for _ in test_corpus]]):
        docs, metadata = zip(*docs_metadata)
        ids = list(zip(*metadata))[0]

        # Update model
        model.update(docs)

        # Get vector representation
        docs = model[docs]

        # Cluster new data
        clustering.add_documents(ids, docs)
        clustering.update()

        # Reduce vector dimension for visualization
        ipca.partial_fit(docs)
        reduced_docs = ipca.transform(docs)

        # Visualization
        visualizer.add_documents(reduced_docs, metadata, t)

        for doc_id, cluster_id in clustering:
            visualizer.set_cluster_for_doc(t, doc_id, cluster_id)

    visualizer.save(visualization_file)
