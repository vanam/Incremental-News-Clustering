#!/usr/bin/env python3

import logging
import os
from pathlib import Path

from gensim.corpora import Dictionary, MmCorpus
from sklearn.decomposition import IncrementalPCA

from clustering_system.clustering.DummyClustering import DummyClustering
from clustering_system.corpus.ArtificialCorpus import ArtificialCorpus
from clustering_system.corpus.FolderAggregatedCorpora import FolderAggregatedCorpora
from clustering_system.corpus.MetaMmCorpusWrapper import MetaMmCorpusWrapper
from clustering_system.corpus.NewsCorpus import NewsCorpus
from clustering_system.corpus.SinglePassCorpusWrapper import SinglePassCorpusWrapper
from clustering_system.corpus.SingletonCorpora import SingletonCorpora
from clustering_system.evaluator.RandomEvaluator import RandomEvaluator
from clustering_system.model.Doc2vec import Doc2vec
from clustering_system.model.Identity import Identity
from clustering_system.model.Lda import Lda
from clustering_system.model.Lsi import Lsi
from clustering_system.visualization.Visualizer import Visualizer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == "__main__":
    # Constants
    # ARTIFICIAL = True
    ARTIFICIAL = False
    K = 3     # Number of clusters
    size = 2  # Size of a feature vector
    language = 'en'

    # Current directory
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Useful directories
    data_dir = os.path.join(dir_path, "..", "data")
    temp_dir = os.path.join(dir_path, "..", "temp")
    temp_corpus_dir = os.path.join(temp_dir, 'corpus')
    temp_model_dir = os.path.join(temp_dir, 'model')
    temp_visualization_dir = os.path.join(temp_dir, 'visualization')

    # Make sure temp directories exist
    Path(temp_corpus_dir).mkdir(parents=True, exist_ok=True)
    Path(temp_model_dir).mkdir(parents=True, exist_ok=True)
    Path(temp_visualization_dir).mkdir(parents=True, exist_ok=True)

    # Paths to data
    training_dir = os.path.join(data_dir, "genuine", "training")
    test_dir = os.path.join(data_dir, "genuine", "test")
    artificial_file = os.path.join(data_dir, "artificial", "02.dat")
    visualization_file = os.path.join(temp_visualization_dir, 'visualization.gexf')

    temp_dictionary_file = os.path.join(temp_corpus_dir, 'dictionary.dict')
    temp_training_corpus_file = os.path.join(temp_corpus_dir, 'training_corpus.mm')
    temp_test_corpus_file = os.path.join(temp_corpus_dir, 'test_corpus.mm')

    ########################
    # Initialization phase #
    ########################

    # Select clustering method
    clustering = DummyClustering(K, size)

    # If artificial data
    if ARTIFICIAL:
        training_corpus = ArtificialCorpus(input=artificial_file)
        test_corpora = SingletonCorpora(SinglePassCorpusWrapper(ArtificialCorpus(input=artificial_file, metadata=True)))
        model = Identity()
    else:  # News data
        # Check if we have already pre-processed the corpus
        if not os.path.exists(temp_training_corpus_file):
            # Load and pre-process corpus
            training_corpus = NewsCorpus(input=training_dir, metadata=True, language=language)

            # Serialize pre-processed corpus and dictionary to temp files
            training_corpus.dictionary.save(temp_dictionary_file)
            MmCorpus.serialize(temp_training_corpus_file, training_corpus, metadata=True)

        # Load corpus and dictionary from temp file
        dictionary = Dictionary.load(temp_dictionary_file)
        training_corpus = MmCorpus(temp_training_corpus_file)

        # Select model
        # model = Lsi(training_corpus, dictionary, temp_model_dir, size=size)
        model = Lda(training_corpus, dictionary, temp_model_dir, size=size)
        # model = Doc2vec()

        # Save trained model to file(s)
        model.save(temp_model_dir)

        # Test corpora
        test_corpora = FolderAggregatedCorpora(test_dir, temp_dir, dictionary, language=language)

    ##############################
    # Online document clustering #
    ##############################

    # Reduce dimension for visualization
    ipca = IncrementalPCA(n_components=2, batch_size=10)
    ipca.fit_transform([vec for vec in model[training_corpus]])

    # Init visualizer
    visualizer = Visualizer()

    # Init evaluator
    evaluator = RandomEvaluator(K, test_corpora)

    # Iterate over test corpora
    for t, docs_metadata in enumerate(test_corpora):
        logging.info("Testing corpus #%d." % t)

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

        ids_clusters = []

        for doc_id, cluster_id in clustering:
            ids_clusters.append((doc_id, cluster_id))
            visualizer.set_cluster_for_doc(t, doc_id, cluster_id)

        evaluator.evaluate(t, ids_clusters, clustering.X, clustering.log_likelihood)

    # Store evaluation and visualization
    logging.info("Storing evaluation")
    evaluator.save(temp_visualization_dir)

    logging.info("Generating visualization")
    visualizer.save(visualization_file)
