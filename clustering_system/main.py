#!/usr/bin/env python3

import logging
import os
import sys
from enum import Enum
from pathlib import Path

import numpy as np
from gensim.corpora import Dictionary, MmCorpus
from sklearn.decomposition import IncrementalPCA

from clustering_system.clustering.DummyClustering import DummyClustering
from clustering_system.clustering.igmm.CrpClustering import CrpClustering
from clustering_system.clustering.igmm.DdCrpClustering import DdCrpClustering, logistic_decay
from clustering_system.clustering.mixture.GaussianMixtureABC import NormalInverseWishartPrior
from clustering_system.corpus.ArtificialCorpus import ArtificialCorpus
from clustering_system.corpus.BowNewsCorpus import BowNewsCorpus
from clustering_system.corpus.FolderAggregatedBowNewsCorpora import FolderAggregatedBowNewsCorpora
from clustering_system.corpus.FolderAggregatedLineNewsCorpora import FolderAggregatedLineNewsCorpora
from clustering_system.corpus.LineCorpus import LineCorpus
from clustering_system.corpus.LineNewsCorpus import LineNewsCorpus
from clustering_system.corpus.SinglePassCorpusWrapper import SinglePassCorpusWrapper
from clustering_system.corpus.SingletonCorpora import SingletonCorpora
from clustering_system.evaluator.RandomEvaluator import RandomEvaluator
from clustering_system.model.Doc2vec import Doc2vec
from clustering_system.model.Identity import Identity
from clustering_system.model.Lda import Lda
from clustering_system.model.Lsi import Lsi
from clustering_system.model.Random import Random
from clustering_system.visualization.ClusterVisualizer import ClusterVisualizer
from clustering_system.visualization.GraphVisualizer import GraphVisualizer
from clustering_system.visualization.LikelihoodVisualizer import LikelihoodVisualizer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Corpus(Enum):
    artificial = 0
    news = 1


class Model(Enum):
    identity = 0
    random = 1
    LSI = 2
    LDA = 3
    doc2vec = 4


if __name__ == "__main__":
    # import random
    # random.seed(0)

    # Only the identity model can be used with artificial corpus
    corpus_type = Corpus.artificial
    model_type = Model.identity

    # Only the LSI/LDA/doc2vec models can be used with news corpus
    # corpus_type = Corpus.news
    # model_type = Model.random
    # model_type = Model.LSI
    # model_type = Model.LDA
    # model_type = Model.doc2vec

    # Constants
    K = 3            # Number of clusters
    size = 2         # Size of a feature vector
    language = 'en'  # Language of news

    # Current directory
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Useful directories
    data_dir = os.path.join(dir_path, "..", "data")
    temp_dir = os.path.join(dir_path, "..", "temp")
    temp_corpus_dir = os.path.join(temp_dir, 'corpus')
    temp_model_dir = os.path.join(temp_dir, 'model')
    temp_visualization_dir = os.path.join(temp_dir, 'visualization')
    temp_cluster_visualization_dir = os.path.join(temp_visualization_dir, 'cluster')

    # Make sure temp directories exist
    Path(temp_corpus_dir).mkdir(parents=True, exist_ok=True)
    Path(temp_model_dir).mkdir(parents=True, exist_ok=True)
    Path(temp_visualization_dir).mkdir(parents=True, exist_ok=True)
    Path(temp_cluster_visualization_dir).mkdir(parents=True, exist_ok=True)

    # Paths to data
    training_dir = os.path.join(data_dir, "genuine", "training")
    test_dir = os.path.join(data_dir, "genuine", "test")
    artificial_file = os.path.join(data_dir, "artificial", "02.dat")
    graph_visualization_file = os.path.join(temp_visualization_dir, 'visualization.gexf')
    likelihood_visualization_file = os.path.join(temp_visualization_dir, 'clustering_likelihood.png')

    temp_dictionary_file = os.path.join(temp_corpus_dir, 'dictionary.dict')
    temp_training_mm_corpus_file = os.path.join(temp_corpus_dir, 'training_corpus.mm')
    temp_training_low_corpus_file = os.path.join(temp_corpus_dir, 'training_corpus.line')
    temp_test_corpus_file = os.path.join(temp_corpus_dir, 'test_corpus.mm')

    ########################
    # Initialization phase #
    ########################
    likelihood_visualizer = LikelihoodVisualizer()

    # Select clustering method

    # Dummy clustering
    clustering = DummyClustering(K, size)

    # ddCRP clustering
    v_0 = size + 1

    prior = NormalInverseWishartPrior(
        np.array([0, 0]),
        0.01,
        0.01*np.eye(size),
        v_0
    )

    # Decay function
    # a = 3  # 3 days
    #
    # def f(d: float):
    #     return logistic_decay(d, a)

    # Decay function for artificial data
    a = 1  # 1 day

    def f(d):
        # Hack distance to look like time
        d = np.math.hypot(d[0], d[1]) * 60*60*24
        return logistic_decay(d, a)

    clustering = DdCrpClustering(size, 0.0001, prior, 20, f, visualizer=likelihood_visualizer)
    clustering = CrpClustering(size, 0.01, prior, 20, visualizer=likelihood_visualizer)
    # clustering = GibbsClustering(K, size, 0.1, prior, 20, visualizer=likelihood_visualizer)

    if corpus_type == Corpus.artificial:
        # For artificial data initialize identity model
        training_corpus = ArtificialCorpus(input=artificial_file)
        test_corpora = SingletonCorpora(SinglePassCorpusWrapper(ArtificialCorpus(input=artificial_file, metadata=True)))
        model = Identity()
    else:
        # For LSI/LDA initialize bag of words corpus
        if model_type in [Model.random, Model.LSI, Model.LDA]:
            # Check if we have already pre-processed the corpus
            if not os.path.exists(temp_training_mm_corpus_file):
                # Load and pre-process corpus
                training_corpus = BowNewsCorpus(input=training_dir, metadata=True, language=language)

                # Serialize pre-processed corpus and dictionary to temp files
                training_corpus.dictionary.save(temp_dictionary_file)
                MmCorpus.serialize(temp_training_mm_corpus_file, training_corpus, id2word=training_corpus.dictionary, metadata=True)

            # Load corpus and dictionary from temp file
            dictionary = Dictionary.load(temp_dictionary_file)
            training_corpus = MmCorpus(temp_training_mm_corpus_file)

            # Initialize correct model
            if model_type == Model.random:
                model = Random(size=size)
            elif model_type == Model.LSI:
                model = Lsi(training_corpus, dictionary, temp_model_dir, size=size)
            elif model_type == Model.LDA:
                model = Lda(training_corpus, dictionary, temp_model_dir, size=size)
            else:
                logging.error("Unknown model type '%s'" % model_type)
                sys.exit(1)
        else:
            # For doc2vec initialize list of words corpus

            # Check if we have already pre-processed the corpus
            if not os.path.exists(temp_training_low_corpus_file):
                # Load and pre-process corpus
                training_corpus = LineNewsCorpus(input=training_dir, metadata=True, language=language)

                # Serialize pre-processed corpus and dictionary to temp files
                training_corpus.dictionary.save(temp_dictionary_file)
                LineCorpus.serialize(temp_training_low_corpus_file, training_corpus, training_corpus.dictionary, metadata=True)

            # Load corpus and dictionary from temp file
            dictionary = Dictionary.load(temp_dictionary_file)
            training_corpus = LineCorpus(temp_training_low_corpus_file)

            model = Doc2vec(training_corpus, temp_model_dir, size=size)

        # Save trained model to file(s)
        model.save(temp_model_dir)

        # Load test corpora
        if model_type == Model.doc2vec:
            test_corpora = FolderAggregatedLineNewsCorpora(test_dir, temp_dir, dictionary, language=language)
        else:
            test_corpora = FolderAggregatedBowNewsCorpora(test_dir, temp_dir, dictionary, language=language)

    ##############################
    # Online document clustering #
    ##############################

    # Reduce dimension for visualization
    ipca = IncrementalPCA(n_components=2, batch_size=10)
    ipca.fit_transform([vec for vec in model[training_corpus]])

    # Init visualizers
    graph_visualizer = GraphVisualizer()
    cluster_visualizer = ClusterVisualizer()

    # Init evaluator
    evaluator = RandomEvaluator(K, test_corpora)

    # Iterate over test corpora
    for t, docs_metadata in enumerate(test_corpora):
        logging.info("Testing corpus #%d." % t)

        docs, metadata = zip(*docs_metadata)

        # Update model
        model.update(docs)

        # Get vector representation
        docs = model[docs]

        # Cluster new data
        clustering.add_documents(docs, metadata)
        clustering.update()

        # Reduce vector dimension for visualization
        ipca.partial_fit(docs)
        reduced_docs = ipca.transform(docs) if corpus_type == Corpus.artificial else docs

        # Visualization
        graph_visualizer.add_documents(reduced_docs, metadata, t)

        ids_clusters = []

        # Iterate over clustered documents
        for doc_id, cluster_id, *rest in clustering:
            ids_clusters.append((doc_id, cluster_id))

            linked_doc_id = rest[0] if rest else None
            graph_visualizer.set_cluster_for_doc(t, doc_id, cluster_id, linked_doc_id=linked_doc_id)

        # Visualize clusters
        k, _, mean, covariance, X = clustering.parameters

        # Reduce vector dimension for visualization
        # TODO mean and cov should be reduced as well
        reduced_X = []
        for i in range(k):
            reduced_X_k = ipca.transform(X[i]) if corpus_type != Corpus.artificial else X[i]
            reduced_X.append(reduced_X_k)

        cluster_visualizer.save(
            os.path.join(temp_cluster_visualization_dir, '{:05d}.png'.format(t)),
            k,
            mean,
            covariance,
            reduced_X
        )

        evaluator.evaluate(t, ids_clusters, clustering.aic, clustering.bic, clustering.likelihood)

    # Store evaluation and visualization
    logging.info("Storing evaluation")
    evaluator.save(temp_visualization_dir)

    logging.info("Generating visualization")
    graph_visualizer.save(graph_visualization_file)
    likelihood_visualizer.save(likelihood_visualization_file)
