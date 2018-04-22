#!/usr/bin/env python3

import argparse
import logging
import itertools
import os
import sys
from enum import Enum
from pathlib import Path

import numpy as np
from gensim.corpora import Dictionary, MmCorpus
from sklearn.decomposition import IncrementalPCA

from clustering_system.clustering.DummyClustering import DummyClustering
from clustering_system.clustering.bgmm.BgmmClustering import BgmmClustering
from clustering_system.clustering.igmm.CrpClustering import CrpClustering
from clustering_system.clustering.igmm.DdCrpClustering import DdCrpClustering, logistic_decay, exponential_decay
from clustering_system.clustering.mixture.GaussianMixtureABC import NormalInverseWishartPrior
from clustering_system.corpus.ArtificialCorpus import ArtificialCorpus
from clustering_system.corpus.FolderAggregatedBowNewsCorpora import FolderAggregatedBowNewsCorpora
from clustering_system.corpus.FolderAggregatedLineNewsCorpora import FolderAggregatedLineNewsCorpora
from clustering_system.corpus.LineCorpus import LineCorpus
from clustering_system.corpus.SinglePassCorpusWrapper import SinglePassCorpusWrapper
from clustering_system.corpus.SingletonCorpora import SingletonCorpora
from clustering_system.evaluator.RandomEvaluator import RandomEvaluator
from clustering_system.model.Doc2vec import Doc2vec
from clustering_system.model.Identity import Identity
from clustering_system.model.Lda import Lda
from clustering_system.model.Lsa import Lsa
from clustering_system.model.Random import Random
from clustering_system.visualization.ClusterVisualizer import ClusterVisualizer
from clustering_system.visualization.GraphVisualizer import GraphVisualizer
from clustering_system.visualization.LikelihoodVisualizer import LikelihoodVisualizer
from evaluator.Evaluator import Evaluator

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Corpus(Enum):
    artificial = 0
    news = 1


class Model(Enum):
    identity = 0
    random = 1
    LSA = 2
    LDA = 3
    doc2vec = 4


class Clustering(Enum):
    dummy = 0
    BGMM = 1
    CRP = 2
    ddCRP = 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run clustering system.')
    parser.add_argument('-a', type=float, help='the alpha hyperparameter')
    parser.add_argument('-c', '--corpus', choices=[c.name for c in Corpus], help='corpus type')
    parser.add_argument('-f', '--fixed-rand', dest='seed', action='store_true', help='fix random seed')
    parser.add_argument('-i', type=int, help='i-th run')
    parser.add_argument('-K', type=int, help='the number of clusters (if applicable)')
    parser.add_argument('-l', '--clustering', choices=[c.name for c in Clustering], help='clustering method')
    parser.add_argument('-m', '--model', choices=[m.name for m in Model], help='document vector representation model')
    parser.add_argument('-n', type=int, help='the number of iterations')
    parser.add_argument('-s', '--size', type=int, choices=[10, 100, 200, 300], help='the size of a feature vector (if applicable)')
    parser.add_argument('-t', '--test', dest='test', action='store_true', help='use test data')
    parser.set_defaults(corpus=Corpus.artificial.name, model=Model.identity.name, clustering=Clustering.ddCRP.name,
                        K=2, size=100, test=False, seed=False, i=0, n=20, a=0.01)
    args = parser.parse_args()

    def has_valid_args(args):
        args.corpus = Corpus[args.corpus]
        args.model = Model[args.model]
        args.clustering = Clustering[args.clustering]

        corpus = args.corpus
        model = args.model

        if corpus == Corpus.artificial and model != Model.identity:
            logging.error("Only identity model can be used with artificial corpus.")
            return False

        elif corpus == Corpus.news and model == Model.identity:
            logging.error("The identity model cannot be used with news corpus.")
            return False

        return True

    if not has_valid_args(args):
        sys.exit(1)

    print("Clustering system")
    print("=================")
    print("Corpus:     %s" % args.corpus.name)
    print("Model:      %s" % args.model.name)
    print("Clustering: %s" % args.clustering.name)
    print("K:          %d" % args.K)
    print("size:       %d" % args.size)
    print("test:       %s" % args.test)
    print("fixed rand: %s" % args.seed)
    print("alpha:      %f" % args.a)
    print("i-th:       %d" % args.i)
    print("iterations: %d" % args.n)

    if args.seed:
        import random
        random.seed(0)

    # Get arguments
    corpus_type = args.corpus
    model_type = args.model
    clustering_type = args.clustering
    K = args.K        # Number of clusters
    size = args.size  # Size of a feature vector
    language = 'en'   # Language of news

    # Current directory
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Useful directories
    data_dir = os.path.join(dir_path, "..", "data")
    temp_dir = os.path.join(dir_path, "..", "temp")
    temp_visualization_dir = os.path.join(temp_dir, 'visualization', '{:02d}'.format(args.i))
    temp_cluster_visualization_dir = os.path.join(temp_visualization_dir, 'cluster')

    # Make sure temp directories exist
    Path(temp_visualization_dir).mkdir(parents=True, exist_ok=True)
    Path(temp_cluster_visualization_dir).mkdir(parents=True, exist_ok=True)

    # Paths to data
    training_dir = os.path.join(data_dir, "genuine", "training")
    heldout_dir = os.path.join(data_dir, "genuine", "heldout", "2017", "10")
    test_dir = os.path.join(data_dir, "genuine", "test", "2017", "10")
    artificial_file = os.path.join(data_dir, "artificial", "02.dat")
    graph_visualization_file = os.path.join(temp_visualization_dir, 'visualization.gexf')
    likelihood_visualization_file = os.path.join(temp_visualization_dir, 'clustering_likelihood.png')

    dictionary_file = os.path.join(training_dir, 'dictionary.dict')
    training_mm_corpus_file = os.path.join(training_dir, 'training_corpus.mm')
    training_low_corpus_file = os.path.join(training_dir, 'training_corpus.line')

    ground_truth_file = os.path.join(data_dir, "genuine", '2017-10-gold.csv')

    ########################
    # Initialization phase #
    ########################

    # Select corpora and model
    if corpus_type == Corpus.artificial:
        # For artificial data initialize identity model
        training_corpus = ArtificialCorpus(input=artificial_file)
        size = training_corpus.size
        corpora = SingletonCorpora(ArtificialCorpus(input=artificial_file, metadata=True))
        model = Identity()
    else:
        if model_type in [Model.random, Model.LSA, Model.LDA]:
            # Load BoW corpus and dictionary from temp files
            dictionary = Dictionary.load(dictionary_file)
            training_corpus = MmCorpus(training_mm_corpus_file)

            # Initialize correct model
            if model_type == Model.random:
                model = Random(size=size)
            elif model_type == Model.LSA:
                model_file = os.path.join(training_dir, 'model_%d.lsa' % size)
                tfidf_file = os.path.join(training_dir, 'model_%d.lsa.tfidf' % size)

                model = Lsa(dictionary, size=size, lsa_filename=model_file, tfidf_filename=tfidf_file)
            elif model_type == Model.LDA:
                model_file = os.path.join(training_dir, 'model_%d.lda' % size)

                model = Lda(dictionary, size=size, lda_filename=model_file)
            else:
                logging.error("Unknown model type '%s'" % model_type)
                sys.exit(1)
        else:
            # Load LoW corpus and dictionary from files
            dictionary = Dictionary.load(dictionary_file)
            training_corpus = LineCorpus(training_low_corpus_file)

            model_file = os.path.join(training_dir, 'model_%d.d2v' % size)

            model = Doc2vec(size=size, d2v_filename=model_file)

        # Load test corpora
        sep_t = None

        if model_type == Model.doc2vec:
            # Heldout LoW corpora
            corpora = FolderAggregatedLineNewsCorpora(heldout_dir, temp_dir, dictionary, language=language)

            # Test LoW corpora
            if args.test:
                sep_t = len(corpora)
                test_corpora = FolderAggregatedLineNewsCorpora(test_dir, temp_dir, dictionary, language=language)
                corpora = itertools.chain(corpora, test_corpora)
        else:
            # Heldout BoW corpora
            corpora = FolderAggregatedBowNewsCorpora(heldout_dir, temp_dir, dictionary, language=language)

            # Test BoW corpora
            if args.test:
                sep_t = len(corpora)
                test_corpora = FolderAggregatedBowNewsCorpora(test_dir, temp_dir, dictionary, language=language)
                corpora = itertools.chain(corpora, test_corpora)

    # Make sure we can see data only once
    corpora = SinglePassCorpusWrapper(corpora)

    # Select clustering algorithm
    likelihood_visualizer = LikelihoodVisualizer()

    # Select clustering method
    if clustering_type == Clustering.dummy:
        clustering = DummyClustering(K, size)
    elif clustering_type == Clustering.BGMM:
        prior = NormalInverseWishartPrior(
            np.zeros(size),
            0.01,
            0.01 * np.eye(size),
            size + 2
        )

        clustering = BgmmClustering(K, size, args.a, prior, args.n, visualizer=likelihood_visualizer)
    elif clustering_type == Clustering.CRP:
        prior = NormalInverseWishartPrior(
            np.zeros(size),
            0.01,
            .45 * np.eye(size),
            size + 2
        )

        clustering = CrpClustering(K, size, args.a, prior, args.n, visualizer=likelihood_visualizer)
    elif clustering_type == Clustering.ddCRP:
        prior = NormalInverseWishartPrior(
            np.zeros(size),
            0.01,
            1 * np.eye(size),
            size + 2
        )

        # Decay function
        a = 1  # 1 day

        def f(d: float):
            # return exponential_decay(d, a)
            return logistic_decay(d, a)

        if corpus_type == Corpus.artificial:
            # Decay function for artificial data
            a = 5  # 5 days

            def f(d):
                # Hack distance to look like time
                d = np.math.hypot(d[0], d[1]) * 60 * 60 * 24
                return logistic_decay(d, a)

        clustering = DdCrpClustering(K, size, args.a, prior, args.n, f, visualizer=likelihood_visualizer)
    else:
        logging.error("Unknown clustering algorithm '%s'" % clustering_type)
        sys.exit(1)
    ##############################
    # Online document clustering #
    ##############################

    # Reduce dimension for visualization
    logging.info("Initializing incremental PCA...")
    ipca = IncrementalPCA(n_components=2)
    # ipca.fit([vec for vec in model[training_corpus]])

    # Init visualizers
    graph_visualizer = GraphVisualizer()
    cluster_visualizer = ClusterVisualizer()

    # Init evaluator
    logging.info("Initializing evaluator...")
    if corpus_type == Corpus.artificial:
        evaluator = RandomEvaluator(K)
    else:
        evaluator = Evaluator(ground_truth_file, language=language)

    # Iterate over heldout/test corpora
    for t, docs_metadata in enumerate(corpora):
        logging.info("Testing corpus at time #%d." % t)

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
        # k, _, mean, covariance, X = clustering.parameters
        #
        # # Reduce vector dimension for visualization
        # # TODO mean and cov should be reduced as well
        # reduced_X = []
        # for i in range(k):
        #     reduced_X_k = ipca.transform(X[i]) if corpus_type != Corpus.artificial else X[i]
        #     reduced_X.append(reduced_X_k)
        #
        # cluster_visualizer.save(
        #     os.path.join(temp_cluster_visualization_dir, '{:05d}.png'.format(t)),
        #     k,
        #     mean,
        #     covariance,
        #     reduced_X
        # )

        evaluator.evaluate(t, ids_clusters, clustering.aic, clustering.bic, clustering.likelihood)

    # Store evaluation and visualization
    logging.info("Storing evaluation")
    evaluator.save(temp_visualization_dir)

    logging.info("Generating visualization")
    graph_visualizer.save(graph_visualization_file)
    likelihood_visualizer.save(likelihood_visualization_file)
