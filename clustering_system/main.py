import logging

from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LsiModel, LdaModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

from clustering_system.clustering.DummyClustering import DummyClustering
from clustering_system.filter.StopWordsFilter import StopWordsFilter
from clustering_system.filter.filters import lower, split
from clustering_system.input.DummyDocumentStream import DummyDocumentStream
from clustering_system.input.ReentrantDocumentStream import ReentrantDocumentStream
from clustering_system.utils import nltk_pos2wn_pos, unwrap_vector
from clustering_system.vector_representation.BowDocumentVectorStream import BowDocumentVectorStream
from clustering_system.vector_representation.RandomDocumentVectorStream import RandomDocumentVectorStream

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

nltk.download('stopwords')                   # Needed for stop word removal
nltk.download('punkt')                       # Needed for POS tagging
nltk.download('averaged_perceptron_tagger')  # Needed for POS tagging
nltk.download('wordnet')                     # Needed for lemmatization

if __name__ == "__main__":
    D = 6
    decay = 0.9

    initialization_documents = [
            "Human machine interface for lab abc computer applications",
            "A survey of user opinion of computer system response time",
            "The generation of random binary unordered trees",
            "The intersection graph of paths in trees",
            "Graph minors A survey",
            "Human machine interface for lab abc computer applications",
            "A survey of user opinion of computer system response time The generaation of random binary unordered trees",
            "The intersection graph of paths in trees Graph minors A survey",
            "Human machine interface for lab abc computer applications A suarvey of user opinion of computer system response time",
            "The generation of random binary unordered trees",
            "The intersection graph of paths in trees",
            "Graph minors A survey Human machine interface for lab abc compuater applications",
            "A survey of user opinion of computer system response time",
            "The generastion of random binary unorsdered trees",
            "The interssection graph of paths in trees",
            "Graph minosrs A survey"
        ]

    documents = [
        "Human machine interface for lab abc computer applications",
        "A survey of user opinion of computer system response time",
        "The EPS user interface management system",
        "System and human system engineering testing of EPS",
        "Relation of user perceived response time to error measurement",
        "The generation of random binary unordered trees",
        "The intersection graph of paths in trees",
        "Graph minors IV Widths of trees and well quasi ordering",
        "Graph minors A survey"
    ]

    ##################################
    # Pre-process incoming documents #
    ##################################
    ids = ReentrantDocumentStream(initialization_documents)
    ds = DummyDocumentStream(documents)

    # Lowercase documents
    ids.add_filter(lower)
    ds.add_filter(lower)

    # Tokenize
    ids.add_filter(nltk.word_tokenize)
    ds.add_filter(nltk.word_tokenize)

    # POS tagging
    ids.add_filter(nltk.pos_tag)
    ds.add_filter(nltk.pos_tag)

    # Lemmatize
    wnl = WordNetLemmatizer()

    def lemmatize(line):
        return [wnl.lemmatize(word, pos=nltk_pos2wn_pos(tag)) for word, tag in line]

    ids.add_filter(lemmatize)
    ds.add_filter(lemmatize)

    # Remove stop words
    stop_list = frozenset(stopwords.words('english'))  # nltk stopwords list
    swf = StopWordsFilter(stop_list)

    ids.add_filter(swf.filter)
    ds.add_filter(swf.filter)

    ##################################
    # Document vector representation #
    ##################################
    # dvs = RandomDocumentVectorStream(dictionary, ds, length=D)

    dictionary = Dictionary(ids)

    ibowdvs = BowDocumentVectorStream(dictionary, ids)
    bowdvs = BowDocumentVectorStream(dictionary, ds)

    tfidf = TfidfModel(dictionary=dictionary)
    corpus_tfidf = tfidf[bowdvs]

    lsi = LsiModel(corpus=tfidf[ibowdvs], id2word=dictionary, num_topics=D, onepass=True, decay=decay)

    #####################
    # Clustering method #
    #####################
    cl = DummyClustering(K=3, D=D)

    ##############################
    # Online document clustering #
    ##############################
    i = 0
    for i, vec in enumerate(corpus_tfidf):
        print("")
        print("Adding document %d" % i)
        # print(vec)
        lsi.add_documents([vec])
        vec = unwrap_vector(lsi[vec])
        # print(vec)

        cl.add_document(vec)

        # Remove every second document
        if i % 2 == 0:
            print("Removing document %d" % i)
            cl.remove_document(vec)
        i += 1

        # Print information about clusters
        print(cl)

        # TODO cluster evaluation
