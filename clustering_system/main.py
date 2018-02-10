import logging

from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LsiModel, LdaModel, Doc2Vec
from gensim.models.deprecated.doc2vec import TaggedDocument
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.decomposition import IncrementalPCA

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
    size = 6
    D = 2
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

    # Init dictionary beforehead
    dictionary = Dictionary(ids)

    ibowdvs = BowDocumentVectorStream(dictionary, ids)
    bowdvs = BowDocumentVectorStream(dictionary, ds)

    # LSI
    #####

    # tfidf = TfidfModel(dictionary=dictionary)
    # corpus_tfidf = tfidf[bowdvs]
    #
    # lsi = LsiModel(corpus=tfidf[ibowdvs], id2word=dictionary, num_topics=size, onepass=True, decay=decay)

    # LDA
    #####

    # extract 6 LDA topics
    lda = LdaModel(corpus=ibowdvs, id2word=dictionary, num_topics=size, passes=1)

    # doc2vec
    #########
    # class TaggedLineSentence(object):
    #     def __init__(self, documents):
    #         self.documents = documents
    #         self.len = len(documents)
    #
    #     def __iter__(self):
    #         for uid, line in enumerate(self.documents):
    #             print("@@@@@@@@@@ %s" % uid)
    #             print("@@@@@@@@@@ %s" % line)
    #             yield TaggedDocument(words=line, tags=['%d' % uid])
    #
    #     def __len__(self):
    #         return self.len
    #
    # sentences = TaggedLineSentence(ids)
    #
    # d2v = Doc2Vec(vector_size=size)
    # d2v.build_vocab(sentences)
    # d2v.train(sentences, total_examples=len(sentences), epochs=10)

    #####################
    # Clustering method #
    #####################
    cl = DummyClustering(K=3, D=D)

    #######################
    # Dimension reduction #
    #######################
    # iX = [unwrap_vector(vec) for vec in lsi[tfidf[ibowdvs]]]  # LSI
    iX = [unwrap_vector(vec) for vec in lda[ibowdvs]]  # LDA
    # iX = [d2v.infer_vector(doc) for doc in ids]  # doc2vec

    ipca = IncrementalPCA(n_components=D, batch_size=10)
    ipca.fit_transform(iX)

    ##############################
    # Online document clustering #
    ##############################
    i = 0
    # for i, vec in enumerate(corpus_tfidf):  # LSI
    for i, vec in enumerate(bowdvs):        # LDA
    # for i, vec in enumerate(ds):            # doc2vec
        print("")
        print("Adding document %d" % i)
        print(vec)

        # LSI
        #####

        # lsi.add_documents([vec])
        # vec = unwrap_vector(lsi[vec])

        # LDA
        #####

        lda.update([vec])
        vec = unwrap_vector(lda[vec])

        # doc2vec
        #########
        # vec = d2v.infer_vector(vec)

        print(vec)

        # Reduce dimmension
        ipca.partial_fit([vec])
        vec = ipca.transform([vec])[0]
        print("iPCA")
        print(vec)

        # Cluster document
        cl.add_document(vec)

        # Remove every second document
        if i % 2 == 0:
            print("Removing document %d" % i)
            cl.remove_document(vec)
        i += 1

        # Print information about clusters
        print(cl)

        # TODO cluster evaluation
