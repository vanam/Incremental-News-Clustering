import logging

from clustering_system.clustering.DummyClustering import DummyClustering
from clustering_system.filter.StopWordsFilter import StopWordsFilter
from clustering_system.filter.filters import lower, split
from clustering_system.input.DummyDocumentStream import DummyDocumentStream
from clustering_system.vector_representation.RandomDocumentVectorStream import RandomDocumentVectorStream

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == "__main__":
    D = 6

    # TODO load stop word list from file
    stop_list = ['graph']

    ##################################
    # Pre-process incoming documents #
    ##################################
    ds = DummyDocumentStream()

    # Lowercase documents
    ds.add_filter(lower)
    # Split by whitespaces
    ds.add_filter(split)
    # Remove stop words
    swf = StopWordsFilter(stop_list)
    ds.add_filter(swf.filter)

    ##################################
    # Document vector representation #
    ##################################
    dvs = RandomDocumentVectorStream(ds, length=D)

    #####################
    # Clustering method #
    #####################
    cl = DummyClustering(K=3, D=D)

    ##############################
    # Online document clustering #
    ##############################
    i = 0
    for i, vec in enumerate(dvs):
        print("")
        print("Adding document %d" % i)
        cl.add_document(vec)

        # Remove every second document
        if i % 2 == 0:
            print("Removing document %d" % i)
            cl.remove_document(vec)
        i += 1

        # Print information about clusters
        print(cl)

        # TODO cluster evaluation
