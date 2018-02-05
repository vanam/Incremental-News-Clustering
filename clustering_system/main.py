from clustering_system.DummyClustering import DummyClustering
from clustering_system.DummyDocumentStream import DummyDocumentStream
from clustering_system.IClustering import IClustering
from clustering_system.RandomDocumentVectorStream import RandomDocumentVectorStream
from clustering_system.IDocumentStream import IDocumentStream
from clustering_system.IDocumentVectorStream import IDocumentVectorStream

if __name__ == "__main__":
    D = 6

    # cl = IClustering()
    cl = DummyClustering(K=3, D=D)

    # ds = IDocumentStream()
    ds = DummyDocumentStream()

    # dvs = IDocumentVectorStream(ds)
    dvs = RandomDocumentVectorStream(ds, length=D)

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
