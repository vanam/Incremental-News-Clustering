from gensim.corpora import Dictionary

from clustering_system.input.IDocumentStream import IDocumentStream
from clustering_system.vector_representation.IDocumentVectorStream import IDocumentVectorStream


class BowDocumentVectorStream(IDocumentVectorStream):

    def __init__(self, dictionary: Dictionary, document_stream: IDocumentStream):
        """
        Stream which generates bag-of-words representation for every document from document stream

        :type dictionary: Dictionary
        :param document_stream: IDocumentStream input document stream
        """
        super().__init__(dictionary, document_stream)

    def __iter__(self):
        """
        Return random vector representation for each document.
        """
        for doc in self.document_stream:
            # print(doc)
            # Don't
            # self.dictionary.add_documents([doc])

            yield self.dictionary.doc2bow(doc)

    # TODO __len__
