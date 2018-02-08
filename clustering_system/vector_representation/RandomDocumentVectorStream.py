import numpy as np
from gensim.corpora import Dictionary

from clustering_system.input.IDocumentStream import IDocumentStream
from clustering_system.vector_representation.IDocumentVectorStream import IDocumentVectorStream


class RandomDocumentVectorStream(IDocumentVectorStream):

    def __init__(self, dictionary: Dictionary, document_stream: IDocumentStream, length: int = 300):
        """
        Stream which generates random vector of given length for every document from document stream

        :param document_stream: input document stream
        :param length: random vector length
        """
        super().__init__(dictionary, document_stream)
        self.length = length

    def __iter__(self):
        """
        Return random vector representation for each document.
        """
        for doc in self.document_stream:
            print(doc)
            yield np.random.random_sample(self.length)
