import numpy as np

from clustering_system.IDocumentStream import IDocumentStream
from clustering_system.IDocumentVectorStream import IDocumentVectorStream


class RandomDocumentVectorStream(IDocumentVectorStream):

    def __init__(self, document_stream: IDocumentStream, length: int = 300):
        """
        Stream which generates random vector of given length for every document from document stream

        :param document_stream: input document stream
        :param length: random vector length
        """
        super(RandomDocumentVectorStream, self).__init__(document_stream)
        self.length = length

    def __iter__(self):
        """
        Return random vector representation for each document.
        """
        for doc in self.document_stream:
            yield np.random.random_sample(self.length)
