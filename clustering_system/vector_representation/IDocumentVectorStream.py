from abc import ABC, abstractmethod

from gensim.corpora import Dictionary

from clustering_system.input.IDocumentStream import IDocumentStream


class IDocumentVectorStream(ABC):

    def __init__(self, dictionary: Dictionary, document_stream: IDocumentStream):
        self.dictionary = dictionary
        self.document_stream = document_stream

    @abstractmethod
    def __iter__(self):
        """
        Return vector representation for each document.
        """
        pass
