from typing import Sequence

from clustering_system.input.IDocumentStream import IDocumentStream


class ReentrantDocumentStream(IDocumentStream):

    def __init__(self, documents: Sequence[str]):
        super().__init__()
        self.documents = documents

    def __iter__(self):
        """
        Return dummy documents one by one containing random string.
        """
        for line in self.documents:
            # assume there's one document per line, tokens separated by whitespace
            yield self._process(line)
