from clustering_system.input.IDocumentStream import IDocumentStream


class DummyDocumentStream(IDocumentStream):

    def __init__(self):
        super().__init__()
        self.i = 0
        self.documents = [
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

    def __iter__(self):
        """
        Return dummy documents one by one containing random string.
        """
        while self.i < len(self.documents):
            line = self.documents[self.i]
            self.i += 1

            # assume there's one document per line, tokens separated by whitespace
            yield self._process(line)
