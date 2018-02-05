import random
import string

from clustering_system.IDocumentStream import IDocumentStream


class DummyDocumentStream(IDocumentStream):

    def __init__(self):
        self.N = 20
        self.i = 0

    def __iter__(self):
        """
        Return dummy documents one by one containing random string.
        """
        while self.i < 10:
            self.i += 1
            yield ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(self.N))
