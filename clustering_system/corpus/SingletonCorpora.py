

class SingletonCorpora:

    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for _ in self.corpus:
            yield [_]
