from clustering_system.corpus.SinglePassCorpusWrapper import SinglePassCorpusWrapper


class TestSinglePassCorpusWrapper:

    corpus = ["Hello", ["cruel", "world", "!"]]

    def test_single_pass(self):
        corpus = SinglePassCorpusWrapper(self.corpus)

        # The first pass
        i = 0
        for _ in corpus:
            i += 1

        # Assert number of times it went through the loop
        assert i == 2

    def test_double_pass(self):
        corpus = SinglePassCorpusWrapper(self.corpus)

        # The first pass
        for _ in corpus:
            pass

        # The second pass
        i = 0
        for _ in corpus:
            i += 1

        # Assert number of times it went through the loop
        assert i == 0
