import os
import tempfile

from gensim.corpora import Dictionary

from clustering_system.corpus.FolderAggregatedBowNewsCorpora import FolderAggregatedBowNewsCorpora


class TestFolderAggregatedBowNewsCorpora:

    def test_create(self):
        # Current directory
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dictionary_file = os.path.join(dir_path, "data", "dictionary.dict")
        dictionary = Dictionary.load(dictionary_file)

        temp_dir = tempfile.TemporaryDirectory()

        corpora = FolderAggregatedBowNewsCorpora(dir_path, temp_dir.name, dictionary, language="en")

        i = 0
        for c in corpora:
            i += 1

            # Traverse corpus
            for _ in c:
                pass

        # Assert number of times it went through the loop
        assert i == 1
