import os

from gensim.corpora import MmCorpus

from clustering_system.corpus.BowNewsCorpus import BowNewsCorpus
from clustering_system.corpus.FolderAggregatedCorporaABC import FolderAggregatedCorporaABC
from clustering_system.corpus.MetaMmCorpusWrapper import MetaMmCorpusWrapper


class FolderAggregatedBowNewsCorpora(FolderAggregatedCorporaABC):
    """
    Groups multiple news documents by folder using MM corpus with metadata
    """

    def _get_group_corpus(self, group_name, group_path):
        """Return MM corpus for a group."""
        temp_corpus_file = os.path.join(self.temp_directory, group_name + '.mm')
        # Check if we have already pre-processed the corpus
        if not os.path.exists(temp_corpus_file):
            corpus = BowNewsCorpus(input=group_path, metadata=True, language=self.language, dictionary=self.dictionary)

            # Serialize pre-processed corpus to temp files
            MmCorpus.serialize(temp_corpus_file, corpus, metadata=True)

        # Load corpus from temp file
        return MetaMmCorpusWrapper(temp_corpus_file)
