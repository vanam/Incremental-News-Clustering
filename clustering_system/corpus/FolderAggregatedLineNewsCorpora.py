import os

from clustering_system.corpus.FolderAggregatedCorporaABC import FolderAggregatedCorporaABC
from clustering_system.corpus.LineCorpus import LineCorpus
from clustering_system.corpus.LineNewsCorpus import LineNewsCorpus
from clustering_system.corpus.MetaLineCorpusWrapper import MetaLineCorpusWrapper


class FolderAggregatedLineNewsCorpora(FolderAggregatedCorporaABC):
    """
    Groups multiple news documents by folder using Line corpus with metadata
    """

    def _get_group_corpus(self, group_name, group_path):
        """Return Line corpus for a group."""
        temp_corpus_file = os.path.join(self.temp_directory, group_name + '.line')
        # Check if we have already pre-processed the corpus
        if not os.path.exists(temp_corpus_file):
            corpus = LineNewsCorpus(input=group_path, metadata=True, language=self.language, dictionary=self.dictionary)

            # Serialize pre-processed corpus to temp files
            LineCorpus.serialize(temp_corpus_file, corpus, self.dictionary, metadata=True)

        # Load corpus from temp file
        return MetaLineCorpusWrapper(temp_corpus_file)