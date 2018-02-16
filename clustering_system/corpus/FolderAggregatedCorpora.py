import logging
import os
from pathlib import Path

from gensim.corpora import MmCorpus, Dictionary

from clustering_system.corpus.MetaMmCorpusWrapper import MetaMmCorpusWrapper
from clustering_system.corpus.NewsCorpus import NewsCorpus


class FolderAggregatedCorpora:
    """
    Groups multiple news documents by folder
    """

    def __init__(self, directory, temp_directory, dictionary: Dictionary, language=None):
        self.directory = directory
        self.temp_directory = os.path.join(temp_directory, 'folder-aggregated-corpora')
        Path(self.temp_directory).mkdir(parents=True, exist_ok=True)
        self.dictionary = dictionary
        self.language = language
        self.group_corpora = self.get_group_corpora()

    def get_group_corpora(self):
        root = self.directory

        groups = []
        for item in os.listdir(root):
            dirpath = os.path.join(root, item)

            # Each directory is a group
            if os.path.isdir(dirpath):
                groups.append((item, dirpath))

        # Sort by directory name
        groups.sort(key=lambda x: x[0])
        logging.info("Found %d groups." % len(groups))

        corpora = []
        # For each group init corpus if not exists
        for group_name, group_path in groups:
            temp_corpus_file = os.path.join(self.temp_directory, group_name + '.mm')
            # Check if we have already pre-processed the corpus
            if not os.path.exists(temp_corpus_file):
                corpus = NewsCorpus(input=group_path, metadata=True, language=self.language, dictionary=self.dictionary)

                # Serialize pre-processed corpus to temp files
                MmCorpus.serialize(temp_corpus_file, corpus, metadata=True)

            # Load corpus from temp file
            corpus = MetaMmCorpusWrapper(temp_corpus_file)

            # Add corpus to list
            corpora.append((group_path, corpus))

        return corpora

    def __iter__(self):
        """
        Iterating over the corpora yields corpus
        """
        for _, corpus in self.group_corpora:
            yield corpus
