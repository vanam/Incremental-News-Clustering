import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

from gensim.corpora import Dictionary


class FolderAggregatedCorporaABC(ABC):
    """
    Groups multiple news documents by folder
    """

    def __init__(self, directory, temp_directory, dictionary: Dictionary, language=None):
        """
        :param directory: A root directory containing group folders
        :param temp_directory: A temp directory for files speeding up repeating execution
        :param dictionary: A dictionary
        :param language: A language of corpora
        """
        self.directory = directory
        self.temp_directory = os.path.join(temp_directory, 'folder-aggregated-corpora')
        Path(self.temp_directory).mkdir(parents=True, exist_ok=True)
        self.dictionary = dictionary
        self.language = language
        self.group_corpora = self.get_group_corpora()

    @abstractmethod
    def _get_group_corpus(self, name, path):
        """Return corpus for a group."""
        pass

    def get_group_corpora(self):
        root = self.directory

        groups = []
        for item in os.listdir(root):
            # Skip __pycache__ folder
            if item == "__pycache__":
                continue

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
            corpus = self._get_group_corpus(group_name, group_path)

            # Add corpus to list
            corpora.append((group_path, corpus))

        return corpora

    def __iter__(self):
        """
        Iterating over the corpora yields corpus
        """
        for _, corpus in self.group_corpora:
            yield corpus

    def __len__(self):
        return len(self.group_corpora)
