from typing import Sequence


class StopWordsFilter:

    def __init__(self, stop_list: frozenset):
        self.stop_list = stop_list

    def filter(self, doc: Sequence[str]) -> Sequence[str]:
        """
        Return documents one by one with stop words removed.
        """
        return [word for word in doc if word not in self.stop_list]
