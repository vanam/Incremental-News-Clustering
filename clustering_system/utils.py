from typing import Tuple, Sequence

import numpy as np


def nltk_pos2wn_pos(nltk_tag: str) -> str:
    """
    Convert NLTK POS tags to WordNet POS tags.

    WordNet only cares about 5 parts of speech.
    The other parts of speech will be tagged as nouns

    :param nltk_pos:
    :return:
    """
    part = {
        'N': 'n',
        'V': 'v',
        'J': 'a',
        'S': 's',
        'R': 'r'
    }

    if nltk_tag[0] in part.keys():
        return part[nltk_tag[0]]
    else:
        # other parts of speech will be tagged as nouns
        return 'n'


def unwrap_vector(gensim_vector: Tuple[int, float]) -> np.ndarray:
    return np.array([f for _, f in gensim_vector])
