import random
from typing import Tuple, List

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


def draw(probabilities: np.ndarray):
    """
    Draw from a discrete random variable with mass in vector `probabilities`.
    Indices returned are between 0 and len(probabilities) - 1.
    """
    # Generate random number in the interval [0, 1)
    k_uni = random.random()

    for i, prob in enumerate(probabilities):
        # Subtract while there is mass left
        k_uni -= prob
        if k_uni < 0:
            return i

    return len(probabilities) - 1


def draw_indexed(index_prob: List[Tuple[int, float]]):
    """
    Draw from a discrete random variable with mass at second position in tuple in list `index_prob`.
    Indices returned are at first position in tuples.
    """
    # Generate random number in the interval [0, 1)
    k_uni = random.random()

    for i, prob in index_prob:
        # Subtract while there is mass left
        k_uni -= prob
        if k_uni < 0:
            return i

    return index_prob[-1][0]
