import random
from copy import copy

import spacy

try:
    NLP = spacy.load("en_core_web_sm")
except IOError:
    from spacy.cli import download
    download("en")
    NLP = spacy.load("en_core_web_sm")

MAX_SHUFFLE_RETRY = 10


def _get_ngrams(tokens, n):
    if n == 1:
        return [[t] for t in tokens]
    ngrams = []
    ngram = []
    for i in range(len(tokens)):
        ngram.append(tokens[i])
        if i % n == n - 1:
            ngrams.append(ngram[:])
            ngram = []
    if ngram:
        ngrams.append(ngram)
    return ngrams


def _shuffle_ngrams(doc: spacy.tokens.Doc, n: int):
    ngrams = _get_ngrams(doc, n)
    if len(ngrams) == 1:
        # shuffle is impossible, no negative text
        return []

    punct = None
    if len(ngrams[-1]) == 1 and ngrams[-1][0].pos_ == "PUNCT":
        if len(ngrams) == 2:
            # avoid the case when only punctuation goes first while remaining the sentence.
            return []
        punct = ngrams[-1][0]
        # shuffle without punctuations first, then insert punctuation on random position
        ngrams = ngrams[:-1]
    random.shuffle(ngrams)
    if punct is not None:
        rand_insert_pos = random.randint(0, len(ngrams))
        if rand_insert_pos == 0:
            ngrams = [[punct]] + ngrams
        else:
            ngrams[rand_insert_pos - 1].append(punct)
    return " ".join([_join_tokens(t) for t in ngrams])


def _join_tokens(tokens):
    joined = ""
    for i, t in enumerate(tokens):
        if i > 0 and t.pos_ != "PUNCT":
            joined += " "
        joined += t.text
    return joined


def _check_same(original, shuffled):

    def _remove_punct(doc):
        return [
            # ignore punctuation in the leftmost and rightmost of the sentence
            t.text for i, t in enumerate(doc) if not (i in [0, len(doc) - 1] and t.pos_ == "PUNCT")
        ]

    # remove punctuations
    original_ = _remove_punct(original)
    shuffled_ = _remove_punct(shuffled)
    return " ".join(original_) == " ".join(shuffled_)


def shuffle_ngrams(doc: spacy.tokens.Doc, n: int | None = None, **kwargs):
    assert n is not None
    assert n in [1, 2, 3, 4, 5]
    shuffled = _shuffle_ngrams(copy(doc), n=n)
    if not shuffled:
        return []  # no negative text

    flag = _check_same(doc, NLP(shuffled))
    i = 0
    while flag:
        shuffled = _shuffle_ngrams(copy(doc), n=n)
        flag = _check_same(doc, NLP(shuffled))
        i += 1
        if i >= MAX_SHUFFLE_RETRY:
            return []

    assert isinstance(shuffled, str)
    return [shuffled]
