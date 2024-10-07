import functools
import random
import re
import unicodedata
from collections.abc import Callable, Iterable, MutableSequence, Sequence
from typing import TypeVar

import nltk

from filelock import FileLock


# See https://stackoverflow.com/a/295466/1165181
def slugify(value, allow_unicode: bool = False) -> str:
    """Taken from https://github.com/django/django/blob/main/django/utils/text.py

    Convert to ASCII if 'allow_unicode' is False.
    Convert spaces or repeated dashes to single dashes.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase.
    Also, strip leading and trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def maybe_nltk_download(id_: str, path: str) -> None:
    try:
        nltk.data.find(path)
    except (LookupError, OSError):
        with FileLock(slugify(id_) + ".lock"):
            nltk.download(id_)


T = TypeVar("T")


def sample_up_to_k(seq: Sequence[T], k: int) -> Sequence[T]:
    return random.sample(seq, k) if len(seq) > k else seq


# From https://stackoverflow.com/a/43649323/1165181
def weighted_random_sample_without_replacement(
    population: Sequence[T],
    weights: Iterable[float | int] | None = None,
    k: int = 1
) -> Sequence[T]:
    weights = (
        (weights if isinstance(weights, MutableSequence) else list(weights)) if weights else
        ([1] * len(population))
    )
    positions = range(len(population))
    indices = []
    while needed := k - len(indices):
        for i in random.choices(positions, weights, k=needed):
            if weights[i]:
                weights[i] = 0
                indices.append(i)
    return [population[i] for i in indices]


def cache_generator_as_list(func: Callable[..., Iterable[T]]) -> Callable[..., Sequence[T]]:
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Sequence[T]:
        key = (args, frozenset(kwargs.items()))
        if key not in cache:
            cache[key] = list(func(*args, **kwargs))
        return cache[key]

    return wrapper


def non_overlapping_consecutive_pairs(seq: Sequence[T]) -> Iterable[tuple[T, T]]:
    return zip(seq[::2], seq[1::2])
