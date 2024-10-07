import functools
import importlib
import itertools
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import partial
from typing import Literal

import pyinflect
import sng_parser
import spacy.matcher
import spacy.tokens
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Lemma, Synset

from .ngram_shuffler import shuffle_ngrams
from .svlc_rule_based import create_rule_based_negatives
from .utils import (
    cache_generator_as_list, maybe_nltk_download, non_overlapping_consecutive_pairs, sample_up_to_k,
    weighted_random_sample_without_replacement,
)

try:
    NLP = spacy.load("en_core_web_sm")
except IOError:
    from spacy.cli import download
    download("en")
    NLP = spacy.load("en_core_web_sm")

VP_MATCHER = spacy.matcher.Matcher(NLP.vocab)
VP_MATCHER.add(
    "Verb phrase", [
        [
            {
                "POS": "VERB",
                "OP": "?"
            }, {
                "POS": "ADV",
                "OP": "*"
            }, {
                "POS": "AUX",
                "OP": "*"
            }, {
                "POS": "VERB",
                "OP": "+"
            }
        ]
    ]
)


def add_random_text_hard_negatives(texts: Sequence[str], style: str, text_batch_size: int):
    if style == "negclip":
        hard_negatives_generator_fn = _generate_hard_negatives_like_negclip

    elif style == "replace":
        maybe_nltk_download("wordnet", "corpora/wordnet.zip")
        hard_negatives_generator_fn = _generate_hard_negatives_like_replace

    elif "ngram_shuffle" in style:
        n = int(style.split("_")[-1])
        hard_negatives_generator_fn = partial(shuffle_ngrams, n=n)

    elif style == "svlc":
        hard_negatives_generator_fn = create_rule_based_negatives

    elif style == "fsc-clip":
        maybe_nltk_download("wordnet", "corpora/wordnet.zip")
        hard_negatives_generator_fn = [
            _generate_hard_negatives_like_negclip,
            _generate_hard_negatives_like_replace,
            partial(shuffle_ngrams, n=2),
        ]

    else:
        raise ValueError(f"Unknown random text hard-negative style: '{style}'")

    if not isinstance(hard_negatives_generator_fn, list):
        hard_negatives_generator_fn = [hard_negatives_generator_fn]

    num_neg_types = len(hard_negatives_generator_fn)
    neg_texts = [[] for _ in range(num_neg_types)]  # per negative types
    neg_masks = [[0] * text_batch_size for _ in range(num_neg_types)]
    for text_ind, doc in enumerate(NLP.pipe(texts, batch_size=text_batch_size)):
        for neg_ind, neg_fn in enumerate(hard_negatives_generator_fn):
            neg_text = neg_fn(doc, n_choices=1)
            if len(neg_text):
                neg_masks[neg_ind][text_ind] = 1
                neg_texts[neg_ind].append(neg_text[0])

    neg_texts = list(itertools.chain.from_iterable(neg_texts))
    neg_masks = list(itertools.chain.from_iterable(neg_masks))

    return neg_texts, neg_masks


def _get_noun_phrases(doc: spacy.tokens.Doc) -> Sequence[spacy.tokens.Span]:
    return [span for span in doc.noun_chunks if len(span) > 2]


def _get_verb_phrases(doc: spacy.tokens.Doc) -> Sequence[spacy.tokens.Span]:
    return VP_MATCHER(doc, as_spans=True)


def _get_nn_list(doc: spacy.tokens.Doc) -> Sequence[spacy.tokens.Span]:
    return [doc[token.i:token.i + 1] for token in doc if token.tag_ == "NN"]


def _get_nns_list(doc: spacy.tokens.Doc) -> Sequence[spacy.tokens.Span]:
    return [doc[token.i:token.i + 1] for token in doc if token.tag_ == "NNS"]


def _get_adv_list(doc: spacy.tokens.Doc) -> Sequence[spacy.tokens.Span]:
    return [doc[token.i:token.i + 1] for token in doc if token.pos_ == "ADV"]


def _get_adj_list(doc: spacy.tokens.Doc) -> Sequence[spacy.tokens.Span]:
    return [doc[token.i:token.i + 1] for token in doc if token.pos_ == "ADJ"]


def _sample_pairs(
    spans: Sequence[spacy.tokens.Span],
    n_choices: int = 1,
) -> Iterable[tuple[spacy.tokens.Span, spacy.tokens.Span]]:
    for span1, span2 in non_overlapping_consecutive_pairs(sample_up_to_k(spans, k=2 * n_choices)):
        if span1.text != span2.text:
            yield span1, span2


def _swap_spans_in_text(
    text: str, first_span: spacy.tokens.Span, second_span: spacy.tokens.Span
) -> str:
    if second_span[0].i < first_span[0].i:
        first_span, second_span = second_span, first_span

    first_start = first_span[0].idx
    first_end = first_span[-1].idx + len(first_span[-1])

    second_start = second_span[0].idx
    second_end = second_span[-1].idx + len(second_span[-1])

    return text[:first_start] + second_span.text + text[first_end:second_start
                                                        ] + first_span.text + text[second_end:]


def _generate_hard_negatives_like_negclip(
    doc: spacy.tokens.Doc,
    n_choices: int = 1,
) -> Sequence[str]:
    # It generates hard negatives by swapping two phrases of the same type, as in NegCLIP.
    # Code adapted from https://gist.github.com/mertyg/03e638fef99cbd8c3a108e8bacd16a6c.
    hard_negatives_swaps = [
        span_pair for span_gen_fn in [
            _get_noun_phrases, _get_verb_phrases, _get_nn_list, _get_nns_list, _get_adv_list,
            _get_adj_list
        ] for span_pair in _sample_pairs(spacy.util.filter_spans(span_gen_fn(doc)))
    ]

    hard_negatives = [
        _swap_spans_in_text(doc.text, *two_spans) for two_spans in hard_negatives_swaps
    ]

    return random.sample(hard_negatives, k=min(n_choices, len(hard_negatives)))


WordNetPos = Literal["a", "n", "r", "v"]
SpaCyPos = Literal["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART",
                   "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]


def _spacy_to_wordnet_pos(spacy_pos: SpaCyPos) -> WordNetPos | None:
    if spacy_pos in {"NOUN", "PRON", "PROPN"}:
        return "n"
    elif spacy_pos == "VERB":
        return "v"
    elif spacy_pos == "ADJ":
        return "a"
    elif spacy_pos == "ADV":
        return "r"
    else:
        return None


def _get_antonyms(synset: Synset) -> Iterator[Lemma]:
    for lemma in synset.lemmas():
        yield from lemma.antonyms()


def _support_adjectives_with_hypernyms(
    func: Callable[[Synset], Iterator[Lemma]]
) -> Callable[[Synset], Iterator[Lemma]]:

    @functools.wraps(func)
    def wrapper(synset: Synset) -> Iterator[Lemma]:
        is_adjective = synset.pos() in {"a", "s"}

        if is_adjective:
            synsets_to_consider = {
                related_lemma.synset()
                for lemma in synset.lemmas()
                for related_lemma in lemma.derivationally_related_forms()
                if related_lemma.synset().pos() == "n"
            }
        else:
            synsets_to_consider = [synset]

        for other_synset in synsets_to_consider:
            for lemma in func(other_synset):
                if is_adjective:
                    for related_lemma in lemma.derivationally_related_forms():
                        if related_lemma.synset() != synset and related_lemma.synset().pos() in {
                            "a", "s"
                        }:
                            yield related_lemma
                else:
                    yield lemma

    return wrapper


@cache_generator_as_list
@_support_adjectives_with_hypernyms
def _get_sibling_or_cousin_co_hyponym_lemmas(synset: Synset) -> Iterator[Lemma]:
    for parent in synset._related("@", sort=False):
        for grandparent in parent._related("@", sort=False):
            for uncle_or_parent in grandparent._related("~", sort=False):
                for cousin_or_sibling in uncle_or_parent._related("~", sort=False):
                    if cousin_or_sibling != synset:
                        yield from cousin_or_sibling.lemmas()


@cache_generator_as_list
@_support_adjectives_with_hypernyms
def _get_co_hyponym_lemmas(synset: Synset) -> Iterator[Lemma]:
    for parent in synset._related("@", sort=False):
        for sibling in parent._related("~", sort=False):
            if sibling != synset:
                yield from sibling.lemmas()


TOTAL_WORDNET_ENGLISH_LEMMA_COUNTS = 248_796


@functools.lru_cache
def _get_prepositions() -> Sequence[tuple[str, int]]:
    # We extrapolate the frequencies of the prepositions from another corpus and take them to WordNet terms.
    #
    # The frequencies obtained with:
    # https://books.google.com/ngrams/json?content={word}&year_start=2018&year_end=2019&corpus=en-2019&smoothing=0`
    return [
        (word, max(round(freq * TOTAL_WORDNET_ENGLISH_LEMMA_COUNTS), 1)) for word, freq in [
            ("aboard", 8.304538823722396e-06),
            ("above", 0.00019195365894120187),
            ("across", 0.0002091274509439245),
            ("after", 0.0006388680194504559),
            ("against", 0.0003930507809855044),
            ("along", 0.00020867066632490605),
            ("alongside", 1.8277087292517535e-05),
            ("amid", 9.313676855526865e-06),
            ("amidst", 4.904808520223014e-06),
            ("among", 0.00023312875418923795),
            ("amongst", 1.7836573533713818e-05),
            ("around", 0.00041722101741470397),
            ("as", 0.005255552940070629),
            ("at", 0.0032681692391633987),
            ("atop", 4.4327798605081625e-06),
            ("before", 0.0006610354757867754),
            ("behind", 0.0001920274953590706),
            ("below", 9.910686640068889e-05),
            ("beneath", 4.395221549202688e-05),
            ("beside", 5.1512502977857366e-05),
            ("besides", 2.0500376194831915e-05),
            ("between", 0.0006798655376769602),
            ("beyond", 0.0001130914461100474),
            ("by", 0.00341515033505857),
            ("down", 0.0006581470370292664),
            ("during", 0.0003052355023100972),
            ("following", 0.00022817311401013285),
            ("for", 0.006153156980872154),
            ("from", 0.003160713240504265),
            ("in", 0.01368333213031292),
            ("in addition to", 1.5962219549692236e-05),
            ("in front of", 6.79560616845265e-05),
            ("in reference to", 2.5002268557727803e-06),
            ("in regard to", 6.212438165675849e-06),
            ("in spite of", 1.7529946489958093e-05),
            ("including", 0.00019620849343482405),
            ("inside", 0.00013211416080594063),
            ("into", 0.0013475629966706038),
            ("like", 0.0010499523486942053),
            ("near", 0.00013061976642347872),
            ("nearer", 1.1801283108070493e-05),
            ("nearest", 1.427516599505907e-05),
            ("next", 0.00025265218573622406),
            ("of", 0.025005823001265526),
            ("off", 0.00044481706572696567),
            ("on", 0.004635580815374851),
            ("on account of", 9.287143257097341e-06),
            ("on side of", 4.1854182342149215e-08),
            ("on the side of", 6.329439202090725e-06),
            ("on top of", 1.6038058674894273e-05),
            ("onto", 7.380569149972871e-05),
            ("opposite", 4.779181472258642e-05),
            ("out", 0.001463941065594554),
            ("outside", 0.00013628744636662304),
            ("over", 0.0008702398044988513),
            ("past", 0.0001756970159476623),
            ("since", 0.00026465917471796274),
            ("through", 0.000747955753467977),
            ("throughout", 7.150941382860765e-05),
            ("thru", 8.848301149555482e-07),
            ("till", 6.974823190830648e-05),
            ("to", 0.019767532125115395),
            ("toward", 0.00016340451838914305),
            ("towards", 0.000129144755192101),
            ("under", 0.0004209426115266979),
            ("underneath", 1.0443360224599019e-05),
            ("unlike", 2.0879459043499082e-05),
            ("until", 0.0002701016201172024),
            ("unto", 7.981383532751352e-05),
            ("up", 0.0014249003725126386),
            ("upon", 0.00035253650275990367),
            ("with", 0.005393100902438164),
            ("with regard to", 1.4580499737348873e-05),
            ("within", 0.00031806135666556656),
            ("without", 0.00041237485129386187),
        ]
    ]


def _prepare_token_for_replacement(
    token_or_span: spacy.tokens.Token | spacy.tokens.Span, new_token_text: str
) -> str:
    token = token_or_span if isinstance(token_or_span, spacy.tokens.Token) else token_or_span.root

    # The following function can return `None`.
    new_token_text = (
        pyinflect.getInflection(new_token_text, tag=token.tag_, inflect_oov=False)
        or [new_token_text]
    )[0]

    new_token_text = new_token_text.replace("_", " ")

    if token_or_span.text.isupper():
        new_token_text = new_token_text.upper()
    elif token_or_span.text.istitle():
        new_token_text = new_token_text.title()

    return new_token_text


def _recreate_text_with_replaced_token(
    token_or_span: spacy.tokens.Token | spacy.tokens.Span, new_token_text: str
) -> str:
    start = token_or_span.idx if isinstance(
        token_or_span, spacy.tokens.Token
    ) else token_or_span.start_char
    end = (start + len(token_or_span)
           ) if isinstance(token_or_span, spacy.tokens.Token) else token_or_span.end_char
    return token_or_span.doc.text[:start] + new_token_text + token_or_span.doc.text[end:]


importlib.import_module("training.text_negatives.spacy_parser")
SCENE_GRAPH_PARSER = sng_parser.Parser("custom-spacy")


@functools.cache
def _get_lemma_count(lemma: Lemma) -> int:
    return lemma.count(
    )  # It seems like it's an expensive operation (I/O bound) to perform repetitively.


@functools.cache
def _get_span_replacements_for_non_adp(text: str, pos: WordNetPos) -> Mapping[str, int]:
    replacements = defaultdict(int)

    synsets = wn.synsets(text, pos=pos)
    synonym_lemmas = frozenset(lemma for synset in synsets for lemma in synset.lemmas())
    for synset in synsets:
        for new_token_text_fn in [_get_antonyms, _get_sibling_or_cousin_co_hyponym_lemmas]:
            for new_token_lemma in new_token_text_fn(synset):
                if new_token_lemma not in synonym_lemmas:
                    replacements[new_token_lemma.name()
                                 ] += max(_get_lemma_count(new_token_lemma), 1)

    return replacements


def _get_span_replacements(
    token_or_span: spacy.tokens.Token | spacy.tokens.Span,
) -> tuple[Sequence[tuple[spacy.tokens.Token | spacy.tokens.Span, str]], Iterable[int | float]]:
    token = token_or_span.root if isinstance(token_or_span, spacy.tokens.Span) else token_or_span

    if token.pos_ == "ADP":
        # Note that here we use the `span`, not the `token`,
        # because we use the more general concept of a PP.
        replacements = [
            (token_or_span, new_token_text) for new_token_text, _ in _get_prepositions()
            if new_token_text != token_or_span.text.lower()
        ]
        weights = (
            count for new_token_text, count in _get_prepositions()
            if new_token_text != token_or_span.text.lower()
        )
    else:
        pos = _spacy_to_wordnet_pos(token.pos_)
        if pos:  # We skip the types of concepts unsupported by WordNet.
            replacement_dict = _get_span_replacements_for_non_adp(token.text, pos)
            replacements = [(token, new_token_text) for new_token_text in replacement_dict.keys()]
            weights = replacement_dict.values()
        else:
            replacements, weights = [], []

    return replacements, weights


def _generate_hard_negatives_like_replace(
    doc: spacy.tokens.Doc,
    n_choices_per_graph_element: int | None = 2,
    n_choices: int = 1,
    filter_by_at_least_2_atoms_and_1_compound: bool = True
) -> Sequence[str]:
    # It generates hard negatives by replacing a token with an antonym or a "cousin" co-hyponym,
    # as in SugarCrepe's REPLACE.
    # Their code is not available, so I reimplemented it based on the paper description and
    # the generated negatives they provide at
    # https://github.com/RAIVNLab/sugar-crepe/issues/4#issuecomment-1747797289.
    # One thing I noticed in their generated negatives is that they have up to 2 per element in
    # the graph (entity, relation, or attribute), so I imitate that here.
    #
    # Some known differences with the original algorithm:
    # * We don't do the BERT filtering because it'd be cumbersome to do during the data loading.
    #   What we do instead is to do a random choice with weights based on prior token probabilities,
    #   obtained from token counts.
    #
    # * Another thing we change is that we try to conjugate the verbs when possible as in the
    # original token, as well as the plurals in nouns, among potential other changes.
    #   We try to keep the casing as the original token as well.
    #
    # * We use a different algorithm for Scene Graph Parsing, as we can't easily access the
    # original one here.
    #
    # * We don't know how they handle the prepositions, so we have our custom way of doing so.
    #
    # * We avoid assigning a token that is a potential synonym (that as a lemma has any
    # synset in common).
    #
    # * We made some modifications to make it efficient to compute during the data loading.
    #   We cache things very heavily (per data loader worker).
    #   In consequence, at the beginning of training, it will be CPU-bound for a few steps,
    #   then gradually using the CPU cores less.
    #
    # I think the reason they use "cousin" co-hyponyms,
    # as opposed to "sibling" ones, is that the colors wouldn't be contemplated.
    # By the way, I'm not sure how they contemplated the colors in the first place,
    # as they are not nouns and thus not part of the hypernym hierarchy.
    # I guess they did similar to us here.

    scene_graph = SCENE_GRAPH_PARSER.parse(doc.text, doc=doc)

    has_at_least_2_atoms = len(scene_graph["entities"]) > 1
    has_at_least_1_compound = scene_graph["relations"] or any(
        e["modifiers"] for e in scene_graph["entities"]
    )

    if filter_by_at_least_2_atoms_and_1_compound and not (
        has_at_least_2_atoms and has_at_least_1_compound
    ):
        return []

    replacements = []

    for token_or_span in itertools.chain(
        (e["span"] for e in scene_graph["entities"]),
        (m for e in scene_graph["entities"] for m in e["modifiers"]),
        (r["span"] for r in scene_graph["relations"])
    ):
        span_replacements, span_replacement_weights = _get_span_replacements(token_or_span)
        span_replacements = weighted_random_sample_without_replacement(
            span_replacements,
            weights=span_replacement_weights,
            k=min(n_choices_per_graph_element, len(span_replacements))
        )
        replacements.extend(span_replacements)

    hard_negatives = [
        _recreate_text_with_replaced_token(
            token_or_span, _prepare_token_for_replacement(token_or_span, new_token_text)
        ) for token_or_span, new_token_text in replacements
    ]

    return random.sample(hard_negatives, k=min(n_choices, len(hard_negatives)))
