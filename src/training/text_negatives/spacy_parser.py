# Adapted from `sng_parser.backend.spacy_parser`. Modified to return the spaCy objects instead of just strings.
# https://github.com/vacancy/SceneGraphParser
import importlib
import logging
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import spacy.tokens
from sng_parser import Parser, database

importlib.import_module("sng_parser.backends.spacy_parser")


def _locate_noun(chunks: Iterable[spacy.tokens.Span], i: int) -> int | None:
    for j, c in enumerate(chunks):
        if c.start <= i < c.end:
            return j
    return None


def _flatten_conjunction(node: spacy.tokens.Token) -> Iterator[spacy.tokens.Token]:
    yield node
    for c in node.children:
        if c.dep_ == "conj":
            yield c


@Parser.register_backend
# We don't use `SpacyParser` as the superclass because of a bug. See https://github.com/vacancy/SceneGraphParser/pull/25
class CustomSpacyParser(Parser._backend_registry["spacy"]):
    __identifier__ = "custom-spacy"

    def parse(
        self,
        sentence: str | None = None,
        doc: spacy.tokens.Doc | None = None,
        return_doc: bool = False
    ) -> Mapping[str, Any] | tuple[Mapping[str, Any], spacy.tokens.Doc]:
        """
        The spaCy-based parser parse the sentence into scene graphs based on the dependency parsing
        of the sentence by spaCy.

        All entities (nodes) of the graph come from the noun chunks in the sentence. And the dependencies
        between noun chunks are used for determining the relations among these entities.

        The parsing is performed in three steps:

            1. Find all the noun chunks as the entities, and resolve the modifiers on them.
            2. Determine the subject of verbs (including nsubj, acl and pobjpass). Please refer to the comments
            in the code for better explanation.
            3. Determine all the relations among entities.
        """
        doc = doc or self.nlp(sentence)

        # Step 1: determine the entities.
        entities = []
        entity_chunks = []
        for entity in doc.noun_chunks:
            # Ignore pronouns such as "it".
            if entity.root.lemma_ == "-PRON-":
                continue

            ent = {"span": entity, "head": [entity.root], "modifiers": []}

            visited_nodes = set()

            def dfs(node: spacy.tokens.Token) -> None:
                if node not in visited_nodes:  # Sometimes the dependency graph is erroneously cyclic.
                    visited_nodes.add(node)

                    for x in node.children:
                        if x.dep_ == "det":
                            ent["modifiers"].append(x)
                        elif x.dep_ == "nummod":
                            ent["modifiers"].append(x)
                        elif x.dep_ == "amod":
                            for y in _flatten_conjunction(x):
                                y.dep = x.dep  # Note we're mutating the original doc tokens.
                                y.dep_ = x.dep_
                                ent["modifiers"].append(y)
                        elif x.dep_ == "compound":
                            ent["head"].insert(0, x)
                            dfs(x)

            try:
                dfs(entity.root)
            except RecursionError:
                logging.error(
                    f"Dependency graph is cyclic when parsing the scene graph."
                    f" Ignoring the subtree rooted at '{entity.root}' for the entity '{entity}'"
                    f" for the doc '{doc}'."
                )

            ent["type"] = "scene" if database.is_scene_noun(
                " ".join(t.lemma_ for t in ent["head"])
            ) else "unknown"

            entities.append(ent)
            entity_chunks.append(entity)

        # Step 2: determine the subject of the verbs.
        # To handle the situation where multiple nouns may be the same word,
        # the tokens are represented by their position in the sentence instead of their text.
        relation_subj = {}
        for token in doc:
            # E.g., A [woman] is [playing] the piano.
            if token.dep_ == "nsubj":
                relation_subj[token.head.i] = token.i
            # E.g., A [woman] [playing] the piano...
            elif token.dep_ == "acl":
                relation_subj[token.i] = token.head.i
            # E.g., The piano is [played] by a [woman].
            elif token.dep_ == "pobj" and token.head.dep_ == "agent" and token.head.head.pos_ == "VERB":
                relation_subj[token.head.head.i] = token.i

        # Step 3: determine the relations.
        relations = []
        fake_noun_marks = set()
        for entity in doc.noun_chunks:
            # Again, the subjects and the objects are represented by their position.
            relation = None

            # E.g., A woman is [playing] the [piano].
            # E.g., The woman [is] a [pianist].
            if entity.root.dep_ in ("dobj", "attr") and entity.root.head.i in relation_subj:
                relation = {
                    "subject": relation_subj[entity.root.head.i],
                    "object": entity.root.i,
                    "span": doc[entity.root.head.i:entity.root.head.i + 1],
                }
            elif entity.root.dep_ == "pobj":
                # E.g., The piano is played [by] a [woman].
                if entity.root.head.dep_ == "agent":
                    pass
                # E.g., A [woman] is playing with the piano in the [room].
                elif (
                    entity.root.head.head.pos_ == "VERB"
                    and entity.root.head.head.i + 1 == entity.root.head.i and database.
                    is_phrasal_verb(entity.root.head.head.lemma_ + " " + entity.root.head.lemma_)
                ) and entity.root.head.head.i in relation_subj:
                    relation = {
                        "subject": relation_subj[entity.root.head.head.i],
                        "object": entity.root.i,
                        "span": doc[entity.root.head.i - 1:entity.root.head.i + 1],
                    }
                # E.g., A [woman] is playing the piano in the [room]. Note that room.head.head == playing.
                # E.g., A [woman] playing the piano in the [room].
                elif (
                    entity.root.head.head.pos_ == "VERB" or entity.root.head.head.dep_ == "acl"
                ) and entity.root.head.head.i in relation_subj:
                    relation = {
                        "subject": relation_subj[entity.root.head.head.i],
                        "object": entity.root.i,
                        "span": doc[entity.root.head.i:entity.root.head.i + 1],
                    }
                # E.g., A [woman] in front of a [piano].
                elif (
                    entity.root.head.head.dep_ == "pobj" and database.is_phrasal_prep(
                        doc[entity.root.head.head.head.i:entity.root.head.i + 1].text.lower()
                    )
                ):
                    fake_noun_marks.add(entity.root.head.head.i)
                    relation = {
                        "subject": entity.root.head.head.head.head.i,
                        "object": entity.root.i,
                        "span": doc[entity.root.head.head.head.i:entity.root.head.i + 1],
                    }
                # E.g., A [piano] in the [room].
                elif entity.root.head.head.pos_ == "NOUN":
                    relation = {
                        "subject": entity.root.head.head.i,
                        "object": entity.root.i,
                        "span": doc[entity.root.head.i:entity.root.head.i + 1],
                    }
                # E.g., A [piano] next to a [woman].
                elif entity.root.head.head.dep_ in (
                    "amod", "advmod"
                ) and entity.root.head.head.head.pos_ == "NOUN":
                    relation = {
                        "subject": entity.root.head.head.head.i,
                        "object": entity.root.i,
                        "span": doc[entity.root.head.head.i:entity.root.head.i + 1],
                    }
                # E.g., A [woman] standing next to a [piano].
                elif (
                    entity.root.head.head.dep_ in ("amod", "advmod")
                    and entity.root.head.head.head.pos_ == "VERB"
                    and entity.root.head.head.head.i in relation_subj
                ):
                    relation = {
                        "subject": relation_subj[entity.root.head.head.head.i],
                        "object": entity.root.i,
                        "span": doc[entity.root.head.head.i:entity.root.head.i + 1],
                    }
                # E.g., A [woman] is playing the [piano] in the room
                elif entity.root.head.head.dep_ == "VERB" and entity.root.head.head.i in relation_subj:
                    relation = {
                        "subject": relation_subj[entity.root.head.head.i],
                        "object": entity.root.i,
                        "span": doc[entity.root.head.i:entity.root.head.i + 1],
                    }
                # E.g., A [piano] is in the [room].
                elif entity.root.head.head.pos_ == "AUX" and entity.root.head.head.i in relation_subj:
                    relation = {
                        "subject": relation_subj[entity.root.head.head.i],
                        "object": entity.root.i,
                        "span": doc[entity.root.head.i:entity.root.head.i + 1],
                    }

            # E.g., The [piano] is played by a [woman].
            elif entity.root.dep_ == "nsubjpass" and entity.root.head.i in relation_subj:
                # Here, we reverse the passive phrase. I.e., subjpass -> obj and objpass -> subj.
                relation = {
                    "subject": relation_subj[entity.root.head.i],
                    "object": entity.root.i,
                    "span": doc[entity.root.head.i:entity.root.head.i + 1],
                }

            if relation:
                relations.append(relation)

        # Apply the `fake_noun_marks`.
        entities = [e for e, ec in zip(entities, entity_chunks) if ec.root.i not in fake_noun_marks]
        entity_chunks = [ec for ec in entity_chunks if ec.root.i not in fake_noun_marks]

        filtered_relations = []
        for r in relations:
            # Use a helper function to map the subj/obj represented by the position
            # back to one of the entity nodes.
            for x in _flatten_conjunction(doc[r["subject"]]):
                for y in _flatten_conjunction(doc[r["object"]]):
                    r_with_indices = {
                        **r,
                        "subject": _locate_noun(entity_chunks, x.i),
                        "object": _locate_noun(entity_chunks, y.i),
                    }
                    if r_with_indices["subject"] is not None and r_with_indices["object"
                                                                                ] is not None:
                        filtered_relations.append(r_with_indices)

        if return_doc:
            return {"entities": entities, "relations": filtered_relations}, doc
        return {"entities": entities, "relations": filtered_relations}
