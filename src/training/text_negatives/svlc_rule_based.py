import random

import numpy as np

from .svlc_learning import (
    action_list, color_list, counts_list, counts_list_capital, material_list, nouns_list, size_list,
    state_list,
)

ATTRIBUTE_LISTS = {
    "color": color_list,
    "action": action_list,
    "size": size_list,
    "state": state_list,
    "material": material_list,
    "noun": nouns_list,
    "counts": counts_list,
    "counts_capital": counts_list_capital,
}


def is_positive(attributes, doc):
    """Check if any of the attributes are mentioned in the text."""
    # Utilize set for faster intersection operation
    attribute_set = set(attributes)
    return any(token.text in attribute_set for token in doc)


def create_rule_based_negatives(doc, n_choices=1):
    """Create negative samples by identifying and replacing attributes in the caption."""
    attribute_types = list(ATTRIBUTE_LISTS.keys())

    present_attributes = [
        is_positive(ATTRIBUTE_LISTS[attr_type], doc) for attr_type in attribute_types
    ]
    possible_types_indices = np.nonzero(present_attributes)[0]

    if not possible_types_indices.size:
        return []  # Return an empty list if no attributes are present

    selected_index = random.choice(possible_types_indices)
    attribute_type = attribute_types[selected_index]
    attributes = ATTRIBUTE_LISTS[attribute_type]
    attribute_set = set(attributes)

    positives_in_caption = [token.text for token in doc if token.text in attribute_set]

    if not positives_in_caption:
        return []  # Return an empty list if no positive matches found

    negatives_text = []
    remaining_attributes = list(set(attributes) - set(positives_in_caption))

    for _ in range(n_choices):
        negative_attribute = random.choice(remaining_attributes)
        attribute_to_replace = random.choice(positives_in_caption)

        new_tokens = []
        for token in doc:
            if token.text == attribute_to_replace:
                new_tokens.append(negative_attribute)
            else:
                new_tokens.append(token.text)

            # Check if the original token is followed by a space
            if token.whitespace_:
                new_tokens.append(" ")

        new_text = "".join(new_tokens).strip()
        negatives_text.append(new_text)

        # Ensure the same negative is not reused
        remaining_attributes.remove(negative_attribute)

    return negatives_text
