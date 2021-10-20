import random

import pkrex.augmentation as pkaug
from pkrex.annotation_preproc import view_all_entities_terminal

EXAMPLE = "2.2 ng.ml/l and 0.4 ng.10-1"
EXPECTED = "2.2 ng·ml/l and 0.4 ng·10-1"


def test_subs_underscore_dot():
    std_text = pkaug.subs_underscore_dot(inp_mention=EXAMPLE)
    assert std_text == EXPECTED


ANNOTATED_EXAMPLE = {
    "text": "The auc was lower than 3.9 ml*h*kg",
    "spans": [
        dict(start=4, end=7, label="PK"),
        dict(start=12, end=22, label="COMPARE"),
        dict(start=23, end=26, label="VALUE"),
        dict(start=27, end=34, label="UNITS"),
    ],
}

DICT_REPLACE = {
    "PK": [
        ["auc", "area under the curve"],
        ["cl", "clearance"],
    ],
    "UNITS": [
        ["ml*h*kg", "milliliters*h*kg-1"]
    ],
    "COMPARE": [
        ["lower than", "<"]
    ]
}


def check_mention_in_sublist(inp_mention, list_of_lists):
    for sublist in list_of_lists:
        if inp_mention in sublist:
            return sublist
    return []


def augment_sentence(inp_anotation):
    if inp_anotation["spans"]:
        all_original_sans = sorted(inp_anotation["spans"], key=lambda anno: anno['start'])
        augmented_text = inp_anotation['text']
        overall_addition = 0
        out_spans = []
        for span in all_original_sans:
            sp_lab = span['label']
            ent_start = span['start'] + overall_addition
            ent_end = span['end'] + overall_addition
            sp_mention = augmented_text[ent_start:ent_end]
            new_span_annotated = dict(start=ent_start, end=ent_end, label=sp_lab)
            if sp_lab in DICT_REPLACE.keys():
                provisional_candidates = check_mention_in_sublist(inp_mention=sp_mention.lower(),
                                                                  list_of_lists=DICT_REPLACE[sp_lab])
                if provisional_candidates:
                    candidates = [x for x in provisional_candidates if x != sp_mention]
                    if candidates:
                        # MAKE REPLACEMENT
                        new_span_mention = random.choice(candidates)
                        original_sp_len = ent_end - ent_start
                        new_sp_len = len(new_span_mention)
                        to_add = new_sp_len - original_sp_len
                        new_span_annotated['end'] += to_add
                        overall_addition += to_add
                        augmented_text = augmented_text[0:ent_start] + new_span_mention + augmented_text[ent_end:]
            out_spans.append(new_span_annotated)
        out_annotation = dict(text=augmented_text,
                              spans=out_spans)
        return out_annotation
    return None


def test_augment_sentence():
    original_sentece = view_all_entities_terminal(inp_text=ANNOTATED_EXAMPLE['text'],
                                                  character_annotations=ANNOTATED_EXAMPLE['spans'])

    print(original_sentece)

    augmented_sentence = augment_sentence(inp_anotation=ANNOTATED_EXAMPLE)
    if augmented_sentence:
        print(view_all_entities_terminal(inp_text=augmented_sentence['text'],
                                         character_annotations=augmented_sentence['spans']))
    a = 1
