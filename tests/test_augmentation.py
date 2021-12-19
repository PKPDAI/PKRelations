import pkrex.augmentation as pkaug
from pkrex.augmentation_syns import AUGMENT_SYNS
from pkrex.annotation_preproc import view_all_entities_terminal

EXAMPLE = "2.2 ng.ml/l and 0.4 ng.10-1"
EXPECTED = "2.2 ng·ml/l and 0.4 ng·10-1"


def test_subs_underscore_dot():
    std_text = pkaug.subs_underscore_dot(inp_mention=EXAMPLE)
    assert std_text == EXPECTED


ANNOTATED_EXAMPLE = {
    "text": "The auc was lower than 3.9 ml·h·kg and AUC ratio was 6 and 3-fold for midazolam. Clearance ranged from 4.1 to 5.",
    "spans": [
        dict(start=4, end=7, label="PK"),
        dict(start=12, end=22, label="COMPARE"),
        dict(start=23, end=26, label="VALUE"),
        dict(start=27, end=34, label="UNITS"),
        dict(start=53, end=54, label="VALUE"),
        dict(start=59, end=65, label="VALUE"),
        dict(start=103, end=111, label="RANGE")
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
EXPECTED_TEXT = "The area under the curve was < 4.14 ml·h·kg and AUC ratio was 9 and 3-fold for midazolam. Clearance ranged from 4 to 5.68."


def test_augment_sentence():
    original_sentece = view_all_entities_terminal(inp_text=ANNOTATED_EXAMPLE['text'],
                                                  character_annotations=ANNOTATED_EXAMPLE['spans'])

    augmented_sentence = pkaug.augment_sentence(inp_anotation=ANNOTATED_EXAMPLE, replacable_dict=DICT_REPLACE)
    if augmented_sentence:
        augmented_sentence_print = view_all_entities_terminal(inp_text=augmented_sentence['text'],
                                                              character_annotations=augmented_sentence['spans'])
        print("\n", original_sentece)
        print(augmented_sentence_print)

    assert augmented_sentence['text'] == EXPECTED_TEXT
    assert augmented_sentence['spans'] == [{'start': 4, 'end': 24, 'label': 'PK'},
                                           {'start': 29, 'end': 30, 'label': 'COMPARE'},
                                           {'start': 31, 'end': 35, 'label': 'VALUE'},
                                           {'start': 36, 'end': 43, 'label': 'UNITS'},
                                           {'start': 62, 'end': 63, 'label': 'VALUE'},
                                           {'start': 68, 'end': 74, 'label': 'VALUE'},
                                           {'start': 112, 'end': 121, 'label': 'RANGE'}]


ANNOTATED_EXAMPLE_2 = {
    "text": "The resulting estimates of hepatic clearance were 3.91, 5.01, and 4.69 L/h/kg using well-stirred, parallel tube and dispersion models, respectively; these estimates are comparable to the clearance following i.v. dosing of SN30000 (5\u201310 L/hr/kg; Table 2).",
    "spans": [{'start': 27, 'end': 44, 'label': 'PK'},
              {'start': 50, 'end': 54, 'label': 'VALUE'},
              {'start': 56, 'end': 60, 'label': 'VALUE'},
              {'start': 66, 'end': 70, 'label': 'VALUE'},
              {'start': 71, 'end': 77, 'label': 'UNITS'},
              {'start': 187, 'end': 196, 'label': 'PK'},
              {'start': 231, 'end': 235, 'label': 'RANGE'},
              {'start': 236, 'end': 243, 'label': 'UNITS'}]
}


def test_augment_big_dict():
    original_sentece = view_all_entities_terminal(inp_text=ANNOTATED_EXAMPLE_2['text'],
                                                  character_annotations=ANNOTATED_EXAMPLE_2['spans'])

    augmented_sentence = pkaug.augment_sentence(inp_anotation=ANNOTATED_EXAMPLE_2, replacable_dict=AUGMENT_SYNS)

    augmented_sentence_print = view_all_entities_terminal(inp_text=augmented_sentence['text'],
                                                          character_annotations=augmented_sentence['spans'])
    print("\n", original_sentece)
    print(augmented_sentence_print)
    a = 1
