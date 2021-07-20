from pkrex.models.utils import bio_to_entity_tokens, assign_index_to_spans, generate_all_possible_rels, \
    filter_not_allowed_rels, get_ctx_token_offsets, get_ent_and_ctx_token_offsets, associate_triplets_with_rels

EXAMPLE = ["B-PK", "B-VALUE", "O", "O", "0", "B-UNITS", "I-UNITS", "I-PK", "B-COMPARE", "I-COMPARE", "O", "O",
           "B-VALUE", "I-VALUE"]
EXAMPLE_2 = ["B-PK", "I-PK", "I-PK", "I-PK", "O", "B-UNITS", "I-UNITS", "O", "B-VALUE"]
EXAMPLE_3 = ["O", "O", "O", "O", "O", "O", "O", "O", "O"]
EXAMPLE_4 = ["O", "O", "O", "O", "O", "B-PK", "O", "O", "B-PK"]


def test_bio_to_entity_tokens():
    entity_tok_idxs = bio_to_entity_tokens(inp_bio_seq=EXAMPLE)
    assert [{'token_start': 0, 'token_end': 0, 'label': 'PK'},
            {'token_start': 1, 'token_end': 1, 'label': 'VALUE'},
            {'token_start': 5, 'token_end': 6, 'label': 'UNITS'},
            {'token_start': 8, 'token_end': 9, 'label': 'COMPARE'},
            {'token_start': 12, 'token_end': 13, 'label': 'VALUE'}] == entity_tok_idxs
    entity_strings = []
    for entity_toks in entity_tok_idxs:
        entity_strings.append(EXAMPLE[entity_toks['token_start']:entity_toks['token_end'] + 1])

    assert entity_strings == [
        ['B-PK'],
        ['B-VALUE'],
        ['B-UNITS', 'I-UNITS'],
        ['B-COMPARE', 'I-COMPARE'],
        ['B-VALUE', 'I-VALUE']
    ]
    assert bio_to_entity_tokens(inp_bio_seq=EXAMPLE_2) == [{'token_start': 0, 'token_end': 3, 'label': 'PK'},
                                                           {'token_start': 5, 'token_end': 6, 'label': 'UNITS'},
                                                           {'token_start': 8, 'token_end': 8, 'label': 'VALUE'}]
    assert bio_to_entity_tokens(inp_bio_seq=EXAMPLE_3) == []
    assert bio_to_entity_tokens(inp_bio_seq=EXAMPLE_4) == [{'token_start': 5, 'token_end': 5, 'label': 'PK'},
                                                           {'token_start': 8, 'token_end': 8, 'label': 'PK'}]


POSSIBLE_RELS_TRUE = [
    {"head": {'token_start': 0, 'token_end': 0, 'label': 'PK', 'ent_id': 0},
     "child": {'token_start': 1, 'token_end': 1, 'label': 'VALUE', 'ent_id': 1}},
    {"head": {'token_start': 0, 'token_end': 0, 'label': 'PK', 'ent_id': 0},
     "child": {'token_start': 5, 'token_end': 6, 'label': 'UNITS', 'ent_id': 2}},
    {"head": {'token_start': 0, 'token_end': 0, 'label': 'PK', 'ent_id': 0},
     "child": {'token_start': 8, 'token_end': 9, 'label': 'COMPARE', 'ent_id': 3}},
    {"head": {'token_start': 0, 'token_end': 0, 'label': 'PK', 'ent_id': 0},
     "child": {'token_start': 12, 'token_end': 13, 'label': 'VALUE', 'ent_id': 4}},

    {"head": {'token_start': 1, 'token_end': 1, 'label': 'VALUE', 'ent_id': 1},
     "child": {'token_start': 5, 'token_end': 6, 'label': 'UNITS', 'ent_id': 2}},
    {"head": {'token_start': 1, 'token_end': 1, 'label': 'VALUE', 'ent_id': 1},
     "child": {'token_start': 8, 'token_end': 9, 'label': 'COMPARE', 'ent_id': 3}},
    {"head": {'token_start': 1, 'token_end': 1, 'label': 'VALUE', 'ent_id': 1},
     "child": {'token_start': 12, 'token_end': 13, 'label': 'VALUE', 'ent_id': 4}},

    {"head": {'token_start': 5, 'token_end': 6, 'label': 'UNITS', 'ent_id': 2},
     "child": {'token_start': 8, 'token_end': 9, 'label': 'COMPARE', 'ent_id': 3}},
    {"head": {'token_start': 5, 'token_end': 6, 'label': 'UNITS', 'ent_id': 2},
     "child": {'token_start': 12, 'token_end': 13, 'label': 'VALUE', 'ent_id': 4}},

    {"head": {'token_start': 8, 'token_end': 9, 'label': 'COMPARE', 'ent_id': 3},
     "child": {'token_start': 12, 'token_end': 13, 'label': 'VALUE', 'ent_id': 4}},

]

POSSIBLE_AND_ALLOWED_TRUE = [
    {"head": {'token_start': 0, 'token_end': 0, 'label': 'PK', 'ent_id': 0},
     "child": {'token_start': 1, 'token_end': 1, 'label': 'VALUE', 'ent_id': 1}},
    {"head": {'token_start': 0, 'token_end': 0, 'label': 'PK', 'ent_id': 0},
     "child": {'token_start': 12, 'token_end': 13, 'label': 'VALUE', 'ent_id': 4}},
    {"head": {'token_start': 1, 'token_end': 1, 'label': 'VALUE', 'ent_id': 1},
     "child": {'token_start': 5, 'token_end': 6, 'label': 'UNITS', 'ent_id': 2}},
    {"head": {'token_start': 1, 'token_end': 1, 'label': 'VALUE', 'ent_id': 1},
     "child": {'token_start': 8, 'token_end': 9, 'label': 'COMPARE', 'ent_id': 3}},
    {"head": {'token_start': 1, 'token_end': 1, 'label': 'VALUE', 'ent_id': 1},
     "child": {'token_start': 12, 'token_end': 13, 'label': 'VALUE', 'ent_id': 4}},
    {"head": {'token_start': 5, 'token_end': 6, 'label': 'UNITS', 'ent_id': 2},
     "child": {'token_start': 12, 'token_end': 13, 'label': 'VALUE', 'ent_id': 4}},
    {"head": {'token_start': 8, 'token_end': 9, 'label': 'COMPARE', 'ent_id': 3},
     "child": {'token_start': 12, 'token_end': 13, 'label': 'VALUE', 'ent_id': 4}}
]

POSSIBLE_AND_ALLOWED_TRUE_2 = [{'head': {'token_start': 0, 'token_end': 3, 'label': 'PK', 'ent_id': 0},
                                'child': {'token_start': 8, 'token_end': 8, 'label': 'VALUE', 'ent_id': 2}},
                               {'head': {'token_start': 5, 'token_end': 6, 'label': 'UNITS', 'ent_id': 1},
                                'child': {'token_start': 8, 'token_end': 8, 'label': 'VALUE', 'ent_id': 2}}]


def test_pairwise_generator():
    entity_tok_idxs = bio_to_entity_tokens(inp_bio_seq=EXAMPLE)
    indexed_spans = assign_index_to_spans(span_list=entity_tok_idxs)
    possible_rels = generate_all_possible_rels(inp_entities=indexed_spans)
    assert len(possible_rels) == len(POSSIBLE_RELS_TRUE)
    possible_and_allowed_rels = filter_not_allowed_rels(inp_possible_rels=possible_rels)
    assert possible_and_allowed_rels == POSSIBLE_AND_ALLOWED_TRUE

    entity_tok_idxs_2 = bio_to_entity_tokens(inp_bio_seq=EXAMPLE_2)
    possible_rels_2 = generate_all_possible_rels(inp_entities=assign_index_to_spans(entity_tok_idxs_2))
    pos_and_allowed_2 = filter_not_allowed_rels(possible_rels_2)
    assert pos_and_allowed_2 == POSSIBLE_AND_ALLOWED_TRUE_2
    pos_and_allowed_3 = filter_not_allowed_rels(
        generate_all_possible_rels(assign_index_to_spans(bio_to_entity_tokens(EXAMPLE_3))))
    assert pos_and_allowed_3 == []
    pos_and_allowed_4 = filter_not_allowed_rels(
        generate_all_possible_rels(assign_index_to_spans(bio_to_entity_tokens(EXAMPLE_4))))
    assert pos_and_allowed_4 == []


def test_intermediate_tokens():
    ctx_tokens_offsets = [get_ctx_token_offsets(inp_rel=r) for r in POSSIBLE_AND_ALLOWED_TRUE]
    assert ctx_tokens_offsets == [
        (0, 1), (0, 12), (1, 5), (1, 8), (1, 12), (6, 12), (9, 12)
    ]
    s, e = get_ctx_token_offsets(inp_rel={"head": {'token_start': 12, 'token_end': 13, 'label': 'VALUE', 'ent_id': 4},
                                          "child": {'token_start': 5, 'token_end': 6, 'label': 'UNITS', 'ent_id': 2}})
    assert s == 6 and e == 12


EXPECTED_TRIPLETS = [
    (['B-PK'],
     [],
     ['B-VALUE']),
    (['B-PK'],
     ['B-VALUE', 'O', 'O', '0', 'B-UNITS', 'I-UNITS', 'I-PK', 'B-COMPARE', 'I-COMPARE', 'O', 'O'],
     ['B-VALUE', 'I-VALUE']),
    (['B-VALUE'],
     ['O', 'O', '0'],
     ['B-UNITS', 'I-UNITS']),
    (['B-VALUE'],
     ['O', 'O', '0', 'B-UNITS', 'I-UNITS', 'I-PK'],
     ['B-COMPARE', 'I-COMPARE']),
    (['B-VALUE'],
     ['O', 'O', '0', 'B-UNITS', 'I-UNITS', 'I-PK', 'B-COMPARE', 'I-COMPARE', 'O', 'O'],
     ['B-VALUE', 'I-VALUE']),
    (['B-UNITS', 'I-UNITS'],
     ['I-PK', 'B-COMPARE', 'I-COMPARE', 'O', 'O'],
     ['B-VALUE', 'I-VALUE']),
    (['B-COMPARE', 'I-COMPARE'],
     ['O', 'O'],
     ['B-VALUE', 'I-VALUE'])
]

EXPECTED_TRIPLETS_2 = [
    (["B-PK", "I-PK", "I-PK", "I-PK"],
     ["O", "B-UNITS", "I-UNITS", "O"],
     ["B-VALUE"]),

    (["B-UNITS", "I-UNITS"],
     ["O"],
     ["B-VALUE"]),
]


def test_all_token_offsets():
    all_triplets = []
    for r in POSSIBLE_AND_ALLOWED_TRUE:
        lo, ro, ctxo = get_ent_and_ctx_token_offsets(r)
        str_triplet = (EXAMPLE[lo[0]:lo[1]], EXAMPLE[ctxo[0]:ctxo[1]], EXAMPLE[ro[0]:ro[1]])
        all_triplets.append(str_triplet)
    assert all_triplets == EXPECTED_TRIPLETS

    all_triplets = []
    for r in POSSIBLE_AND_ALLOWED_TRUE_2:
        lo, ro, ctxo = get_ent_and_ctx_token_offsets(r)
        str_triplet = (EXAMPLE_2[lo[0]:lo[1]], EXAMPLE_2[ctxo[0]:ctxo[1]], EXAMPLE_2[ro[0]:ro[1]])
        all_triplets.append(str_triplet)
    assert all_triplets == EXPECTED_TRIPLETS_2


EXAMPLE_POSSIBLE_TRIPLETS_BIG = [
    ((2, 6), (14, 17), (6, 14)), ((2, 6), (27, 30), (6, 27)), ((2, 6), (47, 50), (6, 47)), ((2, 6), (60, 63), (6, 60)),
    ((2, 6), (80, 83), (6, 80)), ((2, 6), (93, 96), (6, 93)), ((14, 17), (17, 18), (17, 17)),
    ((14, 17), (27, 30), (17, 27)), ((14, 17), (30, 31), (17, 30)), ((14, 17), (47, 50), (17, 47)),
    ((14, 17), (50, 51), (17, 50)), ((14, 17), (60, 63), (17, 60)), ((14, 17), (63, 64), (17, 63)),
    ((14, 17), (80, 83), (17, 80)), ((14, 17), (83, 84), (17, 83)), ((14, 17), (93, 96), (17, 93)),
    ((14, 17), (96, 97), (17, 96)), ((17, 18), (27, 30), (18, 27)), ((17, 18), (47, 50), (18, 47)),
    ((17, 18), (60, 63), (18, 60)), ((17, 18), (80, 83), (18, 80)), ((17, 18), (93, 96), (18, 93)),
    ((27, 30), (30, 31), (30, 30)), ((27, 30), (47, 50), (30, 47)), ((27, 30), (50, 51), (30, 50)),
    ((27, 30), (60, 63), (30, 60)), ((27, 30), (63, 64), (30, 63)), ((27, 30), (80, 83), (30, 80)),
    ((27, 30), (83, 84), (30, 83)), ((27, 30), (93, 96), (30, 93)), ((27, 30), (96, 97), (30, 96)),
    ((30, 31), (47, 50), (31, 47)), ((30, 31), (60, 63), (31, 60)), ((30, 31), (80, 83), (31, 80)),
    ((30, 31), (93, 96), (31, 93)), ((47, 50), (50, 51), (50, 50)), ((47, 50), (60, 63), (50, 60)),
    ((47, 50), (63, 64), (50, 63)), ((47, 50), (80, 83), (50, 80)), ((47, 50), (83, 84), (50, 83)),
    ((47, 50), (93, 96), (50, 93)), ((47, 50), (96, 97), (50, 96)), ((50, 51), (60, 63), (51, 60)),
    ((50, 51), (80, 83), (51, 80)), ((50, 51), (93, 96), (51, 93)), ((60, 63), (63, 64), (63, 63)),
    ((60, 63), (80, 83), (63, 80)), ((60, 63), (83, 84), (63, 83)), ((60, 63), (93, 96), (63, 93)),
    ((60, 63), (96, 97), (63, 96)), ((63, 64), (80, 83), (64, 80)), ((63, 64), (93, 96), (64, 93)),
    ((80, 83), (83, 84), (83, 83)), ((80, 83), (93, 96), (83, 93)), ((80, 83), (96, 97), (83, 96)),
    ((83, 84), (93, 96), (84, 93)), ((93, 96), (96, 97), (96, 96))
]

EXAMPLE_RAW_LABELS_BIG = [
    {'head_span': {'start': 4, 'end': 8, 'token_start': 2, 'token_end': 5, 'label': 'PK', 'ent_id': 0},
     'child_span': {'start': 28, 'end': 33, 'token_start': 14, 'token_end': 16, 'label': 'VALUE', 'ent_id': 1},
     'head_span_idx': 0, 'child_span_idx': 1, 'label': 'C_VAL'},
    {'head_span': {'start': 4, 'end': 8, 'token_start': 2, 'token_end': 5, 'label': 'PK', 'ent_id': 0},
     'child_span': {'start': 62, 'end': 67, 'token_start': 27, 'token_end': 29, 'label': 'VALUE', 'ent_id': 3},
     'head_span_idx': 0, 'child_span_idx': 3, 'label': 'C_VAL'},
    {'head_span': {'start': 4, 'end': 8, 'token_start': 2, 'token_end': 5, 'label': 'PK', 'ent_id': 0},
     'child_span': {'start': 99, 'end': 104, 'token_start': 47, 'token_end': 49, 'label': 'VALUE', 'ent_id': 5},
     'head_span_idx': 0, 'child_span_idx': 5, 'label': 'C_VAL'},
    {'head_span': {'start': 4, 'end': 8, 'token_start': 2, 'token_end': 5, 'label': 'PK', 'ent_id': 0},
     'child_span': {'start': 132, 'end': 137, 'token_start': 60, 'token_end': 62, 'label': 'VALUE', 'ent_id': 7},
     'head_span_idx': 0, 'child_span_idx': 7, 'label': 'C_VAL'},
    {'head_span': {'start': 4, 'end': 8, 'token_start': 2, 'token_end': 5, 'label': 'PK', 'ent_id': 0},
     'child_span': {'start': 169, 'end': 174, 'token_start': 80, 'token_end': 82, 'label': 'VALUE', 'ent_id': 9},
     'head_span_idx': 0, 'child_span_idx': 9, 'label': 'C_VAL'},
    {'head_span': {'start': 4, 'end': 8, 'token_start': 2, 'token_end': 5, 'label': 'PK', 'ent_id': 0},
     'child_span': {'start': 202, 'end': 207, 'token_start': 93, 'token_end': 95, 'label': 'VALUE', 'ent_id': 11},
     'head_span_idx': 0, 'child_span_idx': 11, 'label': 'C_VAL'},
    {'head_span': {'start': 34, 'end': 38, 'token_start': 17, 'token_end': 17, 'label': 'UNITS', 'ent_id': 2},
     'child_span': {'start': 28, 'end': 33, 'token_start': 14, 'token_end': 16, 'label': 'VALUE', 'ent_id': 1},
     'head_span_idx': 2, 'child_span_idx': 1, 'label': 'RELATED'},
    {'head_span': {'start': 68, 'end': 72, 'token_start': 30, 'token_end': 30, 'label': 'UNITS', 'ent_id': 4},
     'child_span': {'start': 62, 'end': 67, 'token_start': 27, 'token_end': 29, 'label': 'VALUE', 'ent_id': 3},
     'head_span_idx': 4, 'child_span_idx': 3, 'label': 'RELATED'},
    {'head_span': {'start': 105, 'end': 109, 'token_start': 50, 'token_end': 50, 'label': 'UNITS', 'ent_id': 6},
     'child_span': {'start': 99, 'end': 104, 'token_start': 47, 'token_end': 49, 'label': 'VALUE', 'ent_id': 5},
     'head_span_idx': 6, 'child_span_idx': 5, 'label': 'RELATED'},
    {'head_span': {'start': 175, 'end': 179, 'token_start': 83, 'token_end': 83, 'label': 'UNITS', 'ent_id': 10},
     'child_span': {'start': 169, 'end': 174, 'token_start': 80, 'token_end': 82, 'label': 'VALUE', 'ent_id': 9},
     'head_span_idx': 10, 'child_span_idx': 9, 'label': 'RELATED'},
    {'head_span': {'start': 208, 'end': 212, 'token_start': 96, 'token_end': 96, 'label': 'UNITS', 'ent_id': 12},
     'child_span': {'start': 202, 'end': 207, 'token_start': 93, 'token_end': 95, 'label': 'VALUE', 'ent_id': 11},
     'head_span_idx': 12, 'child_span_idx': 11, 'label': 'RELATED'},
    {'head_span': {'start': 138, 'end': 142, 'token_start': 63, 'token_end': 63, 'label': 'UNITS', 'ent_id': 8},
     'child_span': {'start': 132, 'end': 137, 'token_start': 60, 'token_end': 62, 'label': 'VALUE', 'ent_id': 7},
     'head_span_idx': 8, 'child_span_idx': 7, 'label': 'RELATED'}]


def test_associate_triplets_with_rels():
    out_labels = associate_triplets_with_rels(inp_rel_labels=EXAMPLE_RAW_LABELS_BIG,
                                              inp_triplets=EXAMPLE_POSSIBLE_TRIPLETS_BIG)

    assert out_labels == ['C_VAL', 'C_VAL', 'C_VAL', 'C_VAL', 'C_VAL', 'C_VAL', 'RELATED', 'NO_RELATION',
                          'NO_RELATION', 'NO_RELATION', 'NO_RELATION', 'NO_RELATION', 'NO_RELATION', 'NO_RELATION',
                          'NO_RELATION', 'NO_RELATION', 'NO_RELATION', 'NO_RELATION', 'NO_RELATION', 'NO_RELATION',
                          'NO_RELATION', 'NO_RELATION', 'RELATED', 'NO_RELATION', 'NO_RELATION', 'NO_RELATION',
                          'NO_RELATION', 'NO_RELATION', 'NO_RELATION', 'NO_RELATION', 'NO_RELATION', 'NO_RELATION',
                          'NO_RELATION', 'NO_RELATION', 'NO_RELATION', 'RELATED', 'NO_RELATION', 'NO_RELATION',
                          'NO_RELATION', 'NO_RELATION', 'NO_RELATION', 'NO_RELATION', 'NO_RELATION', 'NO_RELATION',
                          'NO_RELATION', 'RELATED', 'NO_RELATION', 'NO_RELATION', 'NO_RELATION', 'NO_RELATION',
                          'NO_RELATION', 'NO_RELATION', 'RELATED', 'NO_RELATION', 'NO_RELATION', 'NO_RELATION',
                          'RELATED']



