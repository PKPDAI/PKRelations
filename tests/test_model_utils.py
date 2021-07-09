from pkrex.models.utils import bio_to_entity_tokens, assign_index_to_spans, generate_all_possible_rels, \
    filter_not_allowed_rels

EXAMPLE = ["B-PK",
           "B-VALUE",
           "O", "O" "0",
           "B-UNITS", "I-UNITS",
           "I-PK",
           "B-COMPARE", "I-COMPARE",
           "O", "O",
           "B-VALUE", "I-VALUE"
           ]


def test_bio_to_entity_tokens():
    entity_tok_idxs = bio_to_entity_tokens(inp_bio_seq=EXAMPLE)
    assert [{'token_start': 0, 'token_end': 0, 'label': 'PK'},
            {'token_start': 1, 'token_end': 1, 'label': 'VALUE'},
            {'token_start': 4, 'token_end': 5, 'label': 'UNITS'},
            {'token_start': 7, 'token_end': 8, 'label': 'COMPARE'},
            {'token_start': 11, 'token_end': 12, 'label': 'VALUE'}] == entity_tok_idxs


POSSIBLE_RELS_TRUE = [

    {"head": {'token_start': 0, 'token_end': 0, 'label': 'PK', 'ent_id': 0},
     "child": {'token_start': 1, 'token_end': 1, 'label': 'VALUE', 'ent_id': 1}},
    {"head": {'token_start': 0, 'token_end': 0, 'label': 'PK', 'ent_id': 0},
     "child": {'token_start': 4, 'token_end': 5, 'label': 'UNITS', 'ent_id': 2}},
    {"head": {'token_start': 0, 'token_end': 0, 'label': 'PK', 'ent_id': 0},
     "child": {'token_start': 7, 'token_end': 8, 'label': 'COMPARE', 'ent_id': 3}},
    {"head": {'token_start': 0, 'token_end': 0, 'label': 'PK', 'ent_id': 0},
     "child": {'token_start': 11, 'token_end': 12, 'label': 'VALUE', 'ent_id': 4}},

    {"head": {'token_start': 1, 'token_end': 1, 'label': 'VALUE', 'ent_id': 1},
     "child": {'token_start': 4, 'token_end': 5, 'label': 'UNITS', 'ent_id': 2}},
    {"head": {'token_start': 1, 'token_end': 1, 'label': 'VALUE', 'ent_id': 1},
     "child": {'token_start': 7, 'token_end': 8, 'label': 'COMPARE', 'ent_id': 3}},
    {"head": {'token_start': 1, 'token_end': 1, 'label': 'VALUE', 'ent_id': 1},
     "child": {'token_start': 11, 'token_end': 12, 'label': 'VALUE', 'ent_id': 4}},

    {"head": {'token_start': 4, 'token_end': 5, 'label': 'UNITS', 'ent_id': 2},
     "child": {'token_start': 7, 'token_end': 8, 'label': 'COMPARE', 'ent_id': 3}},
    {"head": {'token_start': 4, 'token_end': 5, 'label': 'UNITS', 'ent_id': 2},
     "child": {'token_start': 11, 'token_end': 12, 'label': 'VALUE', 'ent_id': 4}},

    {"head": {'token_start': 7, 'token_end': 8, 'label': 'COMPARE', 'ent_id': 3},
     "child": {'token_start': 11, 'token_end': 12, 'label': 'VALUE', 'ent_id': 4}},

]

POSSIBLE_AND_ALLOWED_TRUE = [
    {"head": {'token_start': 0, 'token_end': 0, 'label': 'PK', 'ent_id': 0},
     "child": {'token_start': 1, 'token_end': 1, 'label': 'VALUE', 'ent_id': 1}},
    {"head": {'token_start': 0, 'token_end': 0, 'label': 'PK', 'ent_id': 0},
     "child": {'token_start': 11, 'token_end': 12, 'label': 'VALUE', 'ent_id': 4}},
    {"head": {'token_start': 1, 'token_end': 1, 'label': 'VALUE', 'ent_id': 1},
     "child": {'token_start': 4, 'token_end': 5, 'label': 'UNITS', 'ent_id': 2}},
    {"head": {'token_start': 1, 'token_end': 1, 'label': 'VALUE', 'ent_id': 1},
     "child": {'token_start': 7, 'token_end': 8, 'label': 'COMPARE', 'ent_id': 3}},
    {"head": {'token_start': 1, 'token_end': 1, 'label': 'VALUE', 'ent_id': 1},
     "child": {'token_start': 11, 'token_end': 12, 'label': 'VALUE', 'ent_id': 4}},
    {"head": {'token_start': 4, 'token_end': 5, 'label': 'UNITS', 'ent_id': 2},
     "child": {'token_start': 11, 'token_end': 12, 'label': 'VALUE', 'ent_id': 4}},
    {"head": {'token_start': 7, 'token_end': 8, 'label': 'COMPARE', 'ent_id': 3},
     "child": {'token_start': 11, 'token_end': 12, 'label': 'VALUE', 'ent_id': 4}}
]


def test_pairwise_generator():
    entity_tok_idxs = bio_to_entity_tokens(inp_bio_seq=EXAMPLE)
    indexed_spans = assign_index_to_spans(span_list=entity_tok_idxs)
    possible_rels = generate_all_possible_rels(inp_entities=indexed_spans)
    assert len(possible_rels) == len(POSSIBLE_RELS_TRUE)
    possible_and_allowed_rels = filter_not_allowed_rels(inp_possible_rels=possible_rels)
    assert possible_and_allowed_rels == POSSIBLE_AND_ALLOWED_TRUE
    a = 1
