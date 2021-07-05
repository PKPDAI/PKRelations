from pkrex.models.utils import bio_to_entity_tokens

EXAMPLE = ["B-PK", "B-VAL", "O", "0", "B-UNITS", "I-UNITS", "I-PK", "B-COMPARE", "I-COMPARE"]


def test_bio_to_entity_tokens():
    entity_tok_idxs = bio_to_entity_tokens(inp_bio_seq=EXAMPLE)
    assert [{'token_start': 0, 'token_end': 0, 'label': 'PK'},
            {'token_start': 1, 'token_end': 1, 'label': 'VAL'},
            {'token_start': 4, 'token_end': 5, 'label': 'UNITS'},
            {'token_start': 7, 'token_end': 8, 'label': 'COMPARE'}] == entity_tok_idxs


