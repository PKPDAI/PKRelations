from pkrex.utils import resolve_overlapping_spans

ex_sentence_1 = 'For R-flurbiprofen, the rate constant was 1.'
ex_sentence_2 = 'For R-0-flurbiprofen, the rate constant was 1.'

EXAMPLE_NON_OVERLAPPING = [
    {'start': 4, 'end': 18, 'label': 'CHEMICAL', 'ids': ['CHEBI:38666', 'BERN:5816803']},
    {'start': 24, 'end': 37, 'label': 'PK'},
    {'start': 42, 'end': 43, 'label': 'VALUE'},
]

EXAMPLE_OVERLAPPING = [
    {'start': 4, 'end': 20, 'label': 'CHEMICAL', 'ids': ['CHEBI:38666', 'BERN:5816803']},
    {'start': 6, 'end': 7, 'label': 'VALUE'},
    {'start': 26, 'end': 39, 'label': 'PK'},
    {'start': 44, 'end': 45, 'label': 'VALUE'},
]


def test_resolve_overlapping():
    out_spans_2 = resolve_overlapping_spans(EXAMPLE_OVERLAPPING)
    out_spans_1 = resolve_overlapping_spans(EXAMPLE_NON_OVERLAPPING)
    print(out_spans_1)
    print(out_spans_2)
    assert out_spans_1 == EXAMPLE_NON_OVERLAPPING
    assert out_spans_2 == [{'start': 4, 'end': 20, 'label': 'CHEMICAL', 'ids': ['CHEBI:38666', 'BERN:5816803']},
                           {'start': 26, 'end': 39, 'label': 'PK'}, {'start': 44, 'end': 45, 'label': 'VALUE'}]
