import json

from spacy import displacy

from pkrex.annotation_preproc import fix_incorrect_dvals, view_relations_json, get_cval_entities

with open('test_data.json', 'r') as test_file:
    data = test_file.read()
    test_data = json.loads(data)

# parse file

EXAMPLES_BAD_DIRECTION = test_data["EXAMPLES_BAD_DIRECTION"]


def test_swap_direction_dval():
    ex1 = EXAMPLES_BAD_DIRECTION[0]
    view_relations_json(ex1)
    ex1_modified = fix_incorrect_dvals(ex1)
    view_relations_json(ex1_modified)

    assert len(ex1_modified["relations"]) == len(ex1["relations"])
    c_vals = get_cval_entities(ex1_modified)
    assert c_vals == get_cval_entities(ex1)
    head_spans = [relation["head_span"] for relation in ex1_modified["relations"]]
    child_spans = [relation["child_span"] for relation in ex1_modified["relations"]]
    for central_v_span in c_vals:
        assert central_v_span not in head_spans
        assert central_v_span in child_spans
