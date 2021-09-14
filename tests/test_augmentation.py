import pkrex.augmentation as pkaug

EXAMPLE = "2.2 ng.ml/l and 0.4 ng.10-1"
EXPECTED = "2.2 ng·ml/l and 0.4 ng·10-1"


def test_subs_underscore_dot():
    std_text = pkaug.subs_underscore_dot(inp_mention=EXAMPLE)
    assert std_text == EXPECTED

