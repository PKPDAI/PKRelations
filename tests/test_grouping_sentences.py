from pkrex.utils import get_out_sentence

GROUP_SAMPLE_1 = [
    {"metadata": {"pmid": 10101, "SID": 1, "relevant": False}},
    {"metadata": {"pmid": 10101, "SID": 2, "relevant": True}},
    {"metadata": {"pmid": 10101, "SID": 3, "relevant": False}},
    {"metadata": {"pmid": 10101, "SID": 4, "relevant": False}}
]

GROUP_SAMPLE_2 = [
    {"metadata": {"pmid": 10101, "SID": 20, "relevant": False}},
    {"metadata": {"pmid": 10101, "SID": 21, "relevant": True}},
    {"metadata": {"pmid": 10101, "SID": 22, "relevant": False}},
    {"metadata": {"pmid": 10101, "SID": 23, "relevant": True}}
]


def test_get_out_sentence():
    out_groups = get_out_sentence(chunk_sentences=GROUP_SAMPLE_1, sentences_ready=[])
    assert len(out_groups) == 1
    assert len(out_groups[0]['previous_sentences']) == 1
    assert len(out_groups[0]['posterior_sentences']) == 2
    assert out_groups[0]['metadata']['SID'] == 2
    assert out_groups[0]['previous_sentences'][0] == {"metadata": {"pmid": 10101, "SID": 1, "relevant": False}}
    assert out_groups[0]['posterior_sentences'][0] == {"metadata": {"pmid": 10101, "SID": 3, "relevant": False}}
    out_groups_2 = get_out_sentence(chunk_sentences=GROUP_SAMPLE_2, sentences_ready=out_groups)
    assert len(out_groups_2) == 3
    assert out_groups_2[1]['metadata']['SID'] == 21
    assert out_groups_2[2]['previous_sentences'] == GROUP_SAMPLE_2[0:-1]
    assert out_groups_2[2]['posterior_sentences'] == []
