from pkrex.utils import make_nlp_pk_route_ner

TEXT_ENTITIES = 'The mean AUC(0-inf)median of Intravenous midazolam was 4.5 h/l +- 3h/l and aortically 3,9. On the ' \
                'other hand, the clearance ranged from 4 to 9. '


def test_make_nlp_pk_route_ner():
    nlp = make_nlp_pk_route_ner(dictionaries_path='../data/dictionaries/terms.json',
                                pk_ner_path='../data/models/pk_ner_supertok')
    doc = nlp(TEXT_ENTITIES)
    assert doc.ents[0].text == 'mean'
    assert doc.ents[0].label_ == 'TYPE_MEAS'
    assert doc.ents[1].text == 'AUC(0-inf)'
    assert doc.ents[1].label_ == 'PK'
    assert doc.ents[2].text == 'median'
    assert doc.ents[2].label_ == 'TYPE_MEAS'
    assert doc.ents[3].text == 'Intravenous'
    assert doc.ents[3].label_ == 'ROUTE'
    assert doc.ents[4].text == '4.5'
    assert doc.ents[4].label_ == 'VALUE'
    assert doc.ents[5].text == '3'
    assert doc.ents[5].label_ == 'VALUE'
    assert doc.ents[6].text == 'aortically'
    assert doc.ents[6].label_ == 'ROUTE'
    assert doc.ents[7].text == '3,9'
    assert doc.ents[7].label_ == 'VALUE'



