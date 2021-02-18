from pkrex.utils import make_super_tagger

TEXT_ENTITIES_1 = "The mean AUC(0-inf)median of Intravenous midazolam was 4.5 h/l +- 3h/l and aortically 3,9. On the " \
                  "other hand, the CLz/F ranged from 4 to 9 (4-9). Median midazolam clearance was 0.1 to 3e-1."

TEXT_ENTITIES_2 = "The geometric mean of sunitinib and SU12662 AUC24,ss was decreased by 21% and 28% in patients " \
                  "with both gastrectomy and small bowel resection (n = 8) compared to controls (n = 63) " \
                  "for sunitinib (931 ng*hr/mL (95%-CI; 676\u20131283) versus 1177 ng*hr/mL (95%-CI; 1097\u20131263);" \
                  " p < 0.05) and SU12662 (354 ng*hr/mL (95%-CI; 174\u2013720) versus 492 ng*hr/mL" \
                  " (95%-CI; 435\u2013555); p < 0.05)."


def test_entity_matcher():
    nlp = make_super_tagger(dictionaries_path='../data/dictionaries/terms.json',
                            pk_ner_path='../data/models/pk_ner_supertok')
    doc = nlp(TEXT_ENTITIES_1)

    doc1_ents = [(ent.text, ent.label_) for ent in doc.ents]
    print(doc1_ents)

    assert doc1_ents == [('mean', 'TYPE_MEAS'), ('AUC(0-inf)', 'PK'), ('median', 'TYPE_MEAS'),
                         ('Intravenous', 'ROUTE'), ('4.5', 'VALUE'), ('+-', 'COMPARE'), ('3', 'VALUE'),
                         ('aortically', 'ROUTE'), ('3,9', 'VALUE'), ('CLz/F', 'PK'), ('4 to 9', 'RANGE'),
                         ('4-9', 'RANGE'), ('Median', 'TYPE_MEAS'), ('clearance', 'PK'), ('0.1 to 3e-1', 'RANGE')]

    doc2 = nlp(TEXT_ENTITIES_2)

    doc2_ents = [(ent.text, ent.label_) for ent in doc2.ents]
    print(doc2_ents)

    assert doc2_ents == [('geometric mean', 'TYPE_MEAS'), ('AUC24,ss', 'PK'), ('21', 'VALUE'), ('28', 'VALUE'),
                         ('=', 'COMPARE'), ('=', 'COMPARE'), ('63', 'VALUE'), ('931', 'VALUE'), ('95%-CI', 'TYPE_MEAS'),
                         ('676–1283', 'RANGE'), ('1177', 'VALUE'), ('95%-CI', 'TYPE_MEAS'), ('1097–1263', 'RANGE'),
                         ('<', 'COMPARE'), ('0.05', 'VALUE'), ('354', 'VALUE'), ('95%-CI', 'TYPE_MEAS'),
                         ('174–720', 'RANGE'), ('492', 'VALUE'), ('95%-CI', 'TYPE_MEAS'), ('435–555', 'RANGE'),
                         ('<', 'COMPARE'), ('0.05', 'VALUE')]


