from pkrex.utils import make_super_tagger

TEXT_ENTITIES_1 = "The mean AUC(0-inf)median of Intravenous midazolam was 4.5 m/l +- 3m/l and aortically 3,9. On the " \
                  "other hand, the CLz/F ranged from 4 to 9 (4-9). Median midazolam clearance was 0.1 to 3e-1."

TEXT_ENTITIES_2 = "The geometric mean of sunitinib and SU12662 AUC24,ss was decreased by 21% and 28% in patients " \
                  "with both gastrectomy and small bowel resection (n = 8) compared to controls (n = 63) " \
                  "for sunitinib was < 2 (931 ng*hr/mL (95%-CI; 676\u20131283) versus 1177 ng*hr/mL (95%-CI; " \
                  "1097\u20131263);" \
                  " p < 0.05) and SU12662 (354 ng*hr/mL (95%-CI; 174\u2013720) versus 492 ng*hr/mL" \
                  " (95%-CI; 435\u2013555); p < 0.05)."

NER_MODEL = make_super_tagger(dictionaries_path='../data/dictionaries/terms.json',
                              pk_ner_path='../data/models/pk_ner_supertok')


def test_entity_matcher():
    doc = NER_MODEL(TEXT_ENTITIES_1)

    doc1_ents = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ != 'UNITS']
    print(doc1_ents)

    assert doc1_ents == [('mean', 'TYPE_MEAS'), ('AUC(0-inf)', 'PK'), ('median', 'TYPE_MEAS'),
                         ('Intravenous', 'ROUTE'), ('4.5', 'VALUE'), ('+-', 'TYPE_MEAS'), ('3', 'VALUE'),
                         ('aortically', 'ROUTE'), ('3,9', 'VALUE'), ('CLz/F', 'PK'), ('4 to 9', 'RANGE'),
                         ('4-9', 'RANGE'), ('Median', 'TYPE_MEAS'), ('clearance', 'PK'), ('0.1 to 3e-1', 'RANGE')]

    doc2 = NER_MODEL(TEXT_ENTITIES_2)

    doc2_ents = [(ent.text, ent.label_) for ent in doc2.ents if ent.label_ != 'UNITS']
    print(doc2_ents)

    assert doc2_ents == [('geometric mean', 'TYPE_MEAS'), ('AUC24,ss', 'PK'), ('21', 'VALUE'),
                         ('28', 'VALUE'), ('63', 'VALUE'), ('<', 'COMPARE'), ('2', 'VALUE'), ('931', 'VALUE'),
                         ('95%-CI', 'TYPE_MEAS'), ('676–1283', 'RANGE'), ('1177', 'VALUE'),
                         ('95%-CI', 'TYPE_MEAS'), ('1097–1263', 'RANGE'), ('0.05', 'P-VALUE'),
                         ('354', 'VALUE'), ('95%-CI', 'TYPE_MEAS'), ('174–720', 'RANGE'), ('492', 'VALUE'),
                         ('95%-CI', 'TYPE_MEAS'), ('435–555', 'RANGE'), ('0.05', 'P-VALUE')]


EXAMPLE_PVALUE = "The clearance was 54.453 and the p-value < 0.01. The pval was 0.01 . This is true (p>0.01). " \
                 "p-val=0.9, x < 45.4 p=10 (P > 0.1"


def test_p_values():
    doc = NER_MODEL(EXAMPLE_PVALUE)
    doc_ents = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ != 'UNITS']
    assert doc_ents == [('clearance', 'PK'), ('54.453', 'VALUE'), ('0.01', 'P-VALUE'),
                        ('0.01', 'P-VALUE'), ('0.01', 'P-VALUE'), ('0.9', 'P-VALUE'), ('<', 'COMPARE'),
                        ('45.4', 'VALUE'), ('10', 'VALUE'), ('0.1', 'P-VALUE')]


EXAMPLE_WRITTEN_VALUES = "The clearance was three-times higher for condition 1 and two times lower for condition 2."


def test_written_numbers():
    doc = NER_MODEL(EXAMPLE_WRITTEN_VALUES)
    doc_ents = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ != 'UNITS']
    assert doc_ents == [('clearance', 'PK'), ('higher', 'COMPARE'),
                        ('1', 'VALUE'), ('lower', 'COMPARE'), ('2', 'VALUE')]


EXAMPLE_NO_VALUES = "1 The clearance was 3-times (3 times) higher for condition 1. Group 1 had a clearance of 3." \
                    " Table 1 was very explanatory as well as Figure 3. Clearance increased 3-fold. Compound 4 was" \
                    " higher than compound 2. 3 patients undertook the study, each of them weighting 4kg.[14] 5"

EXAMPLE_NO_VALUES_2 = "Group 5 had a clearance of 3"


def test_written_numbers_2():
    doc = NER_MODEL(EXAMPLE_NO_VALUES)
    doc_ents = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ != 'UNITS']
    assert doc_ents == [('1', 'VALUE'), ('clearance', 'PK'), ('higher', 'COMPARE'), ('1', 'VALUE'), ('clearance', 'PK'),
                        ('3', 'VALUE'), ('Clearance', 'PK'), ('higher', 'COMPARE'), ('4', 'VALUE'), ('5', 'VALUE')]
    doc2 = NER_MODEL(EXAMPLE_NO_VALUES_2)
    doc_ents_2 = [(ent.text, ent.label_) for ent in doc2.ents]
    assert doc_ents_2 == [('clearance', 'PK'), ('3', 'VALUE')]
