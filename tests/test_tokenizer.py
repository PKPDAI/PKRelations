import spacy

from pkrex.tokenizer import replace_tokenizer

SAMPLE_SENTENCES = [
    "Study population A (n=8)"
    "The mean AUC(0-inf)x was 4.5h/l +- 3h/l and aortically 3,9.",
    "Additional systemic clearance of 15L/h.37 ",
    "The 90%CI Cmax (18.4%~64.7%), AUC0\u2013t (28.9%~68.5%)and AUC0\u2013\u221e.1",
    "The clearance was 2\u224815.3L/h+-2ml/h.",
    "The clearance was 2\u2248\u224815.3L/h+-2ml/h.",
    "\u0116\u00C0-454\u0104\u0118",
    "After 2 mg/kg intravenous injection, the concentration of DHM reached a maximum of 165.67 \u00b1 16.35 ng/mL, "
    "and t1/2 was 2.05 \u00b1 0.52 h. However, after the oral administration of 20 mg/kg DHM, DHM was not readily "
    "absorbed and reached Cmax 21.63 \u00b1 3.62 ng/mL at approximately 2.67 h, and t1/2 was 3.70 \u00b1 0.99 h. "
]


def test_tokenizer():
    spacy_model = spacy.blank('en')
    spacy_model = replace_tokenizer(spacy_model)
    tok_exp = spacy_model.tokenizer.explain(SAMPLE_SENTENCES[-1])
    print("\n============ NEW TOKENIZER ============\n")
    for t in tok_exp:
        print(t[1], "\t", t[0])

    assert True


"""
Sequence idea: 
1. Split by unicode characters
2. Split by non-alphanumeric
3. Separate numeric and non-numeric parts
"""
