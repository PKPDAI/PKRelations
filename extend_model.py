"""This script gets the pretrained NER model for PK parameters and uses rule-based matching to detect VALUES and
MEAS_TYPE """

import spacy
import os
from spacy.matcher import Matcher
from spacy.tokens import Span


def check_overlap(doc_ent_spans, start_new_entity, end_new_entity):
    """Checks if new entity overlaps with existing ones"""
    for ent_span in doc_ent_spans:
        if (ent_span[0] <= start_new_entity < ent_span[1]) or (ent_span[0] <= end_new_entity < ent_span[1]):
            return True
            break
    return False


def check_restrictions(inp_doc, inp_entity):
    if len(inp_doc) - (len(inp_doc) - inp_entity[0].i) >= 2:
        if (inp_doc[inp_entity.start - 1].text in ["-", "^"]) and (inp_doc[inp_entity.start - 2].text[-1].isalpha()):
            print("Exception here: ")
            print(inp_doc[inp_entity.start - 2:inp_entity.end].text + "\n")
            return True
        else:
            return False
    else:
        return False


def extend_entities(inp_doc):
    new_ents = [tmp_ent for tmp_ent in inp_doc.ents]
    temp_matches = matcher(inp_doc)
    previous_ent_spans = [(inp_ent.start, inp_ent.end) for inp_ent in inp_doc.ents]
    for temp_match_id, temp_start, temp_end in temp_matches:
        temp_string_id = nlp.vocab.strings[temp_match_id]
        new_ent = Span(inp_doc, temp_start, temp_end, label=temp_string_id)
        if (not check_overlap(doc_ent_spans=previous_ent_spans, start_new_entity=temp_start, end_new_entity=temp_end)) \
                and (not check_restrictions(inp_doc=inp_doc, inp_entity=new_ent)):
            new_ents.append(new_ent)
            previous_ent_spans.append((temp_start, temp_end))
    inp_doc.ents = new_ents
    return inp_doc


if __name__ == '__main__':
    nlp = spacy.load(os.path.join("data", "pk_ner_supertok"))
    matcher = Matcher(nlp.vocab)
    pattern = [{'LIKE_NUM': True}]
    pattern1 = [{'LIKE_NUM': True}, {'ORTH': {'IN': ["-", "^-", "^"]}}, {'LIKE_NUM': True}]
    pattern2 = [{'LOWER': {"IN": ["mean", "median", "population", "individual", "estimated", "std", "+-"]}}]
    matcher.add("VALUE", None, pattern1, pattern)  # order matters!
    matcher.add("TYPE_MEAS", None, pattern2)
    nlp.add_pipe(extend_entities, after='ner')

    text3 = 'The accumulation ratio was calculated as the mean ratio of AUC0–τ,ss to AUC0–τ (single dose) which were ' \
            '45.151 87,7 546 4.3 and 15.6h/l-1, and the Population fluctuation ratio was 3333.4, the AUC[AUC(' \
            '0-inf)=342l/h-1] was 34 l^2 10^2, 10^-2'

    print(text3)
    for ent in nlp(text3).ents:
        print(ent.text, ent.label_)
