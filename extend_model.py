"""This script gets the pretrained statistical NER model only for PK parameters and uses rule-based matching to add
 VALUES and MEAS_TYPE as entities"""

import spacy
import os
from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy.util import filter_spans
from spacy.language import Language


def check_overlap(doc_ent_spans, start_new_entity, end_new_entity):
    """Checks if new entity overlaps with existing ones"""
    for ent_span in doc_ent_spans:
        if (ent_span[0] <= start_new_entity < ent_span[1]) or (ent_span[0] <= end_new_entity < ent_span[1]):
            return True
            break
    return False


def check_restrictions(inp_doc, inp_entity):
    if len(inp_doc) - (len(inp_doc) - inp_entity[0].i) >= 2:
        if (((inp_doc[inp_entity.start - 1].text in ["-", "^"]) or (not inp_doc[inp_entity.start - 1].is_ascii))
            and (inp_doc[inp_entity.start - 2].text[-1].isalpha())) or \
                (((inp_doc[inp_entity.start - 1].text in ["-", "^"]) or (not inp_doc[inp_entity.start - 1].is_ascii))
                 and (inp_doc[inp_entity.start - 2].text[-1] in ["(", ")", "[", "]"]) and
                 (inp_doc[inp_entity.start - 3].text[-1].isalpha())):
            #           print("Exception here: ")
            #           print(inp_doc[inp_entity.start - 2:inp_entity.end].text + "\n")
            return True
        else:
            return False
    else:
        return False


class EntityMatcher(object):
    name = "entity_matcher"

    def __init__(self, nlp, pattern_list, labels):
        # patterns = [nlp.make_doc(text) for text in terms]
        self.matcher = Matcher(nlp.vocab)
        for patterns, label in zip(pattern_list, labels):
            self.matcher.add(label, None, *patterns)
        self.vocab = nlp.vocab

    def __call__(self, doc):
        new_ents = [tmp_ent for tmp_ent in doc.ents]
        temp_matches = self.matcher(doc)
        previous_ent_spans = [(inp_ent.start, inp_ent.end) for inp_ent in doc.ents]
        for temp_match_id, temp_start, temp_end in temp_matches:
            temp_string_id = self.vocab.strings[temp_match_id]
            new_ent = Span(doc, temp_start, temp_end, label=temp_string_id)
            if (not check_overlap(doc_ent_spans=previous_ent_spans, start_new_entity=temp_start,
                                  end_new_entity=temp_end)) and (
                    not check_restrictions(inp_doc=doc, inp_entity=new_ent)):
                new_ents.append(new_ent)
        doc.ents = filter_spans(new_ents)  # resolve overlapping
        return doc


if __name__ == '__main__':
    nlp = spacy.load(os.path.join("data", "pk_ner_supertok"))
    pattern = [{'LIKE_NUM': True}]
    pattern1 = [{'LIKE_NUM': True}, {'ORTH': {'IN': ["^-", "^", "(-"]}}, {'LIKE_NUM': True}]
    pattern12 = [{'LIKE_NUM': True}, {'IS_ASCII': False}, {'LIKE_NUM': True}]
    pattern2 = [{'LOWER': {"IN": ["mean", "median", "population", "individual", "estimated", "std", "+-"]}}]

    entity_matcher = EntityMatcher(nlp, [[pattern1, pattern12, pattern], [pattern2]], ["VALUE", "TYPE_MEAS"])
    nlp.add_pipe(entity_matcher, last=True)

    # matcher = Matcher(nlp.vocab)
    # matcher.add("VALUE", None, pattern1, pattern)  # order matters!
    # matcher.add("TYPE_MEAS", None, pattern2)
    # nlp.add_pipe(extend_entities, after='ner')

    text3 = 'The accumulation ratio was calculated as the mean ratio of AUC0–τ,ss to AUC0–τ (single dose) which were ' \
            '45.151 87,7 546 4.3 and 15.6h/l-1, and the Population fluctuation ratio was 3333.4, the AUC[AUC(' \
            '0-inf)=342l/h-1] was 34 l^2 10^2, 10^-2, The final model parameter estimate for CL/F was 2780 L/h, ' \
            '3870 L for V/F, and 0.234 h\u22121 for Ka. The value was 10\u22121(0.1) . 5432 h(-1) 10(-1). Clearance ' \
            'range was 2-3'

    print(text3)
    for ent in nlp(text3).ents:
        print(ent.text, ent.label_)
    Language.factories["entity_matcher"] = lambda nlp, **cfg: EntityMatcher(nlp, **cfg)
    nlp.to_disk('data/ner_ready')

# To include on __init__.py in spacy
# pattern = [{'LIKE_NUM': True}]
# pattern1 = [{'LIKE_NUM': True}, {'ORTH': {'IN': ["-", "^-", "^"]}}, {'LIKE_NUM': True}]
# pattern12 = [{'LIKE_NUM': True}, {'IS_ASCII': False}, {'LIKE_NUM': True}]
# pattern2 = [{'LOWER': {"IN": ["mean", "median", "population", "individual", "estimated", "std", "+-"]}}]
#
#
# def load(**overrides):
#     Language.factories["entity_matcher"] = lambda nlp: EntityMatcher(nlp,
#                                                                      [[pattern1, pattern12, pattern], [pattern2]],
#                                                                      ["VALUE", "TYPE_MEAS"])
#
#     return load_model_from_init_py(__file__, **overrides)
