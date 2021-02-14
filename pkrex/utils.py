import json
import re
import string
from typing import List
import spacy
from spacy.util import filter_spans
from spacy.matcher import Matcher
from spacy.tokens import Span
from itertools import groupby
import ujson
from pathlib import Path


def contains_digit(sentence):
    return any(map(str.isdigit, sentence))


def clean_html(raw_html):
    clean_regex = re.compile('<.*?>')
    return re.sub(clean_regex, '', raw_html)


def check_to_keep(inp_sections):
    """Check whether to keep that sentence depending on the section (essentially removes sentences in the Methods)"""
    keepit = True
    for inp_section in inp_sections:
        inp_section = re.sub(r'\d+', '', inp_section)
        inp_section = inp_section.translate(inp_section.maketrans('', '', string.punctuation))
        inp_section = inp_section.lower().strip()
        if "materials and methods" in inp_section or "patients and methods" in inp_section or "study design" \
                in inp_section or "material and method" in inp_section or "methods and materials" in inp_section:
            keepit = False
        else:
            if "methods" == inp_section:
                keepit = False
    return keepit


def get_output_tmp_file_path(out_path: str, inp_file_name: str):
    output_path_tmp = out_path.split("/")
    output_path_tmp[-1] = inp_file_name
    return "/".join(output_path_tmp)


def has_pk(inp_doc):
    if inp_doc.ents:
        for x in inp_doc.ents:
            if x.label_ == 'PK':
                return True

    return False


class EntityMatcher(object):
    name = "entity_matcher"

    def __init__(self, nlp, pattern_list, labels):
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
            if (not self.check_overlap(doc_ent_spans=previous_ent_spans, start_new_entity=temp_start,
                                       end_new_entity=temp_end)) and (
                    not self.check_restrictions(inp_doc=doc, inp_entity=new_ent)):
                new_ents.append(new_ent)
        doc.ents = filter_spans(new_ents)  # resolve overlapping
        return doc

    @staticmethod
    def check_restrictions(inp_doc, inp_entity):
        if len(inp_doc) - (len(inp_doc) - inp_entity[0].i) >= 2:
            if (((inp_doc[inp_entity.start - 1].text in ["-", "^"]) or (not inp_doc[inp_entity.start - 1].is_ascii))
                and (inp_doc[inp_entity.start - 2].text[-1].isalpha())) or \
                    (((inp_doc[inp_entity.start - 1].text in ["-", "^"]) or (
                            not inp_doc[inp_entity.start - 1].is_ascii))
                     and (inp_doc[inp_entity.start - 2].text[-1] in ["(", ")", "[", "]"]) and
                     (inp_doc[inp_entity.start - 3].text[-1].isalpha())):
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def check_overlap(doc_ent_spans, start_new_entity, end_new_entity):
        """Checks if new entity overlaps with existing ones"""
        for ent_span in doc_ent_spans:
            if (ent_span[0] <= start_new_entity < ent_span[1]) or (ent_span[0] < end_new_entity <= ent_span[1]):
                return True
                break
        return False


def make_nlp_pk_route_ner(dictionaries_path: str, pk_ner_path: str):
    """Gets nlp model that does PK NER and adds rules for detecting VALUES, TYPE_MEASUREMENTS, ROUTES and RANGES"""

    with open(dictionaries_path) as f:
        patterns_dict = json.load(f)
    pattern = [{'LIKE_NUM': True}, {'ORTH': {'IN': ["^-", "^", "(-", "^\u2212"]}}, {'LIKE_NUM': True}]
    pattern1 = [{'LIKE_NUM': True}]
    pattern2 = [{'LOWER': {"IN": patterns_dict['TYPE_MEAS']}}]
    pattern3 = [{"LOWER": {"IN": patterns_dict['ROUTE']}}]
    pattern4 = [{'ORTH': "intra"}, {'ORTH': "-"}, {}]
    nlp = spacy.load(pk_ner_path)
    entity_matcher = EntityMatcher(nlp, [[pattern, pattern1], [pattern2], [pattern3, pattern4]], ["VALUE", "TYPE_MEAS",
                                                                                                  "ROUTE"])
    nlp.add_pipe(component=entity_matcher, name='entity_matcher', after='ner')
    return nlp


def arrange_pk_sentences_abstract_context(all_sentences: List):
    articles_sorted = sorted(all_sentences, key=lambda delement: delement['metadata']['pmid'])
    all_out_ready = []
    for key, values in groupby(articles_sorted, key=lambda delement: delement['metadata']['pmid']):
        abstract_sentences = list(sorted(list(values), key=lambda delement: delement['metadata']['SID']))
        for i, sentence in enumerate(abstract_sentences):
            if sentence['metadata']['relevant']:
                out_sent = sentence.copy()
                out_sent['previous_sentences'] = abstract_sentences[0:i]
                out_sent['posterior_sentences'] = abstract_sentences[i + 1:]
                all_out_ready.append(out_sent)

    return all_out_ready


def arrange_pk_sentences_pmc_context(all_sentences: List):
    articles_sorted = sorted(all_sentences, key=lambda delement: delement['metadata']['pmid'])
    all_out_ready = []
    for key, values in groupby(articles_sorted, key=lambda delement: delement['metadata']['pmid']):
        article_sentences = list(sorted(list(values), key=lambda delement: delement['metadata']['paragraph_id']))
        for subkey, sub_values in groupby(article_sentences, key=lambda delement: delement['metadata']['paragraph_id']):
            paragraph_sentences = list(sorted(list(sub_values), key=lambda delement: delement['metadata']['SID']))
            for i, sentence in enumerate(paragraph_sentences):
                if sentence['metadata']['relevant']:
                    out_sent = sentence.copy()
                    out_sent['previous_sentences'] = paragraph_sentences[0:i]
                    out_sent['posterior_sentences'] = paragraph_sentences[i + 1:]
                    all_out_ready.append(out_sent)

    return all_out_ready


def read_jsonl(file_path):
    """Read a .jsonl file and yield its contents line by line.
    file_path (unicode / Path): The file path.
    YIELDS: The loaded JSON contents of each line.
    """
    with Path(file_path).open('r', encoding='utf8') as f:
        for line in f:
            try:  # hack to handle broken jsonl
                yield ujson.loads(line.strip())
            except ValueError:
                continue


def write_jsonl(file_path, lines):
    """Create a .jsonl file and dump contents.
    file_path (unicode / Path): The path to the output file.
    lines (list): The JSON-serializable contents of each line.
    """
    data = [ujson.dumps(line, escape_forward_slashes=False) for line in lines]
    Path(file_path).open('w', encoding='utf-8').write('\n'.join(data))
