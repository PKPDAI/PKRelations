import hashlib
import json
import re
import string
import random
from typing import List, Dict, Tuple
import spacy
from spacy.util import filter_spans
from spacy.matcher import Matcher
from spacy.tokens import Span
from itertools import groupby
import ujson
from pathlib import Path
from spacy.pipeline import EntityRuler
from azure.storage.blob import BlobClient

random.seed(1)


def make_super_tagger(dictionaries_path: str, pk_ner_path: str):
    """Gets nlp model that does PK NER and adds rules for detecting VALUES, TYPE_MEASUREMENTS, ROUTES and RANGES"""

    with open(dictionaries_path) as f:
        patterns_dict = json.load(f)

    nlp = spacy.load(pk_ner_path)

    # patterns

    value_patterns = [{"label": "VALUE",
                       "pattern": [{'LIKE_NUM': True}, {'ORTH': {'IN': ["^-", "^", "(-", "^\u2212"]}},
                                   {'LIKE_NUM': True}]
                       },

                      {"label": "VALUE",
                       "pattern": [{'LIKE_NUM': True}, {'ORTH': "e"}, {'ORTH': '-'},
                                   {'LIKE_NUM': True}]
                       },

                      {"label": "VALUE",
                       "pattern": [{'LIKE_NUM': True}]
                       }
                      ]

    type_meas_patterns = [{"label": "TYPE_MEAS", "pattern": term} for term in patterns_dict["TYPE_MEAS"]]

    comparative_patterns = [{"label": "COMPARE",
                             "pattern": term} for term in patterns_dict["COMPARE"]]

    route_patterns = [{"label": "ROUTE", "pattern": term} for term in patterns_dict["ROUTE"]]
    route_patterns += [{
        "label": "ROUTE",
        "pattern": [{'ORTH': "intra"}, {'ORTH': "-"}, {}]
    }]

    units_patterns = [{"label": "UNITS", "pattern": term} for term in patterns_dict["UNITS"]]

    all_patterns = value_patterns + type_meas_patterns + route_patterns + comparative_patterns + units_patterns
    # + range_patterns

    ruler = EntityRuler(nlp, phrase_matcher_attr="LOWER")
    ruler.add_patterns(all_patterns)
    nlp.add_pipe(ruler)
    nlp.add_pipe(get_range_entity, last=True)
    nlp.add_pipe(clean_values, last=True)
    return nlp


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


class TokenEntityMatcher(object):
    name = "token_entity_matcher"

    def __init__(self, nlp, pattern_list, labels):
        self.matcher = Matcher(nlp.vocab)
        # self.matcher = PhraseMatcher(nlp.vocab)
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
        return False


# @Language.component("get_range_entity")
def get_range_entity(doc):
    new_ents = []
    for i, ent in enumerate(doc.ents):
        if (ent.label_ == "VALUE") and (ent.end + 2 <= len(doc)) and (i < len(doc.ents)):
            next_token = doc[ent.end]
            next_next_token = doc[ent.end + 1]
            if next_token.text in (
                    "-", "to", "\u2012", "\u2013", "\u2014", "\u2015",) and next_next_token.ent_type_ == "VALUE":
                new_ent = Span(doc, ent.start, doc.ents[i + 1].end, label="RANGE")
                new_ents.append(new_ent)
            else:
                new_ents.append(ent)
        else:
            new_ents.append(ent)
    doc.ents = filter_spans(new_ents)  # resolve overlapping
    return doc


def clean_values(doc):
    """This function gets an input spacy doc and checks for VALUE entities. IF it finds one, it checks whether it's a
    p-value and cleans any VALUE mentions entirely composed by characters, e.g. three, two one etc. Finally it cleans
     comparatives preceding a p-val e.g. for p-val < 0.01, < would not be a comparative """
    new_ents = []
    for i, ent in enumerate(doc.ents):
        if is_pval(inp_ent=ent, inp_doc=doc):  # check whether it's p-value
            new_ent = Span(doc, ent.start, ent.end, label="P-VALUE")
            new_ents.append(new_ent)
        else:
            if ent.label_ == "VALUE":
                if not is_exclusive_value(ent, doc):  # check whether it's an exclusive value
                    new_ents.append(ent)
            else:
                new_ents.append(ent)
    # Clean comparatives that appear before P-VALUE entities
    new_ents_2 = []
    number_of_entities = len(new_ents)
    for i, ent in enumerate(new_ents):
        if ent.label_ == "COMPARE" and ((i + 1) < number_of_entities) and new_ents[i + 1].label_ == "P-VALUE":
            if ent.end != new_ents[i + 1].start:
                new_ents_2.append(ent)
        else:
            new_ents_2.append(ent)

    doc.ents = filter_spans(new_ents_2)
    return doc


def is_exclusive_value(inp_ent, inp_doc):
    """The values that are considered exclusive are those with:
     no digits, Table/Figure/Group/Compound + value, value-fold, value + times/patients or [value], VALUE + - """
    doc_len = len(inp_doc)
    if has_digits(inp_ent.text):
        if all_digits(inp_ent.text):
            prev_tok_idx = inp_ent.start - 1
            prev2_tok_idx = inp_ent.start - 2
            subs_tok_idx = inp_ent.end
            subs2_tok_idx = inp_ent.end + 1
            if prev_tok_idx >= 0 and subs_tok_idx < doc_len:
                if (inp_doc[prev_tok_idx].text == "[") and (inp_doc[subs_tok_idx].text == "]"):
                    # integer value surrounded by parenthesis
                    return True
            if prev_tok_idx >= 0:
                if inp_doc[prev_tok_idx].text.lower() in ["group", "groups", "table", "tables", "compound", "compounds",
                                                          "figure", "figures", "study", "phase", "formulation",
                                                          "product", "fig", "tab", "day", "days", "equation", "eq",
                                                          "trial", "trials", "subject", "fig.", "figs.", "tab.",
                                                          "tabs.", "eq."]:
                    return True

            if prev2_tok_idx >= 0:
                if inp_doc[prev2_tok_idx].text.lower() in ["fig", "eq", "tab"] and inp_doc[prev_tok_idx].text.lower() \
                        in ["."]:
                    return True

                if inp_doc[prev_tok_idx].text.lower() in ["-", "\u223c", "\u20105"] and inp_doc[
                    prev2_tok_idx].text.lower() \
                        not in ["+", "/"]:
                    return True

            if subs_tok_idx < doc_len:
                if inp_doc[subs_tok_idx].text.lower() in ["time", "times", "patient", "patients", "phases", "degrees",
                                                          "sample", "samples", "subject", "subjects", "-", "\u223c"]:
                    return True
            if subs2_tok_idx < doc_len:
                if inp_doc[subs_tok_idx].text.lower() in ["-", "\u223c", "\u20105"] and inp_doc[
                    subs2_tok_idx].text.lower() in \
                        ["time", "times", "fold", "folds"]:
                    return True

    else:
        return True
    return False


def arrange_pk_sentences_abstract_context(all_sentences: List):
    articles_sorted = sorted(all_sentences, key=lambda delement: delement['metadata']['pmid'])
    all_out_ready = []
    # 1) group sentences by pmid
    for key, values in groupby(articles_sorted, key=lambda delement: delement['metadata']['pmid']):
        all_out_ready = get_out_sentence(chunk_sentences=values, sentences_ready=all_out_ready)
    return all_out_ready


def arrange_pk_sentences_pmc_context(all_sentences: List):
    articles_sorted = sorted(all_sentences, key=lambda delement: delement['metadata']['pmid'])
    all_out_ready = []
    # 1) group by pmid
    for key, values in groupby(articles_sorted, key=lambda delement: delement['metadata']['pmid']):
        article_sentences = list(sorted(list(values), key=lambda delement: delement['metadata']['paragraph_id']))
        # 2) sub-group by paragraph id
        for subkey, sub_values in groupby(article_sentences, key=lambda delement: delement['metadata']['paragraph_id']):
            all_out_ready = get_out_sentence(chunk_sentences=sub_values, sentences_ready=all_out_ready)

    return all_out_ready


def get_out_sentence(chunk_sentences, sentences_ready):
    """
    Given a bunch of sentences coming from the same abstract/paragraph, it sorts them based on SID and,
     if the sentence is relevant, it complements that sentence with the ones before and after in the 'previous_sentences
     ' and 'posterior_sentences' and appends that dictionary to sentences_ready
    chunk sentences param: list/iterable of sentences from the same paragraph/abstract
    sentences_ready param: list of sentences selected as 'relevant' (having PK + VALUE/RANGE)
    """
    chunk_sentences_sorted = list(sorted(list(chunk_sentences), key=lambda delement: delement['metadata']['SID']))
    for i, sentence in enumerate(chunk_sentences_sorted):
        if sentence['metadata']['relevant']:
            out_sent = sentence.copy()
            out_sent['previous_sentences'] = chunk_sentences_sorted[0:i]
            out_sent['posterior_sentences'] = chunk_sentences_sorted[i + 1:]
            sentences_ready.append(out_sent)
    return sentences_ready


def read_jsonl(file_path):
    """Read a .jsonl file and yield its contents line by line.
    file_path (unicode / Path): The file path.
    YIELDS: The loaded JSON contents of each line.
    """
    with Path(file_path).open(encoding='utf8') as f:
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


def get_link(inp_sentence: Dict) -> str:
    """Gets an input sentence from PMID/PMC doc in the form of dictionary in which the PMC/PMID is in the metadata key
    and returns the PubMed link to that document"""
    if 'pmc' in inp_sentence['metadata'].keys():
        paper_link = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{}/".format(inp_sentence['metadata']['pmc'])
    else:
        paper_link = "https://www.ncbi.nlm.nih.gov/pubmed/{}".format(inp_sentence['metadata']['pmid'])
    return paper_link


def check_and_resample(sampled_subset: List, main_pool: List, ids_already_sampled: List) -> Tuple[List, List]:
    """
    Checks whether the sampled subset has some sentences that were already present in previous annotations, and if so,
    it replaces that sentence by another one in the main pool
    """
    final_subset_cl = []
    for tmp_sentence in sampled_subset:
        if int(tmp_sentence['metadata']['SID']) in ids_already_sampled:
            """case in which the sentence has already been sampled"""
            unique = False
            while not unique:
                tmp_sentence = random.sample(main_pool, 1)[0]  # sample a new one
                if int(tmp_sentence['metadata']['SID']) not in ids_already_sampled:
                    unique = True

        final_subset_cl.append(tmp_sentence)
        ids_already_sampled.append(int(tmp_sentence['metadata']['SID']))

    assert len(final_subset_cl) == len(sampled_subset)
    return final_subset_cl, ids_already_sampled


def populate_spans(spacy_doc, sentence) -> Dict:
    spans = []
    ent_labels = []
    for entity in spacy_doc.ents:
        spans.append(dict(start=entity.start_char, end=entity.end_char, label=entity.label_, ))
        ent_labels.append(entity.label_)
    sentence['spans'] = spans
    sentence['sentence_hash'] = hashlib.sha1(sentence['text'].encode()).hexdigest()
    sentence['metadata']['relevant'] = False
    if len(sentence['text']) > 5 and ('PK' in ent_labels) and (('VALUE' in ent_labels) or ('RANGE' in ent_labels)):
        sentence['metadata']['relevant'] = True

    return sentence


def sentence_pmid_to_int(sentence_dict: Dict):
    sentence_dict['metadata']['pmid'] = int(sentence_dict['metadata']['pmid'])
    return sentence_dict


def is_pval(inp_ent, inp_doc):
    """
    Gets as an input an entity and the doc where the entity comes from and checks whether that entity is a p-value
    """
    if inp_ent.label_ == "VALUE":  # check that it's a value
        number_as_text = inp_ent.text.replace(',', '.')
        if isfloat(number_as_text):
            if float(number_as_text) < 1:  # check that it's lower than 1
                if (inp_ent.start > 1) and (inp_doc[inp_ent.start - 1].text.lower() in [">", "<", "=", "was"]) and (
                        inp_doc[inp_ent.start - 2].text.lower() in ["p", "pval", "pvalue"]):
                    return True
                else:
                    if (inp_ent.start > 3) and (inp_doc[inp_ent.start - 1].text.lower() in [">", "<", "=", "was"]) and (
                            inp_doc[inp_ent.start - 2].text.lower() in ["value", "val"]) and (
                            inp_doc[inp_ent.start - 3].text in ["-"]) and (inp_doc[inp_ent.start - 4].text.lower() in
                                                                           ["p"]):
                        return True
        else:
            if (inp_ent.start > 1) and (inp_doc[inp_ent.start - 1].text.lower() in [">", "<", "=", "was"]) and (
                    inp_doc[inp_ent.start - 2].text.lower() in ["p", "pval", "pvalue"]):
                return True
            else:
                if (inp_ent.start > 3) and (inp_doc[inp_ent.start - 1].text in [">", "<", "=", "was"]) and (
                        inp_doc[inp_ent.start - 2].text.lower() in ["value", "val"]) and (
                        inp_doc[inp_ent.start - 3].text in ["-"]) and (inp_doc[inp_ent.start - 4].text.lower() in
                                                                       ["p"]):
                    return True
    return False


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def has_digits(input_string):
    return any(char.isdigit() for char in input_string)


def all_digits(input_string):
    return all(char.isdigit() for char in input_string.replace(".", "").replace(",", ""))


def is_sentence_relevant(inp_sentence: Dict) -> bool:
    uq_spans = set([span['label'] for span in inp_sentence['spans']])
    if ('PK' in uq_spans) and (('VALUE' in uq_spans) or ('RANGE' in uq_spans)):
        return True
    else:
        return False


def get_sort_key(tmp_span):
    return tmp_span['end'] - tmp_span['start'], -tmp_span['start']


def resolve_overlapping_spans(inp_spans: List[Dict]) -> List[Dict]:
    """Taken from spacy.util.filter_spans"""
    sorted_spans = sorted(inp_spans, key=get_sort_key, reverse=True)  # sorts spans by length
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        if span["start"] not in seen_tokens and span["end"] - 1 not in seen_tokens:
            result.append(span)
        seen_tokens.update(range(span["start"], span["end"]))  # covers the range of characters occupied by that entity
    result = sorted(result, key=lambda tmp_span: tmp_span["start"])
    return result


def get_blob(inp_blob_name: str):
    blob = BlobClient(
        account_url="https://pkpdaiannotations.blob.core.windows.net",
        container_name="pkpdaiannotations",
        blob_name=inp_blob_name,
        credential="UpC2SPFbEqJdY0tgY91y1oVe3ZcQwxALkJ2QIDTYN17FoTLmpltCFyzxKk13fjrp04y+7K4L6t5KR6VOMUKOqw==")
    return blob


def add_annotator_meta(inp_dict, base_dataset_name):
    """Adds the annotator session id into the prodigy annotated instance dictionary"""
    inp_dict['meta']['source'] = inp_dict['_session_id'].replace(base_dataset_name, "")
    return inp_dict


def print_ner_scores(inp_dict: Dict, is_spacy: bool):
    """
    @param is_spacy: whether the dictionary comes as an output from spacy model
    @param inp_dict: Dictionary with keys corresponding to entity types and subkeys to metrics
    e.g. {'PK': {'ent_type': {..},{'partial': {..},{'strict': {..} }}
    @return: Prints summary of metrics
    """
    if is_spacy:
        token_acc = round(inp_dict['token_acc'] * 100, 2)
        print(f"Token accuracy: {token_acc}")
        per_entity_metrics = inp_dict['ents_per_type']
        for ent_type in per_entity_metrics.keys():
            print(f" ====== Stats for entity {ent_type} ======")
            p = round(per_entity_metrics[ent_type]['p'] * 100, 2)
            r = round(per_entity_metrics[ent_type]['r'] * 100, 2)
            f1 = round(per_entity_metrics[ent_type]['f'] * 100, 2)
            print(f" Precision:\t {p}%")
            print(f" Recall:\t {r}%")
            print(f" F1:\t\t {f1}%")
    else:
        for ent_type in inp_dict.keys():
            print(f"====== Stats for entity {ent_type} ======")
            for metric_type in inp_dict[ent_type].keys():
                if metric_type in ['partial', 'strict']:
                    print(f" === {metric_type} match: === ")
                    precision = inp_dict[ent_type][metric_type]['precision']
                    recall = inp_dict[ent_type][metric_type]['recall']
                    f1 = inp_dict[ent_type][metric_type]['f1']

                    p = round(precision * 100, 2)
                    r = round(recall * 100, 2)
                    f1 = round(f1 * 100, 2)

                    print(f" Precision:\t {p}%")
                    print(f" Recall:\t {r}%")
                    print(f" F1:\t\t {f1}%")

