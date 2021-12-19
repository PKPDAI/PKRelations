import re
from typing import List, Dict
import random

random.seed(48)
TO_REMOVE = []  # ['[', '(', ']', ')']
DOT_SYNS = ['x', '*', '×', '•', ' ', '⋅']
UNIT_SYNONYMS = {
    '·': DOT_SYNS,
    'μg': ['micrograms', 'micro g', 'microg', 'microgram'],
    'h': ['hr', 'hrs', 'hour', 'hours'],
    '%': ['percent', 'percentage'],
    'l': ['liters', 'litre', 'liter', 'litres'],
    'min': ['minutes', 'minute', 'mins'],
    'd': ['days', 'day'],
    'kg': ['kilogram', 'kilograms'],
    's': ['sec'],
    'nM': ['nmol', 'nanomol'],
    'mM': ['mmol', 'milimol'],
    'μM': ['mumol', 'micromol', 'micromols'],
    'pM': ['pmol', 'pmols']
}

MAGNITUDES = {
    'TIME': ['s', 'min', 'h', 'd'],
    'MASS': ['ng', 'μg', 'mg', 'g', 'kg'],
    'VOLUME': ['nl', 'µl', 'ml', 'l'],
    'CONCENTRATION': ['pM', 'nM', 'μM', 'mM', 'M']
}

RANGE_SEPARATORS = [', and ,', '- x (-)', 'to over', '·–·', ',–,', 'and', '–‐', '−,', '%-', 'to', '-', '‐', '−', ';',
                    ',', '–']


# L/h/kg -> Volume/(Time*Mass)

def subs_underscore_dot(inp_mention: str, standard_dot: str = '·') -> str:
    """
    Substitutes '.' by '·' if '.' is not surrounded by numbers
    """
    match_dot = r"(?<!\d)\.(?!\d)|\.(?!\d)|(?<!\d)\."
    inp_mention = re.sub(match_dot, standard_dot, inp_mention)
    return inp_mention


def sub_all_mult(inp_mention: str, standard_dot: str = '·') -> str:
    for x in DOT_SYNS:
        if x in inp_mention:
            inp_mention = inp_mention.replace(x, standard_dot)
    inp_mention = re.sub(r'·+', standard_dot, inp_mention)
    return inp_mention


def check_syns(inp_term: str, replacement_dict: Dict) -> str:
    for main_form, synonyms in replacement_dict.items():
        if inp_term in synonyms:
            return main_form
        synonyms_changed = [x + "-1" for x in synonyms]
        main_form_changed = main_form + "-1"
        if inp_term in synonyms_changed:
            return main_form_changed
    return inp_term


def make_subs_dict(inp_terms: List[str], replacement_dict: Dict) -> List[str]:
    out_terms = [check_syns(inp_term=term, replacement_dict=replacement_dict) for term in inp_terms]
    assert len(inp_terms) == len(out_terms)
    return out_terms


def unit_std_dict(inp_mention: str, standard_dot: str = '·', standard_div: str = '/') -> str:
    subunits_one = inp_mention.split(standard_dot)
    std_subunits_one = []
    subunits_one = ["/" if x == "per" else x for x in subunits_one]
    for subu in subunits_one:
        if standard_div in subu:
            subunits_two = subu.split(standard_div)
            std_subunits_two = [check_syns(inp_term=t, replacement_dict=UNIT_SYNONYMS) for t in subunits_two]
            std_subunits_one.append(f"{standard_div}".join(std_subunits_two))
        else:
            std_subunits_one.append(check_syns(inp_term=subu, replacement_dict=UNIT_SYNONYMS))

    assert len(subunits_one) == len(std_subunits_one)
    return f"{standard_dot}".join(std_subunits_one)


def standardise_unit(inp_mention: str) -> str:
    inp_mention = inp_mention.strip()
    inp_mention = "".join([x.lower() if x != 'M' else x for x in inp_mention])
    inp_mention = inp_mention.replace("per cent", "%")
    inp_mention = inp_mention.replace(" per ", "/")
    inp_mention = inp_mention.replace("per ", "/")
    if '.' in inp_mention:
        inp_mention = subs_underscore_dot(inp_mention=inp_mention)
    for x in TO_REMOVE:
        inp_mention = inp_mention.replace(x, '')
    inp_mention = sub_all_mult(inp_mention=inp_mention)
    inp_mention = unit_std_dict(inp_mention=inp_mention)

    inp_mention = inp_mention.replace("micro·", "μ")
    inp_mention = inp_mention.replace("micro", "μ")
    return inp_mention


def augment_decimal(inp_float, percentage_change, decimal_places):
    assert isinstance(inp_float, float)
    inp_float += inp_float * random.uniform(-percentage_change, percentage_change)
    return str(round(inp_float, decimal_places))


def augment_integer(inp_intg, percentage_change):
    original_intg = inp_intg
    assert isinstance(inp_intg, int)
    inp_intg += inp_intg * random.uniform(-percentage_change, percentage_change)
    inp_intg = round(inp_intg)
    if inp_intg == original_intg:
        return str(inp_intg + 5)
    return str(inp_intg)


def check_mention_in_sublist(inp_mention, list_of_lists):
    for sublist in list_of_lists:
        if inp_mention in sublist:
            return sublist
    return []


def isfloat(inp_str):
    try:
        float(inp_str)
        return True
    except ValueError:
        return False


def create_new_span(new_mention: str, new_span_annot: dict, len_orig_mention: int,
                    start_ent: int, end_ent: int, overall_add: int, aug_text: str):
    new_sp_len = len(new_mention)
    to_add = new_sp_len - len_orig_mention
    new_span_annot['end'] += to_add
    overall_add += to_add
    aug_text = aug_text[0:start_ent] + new_mention + aug_text[end_ent:]
    return new_span_annot, overall_add, aug_text


def split_longest_separator(inp_mention: str, sep_candidates: List[str]):
    for sep_cand in sep_candidates:
        if sep_cand in inp_mention:
            split_childs = [t.strip() for t in inp_mention.split(sep_cand)]
            return split_childs
    return None


def augment_numerical_string(inp_numerical_string: str) -> str:
    assert isfloat(inp_numerical_string)
    if inp_numerical_string.isdigit():
        # case where we deal with an integer
        new_numerical_string = augment_integer(inp_intg=int(inp_numerical_string),
                                               percentage_change=0.9)
    else:
        # case where we deal with a float
        decimal_places = random.choices([1, 2, 3, 4], weights=[0.3, 0.5, 0.15, 0.05], k=1)[0]
        new_numerical_string = augment_decimal(inp_float=float(inp_numerical_string),
                                               percentage_change=0.9,
                                               decimal_places=decimal_places
                                               )
    return new_numerical_string


def augment_sentence(inp_anotation: Dict, replacable_dict: Dict[str, List]):
    if inp_anotation["spans"]:
        if "relations" in inp_anotation.keys():
            if inp_anotation['relations']:
                out_relations = inp_anotation['relations']
        all_original_sans = sorted(inp_anotation["spans"], key=lambda anno: anno['start'])
        augmented_text = inp_anotation['text']
        overall_addition = 0
        out_spans = []
        for span in all_original_sans:
            sp_lab = span['label']
            ent_start = span['start'] + overall_addition
            ent_end = span['end'] + overall_addition
            sp_mention = augmented_text[ent_start:ent_end]
            new_span_annotated = dict(start=ent_start, end=ent_end, label=sp_lab)
            original_sp_len = ent_end - ent_start
            if sp_lab in replacable_dict.keys() or sp_lab in ["VALUE", "RANGE"]:
                if sp_lab in ["VALUE", "RANGE"]:
                    if sp_lab == "VALUE":
                        if isfloat(inp_str=sp_mention):
                            new_span_mention = augment_numerical_string(inp_numerical_string=sp_mention)

                            if new_span_mention != sp_mention:
                                new_span_annotated, overall_addition, augmented_text = create_new_span(
                                    new_mention=new_span_mention,
                                    new_span_annot=new_span_annotated,
                                    len_orig_mention=original_sp_len,
                                    start_ent=ent_start,
                                    end_ent=ent_end,
                                    overall_add=overall_addition,
                                    aug_text=augmented_text
                                )
                    if sp_lab == "RANGE":
                        values_of_range = split_longest_separator(inp_mention=sp_mention,
                                                                  sep_candidates=RANGE_SEPARATORS)
                        if values_of_range and len(values_of_range) == 2:
                            if isfloat(values_of_range[0]) and isfloat(values_of_range[1]):
                                new_val_0 = augment_numerical_string(inp_numerical_string=values_of_range[0])
                                new_val_1 = augment_numerical_string(inp_numerical_string=values_of_range[1])
                                if new_val_0 > new_val_1:
                                    new_span_mention = sp_mention.replace(values_of_range[0], new_val_1)
                                    new_span_mention = new_span_mention.replace(values_of_range[1], new_val_0)
                                else:
                                    new_span_mention = sp_mention.replace(values_of_range[0], new_val_0)
                                    new_span_mention = new_span_mention.replace(values_of_range[1], new_val_1)
                                if new_span_mention != sp_mention:
                                    new_span_annotated, overall_addition, augmented_text = create_new_span(
                                        new_mention=new_span_mention,
                                        new_span_annot=new_span_annotated,
                                        len_orig_mention=original_sp_len,
                                        start_ent=ent_start,
                                        end_ent=ent_end,
                                        overall_add=overall_addition,
                                        aug_text=augmented_text
                                    )

                else:
                    provisional_candidates = check_mention_in_sublist(inp_mention=sp_mention,
                                                                      list_of_lists=replacable_dict[sp_lab])
                    if provisional_candidates:
                        candidates = [x for x in provisional_candidates if x != sp_mention]
                        if candidates:
                            # MAKE REPLACEMENT
                            new_span_mention = random.choice(candidates)
                            new_span_annotated, overall_addition, augmented_text = create_new_span(
                                new_mention=new_span_mention,
                                new_span_annot=new_span_annotated,
                                len_orig_mention=original_sp_len,
                                start_ent=ent_start,
                                end_ent=ent_end,
                                overall_add=overall_addition,
                                aug_text=augmented_text
                            )
            # Update relations if exist
            if "relations" in inp_anotation.keys():
                if inp_anotation['relations']:
                    out_relations = update_relations(out_relations=out_relations, new_span_annotated=new_span_annotated,
                                                     old_span=span)
            out_spans.append(new_span_annotated)

        out_annotation = dict(text=augmented_text, spans=out_spans)
        if "relations" in inp_anotation.keys():
            if inp_anotation['relations']:
                out_annotation['relations'] = out_relations
        return out_annotation
    return None


def update_relations(out_relations, new_span_annotated, old_span):
    out_r = []
    for r in out_relations:
        new_rel = r
        for sp in ['child_span', 'head_span']:
            if r[sp] == old_span:
                r[sp] = new_span_annotated
                new_rel = r
        out_r.append(new_rel)
    return out_r
