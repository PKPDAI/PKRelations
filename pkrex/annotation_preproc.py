import warnings
from typing import Dict, List
from collections import Counter


def fix_incorrect_dvals(inp_annotation: Dict):
    """
    If there is a central measurement pointing towards a deviation measurement in the form of D_VAL relation,
    it swaps the relationship
    """
    # Get all values that have a C_VAL pointing towards them
    c_val_entities = get_cval_entities(inp_annotation)
    if c_val_entities:
        new_relations = []
        for inp_relation in inp_annotation["relations"]:
            is_incorrect = check_incorrect_relation(inp_relation, c_val_entities)
            if is_incorrect:
                assert inp_relation["label"] == "D_VAL"
                assert inp_relation["child_span"]["label"] in ["VALUE", "RANGE"]
                # We are now sure that a D_VAL relation has been assigned from a central value/range to another
                # value/range, so we will flip that relation to make it from the other value towards the central value
                print(f"The following sentence had a D_VAL relation from a central value to a deviation value, "
                      f"we are changing it...\n{inp_annotation['text']}\n{inp_annotation['_input_hash']}"
                      f" {inp_annotation['sentence_hash']}\n Annotator: {inp_annotation['_session_id'].split('-')[-1]}")
                new_relation = flip_relation_entities(inp_relation)
                new_relations.append(new_relation)
            else:
                new_relations.append(inp_relation)
        inp_annotation["relations"] = new_relations
    return inp_annotation


def check_incorrect_relation(inp_relation: Dict, c_val_entities: List) -> bool:
    for central_value in c_val_entities:
        if central_value == inp_relation["head_span"]:  # a central value can't never act as a head span
            return True
    return False


def flip_relation_entities(inp_relation: Dict) -> Dict:
    """Flips the head span for the child span"""
    out_relation = {
        "head": inp_relation["child"],
        "child": inp_relation["head"],
        "head_span": inp_relation["child_span"],
        "child_span": inp_relation["head_span"],
        "color": inp_relation["color"],
        "label": inp_relation["label"]
    }
    return out_relation


def get_cval_entities(inp_annotation: Dict) -> List[Dict]:
    """
    Gets all the VALUE entities that are labelled with a C_VAL relation
    """
    out_c_vals = []
    for relation in inp_annotation["relations"]:
        if relation["label"] == "C_VAL":
            assert relation["head_span"]["label"] == "PK"
            assert relation["child_span"]["label"] in ["VALUE", "RANGE"]
            out_c_vals.append(relation["child_span"])
    return out_c_vals


def simplify_annotations(inp_annotation: Dict) -> Dict:
    """"
    Removes non-useful information from inside entities and relations keys coming out of the prodigy annotations.
    Annotated data is the same so this funcion just removes irrelevant dictionary keys
    All entities/spans become:
    {
        "start": <character_start>,
        "end": <character_end>,
        "label": <label>
    }
    All relations become:
    {
        "head_span": <dictionary of head entity>,
        "child_span": <dictionary of child entity>,
        "label": <relation label>
    }

    """
    new_spans = [{"start": span["start"], "end": span["end"], "label": span["label"]} for span in
                 inp_annotation["spans"]]
    new_relations = [{
        "head_span": {"start": relation["head_span"]["start"], "end": relation["head_span"]["end"],
                      "label": relation["head_span"]["label"]},
        "child_span": {"start": relation["child_span"]["start"], "end": relation["child_span"]["end"],
                       "label": relation["child_span"]["label"]},
        "label": relation["label"]
    }
        for relation in inp_annotation["relations"]]

    inp_annotation["relations"] = new_relations
    inp_annotation["spans"] = new_spans
    return inp_annotation


def view_relations_json(inp_annotation: Dict, print_entities: bool = False):
    sentence_text = inp_annotation['text']
    print(" ======= Example ===== ")
    print(sentence_text)
    if print_entities:
        print("========== Entities ========")
        print("Entity mention | Label")
        for entity in inp_annotation['spans']:
            entity_text = sentence_text[entity['start']:entity['end']]
            entity_label = entity['label']
            print(f"{entity_text} | {entity_label}")

    print("============ Relations ===============")
    for relation in inp_annotation['relations']:
        relation_label = relation['label']
        head_span = relation['head_span']
        child_span = relation['child_span']
        head_span_text = sentence_text[head_span['start']:head_span['end']]
        head_span_label = head_span['label']
        child_span_text = sentence_text[child_span['start']:child_span['end']]
        child_span_label = child_span['label']
        rel_end = max([child_span['end'], head_span['end']])
        rel_start = min([child_span['start'], head_span['start']])
        relation_distance = rel_end - rel_start
        if relation_distance < 15:
            context_span = sentence_text[rel_start:rel_end]
            print(
                f"{relation_label} : {head_span_text} ({head_span_label}) => {child_span_text} ({child_span_label}) : "
                f"{context_span}")

        else:
            print(f"{relation_label} : {head_span_text} ({head_span_label}) => {child_span_text} ({child_span_label})")
    print("\n")


def check_rel_tokens_entities(annotations: List[Dict]):
    """Checks that there are no relations with tokens that are not part of entities"""
    annotations = [simplify_annotations(sentence) for sentence in annotations]
    for sentence in annotations:
        entities = sentence['spans']
        for entity in entities:
            assert len(entity['label']) > 0

        relations = sentence['relations']
        for relation in relations:
            head = relation['head_span']
            child = relation['child_span']

            if (head not in entities) or (child not in entities):
                exc_mg = f"The following sentence contains a relation with a non-entity " \
                         f"token\n:{sentence}\nRelation: {relation}"
                warnings.warn(exc_mg)
                # raise Exception(exc_mg)


def print_rex_stats(annotations: List[Dict]):
    print(f"Dataset has {len(annotations)} sentences")
    c_val_sentences = 0
    annotators = []
    for sentence in annotations:
        if 'source' in sentence['meta'].keys():
            annotators.append(sentence['meta']['source'])
        else:
            annotators.append(sentence['_session_id'].split('-')[-1])
        x = 0
        for relation in sentence['relations']:
            if relation['label'] == 'C_VAL':
                x = 1
        c_val_sentences += x

    print(f"The number of sentences with central values are: {c_val_sentences}, "
          f"{round(c_val_sentences * 100 / len(annotations), 2)}%")

    annotation_freq = Counter(annotators)
    for an, freq in annotation_freq.items():
        print(f"{an} made {freq} annotations ({round(freq * 100 / len(annotations), 2)}%)")


def d_val_pointing(annotations: List[Dict]):
    annotations = [simplify_annotations(sentence) for sentence in annotations]
    dataset = ""
    if annotations[0]['_session_id']:
        dataset = "-".join(annotations[0]['_session_id'].split("-")[0:-1])
    to_correct = []
    for i, sentence in enumerate(annotations):
        c_val_entities = get_cval_entities(sentence)
        if c_val_entities:
            for relation in sentence['relations']:
                assert not check_incorrect_relation(relation, c_val_entities)
                if relation['label'] == "D_VAL":
                    if relation["child_span"] not in c_val_entities:
                        annotator = sentence['_session_id'].split("-")[-1]
                        sentence_hash = sentence['sentence_hash']
                        warnings.warn(f"In the following sentence there is a D_VAL that doesn't point to a central "
                                      f"measurement:\n{sentence}\nRelations{sentence['relations']}\nSpecific "
                                      f"relation: {relation}\nSentence info:\nDataset:{dataset}"
                                      f"\nPosition in dataset: {i}\n Annotator: {annotator}")
                        to_correct.append((i, sentence_hash))
    if to_correct:

        to_correct = set(to_correct)
        warn_message = f"Entries to pay special attention on the review of dataset {dataset}:\n"
        for x in to_correct:
            warn_message += str(x) + "\n"
        warnings.warn(warn_message)


def swap_clear_incorrect(annotation: Dict) -> Dict:
    """
    It gets an annotation and checks its entities. If there are entity mentions that are clearly incorrect,
    it replaces the label by the correct label. The mapper dictionary contains list of mentions that should
    always relate to a specific entity type. It also excludes some entities if their mention should never
     be part of an entity
    """
    mapper = {
        "TYPE_MEAS": ["+-", "±", "\u00b1", "+/-"]
    }
    to_exclude = ["="]
    # Fix entity dictionaries in the spans field
    new_entities = []
    for entity in annotation['spans']:
        entity_text = annotation['text'][entity['start']:entity['end']]
        for desired_entity_label in mapper.keys():
            candidate_tokens = mapper[desired_entity_label]
            entity = assign_label_to_ent(ent=entity, ent_text=entity_text, desired_label=desired_entity_label,
                                         candidate_tokens=candidate_tokens)
            if entity_text not in to_exclude:
                new_entities.append(entity)
    # Fix entity dictionaries in the relations field
    new_relations = []
    for relation in annotation['relations']:
        ent1 = relation["head_span"]
        ent2 = relation["child_span"]
        ent1_text = annotation['text'][ent1['start']:ent1['end']]
        ent2_text = annotation['text'][ent2['start']:ent2['end']]
        for desired_entity_label in mapper.keys():
            candidate_tokens = mapper[desired_entity_label]
            ent1 = assign_label_to_ent(ent=ent1, ent_text=ent1_text, desired_label=desired_entity_label,
                                       candidate_tokens=candidate_tokens)
            ent2 = assign_label_to_ent(ent=ent2, ent_text=ent2_text, desired_label=desired_entity_label,
                                       candidate_tokens=candidate_tokens)
        if ent1_text not in to_exclude and ent2_text not in to_exclude:
            relation["head_span"] = ent1
            relation["child_span"] = ent2
            new_relations.append(relation)

    annotation['spans'] = new_entities
    annotation['relations'] = new_relations
    return annotation


def assign_label_to_ent(ent, ent_text, desired_label, candidate_tokens):
    if ent_text in candidate_tokens:
        ent['label'] = desired_label
    return ent


def check_and_fix_p1(annotations: List[Dict], inspect_mentions=True) -> List[Dict]:
    print("\n")
    annotations = [fix_incorrect_dvals(inp_annotation=annotation) for annotation in annotations]
    annotations = [swap_clear_incorrect(annotation=annotation) for annotation in annotations]
    d_val_pointing(annotations=annotations)
    check_rel_tokens_entities(annotations=annotations)
    uq_entities = set([span["label"] for annotation in annotations for span in annotation["spans"]])
    uq_relations = set([relation["label"] for annotation in annotations for relation in annotation["relations"]])
    print(f"Unique entities: {uq_entities}\nUnique relations: {uq_relations}")
    if inspect_mentions:
        for entity_type in uq_entities:
            print(f"\n========== {entity_type} ==========")
            ent_mentions = set([annotation["text"][span["start"]:span["end"]] for annotation in annotations
                                for span in annotation["spans"] if span["label"] == entity_type])
            [print(mention) for mention in ent_mentions]

    return annotations


def get_all_relevant_related_spans(annotations: List[Dict], label: str) -> List[str]:
    annotations = [simplify_annotations(sentence) for sentence in annotations]
    out_spans = []

    for sentence in annotations:
        c_val_entities = get_cval_entities(sentence)
        if c_val_entities:
            for relation in sentence['relations']:
                if relation['label'] == "RELATED" and relation["child_span"] in c_val_entities and \
                        relation["head_span"]["label"] == label:
                    new_span = sentence['text'][relation["head_span"]["start"]:relation["head_span"]["end"]]
                    out_spans.append(new_span)

    return out_spans


def get_c_val_dicts(annotation: Dict) -> List[Dict]:
    """
    Gets as an input an annotated dataset and it returns a list of central values with all the related information
    in the form of:

    dict(
        parameter=None,
        central_v={
            "value/range": s_text[c_val["start"]:c_val["end"]],
            "units": None,
            "type_meas": None,
            "compare": None
        },
        deviation={
            "value/range": None,
            "units": None,
            "type_meas": None,
            "compare": None
        },
        sentence=annotation["text"],
        source=annotation["meta"]["url"],
                            )

    """
    annotation = simplify_annotations(annotation)
    COMPLEMENTARY_FIELDS = ["TYPE_MEAS", "COMPARE", "UNITS"]
    # 1. Get central values

    c_val_entities = get_cval_entities(annotation)
    output_cvals = []
    s_text = annotation["text"]

    if c_val_entities:
        for c_val in c_val_entities:

            new_cval = dict(
                            parameter=None,
                            central_v={
                                "value/range": s_text[c_val["start"]:c_val["end"]],
                                "units": None,
                                "type_meas": None,
                                "compare": None
                            },
                            deviation={
                                "value/range": None,
                                "units": None,
                                "type_meas": None,
                                "compare": None
                            },
                            sentence=annotation["text"],
                            source=annotation["meta"]["url"],
                            pmid=annotation["metadata"]["pmid"],
                            sentence_hash=annotation["sentence_hash"]
                            )
            relations = annotation["relations"]
            for relation in relations:
                child_span = relation["child_span"]
                head_span = relation["head_span"]
                rel_type = relation["label"]
                if child_span == c_val:
                    if rel_type == "C_VAL":
                        # 2. Get the PK mention relating to the central measurement
                        new_cval["parameter"] = s_text[head_span["start"]:head_span["end"]]
                    if rel_type == "RELATED":
                        # 3. Get complementary info
                        for f in COMPLEMENTARY_FIELDS:
                            if head_span["label"] == f:
                                new_cval["central_v"][f.lower()] = s_text[head_span["start"]:head_span["end"]]
                    if rel_type == "D_VAL":
                        # 4. Find deviation measurement
                        new_cval["deviation"]["value/range"] = s_text[head_span["start"]:head_span["end"]]
                        # 5. Find complementary info for dev measurement
                        for subrel in relations:
                            if subrel["child_span"] == head_span and subrel["label"] == "RELATED":
                                for f in COMPLEMENTARY_FIELDS:
                                    if subrel["head_span"]["label"] == f:
                                        new_cval["deviation"][f.lower()] = s_text[subrel["head_span"]["start"]:
                                                                                  subrel["head_span"]["end"]]
            output_cvals.append(new_cval)
    return output_cvals
