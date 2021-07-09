import gc
import torch
from typing import Dict, List, Tuple, Any
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger
from tokenizers import Encoding
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pkrex.annotation_preproc import view_entities_terminal
import warnings


def get_f1(p, r):
    if p + r == 0.:
        return 0.
    else:
        return (2 * p * r) / (p + r)


def get_ner_metrics(inp_dict):
    p = inp_dict['precision']
    r = inp_dict['recall']
    if "f1" in inp_dict.keys():
        f1 = inp_dict['f1']
    else:
        f1 = get_f1(p=p, r=r)
    return p, r, f1


def get_unique_spans_from_rels(rels: List[Dict]):
    out_spans = []
    for r in rels:
        for i in ['head_span', 'child_span']:
            if r[i] not in out_spans:
                out_spans.append(r[i])
    out_spans = sorted(out_spans, key=lambda d: (d['start'], d['end'], d['label']))
    return out_spans


def assign_index_to_spans(span_list: List[Dict]) -> List[Dict]:
    if 'start' in span_list[0]:
        span_list = sorted(span_list, key=lambda d: (d['start'], d['end'], d['label']))
    else:
        span_list = sorted(span_list, key=lambda d: (d['token_start'], d['token_end'], d['label']))
    out_spans = []
    for idx, sp in enumerate(span_list):
        sp['ent_id'] = idx
        out_spans.append(sp)
    return out_spans


def align_tokens_and_annotations_bilou(tokenized: Encoding, annotations: List[Dict], example: Dict,
                                       relations: List[Dict]):
    annotations_sorted = assign_index_to_spans(sorted(annotations, key=lambda d: (d['start'], d['end'], ['label'])))
    annotations_from_rel = assign_index_to_spans(get_unique_spans_from_rels(rels=relations))
    assert annotations_sorted == annotations_from_rel

    tokens = tokenized.tokens
    aligned_labels_bio = ["O"] * len(tokens)  # Make a list to store our labels the same length as our tokens
    aligned_labels_bilou = ["O"] * len(tokens)

    entity_tokens = []
    for anno in annotations_sorted:
        annotation_token_ix_set = (
            set()
        )  # A set that stores the token indices of the annotation
        for char_ix in range(anno["start"], anno["end"]):

            token_ix = tokenized.char_to_token(char_ix)
            if token_ix is not None:
                annotation_token_ix_set.add(token_ix)

        if not check_correct_alignment(tokenized=tokenized, entity_token_ids=annotation_token_ix_set, annotation=anno):
            example['_task_hash'] = example['_task_hash'] if '_task_hash' in example.keys() else ""
            warnings.warn(f"Careful, some character-level annotations did not align correctly with BERT tokenizer in "
                          f"example with task hash {example['_task_hash']}:"
                          f"\n{view_entities_terminal(example['text'], anno)}")
            print(example["_task_hash"])

        entity_tokens.append(
            dict(start=anno["start"],
                 end=anno["end"],
                 token_start=min(annotation_token_ix_set),
                 token_end=max(annotation_token_ix_set),
                 label=anno["label"],
                 ent_id=anno["ent_id"]
                 )
        )

        if len(annotation_token_ix_set) == 1:
            # If there is only one token
            token_ix = annotation_token_ix_set.pop()
            # bilou
            prefix = "U"  # This annotation spans one token so is prefixed with U for unique
            aligned_labels_bilou[token_ix] = f"{prefix}-{anno['label']}"
            # bio
            prefix = "B"
            aligned_labels_bio[token_ix] = f"{prefix}-{anno['label']}"

        else:

            last_token_in_anno_ix = len(annotation_token_ix_set) - 1
            for num, token_ix in enumerate(sorted(annotation_token_ix_set)):
                if num == 0:
                    prefix = "B"
                elif num == last_token_in_anno_ix:
                    prefix = "L"  # Its the last token
                else:
                    prefix = "I"  # We're inside of a multi token annotation
                aligned_labels_bilou[token_ix] = f"{prefix}-{anno['label']}"
                if prefix == "L":
                    prefix = "I"
                aligned_labels_bio[token_ix] = f"{prefix}-{anno['label']}"

    relations_bert = construct_relations_bert(original_relations=relations, new_entities=entity_tokens)
    assert len(relations) == len(relations_bert)
    return aligned_labels_bilou, aligned_labels_bio, entity_tokens, relations_bert


def construct_relations_bert(original_relations: List[Dict], new_entities: List[Dict]) -> List[Dict]:
    """
    Returns relations with the new entities in the form of
    {
    "head_span": (entity span head),
    "child_span": (entity span child),
    "head_span_idx": int,
    "child_span_idx": int,
    "relation_label": str
    }
    """
    out_relations = []

    for tmp_rel in original_relations:
        hs = tmp_rel['head_span']
        cs = tmp_rel['child_span']
        newhs = map_old_span_to_new(inp_old_span=hs, new_spans=new_entities)
        newcs = map_old_span_to_new(inp_old_span=cs, new_spans=new_entities)
        new_rel = dict(head_span=newhs, child_span=newcs,
                       head_span_idx=newhs['ent_id'], child_span_idx=newcs['ent_id'],
                       label=tmp_rel['label']
                       )
        out_relations.append(new_rel)

    return out_relations


def map_old_span_to_new(inp_old_span: Dict, new_spans: List[Dict]) -> Dict:
    out_sp = None
    for nsp in new_spans:
        if nsp['start'] == inp_old_span['start'] and nsp['end'] == inp_old_span['end']:
            out_sp = nsp
            break
    if out_sp is not None:
        return out_sp
    else:
        raise ValueError("ERROR FINDING RELATION SPAN INTO THE NEW BERT-BASED SPANS")


def check_correct_alignment(tokenized: Encoding, entity_token_ids: set, annotation: Dict):
    """Checks that the original character-level annotations for an entity correspond to the start and end character
    of bert-tokens """

    orig_start_ent_char = annotation["start"]
    orig_end_ent_char = annotation["end"]

    start_char_bert_ent = tokenized.offsets[min(entity_token_ids)][0]
    end_char_bert_ent = tokenized.offsets[max(entity_token_ids)][1]

    if orig_start_ent_char == start_char_bert_ent and orig_end_ent_char == end_char_bert_ent:
        return True
    return False


def bio_to_entity_tokens(inp_bio_seq: List[str]) -> List[Dict]:
    """
    Gets as an input a list of BIO tokens and returns the starting and end tokens of each span
    @return: The return should be a list of dictionary spans in the form of [{"token_start": x,"token_end":y,"label":""]
    """
    out_spans = []

    b_toks = sorted([i for i, t in enumerate(inp_bio_seq) if "B-" in t])  # Get the indexes of B tokens
    sequence_len = len(inp_bio_seq)
    for start_ent_tok_idx in b_toks:
        entity_type = inp_bio_seq[start_ent_tok_idx].split("-")[1]
        end_ent_tok_idx = start_ent_tok_idx + 1
        if start_ent_tok_idx + 1 < sequence_len:  # if it's not the last element in the sequence
            for next_token in inp_bio_seq[start_ent_tok_idx + 1:]:
                if next_token.split("-")[0] == "I" and next_token.split("-")[1] == entity_type:
                    end_ent_tok_idx += 1
                else:
                    break
        out_spans.append(dict(token_start=start_ent_tok_idx, token_end=end_ent_tok_idx - 1, label=entity_type))
    return out_spans


def simplify_labels_and_tokens(sample_tokens: List[str], sample_labels: List[str],
                               irrelevant_label: str) -> Tuple[List[str], List[str]]:
    """

    @param sample_tokens: ["[CLS]", "hi", "##ho","hey", "huu", ".", "[SEP]", "[PAD]", "[PAD]"]
    @param sample_labels: ["-", "B-PK", "-","I-PK", "O", "O", "[SEP]", "[PAD]", "[PAD]"]
    @param irrelevant_label: "-"
    @return: ["hi", "##ho","hey", "huu", "."], ["B-PK", "I-PK","I-PK", "O", "O"]
    """
    sample_tokens_clean = []
    sample_labels_clean = []
    prev_label = None
    for token, label in zip(sample_tokens, sample_labels):
        if token not in ['[CLS]', '[SEP]', '[PAD]']:
            sample_tokens_clean.append(token)
            if label == irrelevant_label:
                if "B" in prev_label:
                    new_label = "I-" + prev_label.split("-")[1]
                else:
                    new_label = prev_label
                sample_labels_clean.append(new_label)
                prev_label = new_label
            else:
                sample_labels_clean.append(label)
                prev_label = label
    assert len(sample_tokens_clean) == len(sample_labels_clean)
    return sample_tokens_clean, sample_labels_clean


def empty_cuda_cache(n_gpus: int):
    torch.cuda.empty_cache()
    gc.collect()
    for x in range(0, n_gpus):
        with torch.cuda.device(x):
            torch.cuda.empty_cache()


def get_tensorboard_logger(log_dir: str, run_name: str) -> LightningLoggerBase:
    return TensorBoardLogger(save_dir=log_dir, name="tensorboard-logs-{}".format(run_name))


def predict_pl_bert_ner(inp_texts, inp_model, inp_tokenizer, batch_size, n_workers):
    encodings = inp_tokenizer(inp_texts, padding=True, truncation=True, max_length=inp_model.seq_len,
                              return_offsets_mapping=True, return_overflowing_tokens=True)
    predict_dataset = PKDatasetInference(encodings=encodings)
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, num_workers=n_workers)
    inp_model.eval()
    predicted_entities = []
    overflow_to_sample = []
    all_seq_end = []

    for idx, batch in tqdm(enumerate(predict_loader)):
        with torch.no_grad():
            batch_logits = inp_model(input_ids=batch['input_ids'],
                                     attention_masks=batch['attention_mask']).to('cpu')

            batch_predicted_entities = predict_bio_tags(model_logits=batch_logits, inp_batch=batch,
                                                        id2tag=inp_model.id2tag)

        for seq_end, omap in zip(batch['offset_mapping'], batch['overflow_to_sample_mapping']):
            all_seq_end.append(seq_end.flatten().max().item())
            overflow_to_sample.append(omap.item())

        predicted_entities += batch_predicted_entities

    predicted_entities = remap_overflowing_entities(predicted_tags=predicted_entities, all_seq_end=all_seq_end,
                                                    overflow_to_sample=overflow_to_sample, original_texts=inp_texts,
                                                    offset_mappings=encodings["offset_mapping"]
                                                    )
    return predicted_entities


def predict_bio_tags(model_logits: torch.Tensor, inp_batch: Dict[str, torch.Tensor],
                     id2tag: Dict[int, str]):
    predictions = model_logits.argmax(dim=2)

    tag_predictions = [[id2tag[prediction.item()] for mask, prediction in zip(att_masks, id_preds) if mask.item() == 1]
                       for att_masks, id_preds in zip(inp_batch["attention_mask"], predictions)]

    return tag_predictions


def remap_overflowing_entities(predicted_tags: List[List[str]], all_seq_end: List[int], overflow_to_sample: List[int],
                               original_texts: List[str], offset_mappings: List[List[Tuple[int, int]]]) -> List[
    List[Dict]
]:
    if len(set(overflow_to_sample)) == len(overflow_to_sample):  # case with no overflowing tokens
        assert len(set(overflow_to_sample)) == len(original_texts)
        tags_per_sentence = predicted_tags
        offset_mappings_rearranged = offset_mappings
    else:
        # Case in which we have overflowing indices
        assert len(all_seq_end) == len(predicted_tags)
        print("Remapping Overflowing")

        all_o_to_s = []
        tags_per_sentence = []
        offset_mappings_rearranged = []
        for i, (ents, send, o_to_s, offsets) in enumerate(zip(predicted_tags, all_seq_end, overflow_to_sample,
                                                              offset_mappings)):
            if o_to_s not in all_o_to_s:
                # tags original sentence
                all_o_to_s.append(o_to_s)
                offset_mappings_rearranged.append(offsets)
                tags_per_sentence.append(ents)
            else:
                # overflowing tags
                new_offsets = offset_mappings_rearranged[-1] + offsets
                new_entities = tags_per_sentence[-1] + ents

                tags_per_sentence = tags_per_sentence[:-1]  # remove last element
                offset_mappings_rearranged = offset_mappings_rearranged[:-1]

                offset_mappings_rearranged.append(new_offsets)
                tags_per_sentence.append(new_entities)  # re-append last element + new one

            assert len(all_o_to_s) == len(set(all_o_to_s))

    entity_tokens = [bio_to_entity_tokens(tag_prediction) for tag_prediction in tags_per_sentence]

    assert len(tags_per_sentence) == len(original_texts) == len(entity_tokens) == len(offset_mappings_rearranged)

    outputs = []
    for offsets, entities, tag in zip(offset_mappings_rearranged, entity_tokens, tags_per_sentence):
        tmp_outputs = []
        for entity in entities:
            entity["start"] = int(offsets[entity["token_start"]][0])
            entity["end"] = int(offsets[entity["token_end"]][1])
            entity["tags"] = tag
            tmp_outputs.append(entity)
        outputs.append(tmp_outputs)

    return outputs


class PKDatasetInference(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["sentence_idx"] = idx
        return item

    def __len__(self):
        return len(self.encodings.encodings)


def generate_all_possible_rels(inp_entities: List[Dict]) -> List[Dict[Any, dict]]:
    """
    Given a list of entities in the form of
    [{'token_start':...,'token_end':...,'label':...,'ent_id':...},{...}]
    it return all the possible pairwise combinations
    """
    possible_rels = []
    inp_entities = sorted(inp_entities, key=lambda d: d['ent_id'])
    for i, ent in enumerate(inp_entities):
        if i + 1 != len(inp_entities):
            for j in range(i + 1, len(inp_entities)):
                possible_rels.append(
                    dict(
                        head=ent,
                        child=inp_entities[j]
                    )
                )
    return possible_rels


ACCEPTABLE_ENTITY_COMBINATIONS = [
    ("PK", "VALUE"), ("PK", "RANGE"), ("VALUE", "PK"), ("RANGE", "PK"),  # C_VAL relations
    ("VALUE", "VALUE"), ("RANGE", "RANGE"), ("RANGE", "VALUE"), ("VALUE", "RANGE"),  # D_VAL relations

    # RELATED relations
    ("UNITS", "VALUE"), ("UNITS", "RANGE"), ("VALUE", "UNITS"), ("RANGE", "UNITS"),
    ("COMPARE", "VALUE"), ("COMPARE", "RANGE"), ("VALUE", "COMPARE"), ("RANGE", "COMPARE")

]


def filter_not_allowed_rels(inp_possible_rels: List[Dict[Any, dict]],
                            allowed_combos=None) -> List[Dict[Any, dict]]:
    """
    Give a list of possible relations  it returns the ones allowed according to the filtering list allowed_combos
    """
    if allowed_combos is None:
        allowed_combos = ACCEPTABLE_ENTITY_COMBINATIONS
    if allowed_combos == "all":
        return inp_possible_rels

    else:
        assert isinstance(allowed_combos, list)
        out_possible_relations = []
        for tmp_rel in inp_possible_rels:
            tmp_combo = (tmp_rel["head"]["label"], tmp_rel["child"]["label"])
            if tmp_combo in allowed_combos:
                out_possible_relations.append(tmp_rel)
    return out_possible_relations
