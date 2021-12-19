"""Functions concerning model operations. Tests available at tests/test_model_utils.py"""
import gc
import torch
from typing import Dict, List, Tuple, Any
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger
from tokenizers import Encoding
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pkrex.annotation_preproc import view_entities_terminal
import warnings
from pkrex.models.sampling import collate_fn_padding

ACCEPTABLE_ENTITY_COMBINATIONS = [
    ("PK", "VALUE"), ("PK", "RANGE"), ("VALUE", "PK"), ("RANGE", "PK"),  # C_VAL relations
    ("VALUE", "VALUE"), ("RANGE", "RANGE"), ("RANGE", "VALUE"), ("VALUE", "RANGE"),  # D_VAL relations

    # RELATED relations
    ("UNITS", "VALUE"), ("UNITS", "RANGE"), ("VALUE", "UNITS"), ("RANGE", "UNITS"),
    ("COMPARE", "VALUE"), ("COMPARE", "RANGE"), ("VALUE", "COMPARE"), ("RANGE", "COMPARE")

]

REX2ID = dict(NO_RELATION=0, C_VAL=1, D_VAL=2, RELATED=3)
ID2REX = {v: k for k, v in REX2ID.items()}


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
    if span_list:
        if 'start' in span_list[0]:
            span_list = sorted(span_list, key=lambda d: (d['start'], d['end'], d['label']))
        else:
            span_list = sorted(span_list, key=lambda d: (d['token_start'], d['token_end'], d['label']))
        out_spans = []
        for idx, sp in enumerate(span_list):
            sp['ent_id'] = idx
            out_spans.append(sp)
        return out_spans
    return []


def cleanspans(inp_spans):
    out_spans = []
    for s in inp_spans:
        new_dict = dict(start=s['start'], end=s['end'], label=s['label'])
        if 'ent_id' in s.keys():
            new_dict['ent_id'] = s['ent_id']
        out_spans.append(new_dict)
    return out_spans


def align_tokens_and_annotations_bilou(tokenized: Encoding, annotations: List[Dict], example: Dict,
                                       relations: List[Dict]):
    annotations_sorted = cleanspans(
        assign_index_to_spans(sorted(annotations, key=lambda d: (d['start'], d['end'], ['label']))))
    annotations_from_rel = cleanspans(assign_index_to_spans(get_unique_spans_from_rels(rels=relations)))

    if annotations_sorted != annotations_from_rel:
        a = 1

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
        end_ent_tok_idx = start_ent_tok_idx
        if start_ent_tok_idx + 1 < sequence_len:  # if it's not the last element in the sequence
            for next_token in inp_bio_seq[start_ent_tok_idx + 1:]:
                if next_token.split("-")[0] == "I" and next_token.split("-")[1] == entity_type:
                    end_ent_tok_idx += 1
                else:
                    break
        out_spans.append(dict(token_start=start_ent_tok_idx, token_end=end_ent_tok_idx, label=entity_type))
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


def add_character_offsets(offset_mappings, entity_tokens, tags_per_sentence):
    batch_ent_offsets = []
    batch_iobs = []
    for offsets, entities, tmp_iobs in zip(offset_mappings, entity_tokens, tags_per_sentence):
        tmp_instance_spans = []
        for entity in entities:
            entity["start"] = int(offsets[entity["token_start"]][0])
            entity["end"] = int(offsets[entity["token_end"]][1])
            tmp_instance_spans.append(entity)
        batch_ent_offsets.append(tmp_instance_spans)
        batch_iobs.append(tmp_iobs)
    return batch_ent_offsets, batch_iobs


def get_tags_per_sentence(batch_logits, inp_att_masks, id2tag):
    ner_predictions = batch_logits.argmax(dim=2)
    tags_per_sentence = [
        [id2tag[tok_pred] if tok_mask != 0 else 'O' for tok_pred, tok_mask in zip(sentence_pred,
                                                                                  sentence_masks)]
        for sentence_pred, sentence_masks in zip(ner_predictions.tolist(), inp_att_masks.tolist())
    ]
    return tags_per_sentence


def rex_pred_checks(original_tuples, original_token_masks, rex_preds_flat):
    assert len(original_tuples) == len(original_token_masks)
    flat_original_tuples = [ent_pair for tup_list in original_tuples if tup_list for ent_pair in tup_list]
    assert len(flat_original_tuples) == len(rex_preds_flat)


def predict_pl_bert_rex(inp_texts, inp_model, inp_tokenizer, batch_size, n_workers):
    """
    Return output for ner and rex in the same way as prodigy annotations
    """
    # Tokenize and construct dataloader
    encodings = inp_tokenizer(inp_texts, padding=True, truncation=True, max_length=inp_model.seq_len,
                              return_offsets_mapping=True, return_overflowing_tokens=False)
    predict_dataset = PKDatasetInference(encodings=encodings)
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=True)

    # Put model in evaluation mode and initialise lists for storing predictions
    inp_model.eval()
    all_ent_offsets = []
    all_rex_predictions = []
    for idx, batch in tqdm(enumerate(predict_loader)):
        with torch.no_grad():
            # =================== 1. Pass through BERT ===============================
            h = inp_model.bert(batch['input_ids'], attention_mask=batch['attention_mask'])[0]
            # =================== 2. Predict entities ===============================
            batch_logits = inp_model.predict_entities(sequence_bert_output=h)
            # 2.1 Get IOBs
            tags_per_sentence = get_tags_per_sentence(batch_logits=batch_logits,
                                                      inp_att_masks=batch['attention_mask'],
                                                      id2tag=inp_model.id2tag)
            # 2.2 Get token offsets Token offsets
            entity_tokens = [bio_to_entity_tokens(tag_prediction) for tag_prediction in tags_per_sentence]
            assert len(tags_per_sentence) == len(entity_tokens) == len(batch['input_ids'])
            # 2.3 Add character offsets
            batch_ent_offsets, batch_iobs = add_character_offsets(offset_mappings=batch["offset_mapping"],
                                                                  entity_tokens=entity_tokens,
                                                                  tags_per_sentence=tags_per_sentence)
            all_ent_offsets += batch_ent_offsets

            # =================== 3. Predict relations ===============================
            # 2.1 Construct samples for batch
            rex_batch_no_pad = [make_rex_pred_batch_sample(inp_iobs=tmp_iobs) for tmp_iobs in batch_iobs]
            # 2.2 Pad batch and get masks (tuples that are not entity candidates ([0,0])
            pred_batch = collate_fn_padding(batch=rex_batch_no_pad)
            rex_masks = inp_model.get_rex_masks(inp_rel_tuples=pred_batch['rel_tuples'])
            # 2.3 Make predictions (this function already applies rex_masks to filter out irrelevant tuples)
            rex_preds_flat = []
            if rex_masks is not None and sum(rex_masks) != 0:
                rex_logits = inp_model.predict_relations(inp_sequence_rep=h, inp_ent_masks=pred_batch['entity_masks'],
                                                         inp_rex_masks=rex_masks, inp_ctx_masks=pred_batch['ctx_mask'],
                                                         inp_rel_tuples=pred_batch['rel_tuples'],
                                                         inp_ctx_width=pred_batch['ctx_len'])

                rex_preds_flat = [ID2REX[x] for x in rex_logits.argmax(dim=1).tolist()]

            # 2.4 Now Map predictions to the way in which prodigy labels are provided (character label per
            # pairs of entities)
            original_tuples = [x['rel_tuples'].tolist() for x in rex_batch_no_pad]
            original_token_masks = [x['entity_masks'] for x in rex_batch_no_pad]
            rex_pred_checks(original_tuples=original_tuples, original_token_masks=original_token_masks,
                            rex_preds_flat=rex_preds_flat)

            rex_pred_batch_ready = map_bert_pred_to_prodigy(rex_preds_flat=rex_preds_flat,
                                                            original_tuples=original_tuples,
                                                            original_token_masks=original_token_masks,
                                                            batch_ent_offsets=batch_ent_offsets)

            all_rex_predictions += rex_pred_batch_ready

    return all_ent_offsets, all_rex_predictions


def map_bert_pred_to_prodigy(rex_preds_flat, original_tuples, original_token_masks, batch_ent_offsets):
    i = 0
    rex_pred_batch = []
    for sample_cand_ents, samle_tok_masks, sample_ent_offsets in zip(original_tuples, original_token_masks,
                                                                     batch_ent_offsets):
        sample_rex_labels = []
        if sample_cand_ents:
            for rel_cand_tuple in sample_cand_ents:
                label = rex_preds_flat[i]
                ent1_tok_idxs = samle_tok_masks[rel_cand_tuple[0]].nonzero()
                ent2_tok_idxs = samle_tok_masks[rel_cand_tuple[1]].nonzero()
                ent1_tok_offsets = (ent1_tok_idxs.min().item(), ent1_tok_idxs.max().item())
                ent2_tok_offsets = (ent2_tok_idxs.min().item(), ent2_tok_idxs.max().item())
                ent1_ready = find_original_entity(inp_tok_offsets=ent1_tok_offsets,
                                                  all_entites_offsets=sample_ent_offsets)
                ent2_ready = find_original_entity(inp_tok_offsets=ent2_tok_offsets,
                                                  all_entites_offsets=sample_ent_offsets)
                assert ent1_ready and ent2_ready
                tmp_pred_rel = dict(left=ent1_ready, right=ent2_ready, label=label)
                sample_rex_labels.append(tmp_pred_rel)
                i += 1
        rex_pred_batch.append(sample_rex_labels)
    return rex_pred_batch


def find_original_entity(inp_tok_offsets: Tuple[int, int], all_entites_offsets: List[Dict]):
    for cand_ent in all_entites_offsets:
        if cand_ent['token_start'] == inp_tok_offsets[0] and cand_ent['token_end'] == inp_tok_offsets[1]:
            return cand_ent


def make_rex_pred_batch_sample(inp_iobs):
    entity_tokens = assign_index_to_spans(bio_to_entity_tokens(inp_bio_seq=inp_iobs))
    candidate_rels = generate_all_possible_rels(inp_entities=entity_tokens)
    candidate_rels = filter_not_allowed_rels(inp_possible_rels=candidate_rels)
    candidate_rels_ent_ids = possible_rels_to_entity_ids_format(inp_entities=entity_tokens,
                                                                inp_rels=candidate_rels)
    pred_ent_masks, pred_ctx_masks, pred_ctx_lengths = get_entities_and_ctx_masks(inp_entities=entity_tokens,
                                                                                  inp_rels=candidate_rels,
                                                                                  max_len=len(inp_iobs))
    tmp_batch_sample = dict(
        entity_masks=torch.tensor(pred_ent_masks),
        rel_tuples=torch.tensor(candidate_rels_ent_ids),
        ctx_mask=torch.tensor(pred_ctx_masks),
        ctx_len=torch.tensor(pred_ctx_lengths)
    )
    return tmp_batch_sample


def create_pred_rex_loader_no_labels(inp_iobs, bs):
    dataset_samples = []
    for iobs in inp_iobs:
        entity_tokens = assign_index_to_spans(bio_to_entity_tokens(inp_bio_seq=iobs))
        candidate_rels = generate_all_possible_rels(inp_entities=entity_tokens)
        candidate_rels = filter_not_allowed_rels(inp_possible_rels=candidate_rels)
        candidate_rels_ent_ids = possible_rels_to_entity_ids_format(inp_entities=entity_tokens,
                                                                    inp_rels=candidate_rels)
        pred_ent_masks, pred_ctx_masks, pred_ctx_lengths = get_entities_and_ctx_masks(inp_entities=entity_tokens,
                                                                                      inp_rels=candidate_rels,
                                                                                      max_len=len(inp_iobs))
        tmp_batch_sample = dict(
            entity_masks=torch.tensor(pred_ent_masks),
            rel_tuples=torch.tensor(candidate_rels_ent_ids),
            ctx_mask=torch.tensor(pred_ctx_masks),
            ctx_len=torch.tensor(pred_ctx_lengths)
        )
        dataset_samples.append(tmp_batch_sample)

    rex_datset = REXInferenceDataset(dataset_samples=dataset_samples)
    rex_loader = DataLoader(rex_datset, batch_size=bs, num_workers=12, collate_fn=collate_fn_padding, pin_memory=True)
    return rex_loader


class REXInferenceDataset(Dataset):
    def __init__(self, dataset_samples: List[Dict[str, torch.Tensor]]):
        self.samples = dataset_samples

    def __getitem__(self, idx):
        return self.samples[idx]


def arrange_from_pl_bert_ner(inp_pred_form_pl_ber_ner, seq_len):
    out_iob = []
    out_ents = []
    for iob_pred in inp_pred_form_pl_ber_ner:
        tmp_iob = ['O'] * seq_len
        tmp_ents = []
        if iob_pred:
            tmp_iob = iob_pred[0]['tags']
            for ent in iob_pred:
                tmp_ents.append(dict(token_start=ent['token_start'], token_end=ent['token_end'],
                                     label=ent['label'], start=ent['start'], end=ent['end'])
                                )
        out_iob.append(tmp_iob)
        out_ents.append(tmp_ents)
    return out_iob, out_ents


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


def arrange_relationship(inp_rel: Dict):
    head_key = 'head'
    child_key = 'child'
    if head_key not in inp_rel.keys():
        assert 'head_span' in inp_rel.keys()
        head_key = 'head_span'
    if child_key not in inp_rel.keys():
        assert 'child_span' in inp_rel.keys()
        child_key = 'child_span'
    hs_start = inp_rel[head_key]['token_start']
    hs_end = inp_rel[head_key]['token_end']
    cs_start = inp_rel[child_key]['token_start']
    cs_end = inp_rel[child_key]['token_end']
    if cs_start > hs_end:
        right_entity = inp_rel[child_key]
        left_entity = inp_rel[head_key]
    else:
        assert hs_start > cs_end
        right_entity = inp_rel[head_key]
        left_entity = inp_rel[child_key]
    return left_entity, right_entity


def get_ctx_token_offsets(inp_rel: Dict) -> Tuple[int, int]:
    """
    inp_rel has the form of:
    example_rel = {"head": {'token_start': 1, 'token_end': 1, 'label': 'VALUE', 'ent_id': 1},
     "child": {'token_start': 4, 'token_end': 5, 'label': 'UNITS', 'ent_id': 2}}
     This function returns the index of the end token of the right-hand sided entity and the index of the
     beginning left-hand sided entity
     example_return = (1,4)
    """
    left_entity, right_entity = arrange_relationship(inp_rel=inp_rel)
    return left_entity['token_end'], right_entity['token_start']


def get_ent_and_ctx_token_offsets(inp_rel: Dict) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Given an input relationship returns a tuple in the form of:
    (
        (tok_start,tok_end), # offsets left-hand side entity
        (tok_start,tok_end), # offsets right-hand side entity
        (tok_start,tok_end) # offsets context tokens between left-hand side and right-hand side entity
    )
    """
    left_entity, right_entity = arrange_relationship(inp_rel=inp_rel)
    left_ent_token_offsets = (left_entity['token_start'], left_entity['token_end'] + 1)
    right_ent_token_offsets = (right_entity['token_start'], right_entity['token_end'] + 1)
    ctx_token_offsets = (left_ent_token_offsets[1], right_ent_token_offsets[0])
    return left_ent_token_offsets, right_ent_token_offsets, ctx_token_offsets


def simplify_relation(inp_rel: Dict) -> Tuple[Tuple[int, int], Tuple[int, int], str]:
    """
    Given an input relation it returns a tuple in the form of
    (
    (tok_start,tok_end), # offsets left-hand side entity
    (tok_start,tok_end), # offsets right-hand side entity
    relation_label # C_VAL etc...
    )
    """
    left_ent, right_ent = arrange_relationship(inp_rel=inp_rel)
    assert left_ent['token_start'] < right_ent['token_start']
    out_tuple = ((left_ent['token_start'], left_ent['token_end'] + 1),
                 (right_ent['token_start'], right_ent['token_end'] + 1),
                 inp_rel['label'])

    return out_tuple


def associate_rel_tuples_with_labels(inp_rel_labels: List[Dict], inp_rel_tuples: List[List[int]]):
    out_labels, i = [], 0
    for rel_indices in inp_rel_tuples:
        relation_label = "NO_RELATION"
        for annotated_rel in inp_rel_labels:
            left_ent, right_ent = arrange_relationship(inp_rel=annotated_rel)
            if rel_indices[0] == left_ent['ent_id'] and rel_indices[1] == right_ent['ent_id']:
                relation_label = annotated_rel['label']
                i += 1
        out_labels.append(relation_label)

    assert len(out_labels) == len(inp_rel_tuples)
    assert i == len(inp_rel_labels)
    return out_labels


def associate_triplets_with_rels(inp_rel_labels: List[Dict], inp_triplets: List[Tuple[Tuple[int, int],
                                                                                      Tuple[int, int],
                                                                                      Tuple[int, int]]]
                                 ):
    simplified_rels = [simplify_relation(inp_rel=r) for r in inp_rel_labels]
    out_labels = []
    i = 0
    for triplet in inp_triplets:
        offsets1 = triplet[0]
        offsets2 = triplet[1]
        relation_label = "NO_RELATION"
        for annotated_rel in simplified_rels:
            if offsets1 == annotated_rel[0] and offsets2 == annotated_rel[1]:
                relation_label = annotated_rel[2]
                i += 1
        out_labels.append(relation_label)

    assert len(out_labels) == len(inp_triplets)
    assert i == len(inp_rel_labels)  # asserts the number of assigned true relations is the same as in the input
    return out_labels


def dynamic_index_maxpool(sentence_rep_batch: torch.Tensor, indices_tensor: torch.Tensor) -> torch.Tensor:
    """
    This function max pools entities in token_start, token_end from a specific sentence
    @param sentence_rep_batch: Batch of sentences composed by token representations.
        The shape is: batch_size * max_length * token_embedding_size
    @param indices_tensor: Tensor with indices of all entities expressed as [token_start,token_end+1]
        The shape is: batch_size * max_n_entities * 2
    @return Tensor of the resulting tokens max pooled.
        The shape is: batch_size * max_n_entities * token_embedding_size
    """
    bs = sentence_rep_batch.shape[0]
    max_len = sentence_rep_batch.shape[1]
    max_n_entities = indices_tensor.shape[1]
    tok_emb_size = sentence_rep_batch.shape[2]
    entity_masks = entity_indices_to_mask(inp_indices=indices_tensor,
                                          batch_size=bs,
                                          max_n_entities=max_n_entities,
                                          max_len=max_len)  # should be boolean and have a shape of

    entity_spans_pool = dpooler(entity_masks=entity_masks, sentence_rep_batch=sentence_rep_batch)
    assert entity_spans_pool.shape == torch.Size([bs, max_n_entities, tok_emb_size])
    # batch_size * max_n_entities * max_len
    return entity_spans_pool


def dpooler(entity_masks, sentence_rep_batch):
    m = (entity_masks.unsqueeze(-1) == 0).float()  # * (-1e30)  # float('-inf')  # (-1e30)
    m[m == 1] = float('-inf')
    entity_spans_pool = m + sentence_rep_batch.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
    entity_spans_pool = entity_spans_pool.max(dim=2)[0]
    return entity_spans_pool


def entity_indices_to_mask(inp_indices: torch.Tensor, batch_size: int, max_n_entities: int, max_len: int):
    assert len(inp_indices.shape) == 3
    out_tensor = []
    for batch_indices in inp_indices:
        batch_tensor = []
        for entity_offsets in batch_indices:
            assert entity_offsets.shape == torch.Size([2])

            start = entity_offsets[0].item()
            end = entity_offsets[1].item()
            if start != end:
                ent_tensor = [True if i in range(start, end) else False for i in range(0, max_len)]
            else:
                ent_tensor = max_len * [False]
            batch_tensor.append(ent_tensor)
        out_tensor.append(batch_tensor)
    out_tensor = torch.Tensor(out_tensor).bool()
    assert out_tensor.shape == torch.Size([batch_size, max_n_entities, max_len])
    return out_tensor


def possible_rels_to_entity_ids_format(inp_entities, inp_rels):
    """
    Makes tuples of relationships between entities in entity list according to their entity id
    """
    ent_indices = [x['ent_id'] for x in inp_entities]
    check_list_is_sorted(inp_list=ent_indices)
    out_rels = []
    for rel in inp_rels:
        left_entity, right_entity = arrange_relationship(inp_rel=rel)
        out_rels.append([left_entity['ent_id'], right_entity['ent_id']])
    return out_rels


def check_list_is_sorted(inp_list: List[int]):
    assert all(inp_list[i] <= inp_list[i + 1] for i in range(len(inp_list) - 1))


def get_entities_and_ctx_masks(inp_entities, inp_rels, max_len) -> Tuple[List[List[bool]], List[List[bool]], List[int]]:
    """
    Returns boolean list corresponding to the entities in the batch and boolean list
    corresponding to the context token in each relation and context token lengths
    """
    ent_masks, ctx_masks, ctx_lengths = [], [], []
    for ent in inp_entities:
        start_tok_offset = ent['token_start']
        end_tok_offset = ent['token_end'] + 1
        assert start_tok_offset < end_tok_offset
        ent_mask = [True if i in range(start_tok_offset, end_tok_offset) else False for i in range(0, max_len)]
        ent_masks.append(ent_mask)

    for r in inp_rels:
        left_entity, right_entity = arrange_relationship(inp_rel=r)
        left_ent_token_offsets = (left_entity['token_start'], left_entity['token_end'] + 1)
        right_ent_token_offsets = (right_entity['token_start'], right_entity['token_end'] + 1)
        ctx_token_offsets = (left_ent_token_offsets[1], right_ent_token_offsets[0])
        assert ctx_token_offsets[0] <= ctx_token_offsets[1]
        ctx_lengths.append(ctx_token_offsets[1] - ctx_token_offsets[0])
        ctx_mask = [True if i in range(ctx_token_offsets[0], ctx_token_offsets[1]) else False
                    for i in range(0, max_len)]
        ctx_masks.append(ctx_mask)
    return ent_masks, ctx_masks, ctx_lengths
