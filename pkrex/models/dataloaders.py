from typing import Dict, List, Tuple
import numpy as np
from termcolor import colored
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding
from tokenizers import Encoding
import torch
from pkrex.models import sampling
from pkrex.models.utils import bio_to_entity_tokens, assign_index_to_spans, generate_all_possible_rels, \
    filter_not_allowed_rels, possible_rels_to_entity_ids_format, get_entities_and_ctx_masks, \
    associate_rel_tuples_with_labels
from pkrex.utils import read_jsonl
import matplotlib.pyplot as plt
from collections import Counter
from itertools import compress

REX2ID = dict(NO_RELATION=0, C_VAL=1, D_VAL=2, RELATED=3)
ID2REX = {v: k for k, v in REX2ID.items()}


def get_val_dataloader(val_data_file: str, tokenizer: BertTokenizerFast, max_len: int, batch_size: int,
                       n_workers: int, tag_type: str, print_tokens: bool, tag2id: Dict[str, int],
                       dataset_name: str, remove_longer_seqs: bool) -> DataLoader:
    labels_key = tag_type + "_tags"
    raw_val_samples = list(read_jsonl(file_path=val_data_file))
    val_samples = [{"tokens": [t["text"] for t in sentence["tokens"]], "text": sentence["text"],
                    "labels": sentence[labels_key], "relations": sentence['relations']} for sentence in raw_val_samples]

    val_dataloader = make_dataloader(inp_samples=val_samples, batch_size=batch_size, inp_tokenizer=tokenizer,
                                     max_len=max_len, shuffle=False, n_workers=n_workers,
                                     tag2id=tag2id, dataset_name=dataset_name, print_tokens=print_tokens,
                                     remove_longer_seqs=remove_longer_seqs)

    return val_dataloader


def get_training_dataloader(training_data_file: str, tokenizer: BertTokenizerFast, max_len: int, batch_size: int,
                            n_workers: int, tag_type: str, print_tokens: bool, dataset_name: str = "training",
                            tag2id: Dict[str, int] = None, rmls: bool = False
                            ) -> Tuple[DataLoader, Dict[str, int], Dict[int, str], Dict[str, float]]:
    """
    @param training_data_file: jsonl file with NER annotations in IOB format
    @param tokenizer: Pre-loaded BERT tokenizer in which all the special tokens (e.g.[unused0]) have been registered
    @param max_len: maximum length for the list of context/mention tokens
    @param batch_size: batch size
    @param n_workers: number of workers for the dataloader
    @param tag_type: either bio or biluo
    @param print_tokens: whether to print tokens
    @param dataset_name: name of the dataset (just for logging purposes)
    @param tag2id: optional to include if loading a pre-trained model that already had some tag2id
    @param rmls: whether to remove annotations with more tokens than max_len
    @return: (1) dataloader for the training data, (2) tag2id converter, (3) id2tag converter, (4) scaling dictionary
    """
    # 1. Read data
    labels_key = tag_type + "_tags"
    raw_train_samples = list(read_jsonl(file_path=training_data_file))

    train_dataloader, tag2id, id2tag, scaling_dict = construct_dataloader_and_mappers(raw_samples=raw_train_samples,
                                                                                      tokenizer=tokenizer,
                                                                                      max_len=max_len,
                                                                                      batch_size=batch_size,
                                                                                      n_workers=n_workers,
                                                                                      labels_key=labels_key,
                                                                                      dataset_name=dataset_name,
                                                                                      print_tokens=print_tokens,
                                                                                      tag2id=tag2id,
                                                                                      remove_longer_seqs=rmls)

    return train_dataloader, tag2id, id2tag, scaling_dict


def get_merged_dataloader(files_to_merge: List[str], tokenizer: BertTokenizerFast,
                          max_len: int, batch_size: int, n_workers: int, tag_type: str, print_tokens: bool,
                          dataset_name: str, tag2id: Dict[str, int] = None, rmls: bool = False
                          ) -> Tuple[DataLoader, Dict[str, int], Dict[int, str], Dict[str, float]]:
    """
    Similar to get_training_dataloader but including all the instances from all_data_files list, which is a list of
    paths to jsonl files that will be merged
    """
    labels_key = tag_type + "_tags"
    all_raw_samples = [x for tmp_file in files_to_merge for x in read_jsonl(tmp_file)]
    merged_dataloader, tag2id, id2tag, scaling_dict = construct_dataloader_and_mappers(raw_samples=all_raw_samples,
                                                                                       tokenizer=tokenizer,
                                                                                       max_len=max_len,
                                                                                       batch_size=batch_size,
                                                                                       n_workers=n_workers,
                                                                                       labels_key=labels_key,
                                                                                       dataset_name=dataset_name,
                                                                                       print_tokens=print_tokens,
                                                                                       tag2id=tag2id,
                                                                                       remove_longer_seqs=rmls)
    return merged_dataloader, tag2id, id2tag, scaling_dict


def construct_dataloader_and_mappers(raw_samples: List[Dict], tokenizer: BertTokenizerFast, max_len: int,
                                     batch_size: int, n_workers: int, labels_key: str, dataset_name: str,
                                     print_tokens: bool, tag2id: Dict[str, int] = None,
                                     remove_longer_seqs: bool = False) -> Tuple[DataLoader, Dict[str, int],
                                                                                Dict[int, str], Dict[str, float]]:
    samples_ready = [{"tokens": [t["text"] for t in sentence["tokens"]], "text": sentence["text"],
                      "labels": sentence[labels_key], "relations": sentence['relations']} for sentence in raw_samples]
    # 2. Compute proportion of labels
    scaling_dict = compute_tag_scaling(inp_training_samples=raw_samples, labels_key=labels_key)
    # 3. Compute tag2id id2tag mappers
    unique_tags = set(tag for doc in raw_samples for tag in doc[labels_key])
    if tag2id is None:
        tag2id = {tag: tag_id for tag_id, tag in enumerate(unique_tags)}
        tag2id["PAD"] = -100  # add padding label to -100 so it can be converted
    id2tag = {tag_id: tag for tag, tag_id in tag2id.items()}
    # 4. Make dataloader
    dataloader_ready = make_dataloader(inp_samples=samples_ready, batch_size=batch_size, inp_tokenizer=tokenizer,
                                       max_len=max_len, shuffle=True, n_workers=n_workers,
                                       tag2id=tag2id, dataset_name=dataset_name, print_tokens=print_tokens,
                                       remove_longer_seqs=remove_longer_seqs)
    return dataloader_ready, tag2id, id2tag, scaling_dict


def make_dataloader(inp_samples: List[Dict], batch_size: int, inp_tokenizer: BertTokenizerFast, max_len: int,
                    shuffle: bool, n_workers: int, tag2id: Dict[str, int],
                    dataset_name: str, print_tokens: bool, remove_longer_seqs: bool) -> DataLoader:
    torch_dataset = process_mention_data(inp_samples=inp_samples, inp_tokenizer=inp_tokenizer, max_len=max_len,
                                         tag2id=tag2id, dataset_name=dataset_name, print_tokens=print_tokens,
                                         remove_longer_seqs=remove_longer_seqs)
    if shuffle:
        loader = DataLoader(torch_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True,
                            collate_fn=sampling.collate_fn_padding, pin_memory=True)

    else:
        loader = DataLoader(torch_dataset, batch_size=batch_size, num_workers=n_workers,
                            collate_fn=sampling.collate_fn_padding, pin_memory=True)

    return loader


def process_mention_data(inp_samples: List[Dict], inp_tokenizer: BertTokenizerFast, max_len: int,
                         tag2id: Dict[str, int], dataset_name: str, print_tokens: bool,
                         remove_longer_seqs: bool) -> Dataset:
    """
    Generates a pytorch dataset containing encoded tokens and labels
    """

    print(f"\n==== {dataset_name.upper()} set ====")
    texts = [sample["text"] for sample in inp_samples]
    ner_labels = [sample["labels"] for sample in inp_samples]
    relation_labels = [sample["relations"] for sample in inp_samples]
    original_tokens = [sample["tokens"] for sample in inp_samples]
    if print_tokens:
        print_token_stats(all_tokens=original_tokens, dataset_name=dataset_name,
                          max_len=max_len, plot_histogram=True)
    if remove_longer_seqs:
        texts, ner_labels, original_tokens, relation_labels = filter_long_seqs(inp_texts=texts, inp_labels=ner_labels,
                                                                               max_len=max_len,
                                                                               original_tokens=original_tokens,
                                                                               relations=relation_labels)
    else:
        ner_labels = reshape_ner_labels(inp_ner_labels=ner_labels, max_len=max_len)

    doc_encodings = inp_tokenizer(texts, padding='max_length', truncation=True, max_length=max_len,
                                  return_overflowing_tokens=True)

    all_tokens = extract_all_tokens_withoutpad(doc_encodings.encodings)

    check_tokens_order(all_tokens, original_tokens)

    print(f"Number of sentences : {len(all_tokens)}")

    check_labels_tokens_alignment(tokens=all_tokens, subword_labels=ner_labels)

    if print_tokens:
        print_few_mentions(all_tokens=all_tokens, labels=ner_labels, n=5)

    encoded_ner_labels = pad_and_encode_ner_labels(all_ner_labels=ner_labels,
                                                   max_len=max_len,
                                                   tag2id=tag2id)

    rex_dataset = RexCollection(doc_encodings=doc_encodings, encoded_ner_labels=encoded_ner_labels, tag2id=tag2id,
                                relation_labels=relation_labels)

    torch_dataset = PKRexDataset(encodings=doc_encodings, labels=encoded_ner_labels,
                                 rex_instances=rex_dataset.rex_instances)

    return torch_dataset


class RexCollection:
    def __init__(self, doc_encodings: BatchEncoding, encoded_ner_labels: List[List[int]], tag2id: Dict[str, int],
                 relation_labels: List[List[Dict]]):
        if len(doc_encodings['input_ids']) == len(encoded_ner_labels) == len(relation_labels):
            self.n_instances = len(relation_labels)
            self.docs = doc_encodings
            self.seq_len = len(self.docs.tokens())
            self.ner_labels = encoded_ner_labels
            self.tag2id = tag2id
            self.relation_labels = relation_labels
            self.id2tag = {tag_id: tag for tag, tag_id in self.tag2id.items()}
            self.rex_instances = None
            self.possible_rels_per_sample = None
            self.rex_instances, self.possible_rels_per_sample = self.compute_rex_samples()

        else:
            raise ValueError("The number of input samples, ner labels and relation labels does not correspond")

    def compute_rex_samples(self):
        n_sample_rels_per_instance = []
        rex_instances = []
        for i, (enc_ner_labs, annotated_rel_labs) in enumerate(zip(self.ner_labels, self.relation_labels)):
            # 1. Get IOB labels
            str_ner_labs = [self.id2tag[x] for x in enc_ner_labs]
            # 2. Get entities as a list of dictionaries and assign index to each entity
            # (in order of appearance in the sentence)
            entity_tokens = bio_to_entity_tokens(inp_bio_seq=str_ner_labs)
            indexed_spans = assign_index_to_spans(span_list=entity_tokens)
            # 3. Generate candidate relations given by all the allowed entity pairs
            candidate_rels = generate_all_possible_rels(inp_entities=indexed_spans)
            candidate_rels = filter_not_allowed_rels(inp_possible_rels=candidate_rels)
            candidate_rels_ent_ids = possible_rels_to_entity_ids_format(inp_entities=indexed_spans,
                                                                        inp_rels=candidate_rels)
            # 4. For every relation and entity generate entity masks, context masks, and store the length of the context
            ent_masks, ctx_masks, ctx_lengths = get_entities_and_ctx_masks(inp_entities=indexed_spans,
                                                                           inp_rels=candidate_rels,
                                                                           max_len=self.seq_len)

            assert len(candidate_rels_ent_ids) == len(ctx_masks) == len(ctx_lengths)
            assert len(ent_masks) == len(indexed_spans)
            n_sample_rels_per_instance.append(len(candidate_rels))  # store # of allowed rels in sentence
            #    rels_triplet = [get_ent_and_ctx_token_offsets(r) for r in possible_and_allowed_rels]
            #    assert len(rels_triplet) == len(possible_and_allowed_rels)
            #    if not possible_and_allowed_rels:
            #        assert possible_and_allowed_rels == raw_rel_labs
            #    assert len(possible_and_allowed_rels) >= len(raw_rel_labs)

            # 5. Map the labels of the candidate relationships to the annotated relations
            tmp_cand_rex_labels = []
            if candidate_rels:
                check_all_tuples(inp_rels=candidate_rels_ent_ids, iob_labels=str_ner_labs, ent_masks=ent_masks)

                tmp_cand_rex_labels = associate_rel_tuples_with_labels(inp_rel_labels=annotated_rel_labs,
                                                                       inp_rel_tuples=candidate_rels_ent_ids)
            # 6. Construct instance class
            tmp_rex_instance = RexInstance(candidate_rel_tuples=candidate_rels_ent_ids,
                                           rex_labels=tmp_cand_rex_labels,
                                           ner_labels=enc_ner_labs,
                                           ner_masks=ent_masks,
                                           context_masks=ctx_masks,
                                           id2tag=self.id2tag,
                                           tokens_str=self.docs.tokens(i),
                                           )
            #  if tmp_rex_instance.rel_tuples:
            #      print("== New sentence ==")
            #      tmp_rex_instance.print_relations()
            rex_instances.append(tmp_rex_instance)

            #     for trp in rels_triplet:
            #         check_correct_triplet(inp_bio_labels=str_ner_labs, inp_triplet=trp)
            #     tmp_rex_labels = associate_triplets_with_rels(inp_rel_labels=raw_rel_labs,
            #                                                   inp_triplets=rels_triplet)
            #
            #          tmp_rex_instance = RexInstance(possible_rel_triplets=rels_triplet, rex_labels=tmp_rex_labels,
            #                                         ner_labels=enc_ner_labs, id2tag=self.id2tag)

        assert len(rex_instances) == self.n_instances == len(n_sample_rels_per_instance)
        return rex_instances, n_sample_rels_per_instance


class RexInstance:
    def __init__(self, candidate_rel_tuples: List[List[int]], rex_labels: List[str], ner_labels: List[int],
                 ner_masks: List[List[bool]], context_masks: List[List[bool]],
                 id2tag: Dict[int, str] = None, tokens_str: List[str] = None):
        assert len(candidate_rel_tuples) == len(rex_labels)
        self.rel_tuples = candidate_rel_tuples
        self.cand_rel_labels = rex_labels
        self.encoded_rel_labels = [REX2ID[x] for x in self.cand_rel_labels]
        self.ner_labels = ner_labels
        self.id2tag = id2tag
        self.tokens_str = tokens_str
        self.ner_masks = ner_masks
        self.context_masks = context_masks
        self.context_lengths = [sum(c) for c in context_masks]

    def print_relations(self):
        if self.rel_tuples:
            for r, ctx, label in zip(self.rel_tuples, self.context_masks, self.cand_rel_labels):
                left_tokens = "".join(
                    [tok.replace("##", "") if tok.startswith("##") or i == 0 else " " + tok for i, tok in
                     enumerate(self.tokens_str[0: min([i for i, v in enumerate(self.ner_masks[r[0]]) if v])])])

                ent1str = "".join([tok.replace("##", "") if tok.startswith("##") or i == 0 else " " + tok for i, tok in
                                   enumerate(list(compress(self.tokens_str, self.ner_masks[r[0]])))])
                ctx_str = "".join([tok.replace("##", "") if tok.startswith("##") or i == 0 else " " + tok for i, tok in
                                   enumerate(list(compress(self.tokens_str, ctx)))])
                ent2str = "".join([tok.replace("##", "") if tok.startswith("##") or i == 0 else " " + tok for i, tok in
                                   enumerate(list(compress(self.tokens_str, self.ner_masks[r[1]])))])

                right_tokens = "".join(
                    [tok.replace("##", "") if tok.startswith("##") or i == 0 else " " + tok for i, tok in
                     enumerate(self.tokens_str[max([i for i, v in enumerate(self.ner_masks[r[1]]) if v]) + 1:])])
                ent1_str = colored(ent1str, 'green', attrs=['reverse', 'bold'])
                ctx_str = colored(ctx_str, 'grey', attrs=['reverse', 'bold'])
                ent2_str = colored(ent2str, 'red', attrs=['reverse', 'bold'])
                print(label, ":")
                print(left_tokens + " " + ent1_str + ctx_str + ent2_str + " " + right_tokens)
        else:
            print("There are no relations in this instance")


class PKRexDataset(Dataset):
    def __init__(self, encodings, labels, rex_instances: List[RexInstance]):
        self.encodings = encodings
        self.labels = labels
        self.rex_instances = rex_instances

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        rex_instance = self.rex_instances[idx]
        item['entity_masks'] = torch.tensor(rex_instance.ner_masks)
        item['rel_tuples'] = torch.tensor(rex_instance.rel_tuples)
        item['ctx_mask'] = torch.tensor(rex_instance.context_masks)
        item['rel_labels'] = torch.tensor(rex_instance.encoded_rel_labels)
        item['ctx_len'] = torch.tensor(rex_instance.context_lengths)
        return item

    def __len__(self):
        return len(self.labels)


def check_all_tuples(inp_rels, iob_labels, ent_masks):
    for tmp_rel_idxs in inp_rels:
        check_correct_rel_tuple(inp_bio_labels=iob_labels, inp_tuple=tmp_rel_idxs,
                                inp_entity_masks=ent_masks)


def check_correct_triplet(inp_bio_labels: List[str], inp_triplet: Tuple[Tuple[int, int], Tuple[int, int],
                                                                        Tuple[int, int]]):
    ent1 = inp_bio_labels[inp_triplet[0][0]:inp_triplet[0][1]]
    ent2 = inp_bio_labels[inp_triplet[1][0]:inp_triplet[1][1]]
    check_correct_ents_iobs(ent1=ent1, ent2=ent2)


def check_correct_rel_tuple(inp_bio_labels: List[str], inp_tuple: List[int], inp_entity_masks: List[List[bool]]):
    assert len(inp_bio_labels) == len(inp_entity_masks[0])
    assert len(inp_tuple) == 2
    assert inp_tuple[0] < inp_tuple[1]
    ent1 = list(compress(inp_bio_labels, inp_entity_masks[inp_tuple[0]]))
    ent2 = list(compress(inp_bio_labels, inp_entity_masks[inp_tuple[1]]))
    check_correct_ents_iobs(ent1=ent1, ent2=ent2)


def check_correct_ents_iobs(ent1, ent2):
    assert "B" in ent1[0] and "B" in ent2[0]
    assert len(set([x.split("-")[1] for x in ent1])) == 1
    assert len(set([x.split("-")[1] for x in ent2])) == 1
    if len(ent1) > 1:
        for x in ent1[1:]:
            assert x.split("-")[0] == "I"
    if len(ent2) > 1:
        for x in ent2[1:]:
            assert x.split("-")[0] == "I"


def pad_and_encode_ner_labels(all_ner_labels: List[List[str]], max_len: int,
                              tag2id: Dict[str, int]):
    """PADS and transforms labels"""
    all_padded_labels = []
    for seq_lab in all_ner_labels:
        padding = ["PAD"] * (max_len - len(seq_lab))
        padded_labels = seq_lab + padding
        assert len(padded_labels) == max_len
        all_padded_labels.append(padded_labels)

    encoded_labels = [[tag2id[label] for label in pl] for pl in all_padded_labels]
    return encoded_labels


def print_few_mentions(all_tokens, labels, n):
    i = 0
    for tokens, l in zip(all_tokens, labels):
        if i > n:
            break
        entity_tokens = bio_to_entity_tokens(l)
        if entity_tokens:
            i += 1
            for span in entity_tokens:
                mention = make_seq(tokens[span["token_start"]:span["token_end"] + 1])
                print(mention)


def print_token_stats(all_tokens: List[List[str]], dataset_name: str, max_len: int,
                      plot_histogram: bool = False):
    n_tokens = []
    sentences_with_more_than_max = 0
    overlflowing_sentences = []
    for tokens in all_tokens:
        nt = len(tokens)
        n_tokens.append(nt)
        if nt > max_len:
            sentences_with_more_than_max += 1
            overlflowing_sentences.append(make_seq(bert_tokens=tokens))
    if plot_histogram:
        plt.hist(n_tokens, bins=50)
        plt.title(f"Number of bert tokens in the {dataset_name.upper()} set")
        plt.xlabel("# tokens")
        plt.ylabel("# sentences")
        plt.show()
        plt.close()
    print(f"There were {sentences_with_more_than_max} sentences with more than {max_len} tokens"
          f" ({round(sentences_with_more_than_max * 100 / len(all_tokens), 2)}%): ")
    for s in overlflowing_sentences:
        print(s)


def get_token_stats(docs_encodings: BatchEncoding, dataset_name: str, print_truncated: bool,
                    plot_histogram: bool = False):
    """
    Print some dataset statistics
    @param plot_histogram: whether to plot a token histogram
    @param docs_encodings: documents after passing through bert fast tokenizer
    @param dataset_name: name of the dataset e.g. training, valid, test
    @param print_truncated: whether to print some sentences from the ones that have been truncated
    (purely for visual inspection)
    @return:
    """
    overflowing_sentences = [x for x in docs_encodings.encodings if x.overflowing]
    print(f"Number of sentences overflowing in {dataset_name} set: {len(overflowing_sentences)} from "
          f"{len(docs_encodings.encodings)} "
          f"({round(len(overflowing_sentences) * 100 / len(docs_encodings.encodings), 2)}%)")

    number_of_bert_tokens = [len([token for token in doc.tokens if token != '[PAD]']) for doc in
                             docs_encodings.encodings if
                             '[PAD]' in doc.tokens]
    if plot_histogram:
        plt.hist(number_of_bert_tokens)
        plt.title(f"Number of bert tokens in the {dataset_name} set")
        plt.xlabel("# tokens")
        plt.ylabel("# sentences")
        plt.show()
        plt.close()

    if print_truncated:
        max_print = 5
        if len(overflowing_sentences) < 5:
            max_print = len(overflowing_sentences)
        example_sentences = overflowing_sentences[0:max_print]
        for example in example_sentences:
            print(make_seq(example.tokens))


def make_seq(bert_tokens: List[str]):
    """Very simple function to return a sequence (not the original one) given input bert tokens"""
    seq = bert_tokens[0]
    for tok in bert_tokens[1:]:
        if "##" in tok:
            seq += tok.replace("##", "")
        else:
            seq += f" {tok}"
    return seq


def read_dataset(data_dir_inp: str, dataset_name: str) -> List[Dict]:
    file_name = f"{dataset_name}.jsonl"
    file_path = os.path.join(data_dir_inp, file_name)
    return read_jsonl(file_path=file_path)


def check_labels_tokens_alignment(tokens: List[List[str]], subword_labels: List[List[str]]):
    for toks, seq_label in zip(tokens, subword_labels):
        if len(toks) != len(seq_label):
            raise ValueError(f"The number of tokens and the number of labels do not correspond.")


def encode_and_align_labels(tokenized_inputs: BatchEncoding, original_labels: List[List[str]],
                            tag2id: Dict[str, int]) -> List[List[int]]:
    """
    Returns original labels encoded in a numerical form and aligned with bert word-pieces. Adapted from
    https://huggingface.co/transformers/custom_datasets.html#token-classification-with-w-nut-emerging-entities
    If the original label of the token "@HuggingFace" was 3 and this token gets split by BERT into sub-words
    e.g.  ['@', 'hugging', '##face']
    This function would return
    ['@', 'hugging', '##face']
    [3, -100, -100]
    Assigning only the label to the first token and ignoring the subsequent sub-word pieces at training time.

    @param tokenized_inputs: Inputs tokenized by huggingface BertTokenizerFast
    @param original_labels: labels in the string form [["B-PK", "I-PK", "O-"], ["O-","B-PK", "I-PK"]]
    @param tag2id: Dictionary to convert string labels to numerical integers e.g. {"B-PK": 0, "I-PK": 1}
    @return: the original labels encoded in a numerical form and aligned with bert word-pieces
    """
    labels = [[tag2id[tag] for tag in doc] for doc in original_labels]  # transform string tags to numerical labels
    assert len(labels) == len(tokenized_inputs.offset_mapping)

    encoded_labels = []
    for i, (doc_labels, doc_offset) in tqdm(enumerate(zip(labels, tokenized_inputs.offset_mapping))):
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)

        if len(doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)]) == len(doc_labels):
            doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        else:
            # most likely a case of truncation in which the labels were not truncated
            assert doc_offset[-2] != (0, 0)  # assert that this is a truncation case
            n_exp_labels = len(doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)])
            doc_labels = doc_labels[0:n_exp_labels]  #
            assert len(doc_labels) == len(doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)])

        encoded_labels.append(doc_enc_labels.tolist())

    assert len(encoded_labels) == len(tokenized_inputs.encodings)

    return encoded_labels


def extract_all_tokens_withoutpad(docencodings: List[Encoding]):
    return [[t for t in enc.tokens if t != '[PAD]'] for enc in docencodings]


def filter_long_seqs(inp_texts: List[str], inp_labels: List[List[str]], max_len: int,
                     original_tokens: List[List[str]], relations: List[List[Dict]]):
    assert len(inp_texts) == len(inp_labels) == len(original_tokens) == len(relations)
    out_texts, out_labels, out_org_toks, out_relations = [], [], [], []
    i = 0
    for tmp_text, tmp_labels, tmp_or_toks, tmp_rels in zip(inp_texts, inp_labels, original_tokens, relations):
        if len(tmp_labels) > max_len:
            i += 1
        else:
            out_texts.append(tmp_text)
            out_labels.append(tmp_labels)
            out_org_toks.append(tmp_or_toks)
            out_relations.append(tmp_rels)
    assert len(out_texts) == len(out_labels) == len(out_org_toks) == len(out_relations)
    print(f"A total of {i} sentences were removed")
    return out_texts, out_labels, out_org_toks, out_relations


def reshape_ner_labels(inp_ner_labels: List[List[str]], max_len: int):
    """
    Given a sequence of IOB tags, if the length of the sequence > max_len, it splits them into two and adds labels
    for the new [CLS] AND [SEP] tokens
    """
    reshaped_labels = []
    for seq_labels in inp_ner_labels:
        if len(seq_labels) > max_len:
            base = 0
            ending = max_len - 1
            remaining_seq = seq_labels
            while len(remaining_seq) > max_len:
                new_labels = remaining_seq[base:ending] + ['O']  # append a final O since there will be an extra sep
                # token
                assert len(new_labels) == max_len
                reshaped_labels.append(new_labels)
                remaining_seq = ['O'] + remaining_seq[ending:]  # add the label for the new CLS token
            reshaped_labels.append(remaining_seq)
        else:
            reshaped_labels.append(seq_labels)
    return reshaped_labels


def compute_tag_scaling(inp_training_samples: List[Dict], labels_key: str) -> Dict[str, float]:
    all_training_tags = [tag for doc in inp_training_samples for tag in doc[labels_key]]
    tag_freqs = Counter(all_training_tags).most_common()
    most_freq_tag_count = tag_freqs[0][1]
    return {tmp_key: most_freq_tag_count / tmp_freq for tmp_key, tmp_freq in tag_freqs}


def check_tokens_order(new_toks, orig_toks):
    for sub1, sub2 in zip(new_toks, orig_toks):
        for tok1, tok2 in zip(sub1, sub2):
            assert tok1 == tok2
