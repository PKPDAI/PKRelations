"""Gets a jsonl file and removes entities that are not part of relations before we can use it for training"""
import hashlib
from pathlib import Path
from typing import Dict, List
import prodigy
import typer
from pkrex.utils import read_jsonl, write_jsonl
from tqdm import tqdm

from pkrex.annotation_preproc import fix_incorrect_dvals, swap_clear_incorrect, check_rel_tokens_entities, \
    d_val_pointing, remove_irrelevant_entities, visualize_relations_brat, simplify_annotation, keep_relevant_fields, \
    remove_ent_by_type


def remove_duplicated_annotations(inp_annotations: List[Dict], unique_key: str):
    print(f"Length before removing duplicates: {len(inp_annotations)}")
    uq_ids = []
    out_annot = []
    for ann in inp_annotations:
        if ann[unique_key] not in uq_ids:
            out_annot.append(ann)
            uq_ids.append(ann[unique_key])
    print(f"Length after removing duplicates: {len(out_annot)}")
    return out_annot


REPL_DICT = {"\u223c": "~",
             "\u2061": " "}


def replace_conflicting_tokens(inp_annotation):
    initial_len = len(inp_annotation["text"])
    for key in REPL_DICT.keys():
        if key in inp_annotation["text"]:
            inp_annotation["text"] = inp_annotation["text"].replace(key, REPL_DICT[key])
            if "tokens" in inp_annotation.keys():
                new_tokens = []
                for token in inp_annotation["tokens"]:
                    if key in token["text"]:
                        token["text"] = token["text"].replace(key, REPL_DICT[key])
                    new_tokens.append(token)
                inp_annotation["tokens"] = new_tokens
    assert len(inp_annotation["text"]) == initial_len
    return inp_annotation


def main(
        input_file_path: Path = typer.Option(default="train-extra-20.jsonl",
                                             help="File that we want to preprocess"),

        output_file_path: Path = typer.Option(
            default="data/annotations/P1/ready/train-extra-20-all-reviewed.jsonl",
            help="Path to the output file"),

        remove_ents=typer.Option(default=["TYPE_MEAS"],
                                 help="Entities and relations to remove"),

        keep_pk_ent: bool = typer.Option(default=False,
                                         help="Whether to keep pk entities that are not part of a relationship"),

        display_relations: bool = typer.Option(default=False,
                                               help="Whether to review the annotated data"),

        remove_sentences_with_latex_tags: bool = typer.Option(default=True,
                                                              help="Remove the sentences with usepackage{ inside")

):
    output_dataset = [preprocess_review_sentence(prodigy_annotation=sentence, idx=idx, keep_pk=keep_pk_ent,
                                                 remove_ents=remove_ents, keep_tokens=True)
                      for idx, sentence in tqdm(enumerate(read_jsonl(input_file_path)))]
    if remove_sentences_with_latex_tags:
        output_dataset = [x for x in output_dataset if "usepackage{" not in x["text"]]

    output_dataset = remove_duplicated_annotations(inp_annotations=output_dataset, unique_key="_input_hash")
    a = []
    for x in output_dataset:
        if x["metadata"] not in a:
            a.append(x["metadata"])
        else:
            print("Potential duplicate: ")
            print(x)
    if display_relations:
        visualize_relations_brat(inp_annotations=output_dataset, file_path="brat/rel_brat_template.html")

    write_jsonl(output_file_path, output_dataset)


def preprocess_review_sentence(prodigy_annotation: Dict, idx: int, keep_tokens: bool = True,
                               keep_pk: bool = True, remove_ents=None) -> Dict:
    print(idx)
    # 1. Rehash sentence hash based on text input
    prodigy_annotation = replace_conflicting_tokens(prodigy_annotation)
    prodigy_annotation['sentence_hash'] = hashlib.sha1(prodigy_annotation['text'].encode()).hexdigest()  # rehash

    # 2. Remove non-useful information from inside "spans" and "relations" fields
    prodigy_annotation = simplify_annotation(prodigy_annotation, keep_tokens=keep_tokens)  # removes non-useful info

    # 3. Remove entities that are not part of central value relationships
    prodigy_annotation = remove_irrelevant_entities(inp_annotation=prodigy_annotation, preserve_pk=keep_pk,
                                                    keep_tokens=keep_tokens)

    # 4. If we want to remove some specific entity types, remove them
    if remove_ents and (prodigy_annotation["relations"] or prodigy_annotation["spans"]):
        prodigy_annotation = remove_ent_by_type(prodigy_annotation, remove_ents)

    # 5. Swap D_VALs that are pointing towards another
    prodigy_annotation = fix_incorrect_dvals(inp_annotation=prodigy_annotation)

    # 6. If there are entity mentions that are clearly incorrect, it replaces the label by the correct label
    prodigy_annotation = swap_clear_incorrect(annotation=prodigy_annotation)

    # 3. Check incorrect labelling
    d_val_pointing(annotations=[prodigy_annotation], idx=idx, keep_tokens=keep_tokens)

    # 4. Checks that there are no relations with tokens that are not part of entities
    check_rel_tokens_entities(annotations=[prodigy_annotation], keep_tokens=keep_tokens)

    prodigy_annotation = keep_relevant_fields(inp_annotation=prodigy_annotation)

    prodigy_annotation = prodigy.util.set_hashes(prodigy_annotation)

    return prodigy_annotation
    # print(f"After:\n{view_all_entities_terminal(inp_annotation['text'], inp_annotation['spans'])}")


if __name__ == "__main__":
    typer.run(main)
