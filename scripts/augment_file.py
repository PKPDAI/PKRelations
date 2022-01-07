import json
import typer
from pathlib import Path
from pkrex.utils import read_jsonl, write_jsonl
from pkrex.augmentation_syns import AUGMENT_SYNS
from pkrex.annotation_preproc import view_all_entities_terminal
import pkrex.augmentation as pkaug
import copy


def main(
        input_file: Path = typer.Option(default="data/annotations/P1/ready/train-all-reviewed.jsonl"),
        out_file: Path = typer.Option(default="data/annotations/P1/ready/train-all-reviewed-augmented.jsonl")
):
    annotations = list(read_jsonl(input_file))
    a = 1
    rels = 0
    original_annotations = []
    augmented_annotations = []
    for ann in annotations:
        original_annotations.append(ann)
        ann_to_change = copy.deepcopy(ann)
        if ann_to_change['relations']:
            original_sentece = view_all_entities_terminal(inp_text=ann_to_change['text'],
                                                          character_annotations=ann_to_change['spans'])
            augmented_sentence = pkaug.augment_sentence(inp_anotation=ann_to_change, replacable_dict=AUGMENT_SYNS)
            if augmented_sentence['text'] != ann['text']:
                augmented_annotations.append(augmented_sentence)
                augmented_sentence_print = view_all_entities_terminal(inp_text=augmented_sentence['text'],
                                                                      character_annotations=augmented_sentence['spans'])

                print(original_sentece)
                print(augmented_sentence_print)
                print("\n")

    out_annot = original_annotations + augmented_annotations
    print(rels * 100 / len(annotations))
    write_jsonl(file_path=out_file, lines=out_annot)


if __name__ == "__main__":
    typer.run(main)
