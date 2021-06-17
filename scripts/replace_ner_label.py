from pathlib import Path
from typing import List, Dict
import typer
from pkrex.annotation_preproc import fix_incorrect_dvals, swap_clear_incorrect
from pkrex.utils import read_jsonl, write_jsonl


def replace_ner(annotations: List[Dict]) -> List[Dict]:
    out_annotations = []
    for annotation in annotations:
        annotation = fix_incorrect_dvals(annotation)
        annotation = swap_clear_incorrect(annotation=annotation)
        out_annotations.append(annotation)
    return out_annotations


def main(
        input_file: Path = typer.Option(default="data/annotations/P1/reviewed/test-0-200-reviewed.jsonl",
                                        help="file to modify"),
        output_file: Path = typer.Option(default="data/annotations/P1/reviewed/test-0-200-reviewed.jsonl",
                                         help="Directory of the output file ready for Part 2 annotations")
):
    annotations = list(read_jsonl(input_file))

    annotations_clean = replace_ner(annotations)
    write_jsonl(file_path=output_file, lines=annotations_clean)


if __name__ == "__main__":
    typer.run(main)
