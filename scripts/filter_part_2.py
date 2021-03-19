import os
from pathlib import Path
from typing import List, Dict

import typer
from pkrex.utils import read_jsonl, write_jsonl


def filter_part_2(inp_annotations: List[Dict]) -> List[Dict]:
    out_annotations = []
    for annotation in inp_annotations:
        if annotation['relations']:
            uq_relation_types = set([relation['label'] for relation in annotation['relations']])
            if "C_VAL" in uq_relation_types:
                # clean_relations = [{k: rel[k] for k in rel.keys() if k != 'color'} for rel in annotation['relations']]
                clean_relations = []
                for rel in annotation['relations']:
                    if rel['label'] in ['RELATED', 'D_VAL']:
                        out_color = "#dddddd"
                    else:
                        out_color = "#f51307"
                    rel["color"] = out_color
                    clean_relations.append(rel)

                annotation['relations'] = clean_relations
                out_annotations.append(annotation)
    return out_annotations


def main(
        input_file: Path = typer.Option(default="data/annotations/dev/1/rex-dev0-200.jsonl",
                                        help="Path to the input file after being labelled in Part 1"),
        output_dir: Path = typer.Option(default="data/part2/dev/",
                                        help="Directory of the output file ready for Part 2 annotations")
):
    """
    Apply your trained NER model to the test/development set
    """
    p1_annotations = list(read_jsonl(input_file))
    print("{} P1 annotated sentences".format(len(p1_annotations)))
    p2_annotations = filter_part_2(inp_annotations=p1_annotations)
    print(f"{round(len(p2_annotations) * 100 / len(p1_annotations), 2)} % of samples with C_VAL relation")
    out_name = str(input_file).split("/")[-1].split(".")[0] + "-p2.jsonl"
    out_path = os.path.join(output_dir, out_name)
    print(f"Writing {len(p2_annotations)} annotations in {out_path}")
    write_jsonl(file_path=out_path, lines=p2_annotations)


if __name__ == "__main__":
    typer.run(main)
