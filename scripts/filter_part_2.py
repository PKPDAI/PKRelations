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
        input_file: Path = typer.Option(default="data/annotations/P1/reviewed/dev-0-200-reviewed.jsonl",
                                        help="Path to the input file after being labelled in Part 1"),
        output_dir: Path = typer.Option(default="data/annotations/P2/to_annotate/",
                                        help="Directory of the output file ready for Part 2 annotations")
):
    """
    Apply your trained NER model to the test/development set
    """

    p1_annotations = list(read_jsonl(input_file))
    print("{} P1 annotated sentences".format(len(p1_annotations)))
    p2_annotations = filter_part_2(inp_annotations=p1_annotations)
    print(f"{round(len(p2_annotations) * 100 / len(p1_annotations), 2)} % of samples with C_VAL relation")
    out_name = str(input_file).split("/")[-1].split(".")[0] + "to-annotate-p2.jsonl"
    if "reviewed" in out_name:
        out_name = out_name.replace("reviewed", "")
    out_path = os.path.join(output_dir, out_name)
    print(f"Writing {len(p2_annotations)} annotations in {out_path}")
    write_jsonl(file_path=out_path, lines=p2_annotations)

    out_db = "-".join(str(input_file).split("/")[-1].split("-")[0:3]) + "-p2-trials-1"
    prodigy_command = f"PRODIGY_ALLOWED_SESSIONS=ferran PRODIGY_PORT=8001 prodigy custom.rel.manual " \
                      f"{out_db} data/models/tokenizers/super-tokenizer " \
                      f"{out_path} --label RELATED," \
                      "DOSE,C_VAL,D_VAL --wrap --span-label CONTEXT,ROUTE,CHEMICAL,DISEASE,SPECIES,UNITS,PK," \
                      "TYPE_MEAS,COMPARE,RANGE,VALUE --wrap -F recipes/rel_custom_part_2.py "
    print(f"To try labelling in prodigy run:\n{prodigy_command}")


if __name__ == "__main__":
    typer.run(main)
