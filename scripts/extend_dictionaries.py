import json
import os
from pathlib import Path
from typing import List, Dict

import typer
from pkrex.annotation_preproc import get_all_relevant_related_spans
from pkrex.utils import read_jsonl


def get_all_spans(annotations: List[Dict], label: str) -> List[str]:
    return list(set(
        [sentence['text'][entity['start']:entity['end']] for sentence in annotations for entity in sentence['spans'] if
         entity['label'] == label]))


def main(
        input_dir: Path = typer.Option(default="data/annotations/P1/reviewed/",
                                       help="File to perform checks on"),

        base_dicts: Path = typer.Option(default="data/dictionaries/terms.json",
                                        help="File with dictionaries")

):
    entity_dictionaries = json.load(open(base_dicts, 'r', encoding='utf8'))

    all_units = entity_dictionaries['UNITS']
    all_type_meas = entity_dictionaries['TYPE_MEAS']
    all_compare = entity_dictionaries['COMPARE']
    compare_remove = ["between"]
    type_meas_remove = ["range", "approximately"]
    units_remove = ['y', '90% confidence intervals (cis)', 'generic formulations', 'fold', 'ratio', 'a',
                    '90% confidence intervals', 'g', 's']
    for inp_file in os.listdir(input_dir):
        print(f"====== Analysing {inp_file} ==============")
        file_path = os.path.join(input_dir, inp_file)
        annotations = list(read_jsonl(file_path))

        tmp_units = [token.lower() for token in get_all_spans(annotations=annotations, label='UNITS')]
        all_units = [x for x in sorted(list(set(all_units + tmp_units)), key=len) if x not in units_remove]

        tmp_type_meas = [token.lower() for token in get_all_spans(annotations=annotations, label='TYPE_MEAS')]
        all_type_meas = [x for x in sorted(list(set(all_type_meas + tmp_type_meas)), key=len) if
                         x not in type_meas_remove]

        tmp_compare = [token.lower() for token in get_all_relevant_related_spans(annotations=annotations, label='COMPARE')]
        all_compare = [x for x in sorted(list(set(all_compare + tmp_compare)), key=len) if x not in compare_remove]

    print(all_units)
    print(all_type_meas)
    print(all_compare)

    entity_dictionaries["TYPE_MEAS"] = all_type_meas
    entity_dictionaries["COMPARE"] = all_compare
    entity_dictionaries["UNITS"] = all_units
    with open(base_dicts, "w", encoding="utf8") as outfile:
        json.dump(entity_dictionaries, outfile, indent=4)


if __name__ == "__main__":
    typer.run(main)
