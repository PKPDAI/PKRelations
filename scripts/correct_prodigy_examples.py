import os
from typing import List
import typer
from prodigy.components.db import connect
import prodigy
from pkrex.utils import read_jsonl, write_jsonl
from recipes.rel_custom import custom_rel_manual


def main(

        base_file: str = typer.Option(default="data/annotations/P1/ready/train-all-reviewed.jsonl",
                                      help="Local output directory to save the model and files"),
        new_dataset_name: str = typer.Option(default="train-all-reviewed-444",
                                             help="Whether to save the model to azure"),
        hash_list: List[int] = typer.Option(
            default=[-1294653114, -1372168709, -363191018, -2081848871, -1597632365],
            help="Task hash of examples that we would like to re-annotate"),

):
    db = connect()
    if db.get_dataset(new_dataset_name):
        db.drop_dataset(new_dataset_name)
    db.add_dataset(new_dataset_name)

    inp_dataset = list(read_jsonl(base_file))

    annotated_examples_to_keep = [x for x in inp_dataset if
                                  x['_task_hash'] not in hash_list]

    annotated_examples_to_correct = [x for x in inp_dataset if x['_task_hash'] in hash_list]

    assert len(inp_dataset) == len(annotated_examples_to_correct) + len(annotated_examples_to_keep)

    db.add_examples(annotated_examples_to_keep, [new_dataset_name])
    print(f"When finished run\nprodigy db-out {new_dataset_name} > {new_dataset_name}.jsonl")

    # Make temp file
    tmp_path = "tmp_prodigy.jsonl"
    write_jsonl(file_path=tmp_path, lines=annotated_examples_to_correct)
    prodigy_command = f"custom.rel.manual " \
                      f"{new_dataset_name} data/models/tokenizers/super-tokenizer " \
                      f"{tmp_path} " \
                      f"--label C_VAL,D_VAL,RELATED " \
                      f"--wrap --span-label UNITS,PK,TYPE_MEAS,COMPARE,RANGE,VALUE --wrap"

    prodigy.serve(prodigy_command, port=8001)

    os.remove(tmp_path)


if __name__ == '__main__':
    typer.run(main)
