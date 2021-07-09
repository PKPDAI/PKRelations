import os
from typing import List
import typer
from prodigy.components.db import connect
import prodigy
from pkrex.utils import read_jsonl, write_jsonl
from recipes.rel_custom import custom_rel_manual


def main(

        base_file: str = typer.Option(default="data/annotations/P1/ready/test-all-ready-fixed-5.jsonl",
                                      help="Local output directory to save the model and files"),
        new_dataset_name: str = typer.Option(default="test-all-ready-fixed-6",
                                             help="Whether to save the model to azure"),
        hash_list: List[int] = typer.Option(default=[-1998249304, 1658067852, 1683524666, -246653432, 955651313,
                                                     1518893380, -1680488763, 703404576, -1360392228, -116922062,
                                                     511158322, 1755309756, -80323966, 1267782115, -1135872753,
                                                     -29179229, 1979122051, 1362797252, 2113730823, 1028981741,
                                                     266978274, 1490494434, -1021806704, 960827863, 1421861594,
                                                     -608966871, -1003213120, -1584755397, -1505317042, 21358307,
                                                     2112988683, -732026330, -391369328, 512208518, -2068375045,
                                                     -439308425, 2006811153, -1171100371, 1202382928, 1405683981,
                                                     -1222961202, -1694101210, 3312291, -383127636, 1226119001,
                                                     708069337, 437562362, -745640013, -675742774, 519083198,
                                                     -249903873, 1807679323, -1100502771, -1056412093, 1606036041,
                                                     808707243, 1476507720, -1191665966, -2047422471, -1899215155,
                                                     1322026208, 817268931, 1840148650, 1663319374, 1117315234,
                                                     -709320724, -524962873, 1407003057, -1644033144, -336211990,
                                                     -155874834, 2131022791, -1512350671, 1862317801, 1118048020,
                                                     -1242578126, -2085323251, -772947785, 709856478, 2042509145,
                                                     2122674236, -1827205875, 1182752626],
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
