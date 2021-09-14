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
        new_dataset_name: str = typer.Option(default="test-all-reviewed-2",
                                             help="Whether to save the model to azure"),
        hash_list: List[int] = typer.Option(default=[-1795520872, -784326854, -50538781, -2034452386, 21173666,
                                                     538793189, -526810301, 835803611, -1141047764, -1342741213,
                                                     -182403480, 1489447781, 1847590831, -1244967174, -1345379501,
                                                     176519790, 1118903905, -271770009, -674663106, 1366463053,
                                                     -76230883, 1116634145, 1788031430, -696548362, -754656065,
                                                     504142560, 1140028784, 594490673, 1451306311, 1271277114,
                                                     644898881, -596274663, 137325822, 2119107551, -1602320468,
                                                     -448670654, -1016400983, -197697156, -1888315032, -1972296656,
                                                     -1666356950, -586867324, -1624553884, -1949016288, -1720436888,
                                                     819132548, 593615948, -2081848871, 1236864598, 10504965, 412247270,
                                                     2099709336, -1432290358, 1544912066, -778177641, 1045430870,
                                                     516294806, -228683933, -582503469, -1867143482, -1993566842,
                                                     636219836, -1769108535, 1705704901, -99899954, -278614581,
                                                     -1346198158, 970933439, -1168656442, 1584861580, -570437360,
                                                     248254737, -1825963520, -698238180, 770597884, 1803326550,
                                                     -1299217880, -1473635621, -410551200, 277353764, -1564805276,
                                                     -505059157, -2062199833, 818751286, 213630191, 842465359,
                                                     -765603333, 1342496236, -1200897608, -1595607910, -44238274,
                                                     -1981298226, 1391942156, -1485674192, 811887785, -912297296,
                                                     1291792258, 5081044, -883249704, -1610576995],
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
