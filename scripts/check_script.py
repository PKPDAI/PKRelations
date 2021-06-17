import os
from pathlib import Path
import typer
from pkrex.annotation_preproc import print_rex_stats, check_rel_tokens_entities, d_val_pointing
from pkrex.utils import read_jsonl


def main(
        input_dir: Path = typer.Option(default="data/annotations/P1/reviewed/",
                                       help="Directory with the files to perform checks on"),
        insect_mentions: bool = typer.Option(default=False,
                                             help="Whether to print mentions of each entity type")

):
    all_files = [x for x in os.listdir(input_dir) if ".jsonl" in x]
    for inp_file in all_files:
        print(f"====== Analysing {inp_file} ==============")
        file_path = os.path.join(input_dir, inp_file)
        annotations = list(read_jsonl(file_path))
        print("============ Printing dataset stats ==============")
        print_rex_stats(annotations=annotations)
        print("\n============ Checking that all relations are formed by entity tokens ==============")
        check_rel_tokens_entities(annotations=annotations)
        print("Check passed!")
        print("\n============ Checking that D_VALs always point to C_VALs ==============")
        d_val_pointing(annotations=annotations)
        print("Check passed!")
        print("All checks passed!")
        uq_entities = set([span["label"] for annotation in annotations for span in annotation["spans"]])
        uq_relations = set([relation["label"] for annotation in annotations for relation in annotation["relations"]])
        print(f"Unique entities: {uq_entities}\nUnique relations: {uq_relations}")
        if insect_mentions:
            for entity_type in uq_entities:
                if entity_type not in ["VALUE"]:
                    print(f"\n========== {entity_type} ==========")
                    ent_mentions = set([annotation["text"][span["start"]:span["end"]] for annotation in annotations
                                        for span in annotation["spans"] if span["label"] == entity_type])
                    [print(mention) for mention in ent_mentions]


if __name__ == "__main__":
    typer.run(main)
