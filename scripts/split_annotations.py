"""This script splits a jsonl file with prodigy annotated data in multiple files, one per annotator"""
import argparse
from prodigy.components.db import connect
from prodigy.util import read_jsonl, write_jsonl
import os
from pkrex.utils import get_blob


def run(azure_file_name: str, save_local: bool, out_dir: str):
    blob = get_blob(inp_blob_name=azure_file_name)

    filename = "temp_annotations.jsonl"
    with open(filename, "wb") as f:
        f.write(blob.download_blob().readall())

    annotations = list(read_jsonl(filename))

    write = False
    if not save_local:
        os.remove(filename)
        os.remove(os.path.join("..", filename))

    else:
        write = True

    uq_annotators = set([x["_session_id"] for x in annotations])
    db = connect()
    dataset_names = []
    for annotator in uq_annotators:

        annotator_dataset_name = annotator + "-done"

        sub_annotations = [an for an in annotations if an["_session_id"] == annotator]
        if db.get_dataset(annotator_dataset_name):
            db.drop_dataset(annotator_dataset_name)
        if write:
            write_jsonl(os.path.join(out_dir, annotator_dataset_name + ".jsonl"), sub_annotations)

        db.add_dataset(annotator_dataset_name)
        dataset_names.append(annotator_dataset_name)
        assert annotator_dataset_name in db  # check  dataset was added
        sub_annotations_out = [{k: v for k, v in d.items() if k != "_session_id"} for d in sub_annotations]
        db.add_examples(sub_annotations_out, [annotator_dataset_name])  # add examples to dataset
        dataset = db.get_dataset(annotator_dataset_name)  # retrieve a dataset
        print("=====", annotator.split("-")[-1], " made: ", len(dataset), " annotations ========")
    print("The new Prodigy Dataset names are: \n:", dataset_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--azure-file-name", type=str, help="Annotated file that we want to split",
                        default='rex-pilot-ferran-output.jsonl'
                        )
    parser.add_argument("--save-local", type=bool, help="Whether to save the jsonl file locally",
                        default=False
                        )
    parser.add_argument("--out-dir", type=str, help="Dir to write files",
                        default='../data/annotations/pilot'
                        )

    args = parser.parse_args()

    run(azure_file_name=args.azure_file_name, save_local=args.save_local, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
