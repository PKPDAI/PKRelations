"""This script splits a jsonl file with prodigy annotated data in multiple files, one per annotator"""
import argparse
from prodigy.components.db import connect
from prodigy.util import read_jsonl
from azure.storage.blob import BlobClient
import os


def run(azure_file_name: str, save_local: bool):
    blob = BlobClient(
        account_url="https://pkpdaiannotations.blob.core.windows.net",
        container_name="pkpdaiannotations",
        blob_name=azure_file_name,
        credential="UpC2SPFbEqJdY0tgY91y1oVe3ZcQwxALkJ2QIDTYN17FoTLmpltCFyzxKk13fjrp04y+7K4L6t5KR6VOMUKOqw==")

    filename = "temp_annotations.jsonl"
    with open(filename, "wb") as f:
        f.write(blob.download_blob().readall())

    annotations = list(read_jsonl(filename))
    if not save_local:
        os.remove(filename)
    uq_annotators = set([x["_session_id"] for x in annotations])
    db = connect()
    dataset_names = []
    for annotator in uq_annotators:
        sub_annotations = [an for an in annotations if an["_session_id"] == annotator]
        if db.get_dataset(annotator):
            db.drop_dataset(annotator)
        annotator_dataset_name = annotator + "-done"
        db.add_dataset(annotator_dataset_name)
        dataset_names.append(annotator_dataset_name)
        assert annotator_dataset_name in db  # check  dataset was added
        sub_annotations_out = [{k: v for k, v in d.items() if k != "_session_id"} for d in sub_annotations]
        db.add_examples(sub_annotations_out, [annotator])  # add examples to dataset
        dataset = db.get_dataset(annotator)  # retrieve a dataset
        print("=====", annotator.split("-")[-1], " made: ", len(dataset), " annotations ========")
    print("The new Prodigy Dataset names are: \n:", dataset_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--azure-file-name", type=str, help="Annotate file that we want to split",
                        default='tableclass-trials-1-output.jsonl'
                        )
    parser.add_argument("--save-local", type=bool, help="Whether to save the jsonl file locally",
                        default=False
                        )
    args = parser.parse_args()

    run(azure_file_name=args.azure_file_name, save_local=args.save_local)


if __name__ == '__main__':
    main()
