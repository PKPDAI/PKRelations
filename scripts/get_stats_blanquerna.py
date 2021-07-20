import os
from typing import List, Dict

import typer
from tqdm import tqdm

from pkrex.annotation_preproc import get_c_val_dicts
from pkrex.utils import read_jsonl, write_jsonl
from collections import Counter
import pandas as pd
import pickle


def get_all_spans(annotations: List[Dict], label: str) -> List[str]:
    output = []
    for sentence in annotations:
        for entity in sentence["spans"]:
            if entity["label"] == label:
                if "start" in entity.keys() and "end" in entity.keys():
                    mention = sentence["text"][entity["start"]:entity["end"]]
                    output.append(mention)
                else:
                    print(f"Careful, the following sentence has one entity without starting and end characters:\n"
                          f"{sentence}\nSentence hash: {sentence['_task_hash']}\nEntity: {entity}")
    return output


def main(
        input_file: str = typer.Option(default="data/annotations/P1/preprocessed/test-all-ready.jsonl",
                                       help="File to perform checks on"),

        out_dir: str = typer.Option(default="data/stats/test_set",
                                    help="File with dictionaries")

):
    dataset = list(read_jsonl(input_file))
    uq_ent_types = set([s["label"] for annotation in dataset for s in annotation["spans"]])

    for entity in uq_ent_types:
        print(entity)
        tmp_mentions = [token.lower() for token in get_all_spans(annotations=dataset, label=entity)]
        n = len(tmp_mentions)
        counts = Counter(tmp_mentions).most_common()
        ent_list = []
        for x in counts:
            ent_list.append(dict(mention=x[0], count=x[1], percentage_over_all_mentions=round(x[1] * 100 / n, 2)))
        pd.DataFrame(ent_list).to_csv(os.path.join(out_dir, f"mentions_{entity.lower()}.csv"))

    with open("data/ner_list_bulk.pkl", 'rb') as f:
        pmid_tagged = pickle.load(f)

    all_cval_dicts = []
    for annotation in tqdm(dataset):
        c_val_dicts = get_c_val_dicts(annotation)
        if c_val_dicts:
            c_val_dicts_additional = []
            for cv in c_val_dicts:
                cv["drugs"] = None
                cv["species"] = None
                cv["diseases"] = None
                for entry in pmid_tagged:
                    if entry["pmid"] == str(cv["pmid"]):
                        cv["drugs"] = entry["drugs"]
                        cv["species"] = entry["species"]
                        cv["diseases"] = entry["diseases"]
                        break
                c_val_dicts_additional.append(cv)
            all_cval_dicts += c_val_dicts_additional

    write_jsonl(file_path=os.path.join(out_dir, input_file.split("/")[-1].split(".")[0] + "-cvals.jsonl"),
                lines=all_cval_dicts)

    allentries = []
    for entry in all_cval_dicts:
        allentries.append(dict(parameter=entry["parameter"], central_val=entry["central_v"]["value/range"],
                               central_val_units=entry["central_v"]["units"],
                               central_val_type_meas=entry["central_v"]["type_meas"],
                               central_val_compare=entry["central_v"]["compare"],
                               deviation_val=entry["deviation"]["value/range"],
                               deviation_val_units=entry["deviation"]["units"],
                               deviation_val_type_meas=entry["deviation"]["type_meas"],
                               deviation_val_compare=entry["deviation"]["compare"], original_sentence=entry["sentence"],
                               pmid=entry["pmid"], sentence_id=entry["sentence_hash"], related_drugs=entry["drugs"],
                               related_species=entry["species"], related_diseases=entry["diseases"])
                          )

    pd.DataFrame(allentries).to_csv(os.path.join(out_dir, f"all_values_context.csv"))


if __name__ == "__main__":
    typer.run(main)
