"""Gets a jsonl file that previously had entities and adds them again if they have been lost"""

from pathlib import Path
from typing import List, Dict

import typer
from prodigy.util import read_jsonl, write_jsonl
from tqdm import tqdm


def main(
        input_file_path: Path = typer.Option(default="data/annotations/P1/reviewed/dev-0-200-reviewed.jsonl",
                                             help="File that we want to add the entities to"),

        base_file_path: Path = typer.Option(default="data/gold/dev0-200.jsonl",
                                            help="File with all base entities anntated"),

        base_entities=typer.Option(default=['CHEMICAL', 'SPECIES', 'DISEASE', 'ROUTE'],
                                   help="Entities to add")
):
    input_file = list(read_jsonl(input_file_path))
    base_file = list(read_jsonl(base_file_path))

    out_instances = []
    for tmp_instance in tqdm(input_file):
        tmp_hash = tmp_instance["sentence_hash"]
        base_sentences = []
        for tmp_base_instance in base_file:
            if tmp_base_instance["sentence_hash"] == tmp_hash:
                base_sentences.append(tmp_base_instance)
        assert len(base_sentences) == 1
        base_sentence = base_sentences[0]
        # Select additional entities
        existing_labels = set([x['label'] for x in tmp_instance['spans']])
        existing_entities = tmp_instance['spans']
        additional_entities = [span for span in base_sentence['spans'] if
                               ((span['label'] in base_entities) and (span['label'] not in existing_labels))]

        final_entities = add_non_overlapping_spans(base_spans=existing_entities, additional_spans=additional_entities)
        assert len(final_entities) >= len(existing_entities)
        tmp_instance["spans"] = final_entities
        out_instances.append(tmp_instance)
    write_jsonl(location=input_file_path,lines=out_instances)


def add_non_overlapping_spans(base_spans: List[Dict], additional_spans: List[Dict]) -> List[Dict]:
    """Adds the additional spans only if they don't overlap with existing even if longer"""
    seen_tokens = set()
    result = []
    for span in base_spans:
        seen_tokens.update(range(span["start"], span["end"]))
        result.append(span)
    for additional_span in additional_spans:
        if additional_span["start"] not in seen_tokens and additional_span["end"] - 1 not in seen_tokens:
            seen_tokens.update(range(additional_span["start"], additional_span["end"]))
            result.append(additional_span)

    result = sorted(result, key=lambda tmp_span: tmp_span["start"])
    return result


if __name__ == "__main__":
    typer.run(main)
