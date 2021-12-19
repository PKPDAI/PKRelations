"""This script gets an input JSONL in which we've already lablled PK, VALUE, RANGE, P-VALUE, ROUTE,COMPARE,
TYPE_MEAS some UNITS and appends DRUGS, SPECIES, DISEASES by using BERN and resolving overlapping entities if needed"""
import argparse
from pkrex.utils import read_jsonl, write_jsonl
from pkrex.bern import get_drugs_diseases_species
import spacy
from tqdm import tqdm

from pkrex.utils import is_sentence_relevant, resolve_overlapping_spans


def run(path_inp_file: str, resolve_overlapping: bool, drop_irrelevant: bool):
    nlp_drugs = spacy.load("en_ner_bc5cdr_md")
    out_sentences = []
    old_sentences = list(read_jsonl(path_inp_file))

    for sentence in tqdm(old_sentences):
        extra_spans = get_drugs_diseases_species(inp_sentence=sentence["text"],
                                                 inp_entities=["CHEMICAL", "DISEASE", "SPECIES", "GENE", "MUTATION"],
                                                 inp_model=nlp_drugs,
                                                 use_local_bern=True)


        if extra_spans:
            all_spans = sentence["spans"] + extra_spans
            if resolve_overlapping:
                all_spans = resolve_overlapping_spans(all_spans)
            sentence["spans"] = all_spans
            sentence['metadata']['relevant'] = is_sentence_relevant(sentence)
        if drop_irrelevant:
            if sentence['metadata']['relevant']:
                out_sentences.append(sentence)
            else:
                print("Hey, the following sentence was relevant and now it's not: {}".format(sentence['text']))
        else:
            out_sentences.append(sentence)

    print(f"Writing to {path_inp_file}")
    write_jsonl(path_inp_file, out_sentences)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-inp-file", type=str, help="Annotated file that we want to add drugs, diseases and "
                                                          "species",
                        default='data/gold/test800-1000.jsonl'
                        )
    parser.add_argument("--resolve-overlapping", type=bool, help="Whether to resolve overlapping spans",
                        default=True
                        )

    parser.add_argument("--drop-irrelevant", type=bool, help="Whether to drop irrelevant sentences",
                        default=True
                        )

    args = parser.parse_args()

    run(path_inp_file=args.path_inp_file, resolve_overlapping=args.resolve_overlapping,
        drop_irrelevant=args.drop_irrelevant)


if __name__ == '__main__':
    main()
