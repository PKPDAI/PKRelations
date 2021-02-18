"""This script makes all the data needed to subset potential sentences for relation extraction"""
import argparse
from pkrex.utils import read_jsonl, write_jsonl, make_super_tagger
import random
from tqdm import tqdm

random.seed(1)


def run(path_inp_file, path_out_file, path_base_model, path_ner_dict):
    spacy_model = make_super_tagger(dictionaries_path=path_ner_dict, pk_ner_path=path_base_model)
    inp_data = list(read_jsonl(path_inp_file))
    raw_text = [x['text'] for x in inp_data]
    out_sentences = []
    for n, (sentence, spacy_doc) in tqdm(enumerate(zip(inp_data, spacy_model.pipe(raw_text, n_process=12)))):
        spans = []
        for entity in spacy_doc.ents:
            spans.append(dict(start=entity.start_char, end=entity.end_char, label=entity.label_, ))
        sentence['spans'] = spans
        out_sentences.append(sentence)

    write_jsonl(file_path=path_out_file, lines=out_sentences)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-inp-file", type=str, help="File that we want to re-tagg",
                        default='../data/gold/rex-minipilot.jsonl'
                        )
    parser.add_argument("-path-out-file", type=str, help="Path of the re-tagged file",
                        default='../data/gold/rex-minipilot2.jsonl'
                        )

    parser.add_argument("--path-base-model", type=str, help="Base model",
                        default='../data/models/pk_ner_supertok'
                        )
    parser.add_argument("--path-ner-dict", type=str, help="JSON dictionary of NER terms",
                        default='../data/dictionaries/terms.json'
                        )

    args = parser.parse_args()

    run(path_inp_file=args.path_inp_file, path_out_file=args.path_out_file, path_base_model=args.path_base_model,
        path_ner_dict=args.path_ner_dict)


if __name__ == '__main__':
    main()
