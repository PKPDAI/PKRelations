"""This is used to construct a blank spaCy model with a tokenizer that splits by unicodes,
single non-alphanumeric characters, sequences of digits/alpha"""
import argparse
import spacy

from pkrex.tokenizer import replace_tokenizer


def run(out_path: str):
    blank_model = spacy.blank('en')
    final_model = replace_tokenizer(blank_model)
    final_model.meta['name'] = out_path.split('/')[-1]
    final_model.to_disk(out_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out-path", type=str, help="Output path of the spacy model",
                        default='../data/models/tokenizers/super-tokenizer'
                        )

    args = parser.parse_args()
    run(out_path=args.out_path)


if __name__ == '__main__':
    main()
