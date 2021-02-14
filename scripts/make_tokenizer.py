"""This is used to extract a simpler spaCy model that only has the tokenizer give an input model"""
import argparse
import spacy


def run(path_base_model: str, out_dir: str, out_model_name: str):
    base_model = spacy.load(path_base_model)
    blank_model = spacy.blank('en')
    blank_model.tokenizer = base_model.tokenizer
    #new_name = 'tokenizer_'+path_base_model.split('/')[-1]

    blank_model.meta['name'] = out_model_name
    blank_model.to_disk(out_dir + out_model_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-base-model", type=str, help="Path to the base spaCy model",
                        default='../data/models/pk_ner_supertok'
                        )

    parser.add_argument("--out-dir", type=str, help="Path to the base spaCy model",
                        default='../data/models/tokenizers/'
                        )

    parser.add_argument("--out-model-name", type=str, help="Path to the base spaCy model",
                        default='rex-tokenizer'
                        )

    args = parser.parse_args()
    run(path_base_model=args.path_base_model, out_dir=args.out_dir, out_model_name=args.out_model_name)


if __name__ == '__main__':
    main()
