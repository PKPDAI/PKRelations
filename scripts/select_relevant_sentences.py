"""This script makes all the data needed to subset potential sentences for relation extraction"""
import argparse
import os
from tqdm import tqdm
from pkrex.utils import clean_html, make_nlp_pk_route_ner, arrange_pk_sentences_abstract_context, read_jsonl, \
    write_jsonl, check_to_keep, arrange_pk_sentences_pmc_context
import hashlib


def filter_pmid_articles(inp_path, inp_model, output_path):
    """
    :param inp_model: NER model for PK, VALUE and ROUTE
    :param output_path: path of the output dir
    :param inp_path: string input path
    """
    all_annotations = []
    for current_file in tqdm(os.listdir(inp_path)):
        # =============== 1. Read sentences in from PubMed abstracts and remove titles ===========================
        # output_path_tmp = get_output_tmp_file_path(out_path=output_path, inp_file_name=current_file)
        tmp_sentences = [sentence for sentence in read_jsonl(os.path.join(inp_path, current_file)) if
                         not sentence['metadata']['istitle']]
        print("Number of sentences to filter:", len(tmp_sentences))
        # =============== 2. Clean potential html tags ===========================
        raw_texts = [clean_html(sentence['text']) for sentence in tmp_sentences]
        out_sentences = []
        for n, (sentence, spacy_doc) in enumerate(zip(tmp_sentences, inp_model.pipe(raw_texts, n_process=12))):
            spans = []
            ent_labels = []
            for entity in spacy_doc.ents:
                spans.append(dict(start=entity.start_char, end=entity.end_char, label=entity.label_, ))
                ent_labels.append(entity.label_)
            sentence['spans'] = spans
            sentence['sentence_hash'] = hashlib.sha1(sentence['text'].encode()).hexdigest()
            sentence['metadata']['relevant'] = False
            if len(sentence['text']) > 5 and ('PK' in ent_labels) and ('VALUE' in ent_labels):
                sentence['metadata']['relevant'] = True
            out_sentences.append(sentence)

        sentences_ready = arrange_pk_sentences_abstract_context(out_sentences)
        all_annotations = all_annotations + sentences_ready
    print("Total number of sentences: {}".format(len(all_annotations)))
    write_jsonl(file_path=output_path, lines=all_annotations)


# "spans":[{"start":65,"end":70,"label":"CHEMICAL","ids":["CUI-less"]},
#         {"start":71,"end":76,"label":"CHEMICAL","ids":["CHEBI:15618","BERN:4259103"]},
#         {"start":109,"end":114,"label":"CHEMICAL","ids":["CHEBI:15618","BERN:4259103"]}


def filter_pmc_articles(inp_path: str, inp_model: str, output_path: str):
    all_sentences = list(read_jsonl(inp_path))

    tmp_sentences = [sentence for sentence in tqdm(all_sentences) if check_to_keep(sentence['metadata']['sections'])]
    raw_texts = [clean_html(sentence['text']) for sentence in tmp_sentences]
    print("Starting processing ", len(raw_texts), " documents")
    out_sentences = []
    for n, (sentence, spacy_doc) in tqdm(enumerate(zip(tmp_sentences, inp_model.pipe(raw_texts, n_process=12)))):
        spans = []
        ent_labels = []
        for entity in spacy_doc.ents:
            spans.append(dict(start=entity.start_char, end=entity.end_char, label=entity.label_, ))
            ent_labels.append(entity.label_)
        sentence['spans'] = spans
        sentence['sentence_hash'] = hashlib.sha1(sentence['text'].encode()).hexdigest()
        sentence['metadata']['relevant'] = False
        if len(sentence['text']) > 5 and ('PK' in ent_labels) and ('VALUE' in ent_labels):
            sentence['metadata']['relevant'] = True
        out_sentences.append(sentence)

    final_sentences = arrange_pk_sentences_pmc_context(out_sentences)
    write_jsonl(output_path, final_sentences)


def run(path_model: str, path_ner_dict: str, path_pmid: str, path_pmc: str, out_dir: str):
    spacy_model = make_nlp_pk_route_ner(dictionaries_path=path_ner_dict, pk_ner_path=path_model)

    # out_pmid_path = os.path.join(out_dir, 'all_selected_pmid.jsonl')
    # print('Planning to write to: {}'.format(out_pmid_path))
    # filter_pmid_articles(inp_path=path_pmid, inp_model=spacy_model, output_path=out_pmid_path)

    out_pmc_path = os.path.join(out_dir, 'all_selected_pmc.jsonl')
    print('Planning to write to: {}'.format(out_pmc_path))
    filter_pmc_articles(inp_path=path_pmc, inp_model=spacy_model, output_path=out_pmc_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-model", type=str, help="Path to spaCy NER model for PK parameter mentions.",
                        default='../data/models/pk_ner_supertok'
                        )
    parser.add_argument("--path-ner-dict", type=str, help="Path to json file with dictionaries for dictionary-based "
                                                          "entities.",
                        default='../data/dictionaries/terms.json'
                        )
    parser.add_argument("--path-pmid", type=str, help="Path to the directory with all the PK relevant files in xml",
                        default='../data/all_sentences/pmids'
                        )
    parser.add_argument("--path-pmc", type=str, help="Path to the file with all the PMC sentences",
                        default='../data/all_sentences/raw/all_sentences.jsonl'
                        )

    parser.add_argument("--out-dir", type=str, help="Path to the output directory.",
                        default='../data/all_sentences/selected/clean'
                        )
    args = parser.parse_args()
    run(path_model=args.path_model, path_ner_dict=args.path_ner_dict, path_pmid=args.path_pmid, path_pmc=args.path_pmc,
        out_dir=args.out_dir)


if __name__ == '__main__':
    main()
