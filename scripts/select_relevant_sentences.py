"""This script makes all the data needed to subset potential sentences for relation extraction"""
import argparse
import os
from tqdm import tqdm
from pkrex.utils import clean_html, make_super_tagger, arrange_pk_sentences_abstract_context, read_jsonl, \
    write_jsonl, check_to_keep, arrange_pk_sentences_pmc_context, populate_spans, get_link, sentence_pmid_to_int


def filter_pmid_articles(inp_path, inp_model, output_path, relevant_pmids):
    """
    :param inp_model: NER model for PK, VALUE and ROUTE
    :param output_path: path of the output dir
    :param inp_path: string input path
    :param relevant_pmids: List list of relevant pmids
    """
    all_annotations = []
    for current_file in tqdm(os.listdir(inp_path)):
        # =============== 1. Read sentences in from PubMed abstracts and remove titles ===========================
        # output_path_tmp = get_output_tmp_file_path(out_path=output_path, inp_file_name=current_file)
        tmp_sentences = [sentence_pmid_to_int(sentence) for sentence in
                         read_jsonl(os.path.join(inp_path, current_file)) if not sentence['metadata']['istitle']]
        tmp_sentences = [sentence for sentence in tqdm(tmp_sentences) if sentence['metadata']['pmid'] in relevant_pmids]

        print("Number of sentences to filter:", len(tmp_sentences))
        # =============== 2. Clean potential html tags ===========================
        raw_texts = [clean_html(sentence['text']) for sentence in tmp_sentences]
        # =============== 3. Apply model to all sentences and append entity spans ===========================
        out_sentences = []
        for n, (sentence, spacy_doc) in enumerate(zip(tmp_sentences, inp_model.pipe(raw_texts, n_process=12))):
            sentence = populate_spans(spacy_doc=spacy_doc, sentence=sentence)
            sentence['meta'] = {"url": get_link(sentence)}
            out_sentences.append(sentence)
        # =============== 4. Arrange and filter relevant ones ===========================
        sentences_ready = arrange_pk_sentences_abstract_context(out_sentences)
        all_annotations = all_annotations + sentences_ready
    print("Total number of sentences: {}".format(len(all_annotations)))
    write_jsonl(file_path=output_path, lines=all_annotations)


# "spans":[{"start":65,"end":70,"label":"CHEMICAL","ids":["CUI-less"]},
#         {"start":71,"end":76,"label":"CHEMICAL","ids":["CHEBI:15618","BERN:4259103"]},
#         {"start":109,"end":114,"label":"CHEMICAL","ids":["CHEBI:15618","BERN:4259103"]}


def filter_pmc_articles(inp_path, inp_model, output_path, relevant_pmids):
    # ================== 1. Read from PMC jsonl file =================== #
    all_sentences = [sentence_pmid_to_int(sentence) for sentence in read_jsonl(inp_path)]
    all_sentences = [sentence for sentence in tqdm(all_sentences) if sentence['metadata']['pmid'] in relevant_pmids]
    # ================== 2. Remove methods sections and clean html=================== #
    tmp_sentences = [sentence for sentence in tqdm(all_sentences) if check_to_keep(sentence['metadata']['sections'])]
    raw_texts = [clean_html(sentence['text']) for sentence in tmp_sentences]
    # ================== 3. Apply model to all sentences and append entity spans =================== #
    print("Starting processing ", len(raw_texts), " documents")
    out_sentences = []
    for n, (sentence, spacy_doc) in tqdm(enumerate(zip(tmp_sentences, inp_model.pipe(raw_texts, n_process=12)))):
        sentence = populate_spans(spacy_doc=spacy_doc, sentence=sentence)
        sentence['meta'] = {"url": get_link(sentence)}
        out_sentences.append(sentence)
    # =============== 4. Arrange and filter relevant ones ===========================
    final_sentences = arrange_pk_sentences_pmc_context(out_sentences)
    write_jsonl(output_path, final_sentences)


def run(path_model: str, path_ner_dict: str, path_pmid: str, path_pmc: str, out_dir: str, path_relevant_pmids: str):
    spacy_model = make_super_tagger(dictionaries_path=path_ner_dict, pk_ner_path=path_model)

    with open(path_relevant_pmids) as f:
        relevant_pmids = [int(pmid.replace('\n', '')) for pmid in list(f)]

    out_pmid_path = os.path.join(out_dir, 'all_selected_pmid.jsonl')
    print('Planning to write to: {}'.format(out_pmid_path))
    filter_pmid_articles(inp_path=path_pmid, inp_model=spacy_model, output_path=out_pmid_path,
                         relevant_pmids=relevant_pmids)

    out_pmc_path = os.path.join(out_dir, 'all_selected_pmc.jsonl')
    print('Planning to write to: {}'.format(out_pmc_path))
    filter_pmc_articles(inp_path=path_pmc, inp_model=spacy_model, output_path=out_pmc_path,
                        relevant_pmids=relevant_pmids)


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
                        default='../data/raw/pmids'
                        )
    parser.add_argument("--path-pmc", type=str, help="Path to the file with all the PMC sentences",
                        default='../data/raw/pmcs/all_sentences.jsonl'
                        )

    parser.add_argument("--path-relevant-pmids", type=str, help="Path to the txt filed with the list of PMIDs "
                                                                "considered relevant by the algorithm",
                        default='../data/raw/allPapersPMIDS.txt'
                        )

    parser.add_argument("--out-dir", type=str, help="Path to the output directory.",
                        default='../data/raw/selected/'
                        )

    args = parser.parse_args()
    run(path_model=args.path_model, path_ner_dict=args.path_ner_dict, path_pmid=args.path_pmid, path_pmc=args.path_pmc,
        out_dir=args.out_dir, path_relevant_pmids=args.path_relevant_pmids)


if __name__ == '__main__':
    main()
