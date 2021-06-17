"""This script makes all the data needed to subset potential sentences for relation extraction"""
import argparse
from typing import List
from pkrex.utils import read_jsonl, write_jsonl, check_and_resample, sentence_pmid_to_int
import os
import random
from tqdm import tqdm

random.seed(1)


def run(path_jsonl_pmids: str, path_jsonl_pmcs: str, slice_sizes: List, slice_names: List, output_dir: str,
        path_already_sampled: str):
    assert len(slice_sizes) == len(slice_names)

    # ===== 1. Read sentences from main pools in ===============
    pmc_sentences = [sentence_pmid_to_int(sentence) for sentence in read_jsonl(path_jsonl_pmcs)]
    full_text_pmids = set([x['metadata']['pmid'] for x in pmc_sentences])
    pmid_sentences = [x for x in list(read_jsonl(path_jsonl_pmids)) if x['metadata']['pmid'] not in full_text_pmids]

    # ===== 2. Read list of sentence IDs already sampled and relevant pmids ===============
    with open(path_already_sampled) as f:
        sids_already_sampled = [int(sid) for sid in list(f)]

    # ===== 4. Randomly shuffle sentences ===============
    random.shuffle(pmc_sentences)
    random.shuffle(pmid_sentences)

    previous_size = 0
    for size, name in tqdm(zip(slice_sizes, slice_names)):
        half_size = round(size / 2)
        # ===== 5. Take and check PMID sentences ===============
        pmid_subset = pmid_sentences[previous_size:previous_size + half_size]

        pmid_subset, sids_already_sampled = check_and_resample(sampled_subset=pmid_subset, main_pool=pmid_sentences,
                                                               ids_already_sampled=sids_already_sampled)

        # ===== 6. Take and check PMC sentences ===============
        pmc_subset = pmc_sentences[previous_size:previous_size + half_size]

        pmc_subset, sids_already_sampled = check_and_resample(sampled_subset=pmc_subset, main_pool=pmc_sentences,
                                                              ids_already_sampled=sids_already_sampled)

        # ===== 7. Join PMID and PMC sampled subsets ===============
        final_subset = pmid_subset + pmc_subset

        # ===== 8. Rewrite sids_already_sampled ===============
        with open(path_already_sampled, 'w') as file:
            file.write('\n'.join([str(sid) for sid in sids_already_sampled]))

        # ===== 9. Write new jsonl file ===============
        random.shuffle(final_subset)  # shuffle them again before writing
        out_file = os.path.join(output_dir, name + '.jsonl')
        write_jsonl(out_file, final_subset)
        previous_size += half_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-jsonl-pmids", type=str, help="Path to the file with all the relevant sentences from "
                                                             "abstracts (pmid)",
                        default='../data/gold/base_files/all_selected_pmid.jsonl'
                        )
    parser.add_argument("--path-jsonl-pmcs", type=str, help="Path to the file with all the relevant sentences from "
                                                            "full-text (pmc)",
                        default='../data/gold/base_files/all_selected_pmc.jsonl'
                        )

    parser.add_argument("--slice-sizes", nargs='+', help="Number of examples in each output file",
                        default=[310, 310, 210], type=int
                        )
    parser.add_argument("--slice-names", nargs='+', help="Names of the output files",
                        default=['train1500-1800', 'train1800-2100', 'test800-1000'], type=str
                        )

    parser.add_argument("--out-dir", type=str, help="Path to the output directory",
                        default='../data/gold/'
                        )

    parser.add_argument("--path-already-sampled", type=str, help="Path to the txt filed with the list of PMIDs already "
                                                                 "sampled from that distribution",
                        default='../data/gold/already_sampled.txt'
                        )

    args = parser.parse_args()
    run(path_jsonl_pmids=args.path_jsonl_pmids, path_jsonl_pmcs=args.path_jsonl_pmcs, slice_sizes=args.slice_sizes,
        slice_names=args.slice_names, output_dir=args.out_dir, path_already_sampled=args.path_already_sampled)


if __name__ == '__main__':
    main()
