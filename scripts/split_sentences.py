"""This script makes all the data needed to subset potential sentences for relation extraction"""
import argparse
from typing import List
from prodigy.util import read_jsonl, write_jsonl
import os
import random
from tqdm import tqdm

random.seed(1)


def run(path_jsonl_pmids: str, path_jsonl_pmcs: str, slice_sizes: List, slice_names: List, output_dir: str):
    assert slice_sizes.__len__() == slice_names.__len__()
    pmid_sentences = list(read_jsonl(path_jsonl_pmids))
    pmc_sentences = [x for x in read_jsonl(path_jsonl_pmcs) if
                     x['metadata']['sections'][0] in ['Abstract', 'abstract', 'ABSTRACT']]
    random.shuffle(pmc_sentences)
    random.shuffle(pmid_sentences)

    previous_size = 0
    for size, name in tqdm(zip(slice_sizes, slice_names)):
        half_size = round(size / 2)
        subset1 = pmid_sentences[previous_size:previous_size+half_size]
        subset2 = pmc_sentences[previous_size:previous_size+half_size]
        subset3 = subset1 + subset2
        out_file = os.path.join(output_dir, name + '.jsonl')
        write_jsonl(out_file, subset3)
        previous_size = half_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-jsonl-pmids", type=str, help="Path to the file with all the relevant sentences from "
                                                             "abstracts (pmid)",
                        default='../data/gold/base_files/all_selected_pmid.jsonl'
                        )
    parser.add_argument("-path-jsonl-pmcs", type=str, help="Path to the file with all the relevant sentences from "
                                                           "full-text (pmc)",
                        default='../data/gold/base_files/all_selected_pmc.jsonl'
                        )

    parser.add_argument("--slice-sizes", nargs='+', help="Number of examples in each output file",
                        default=[1000, 20]
                        )
    parser.add_argument("--slice-names", nargs='+', help="Names of the output files",
                        default=['rex-pilot', 'rex-minipilot']
                        )

    parser.add_argument("-out-dir", type=str, help="Path to the output directory",
                        default='../data/gold/'
                        )

    args = parser.parse_args()
    run(path_jsonl_pmids=args.path_jsonl_pmids, path_jsonl_pmcs=args.path_jsonl_pmcs, slice_sizes=args.slice_sizes,
        slice_names=args.slice_names, output_dir=args.out_dir)


if __name__ == '__main__':
    main()
