# PKRelations
[**Download annotations**](#download-annotations)| [**Model development**](#model-development)| [**Pre-processing**](#pre-processing)

## Download annotations

````bash
bash scripts/download-annotations.sh
````

## Model development

Launch training:

````sh
python scripts/train_pkrex.py \
   --training-file-path data/biobert_tokenized/train-all-reviewed.jsonl
   --val-file-path data/biobert_tokenized/dev-all-reviewed.jsonl \
   --output-dir results \
   --model-config-file configs/config-biobert.json
````

Output format expectation:
````python
var = 
    {
        "parameter_mention": "renal CL",

        "central_measurement": {
            "value/range": 3.0,
            "units": "mL/min",
            "comparative_term": "higher",
            "type_measurement": "median"
        },

        "deviation_measurement": {
            "value/range": 0.2,
            "units": "mL/min",
            "comparative_term": "",
            "type_measurement": "+-"
        }
    }
````

## Pre-processing

1. Tag sentences across all articles/abstracts and select sentences with at least 1 PK and 1 VALUE/RANGE entity (~1.1h with 12 cores):

````bash
python scripts/select_relevant_sentences.py \
   --path-model data/models/pk_ner_supertok \
   --path-ner-dict data/dictionaries/terms.json \
   --path-pmid data/raw/pmids \
   --path-pmc data/raw/pmcs/all_sentences.jsonl \
   --path-relevant-pmids data/raw/allPapersPMIDS.txt \
   --out-dir data/gold/base_files/
````


2. Sample sentences to annotate from main pool of sentences:

````bash
python scripts/sample_sentences.py \
   --path-jsonl-pmids data/gold/base_files/all_selected_pmid.jsonl \
   --path-jsonl-pmcs data/gold/base_files/all_selected_pmc.jsonl \
   --slice-sizes 305 \
   --slice-names train450-750\
   --out-dir data/gold/ \
   --path-already-sampled data/gold/already_sampled.txt
````

Re-tagg some file that was already tagged or just attach article link:

````bash
python scripts/retagg_jsonl.py \
   --path-inp-file data/gold/base_files/all_selected_pmc.jsonl \
   --path-out-file data/gold/base_files/all_selected_pmc.jsonl \
   --path-base-model data/models/pk_ner_supertok \
   --path-ner-dict data/dictionaries/terms.json 
````

````bash
python scripts/retagg_jsonl.py \
   --path-inp-file data/gold/base_files/all_selected_pmid.jsonl \
   --path-out-file data/gold/base_files/all_selected_pmid.jsonl \
   --path-base-model data/models/pk_ner_supertok \
   --path-ner-dict data/dictionaries/terms.json 
````

Add bern entities

````bash
python scripts/add_bern.py \
   --path-inp-file data/gold/test200-500.jsonl \
   --resolve-overlapping true
````

````bash
python scripts/add_bern.py \
   --path-inp-file data/gold/train250-450.jsonl \
   --resolve-overlapping true
````

````bash
python scripts/add_bern.py \
   --path-inp-file data/gold/train250-450.jsonl \
   --resolve-overlapping true
````

Make tokenizer ready for prodigy usage: 

````bash
python scripts/make_destructive_tokenizer.py \
   --out-path data/models/tokenizers/super-tokenizer
````
 
Filter sentences annotated in P1 ready for P2

````bash
python scripts/filter_part_2.py \
   --input-file data/annotations/dev/1/rex-dev0-200.jsonl \
   --output-dir data/part2/dev/
````
