# PKRelations
[**Pre-processing**](#pre-processing)| [**Annotations**](#annotations) | [**Model development**](#model-development)
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

## Annotations

Annotation recipes are stored at the `recipes` folder. To launch the recipe run: 

````bash
PRODIGY_ALLOWED_SESSIONS=ferran,vicky,joe,frank PRODIGY_PORT=8001 prodigy custom.rel.manual rex-pilot-50 data/models/tokenizers/super-tokenizer data/gold/train0-200.jsonl --label C_VAL,D_VAL,RELATED --wrap --span-label UNITS,PK,TYPE_MEAS,COMPARE,RANGE,VALUE  --wrap -F recipes/rel_custom.py
````

#### Review annotated data

Run this to retrieve a file from azure storage containing annotations:

````bash
python scripts/split_annotations.py \
   --azure-file-name rex-pilot-ferran-output.jsonl \
   --save-local true \
   --out-dir data/annotations/pilot
````

This will create different prodigy datasets on your local machine, one for each annotator.

Launch review:

````bash
prodigy review rex-pilot-reviewed rex-pilot-ferran-frank-done,rex-pilot-ferran-ferran-done,rex-pilot-ferran-simon-done --view-id relations 
````

Review existing annotations from a single file: 

````bash
prodigy custom.rel.manual rex-simon-reviewed data/models/tokenizers/rex-tokenizer data/rex-pilot-ferran-simon-done.jsonl --label C_VAL,D_VAL,RELATED --wrap --span-label UNITS,PK,TYPE_MEAS,COMPARE,RANGE,VALUE  --wrap -F recipes/rel_custom.py
````

## Model development

Output format expectation:
````
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

Render template

https://stackoverflow.com/questions/31965558/how-to-display-a-variable-in-html

## TO DO

```` shell
python scripts/train_pkrex.py \
   --training-file-path data/pubmedbert_tokenized/train-all-reviewed-clean-4.jsonl \
   --val-file-path data/pubmedbert_tokenized/test-all-ready-fixed-6.jsonl \
   --output-dir results \
   --model-config-file configs/config-biobert.json
````


training_almost_ready includes:

0-200

200-250

250-450

450-750

750-1150

1150-1500

Missing 1500-1800

1800-2100



### Functions missing

1) Prediction step

#### Analyses to do

1) Pre-compute negatives
2) Compute negatives on-the-fly