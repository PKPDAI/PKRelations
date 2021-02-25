# PKRelations
[**Pre-processing**](#pre-proessing)| [**Annotations**](#annotations) | [**Model development**](#model-development)
## Pre-proessing

1. Tag sentences across all articles/abstracts and select sentences with at least 1 PK and 1 VALUE entity (~1.1h with 12 cores):

````bash
python scripts/select_relevant_sentences.py \
   --path-model data/models/pk_ner_supertok \
   --path-ner-dict data/dictionaries/terms.json \
   --path-pmid data/raw/pmids \
   --path-pmc data/raw/pmcs/all_sentences.jsonl \
   --out-dir data/gold/base_files/
````


2. Split from main pool:

````
python scripts/split_sentences.py \
   --path-jsonl-pmids data/gold/base_files/all_selected_pmid.jsonl \
   --path-jsonl-pmcs data/gold/base_files/all_selected_pmc.jsonl \
   --slice-sizes 1000,20 \
   --slice-names rex-pilot,rex-minipilot \
   --out-dir data/gold/
````


3. Re-tag some file that was already tagged:

````
python scripts/rematch_jsonl.py \
   --path-inp-file data/gold/rex-minipilot.jsonl \
   --path-out-file data/gold/rex-minipilot2.jsonl \
   --path-base-model data/models/pk_ner_supertok \
   --path-ner-dict data/dictionaries/terms.json
````

## Annotations

Annotation recipes are stored at the `recipes` folder. To launch the recipe run: 

````
PRODIGY_ALLOWED_SESSIONS=ferran,vicky,joe,frank PRODIGY_PORT=8001 prodigy custom.rel.manual rex-pilot-50 data/models/tokenizers/rex-tokenizer data/gold/rex-minipilot2.jsonl --label C_VAL,D_VAL,RELATED --wrap --span-label UNITS,PK,TYPE_MEAS,COMPARE,RANGE,VALUE  --wrap -F recipes/rel_custom.py
````

#### Review annotated data

Run this to retrieve a file from azure storage containing annotations:

````
python scripts/split_annotations.py \
   --azure-file-name rex-pilot-ferran-output.jsonl \
   --save-local False
````

This will create different prodigy datasets on your local machine, one for each annotator.

Launch review:

````
prodigy review rex-pilot-reviewed rex-pilot-ferran-frank-done,rex-pilot-ferran-ferran-done,rex-pilot-ferran-simon-done --view-id blocks
````
Review existing annotations from a single file: 

````
prodigy custom.rel.manual rex-simon-reviewed data/models/tokenizers/rex-tokenizer data/rex-pilot-ferran-simon-done.jsonl --label C_VAL,D_VAL,RELATED --wrap --span-label UNITS,PK,TYPE_MEAS,COMPARE,RANGE,VALUE  --wrap -F recipes/rel_custom.py
````

## Model development

### Types of entities

### Relation types


<!--
![alt text](example.png) 
-->