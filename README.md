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
   --slice-sizes 205 \
   --slice-names dev0-200\
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
   --path-inp-file data/gold/dev0-200.jsonl \
   --resolve-overlapping true
````

````bash
python scripts/add_bern.py \
   --path-inp-file data/gold/test0-200.jsonl \
   --resolve-overlapping true
````

Make tokenizer ready for prodigy usage: 

````bash
python scripts/make_destructive_tokenizer.py \
   --out-path data/models/tokenizers/super-tokenizer
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

### Types of entities

### Relation types


<!--
![alt text](example.png) 
-->