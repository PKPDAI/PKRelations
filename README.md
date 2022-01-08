# PKRelations
[**Pre-processing**](#pre-processing)| [**Annotations**](#annotations) | [**Model development**](#model-development)

## Download annotations

````bash
bash scripts/download-annotations.sh
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

## Annotations

Annotation recipes are stored at the `recipes` folder. To launch the recipe run: 

````bash
PRODIGY_ALLOWED_SESSIONS=ferran,vicky,joe,frank PRODIGY_PORT=8001 prodigy custom.rel.manual rex-pilot-50 data/models/tokenizers/super-tokenizer data/gold/train0-200.jsonl --label C_VAL,D_VAL,RELATED --wrap --span-label UNITS,PK,TYPE_MEAS,COMPARE,RANGE,VALUE  --wrap -F recipes/rel_custom.py
````

PRODIGY_ALLOWED_SESSIONS=ferran PRODIGY_PORT=8001 prodigy custom.rel.manual aug_check data/models/tokenizers/super-tokenizer data/annotations/P1/ready/train-all-reviewed-augmented.jsonl --label C_VAL,D_VAL,RELATED --wrap --span-label UNITS,PK,TYPE_MEAS,COMPARE,RANGE,VALUE,CHEMICAL,DISEASE,SPECIES,ROUTE  --wrap -F recipes/rel_custom.py


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
   --training-file-path data/biobert_tokenized/train-all-reviewed.jsonl \
   --val-file-path data/biobert_tokenized/dev-all-reviewed.jsonl \
   --output-dir results \
   --model-config-file configs/config-biobert.json
````

```` shell
python scripts/train_pkrex.py \
   --training-file-path data/biobert_tokenized/test-all-reviewed.jsonl \
   --val-file-path data/biobert_tokenized/dev-all-reviewed.jsonl \
   --output-dir results \
   --model-config-file configs/config-biobert.json
````

```` shell
python scripts/train_pkrex.py \
   --training-file-path data/biobert_tokenized/train.jsonl \
   --val-file-path data/biobert_tokenized/dev.jsonl \
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

#### Correct the following examples training set:

-1795520872
-784326854
-50538781 # range
-2034452386 # fractional insulin half-life
21173666
538793189 # is not a deviation
-526810301 # pfs
835803611 clcr nothing
-1141047764 # auc of change
-1342741213 # than remove
-182403480 # apparent median clearance
1489447781 # unbound
1847590831 ratio
-1244967174 Cmin
-1345379501 kp hiher
176519790 # three to five days
1118903905 total body water
-271770009 # terminal half-life,
-674663106
1366463053
-76230883
1116634145 circulation half-life
1788031430 # F
-696548362 # availability
-754656065 # clearance of plasma
504142560
1140028784 # metabolic clearance
594490673
1451306311 # less
1271277114 # bone half-life
644898881 # biological
-596274663 # less
137325822
2119107551
-1602320468 # weird range
-448670654 # initial
-1016400983 # pharmacokinetic
-197697156 # distribution
-1888315032 # absorption
-1972296656
-1666356950
-586867324 # oral midazolam
-1624553884
-1949016288 # extra coma
-1720436888 # body
819132548 # of elimination
593615948 ec50
-2081848871 # protein
1236864598 # cl
10504965 range
412247270 # whole span
2099709336 less than
-1432290358
1544912066 # ratios
-778177641 # maximum..
1045430870 
516294806
-228683933
-582503469
-1867143482 # steady state
-1993566842
636219836
-1769108535
1705704901
-99899954 # half-life
-278614581
-1346198158
970933439
-1168656442
1584861580
-570437360
248254737
-1825963520
-698238180
770597884
1803326550 parenthesis
-1299217880
-1473635621 # body-weight
-410551200
277353764 first-order
-1564805276 # apparent initial half-life
-505059157
-2062199833
818751286
213630191
842465359
-765603333
1342496236
-1200897608
-1595607910
-44238274
-1981298226
1391942156
-1485674192
811887785
-912297296
1291792258
5081044
-883249704 # between?
-1610576995


GFR no - check
crcl no - check
concentrations - check
max doses - check



TEST TO CORRECT:

188373212
-72628336
932029833
-424252256 cbw

DEV TO CORRECT:

-1986709618
-157678222 response rate?
1570154742 parenthesis
-1592768135 late phase
1164562386 units
1611056298
-342240023
1227138670 units
-2049618001 slowest
-457441628 range
632683494
618554853
-2000507776 43
720821972 greater
1120077822 under

CHECK TRAINING


799343678 edit spacing
1604399443 edit







Blob storages:

URL:
https://pkannotations.blob.core.windows.net/pkrex/gold/train.jsonl?sv=2020-10-02&st=2022-01-07T14%3A01%3A51Z&se=2124-07-20T16%3A01%3A00Z&sr=c&sp=racwl&sig=Rf0C7R5cGJNq90Te3hlJmOTSTAQwHGzo0qiVyDvBThk%3D
QUERY:
?sv=2020-10-02&st=2022-01-07T14%3A01%3A51Z&se=2124-07-20T16%3A01%3A00Z&sr=c&sp=racwl&sig=Rf0C7R5cGJNq90Te3hlJmOTSTAQwHGzo0qiVyDvBThk%3D