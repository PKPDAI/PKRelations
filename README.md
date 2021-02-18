# PKRelations

## 1. Getting pool of sentences ready for annotations and performing initial NER

From all the relevant PK papers we selected sentence with.....

````
python scripts/select_relevant_sentences.py \
   --path-model data/models/pk_ner_supertok \
   --path-ner-dict data/dictionaries/terms.json \
   --path-pmid data/all_sentences/pmids \
   --path-pmc data/data/all_sentences/raw/all_sentences.jsonl \
   --out-dir data/all_sentences/selected/clean
````

**TODO:** make sure pmids don't overlap with pmcs and that pmids are just the relevant ones (there seem to be many)

### Split from main pool:

````
python scripts/split_sentences.py \
   --path-jsonl-pmids data/models/pk_ner_supertok \
   --path-jsonl-pmcs data/dictionaries/terms.json \
   --slice-sizes 1000,20 \
   --slice-names rex-pilot,rex-minipilot \
   --out-dir data/gold/
````

### Re-tag some file that was already tagged:

````
python scripts/rematch_jsonl.py \
   --path-inp-file data/gold/rex-minipilot.jsonl \
   --path-out-file data/gold/rex-minipilot2.jsonl \
   --path-base-model data/models/pk_ner_supertok \
   --path-ner-dict data/dictionaries/terms.json
   
````


### Create multiple datasets from single jsonl for review

1. Run this: 
````
python scripts/split_annotations.py \
   --azure-file-name rex-pilot-ferran-output.jsonl \
   --save-local False
   
````

2. Launch review: 

For instance:

````
prodigy review table-trials-1-review tableclass-trials-1-ferran,tableclass-trials-1-frank,tableclass-trials-1-gill,tableclass-trials-1-joe --view-id choice
   
````

## 2. Prodigy interface for annotations

````
prodigy custom.rel.manual rel_trials_Simon_1108 en_core_sci_lg training_tagged.jsonl --label C_VAL,D_VAL,RELATED,DOSAGE,C_MIN,C_MAX,D_MIN,D_MAX,COMPLEMENT --wrap --span-label UNITS,COVARIATES,COMPARATIVE,TYPE_MEAS,VALUE,PK,DISEASES,SPECIES,CHEMICAL,ROUTE --add-ents --wrap -F rel_custom.py

````

## 3. Model development

## Types of entities:

1. **PK**: Mention of a pharmacokinetic parameter 

2. **C_VAL**: Central value of a PK parameter

3. **C_TYPE**: Type of central value, e.g. mean, median, population, individual

4. **D_VAL**: Deviation measurement

5. **D_TYPE**: Type of deviation measurement, e.g. variance, standard deviation, IIV, +-

6. **RANGE_MIN**: PK minimum value when a range is expressed e.g. midazolam's renal clearance went from _**4**_ to 7 mg/hL

7. **RANGE_MAX**: PK maximum value when a range is expressed e.g. midazolam's renal clearance went from 4 to _**7**_ mg/hL

8. **COMPARATIVE**: That's a complement of C_TYPE to specify whether instead of purely a central value it says: "higher", "less", ">","<". e.g. Clearance rate was ">"20mg/hL

9. **UNITS**: **Any** units mentioned in the sentence 

10. **COV**: Covariate that complements the value of the parameter

11. **CHEMICAL**: Pre-highlighted by bio-bert but worth to check/modify (screenshot if that happens)
12. **DISEASE**: Pre-highlighted by bio-bert but worth to check/modify (screenshot if that happens)
13. **SPECIES**: Pre-highlighted by bio-bert but worth to check/modify (screenshot if that happens)

14. **DOSAGE**?: Central value of a dosage administered to a patient (the relation between this and the units will be considered by the RELATED type)

#### Relation types:

1. **RELATED**: Main type of relation. Usually this gets assigned between all entities and C_VAL or RANGE_MIN, RANGE_MAX. Also RANGE_MIN should almost always have a RELATED that points towards RAGE_MAX.
2. **COMPLEMENT**: This happens when a single covariate needs to be splitted into two or more tokens, in this case the main covariate complements the type. Example: 

![alt text](example.png)