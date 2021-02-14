#prodigy rel.manual rel_trials_Simon_0408 en_core_sci_lg training_tagged.jsonl --label C_VAL,D_VAL,RELATED,DOSAGE,C_MIN,C_MAX,D_MIN,D_MAX,COMPLEMENT --span-label UNITS,COVARIATES,COMPARATIVE,TYPE_MEAS,VALUE,PK,DISEASES,SPECIES,CHEMICAL --add-ents --disable-patterns rules_disable.jsonl --wrap
prodigy custom.rel.manual rex-trials-ferran-3 data/models/tokenizers/tokenizer_pk_ner_supertok data/gold/rex-minipilot.jsonl --label C_VAL,D_VAL,RELATED,C_MIN,C_MAX,D_MIN,D_MAX --wrap --span-label UNITS,TYPE_MEAS,VALUE,PK --wrap -F recipes/rel_custom.py


