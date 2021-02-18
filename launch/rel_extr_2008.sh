#prodigy rel.manual rel_trials_Simon_0408 en_core_sci_lg training_tagged.jsonl --label C_VAL,D_VAL,RELATED,DOSAGE,C_MIN,C_MAX,D_MIN,D_MAX,COMPLEMENT --span-label UNITS,COVARIATES,COMPARATIVE,TYPE_MEAS,VALUE,PK,DISEASES,SPECIES,CHEMICAL --add-ents --disable-patterns rules_disable.jsonl --wrap
PRODIGY_ALLOWED_SESSIONS=ferran,vicky,joe,frank,julia PRODIGY_PORT=8001 prodigy custom.rel.manual rex-pilot-50 data/models/tokenizers/rex-tokenizer data/gold/rex-minipilot2.jsonl --label C_VAL,D_VAL,RELATED --wrap --span-label UNITS,PK,TYPE_MEAS,COMPARE,RANGE,VALUE,  --wrap -F recipes/rel_custom.py


