#PRODIGY_ALLOWED_SESSIONS=ferran,palang,pum PRODIGY_PORT=8001 prodigy custom.rel.manual rex-pilot-300 data/models/tokenizers/super-tokenizer data/gold/dev200-500.jsonl --label C_VAL,D_VAL,RELATED --wrap --span-label UNITS,PK,TYPE_MEAS,COMPARE,RANGE,VALUE  --wrap -F recipes/rel_custom.py
PRODIGY_ALLOWED_SESSIONS=ferran PRODIGY_PORT=8001 prodigy custom.rel.manual aug_check data/models/tokenizers/super-tokenizer data/annotations/P1/ready/train-all-reviewed-augmented.jsonl --label C_VAL,D_VAL,RELATED --wrap --span-label UNITS,PK,TYPE_MEAS,COMPARE,RANGE,VALUE,CHEMICAL,DISEASE,SPECIES,ROUTE  --wrap -F recipes/rel_custom.py


