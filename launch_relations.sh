PRODIGY_ALLOWED_SESSIONS=ferran,frank,simon,silke,joe,vicky PRODIGY_PORT=8001 prodigy custom.rel.manual rex-pilot-300 data/models/tokenizers/rex-tokenizer data/gold/train0-200.jsonl --label C_VAL,D_VAL,RELATED --wrap --span-label UNITS,PK,TYPE_MEAS,COMPARE,RANGE,VALUE  --wrap -F recipes/rel_custom.py


