PRODIGY_ALLOWED_SESSIONS=ferran,gill PRODIGY_PORT=8001 prodigy custom.rel.manual dev-0-200-p2-trials-1 data/models/tokenizers/super-tokenizer data/annotations/P2/to_annotate/dev-0-200-to-annotate-p2.jsonl --label RELATED,DOSE,C_VAL,D_VAL --wrap --span-label CONTEXT,ROUTE,CHEMICAL,DISEASE,SPECIES,UNITS,PK,TYPE_MEAS,COMPARE,RANGE,VALUE --wrap -F recipes/rel_custom_part_2.py





