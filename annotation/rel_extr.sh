#prodigy rel.manual rel_trials /home/ferran/Dropbox/PKRelations/data/pk_ner_supertok data/all_sentences/selected/nocontext/ready/training_tagged.jsonl --label RELATED,COMPLEMENT --span-label PK,C_VAL,C_TYPE,D_VAL,D_TYPE,RANGE_MIN,RANGE_MAX,COMPARATIVE,UNITS,COV,CHEMICAL,DISEASE,SPECIES,DOSAGE --add-ents --disable-patterns rules_disable.jsonl --wrap
prodigy rel.manual rel_trials323 en_core_sci_lg ../data/all_sentences/selected/nocontext/ready/training_tagged.jsonl --label C_VAL,D_VAL,RELATED,DOSAGE,C_MIN,C_MAX,D_MIN,D_MAX,COMPLEMENT --span-label UNITS,COMPLEMENT,COMPARATIVE,TYPE_MEAS,VALUE,PK,DISEASES,SPECIES,CHEMICAL --add-ents --wrap
