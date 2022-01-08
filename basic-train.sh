#!/bin/bash
/home/waty/anaconda3/envs/PKRelations/bin/python scripts/train_pkrex.py \
--training-file-path data/biobert-tokenized/train.jsonl \
--val-file-path data/biobert-tokenized/dev.jsonl \
--output-dir results \
--model-config-file configs/config-biobert.json
