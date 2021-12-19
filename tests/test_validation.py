BASE_CONFIG = {'run_name': 'pkner-rex-augmented-128-clean',
               'base_model': 'data/models/pk-bert',
               'n_workers_dataloader': 12,
               'gpus': True,
               'seed': 1,
               'learning_rate': 2e-05,
               'max_epochs': 20,
               'max_length': 128,
               'batch_size': 3,
               'val_batch_size': 4,
               'eps': 1e-08,
               'ctx_embedding_size': 4,
               'tensorboard_logger': True,
               'tag_type': 'bio',
               'early_stopping': False,
               'early_stopping_patience': 5,
               'gradient_clip_val': 1.0,
               'weight_decay': 0.01,
               'stochastic_weight_avg': True,
               'remove_longer_seqs': True,
               'lr_warmup': False,
               'weighted_loss': False,
               'final_train': False,
               'training_file_path': 'data/pk-bert-tokenized/train-all-reviewed-augmented.jsonl',
               'val_file_path': 'data/pk-bert-tokenized/test-all-reviewed.jsonl',
               'output_dir': 'results',
               'scaling_dict': {'O': 1.0,
                                'I-PK': 9.568742418115649,
                                'I-VALUE': 12.519266378626224,
                                'I-UNITS': 15.075493735400297,
                                'B-VALUE': 21.245099506209787,
                                'B-UNITS': 32.965172974228004,
                                'B-PK': 39.893509412756394,
                                'I-RANGE': 51.14589337175793,
                                'B-RANGE': 236.2412645590682,
                                'B-COMPARE': 1434.1515151515152,
                                'I-COMPARE': 6761.0}}

def test_validation_run():
