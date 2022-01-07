import os
import torch
import pytorch_lightning as pl
from pkrex.models.dataloaders import get_training_dataloader, get_val_dataloader
from pkrex.models.utils import empty_cuda_cache, get_tensorboard_logger
from transformers import BertTokenizerFast
from pkrex.models.bertpkrex import BertPKREX
from pytorch_lightning.core.decorators import auto_move_data

BASE_CONFIG = {'run_name': 'pkner-rex-augmented-128-clean',
               'base_model': 'data/models/pk-bert',
               'n_workers_dataloader': 12,
               'gpus': True,
               'seed': 1,
               'learning_rate': 2e-05,
               'max_epochs': 2,
               'max_length': 128,
               'batch_size': 2,
               'val_batch_size': 2,
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
               'training_file_path': 'data/pk-bert-tokenized/train-all-reviewed.jsonl',
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def test_full_validation_run():
    gpus = 0
    if BASE_CONFIG['gpus']:
        gpus = torch.cuda.device_count()
        empty_cuda_cache(gpus)

    tensorboard_logger = get_tensorboard_logger(log_dir=os.path.join(BASE_CONFIG['output_dir'], "logs"),
                                                run_name=BASE_CONFIG['run_name'])

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='cval_f1',
                                                       dirpath=os.path.join(BASE_CONFIG['output_dir'], 'checkpoints'),
                                                       filename=BASE_CONFIG['run_name'] + '-{epoch:04d}-{'
                                                                                          'cval_f1:.2f}',
                                                       save_top_k=2,
                                                       mode='max'
                                                       )
    tokenizer = BertTokenizerFast.from_pretrained(BASE_CONFIG['base_model'])
    train_dataloader, tag2id, id2tag, scaling_dict = get_training_dataloader(
        training_data_file=BASE_CONFIG['training_file_path'],
        tokenizer=tokenizer,
        max_len=BASE_CONFIG['max_length'],
        batch_size=BASE_CONFIG['batch_size'],
        n_workers=BASE_CONFIG['n_workers_dataloader'],
        tag_type=BASE_CONFIG['tag_type'],
        print_tokens=False,
        rmls=BASE_CONFIG["remove_longer_seqs"]
    )
    val_dataloader = get_val_dataloader(
        val_data_file=BASE_CONFIG['val_file_path'],
        tokenizer=tokenizer,
        max_len=BASE_CONFIG['max_length'],
        batch_size=BASE_CONFIG['val_batch_size'],
        n_workers=BASE_CONFIG['n_workers_dataloader'],
        tag_type=BASE_CONFIG['tag_type'],
        print_tokens=False,
        tag2id=tag2id,
        dataset_name="validation",
        remove_longer_seqs=BASE_CONFIG["remove_longer_seqs"]
    )

    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=BASE_CONFIG['max_epochs'],
        logger=tensorboard_logger,
        callbacks=[checkpoint_callback],
        deterministic=True,
        log_every_n_steps=1,
        gradient_clip_val=BASE_CONFIG["gradient_clip_val"],
        stochastic_weight_avg=BASE_CONFIG["stochastic_weight_avg"],
        limit_train_batches=0.01,
        limit_val_batches=1.
    )

    total_training_steps = len(train_dataloader) * BASE_CONFIG["max_epochs"]
    # ============ 4. Get model =============== #
    BertPKREX.forward = auto_move_data(BertPKREX.forward)  # auto move data to the correct device
    BertPKREX.predict_entities = auto_move_data(BertPKREX.predict_entities)
    BertPKREX.predict_relations = auto_move_data(BertPKREX.predict_relations)
    model = BertPKREX(config=BASE_CONFIG, id2tag=id2tag, n_training_steps=total_training_steps)

    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

