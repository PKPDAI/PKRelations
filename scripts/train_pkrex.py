import json
import pytorch_lightning as pl
import torch
from pytorch_lightning.core.decorators import auto_move_data
from transformers import BertTokenizerFast
from pkrex.models.bertpkrex import BertPKREX
from pkrex.models.dataloaders import get_training_dataloader, get_val_dataloader, get_merged_dataloader
from pkrex.models.utils import empty_cuda_cache, get_tensorboard_logger
import typer
from pathlib import Path
from datetime import datetime
import os


def main(
        training_file_path: Path = typer.Option(default="data/pk-bert-tokenized/train-augmented.jsonl",
                                                help="Path to the jsonl file with the training data"),

        val_file_path: Path = typer.Option(default="data/pk-bert-tokenized/test.jsonl",
                                           help="Path to the jsonl file with the development data"),

        output_dir: Path = typer.Option(default="results/",
                                        help="Output directory"),
        model_config_file: Path = typer.Option(default="configs/config-biobert.json"),
        print_tokens: bool = typer.Option(default=False),

        debug_mode: bool = typer.Option(default=False)

):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    with open(model_config_file) as cf:
        config = json.load(cf)

    limit_train_batches = 1.
    limit_val_batches = 1.
    if debug_mode:
        config["n_workers_dataloader"] = 0
        config["gpus"] = False
        limit_train_batches = 0.05
        limit_val_batches = 1.
    print(f"Debug mode is {str(debug_mode)}")

    assert config['tag_type'] in ["bio", "biluo"]

    config['training_file_path'] = str(training_file_path)
    config['val_file_path'] = str(val_file_path)
    config['output_dir'] = str(output_dir)
    current_time = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
    out_config_path = os.path.join(config['output_dir'], "configs", f"config-{config['run_name']}-{current_time}.json")
    if not os.path.exists(os.path.join(config['output_dir'], "configs")):
        os.makedirs(os.path.join(config['output_dir'], "configs"))

    with open(out_config_path, 'w') as fp:
        json.dump(config, fp, indent=3)

    # ============ 1. Seed everything =============== #
    pl.seed_everything(config['seed'])
    # ============ 2. Load tokenizer and make =========== #
    tokenizer = BertTokenizerFast.from_pretrained(config['base_model'])

    # ============ 3. Get data loaders and tag-id converters =============== #
    print("=========== Constructing DataLoaders ===========")
    if not config["final_train"]:
        train_dataloader, tag2id, id2tag, scaling_dict = get_training_dataloader(
            training_data_file=config['training_file_path'],
            tokenizer=tokenizer,
            max_len=config['max_length'],
            batch_size=config['batch_size'],
            n_workers=config['n_workers_dataloader'],
            tag_type=config['tag_type'],
            print_tokens=print_tokens,
            rmls=config["remove_longer_seqs"]
        )
    else:
        to_print = 20 * "=" + " Doing final train with training + validation dataset ".upper() + 20 * "="
        print("\n", len(to_print) * "=", "\n", to_print, "\n", len(to_print) * "=")
        train_dataloader, tag2id, id2tag, scaling_dict = get_merged_dataloader(
            files_to_merge=[config['training_file_path'], config['val_file_path']],
            tokenizer=tokenizer,
            max_len=config['max_length'],
            batch_size=config['batch_size'],
            n_workers=config['n_workers_dataloader'],
            tag_type=config['tag_type'],
            print_tokens=print_tokens,
            dataset_name="training + validation",
            rmls=config["remove_longer_seqs"]
        )

    val_dataloader = get_val_dataloader(
        val_data_file=config['val_file_path'],
        tokenizer=tokenizer,
        max_len=config['max_length'],
        batch_size=config['val_batch_size'],
        n_workers=config['n_workers_dataloader'],
        tag_type=config['tag_type'],
        print_tokens=print_tokens,
        tag2id=tag2id,
        dataset_name="validation",
        remove_longer_seqs=config["remove_longer_seqs"]
    )

    config["scaling_dict"] = scaling_dict

    total_training_steps = len(train_dataloader) * config["max_epochs"]
    # ============ 4. Get model =============== #
    BertPKREX.forward = auto_move_data(BertPKREX.forward)  # auto move data to the correct device
    BertPKREX.predict_entities = auto_move_data(BertPKREX.predict_entities)
    BertPKREX.predict_relations = auto_move_data(BertPKREX.predict_relations)
    model = BertPKREX(config=config, id2tag=id2tag, n_training_steps=total_training_steps)

    # ============ 5. Define trainer =============== #

    tensorboard_logger = get_tensorboard_logger(log_dir=os.path.join(config['output_dir'], "logs"),
                                                run_name=config['run_name'])

    gpus = 0
    if config['gpus']:
        gpus = torch.cuda.device_count()
        empty_cuda_cache(gpus)

    if not config["final_train"]:

        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='cval_f1',
                                                           dirpath=os.path.join(config['output_dir'], 'checkpoints'),
                                                           filename=config['run_name'] + '-{epoch:04d}-{'
                                                                                         'cval_f1:.2f}',
                                                           save_top_k=2,
                                                           mode='max'
                                                           )
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=config['max_epochs'],
            logger=tensorboard_logger,
            callbacks=[checkpoint_callback],
            deterministic=True,
            log_every_n_steps=1,
            gradient_clip_val=config["gradient_clip_val"],
            stochastic_weight_avg=config["stochastic_weight_avg"],
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches
        )
    else:
        to_print = 20 * "=" + " Doing final train with training + validation dataset ".upper() + 20 * "="
        print("\n", len(to_print) * "=", "\n", to_print, "\n", len(to_print) * "=")

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(config['output_dir'], 'checkpoints'),
            filename=config['run_name'] + 'train_dev',
            save_last=True
        )
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=config['max_epochs'],
            logger=tensorboard_logger,
            callbacks=[checkpoint_callback],
            deterministic=True,
            log_every_n_steps=1,
            gradient_clip_val=config["gradient_clip_val"],
            limit_val_batches=0,  # do not validate
            limit_train_batches=limit_train_batches
        )

    if config['early_stopping']:
        early_stop = pl.callbacks.EarlyStopping(monitor='val_loss', patience=config['early_stopping_patience'],
                                                strict=False, verbose=True, mode='min')
        trainer.callbacks += [early_stop]

    # ============ 6. Train model =============== #
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':
    typer.run(main)
