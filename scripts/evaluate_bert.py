import os
from pathlib import Path
import typer
from transformers import BertTokenizerFast
from pkrex.models.bertpkrex import load_pretrained_model
from pkrex.annotation_preproc import view_all_entities_terminal, clean_instance_span
from pkrex.utils import read_jsonl, print_ner_scores

from pkrex.models.utils import predict_pl_bert_ner
from nervaluate import Evaluator


def main(
        model_checkpoint: Path = typer.Option(
            default="results/checkpoints/first-train-all-128-epoch=0004-val_pk_value_strict=0.87.ckpt",
            help="Path to the input model"),

        predict_file_path: Path = typer.Option(default="data/pubmedbert_tokenized/test-all-ready-fixed-5.jsonl",
                                               help="Path to the jsonl file of the test/evaluation set"),

        display_errors: bool = typer.Option(default=True, help="Whether to display sentences with errors"),

        display_all: bool = typer.Option(default=False, help="Whether to display all sentences "),

        batch_size: int = typer.Option(default=256, help="Batch size"),

        gpu: bool = typer.Option(default=True, help="Whether to use GPU for inference"),

        n_workers: int = typer.Option(default=12, help="Number of workers to use for the dataloader"),

):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # ============== 1. Load model and tokenizer ========================= #
    # with open(model_config_file) as cf:
    #     config = json.load(cf)
    # id2tag = {0: 'B-PK', 1: 'I-PK', 2: 'O', -100: 'PAD'}
    pl_model = load_pretrained_model(model_checkpoint_path=model_checkpoint, gpu=gpu)
    tokenizer = BertTokenizerFast.from_pretrained(pl_model.bert.name_or_path)

    # ============= 2. Load corpus  ============================ #
    predict_sentences = list(read_jsonl(file_path=predict_file_path))
    true_entities = [clean_instance_span(x["spans"]) for x in predict_sentences]
    texts_to_predict = [sentence["text"] for sentence in predict_sentences]

    # ============= 4. Predict  ============================ #
    predicted_entities = predict_pl_bert_ner(inp_texts=texts_to_predict, inp_model=pl_model, inp_tokenizer=tokenizer,
                                             batch_size=batch_size, n_workers=n_workers)

    predicted_entities_offsets = [clean_instance_span(x) for x in predicted_entities]

    # ============= 5. Evaluate  ============================ #
    all_ent_types = list(set([x['label'] for s in true_entities for x in s]))
    assert len(predicted_entities_offsets) == len(true_entities)
    evaluator = Evaluator(true_entities, predicted_entities_offsets, tags=all_ent_types)
    _, results_agg = evaluator.evaluate()

    # print(results_agg)
    print_ner_scores(inp_dict=results_agg, is_spacy=False)
    if display_errors or display_all:
        i = 0
        for instance, predicted_ent, true_ent in zip(predict_sentences, predicted_entities_offsets, true_entities):
            sentence_text = instance["text"]
            predicted_ent = sorted(predicted_ent, key=lambda k: (k['start'], k['end'], k['label']))
            true_ent = sorted(true_ent, key=lambda k: (k['start'], k['end'], k['label']))
            if display_all or (predicted_ent != true_ent):
                instance["_task_hash"] = 8888 if "_task_hash" not in instance.keys() else instance["_task_hash"]
                print(10 * "=", f"Example with task hash {instance['_task_hash']} n={i}", 10 * "=")
                print("REAL LABELS:")
                print(view_all_entities_terminal(inp_text=sentence_text, character_annotations=true_ent))
                print("MODEL PREDICTIONS:")
                print(view_all_entities_terminal(inp_text=sentence_text, character_annotations=predicted_ent))


if __name__ == "__main__":
    typer.run(main)
