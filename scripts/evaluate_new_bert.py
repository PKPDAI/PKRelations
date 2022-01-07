import os
from pathlib import Path
from typing import List, Dict
import typer
from sklearn.metrics import classification_report
from transformers import BertTokenizerFast
from pkrex.models.bertpkrex import load_pretrained_model, get_avg_ner_metrics
from pkrex.annotation_preproc import view_all_entities_terminal, clean_instance_span, visualize_relations_brat
from pkrex.utils import read_jsonl, print_ner_scores
from pkrex.models.utils import predict_pl_bert_rex, arrange_relationship
from nervaluate import Evaluator


def compute_micro_macro(inp_res, inp_types):
    valid_values = [v for v in inp_res.values() if v['support'] > 0]
    for metric_type in inp_types:
        print(f"{metric_type} match:")
        macro = sum([v[metric_type]['f1'] for v in valid_values]) / len(valid_values)
        total_support = sum(v['support'] for v in valid_values)
        micro = sum([v[metric_type]['f1'] * (v['support'] / total_support) for v in valid_values])
        print(f"Micro: {micro}")
        print(f"Macro: {macro}")


def main(
        model_checkpoint: Path = typer.Option(
            default="results/checkpoints/biobert-rex-new-128-clean-epoch=0010-cval_f1=0.90.ckpt",
            help="Path to the input model"),

        predict_file_path: Path = typer.Option(default="data/biobert_tokenized/test-all-reviewed.jsonl",
                                               help="Path to the jsonl file of the test/evaluation set"),

        display_errors: bool = typer.Option(default=True, help="Whether to display sentences with errors"),

        display_all: bool = typer.Option(default=False, help="Whether to display all sentences "),

        batch_size: int = typer.Option(default=32, help="Batch size"),

        gpu: bool = typer.Option(default=True, help="Whether to use GPU for inference"),

        n_workers: int = typer.Option(default=12, help="Number of workers to use for the dataloader"),

        debug: float = typer.Option(default=False)

):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # ============== 1. Load model and tokenizer ========================= #
    pl_model = load_pretrained_model(model_checkpoint_path=model_checkpoint, gpu=gpu)
    tokenizer = BertTokenizerFast.from_pretrained(pl_model.bert.name_or_path)

    # ============= 2. Load corpus  ============================ #
    predict_sentences = [x for x in read_jsonl(file_path=predict_file_path) if len(x['tokens']) <= pl_model.seq_len]
    if debug:
        predict_sentences = predict_sentences[0:50]
    true_entities = [clean_instance_span(x["spans"]) for x in predict_sentences]
    texts_to_predict = [sentence["text"] for sentence in predict_sentences]

    # ============= 4. Predict  ============================ #
    predicted_entities, predicted_rex = predict_pl_bert_rex(inp_texts=texts_to_predict, inp_model=pl_model,
                                                            inp_tokenizer=tokenizer, batch_size=batch_size,
                                                            n_workers=n_workers)

    predicted_entities_offsets = [clean_instance_span(x) for x in predicted_entities]

    # ============= 5. Evaluate NER  ============================ #
    all_ent_types = list(set([x['label'] for s in true_entities for x in s]))
    assert len(predicted_entities_offsets) == len(true_entities)
    evaluator = Evaluator(true_entities, predicted_entities_offsets, tags=all_ent_types)
    _, results_agg = evaluator.evaluate()
    print_ner_scores(inp_dict=results_agg, is_spacy=False)
    avg_metrics = get_avg_ner_metrics(results_agg, all_ent_types)[0]
    compute_micro_macro(avg_metrics, ['strict', 'partial'])

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

    # ============== 6. Evaluate REX ====================== #
    # 6.1. Do printing and HTML in BRAT (separate files)
    predicted_rex = predicted_rex
    true_rex = [x['relations'] for x in predict_sentences]
    assert len(true_rex) == len(predicted_rex)
    # 6.2. transform formats
    pred_rex_ready, true_rex_ready = transform_rex_formats(predicted_rex=predicted_rex, true_rex=true_rex)
    # 6.3 Get them ready for BRAT
    pred_brat_annot = []
    true_brat_annot = []
    for pred_relations, true_relations, original_annot in zip(pred_rex_ready, true_rex_ready, predict_sentences):
        pred_relations = [rel for rel in pred_relations if rel['label'] != "NO_RELATION"]
        true_relations = [rel for rel in true_relations if rel['label'] != "NO_RELATION"]
        pred_relations = sorted(pred_relations, key=lambda d: (d['head_span']['start'], d['child_span']['start']))
        true_relations = sorted(true_relations, key=lambda d: (d['head_span']['start'], d['child_span']['start']))
        if pred_relations != true_relations:
            pred_brat_annot.append(get_ready_brat(original_annot=original_annot, pred_relations=pred_relations))
            true_brat_annot.append(get_ready_brat(original_annot=original_annot, pred_relations=true_relations))
    print(f"Sentences with different predictions: {len(pred_brat_annot)} of {len(pred_rex_ready)}")
    visualize_relations_brat(inp_annotations=pred_brat_annot, file_path="brat/rex_pred.html")
    visualize_relations_brat(inp_annotations=true_brat_annot, file_path="brat/rex_true.html")
    # 6.4 Compute F1 scores
    all_rex_labels, all_rex_predictions = [], []
    pred_rex_ready = [encode_rels(x) for x in pred_rex_ready]
    true_rex_ready = [encode_rels(x) for x in true_rex_ready]
    for sample_pred, sample_true in zip(pred_rex_ready, true_rex_ready):
        tmp_pred_labels = []
        tmp_true_labels = []
        if sample_pred:
            # predicted relations
            if sample_true:
                # predicted and ground truth
                for predicted_rel_id in sample_pred.keys():
                    if predicted_rel_id in sample_true.keys():
                        tmp_true_labels.append(sample_true[predicted_rel_id])
                        tmp_pred_labels.append(sample_pred[predicted_rel_id])
                    else:
                        tmp_pred_labels.append(sample_pred[predicted_rel_id])
                        tmp_true_labels.append("NO_RELATION")
                # catch missing
                for annotated_rel_id in sample_true.keys():
                    if annotated_rel_id not in sample_pred.keys():
                        tmp_true_labels.append(sample_true[annotated_rel_id])
                        tmp_pred_labels.append("NO_RELATION")
            else:
                # predicted but not ground truth
                tmp_pred_labels = list(sample_pred.values())
                tmp_true_labels = ["NO_RELATION" for _ in tmp_pred_labels]

        else:
            # no predicted relations
            if sample_true:
                # no predicted but ground truth
                tmp_true_labels = list(sample_true.values())
                tmp_pred_labels = ["NO_RELATION" for _ in tmp_true_labels]

        all_rex_predictions += tmp_pred_labels
        all_rex_labels += tmp_true_labels
    clas_report = classification_report(y_true=all_rex_labels, y_pred=all_rex_predictions,
                                        zero_division=0)
    print(clas_report)


def get_ready_brat(original_annot, pred_relations):
    out_pr_rex = dict(text=original_annot['text'], metadata=original_annot['metadata'])
    pred_relations = [rel for rel in pred_relations if rel['label'] != "NO_RELATION"]
    pred_spans = get_spans_from_rex(rex_annotations=pred_relations)
    out_pr_rex['relations'] = pred_relations
    out_pr_rex['spans'] = pred_spans
    out_pr_rex['_task_hash'] = 1010
    return out_pr_rex


def get_spans_from_rex(rex_annotations: List[Dict]):
    out_spans = []
    for sps in rex_annotations:
        if sps['head_span'] not in out_spans:
            out_spans.append(sps['head_span'])
        if sps['child_span'] not in out_spans:
            out_spans.append(sps['child_span'])
    return out_spans


def transform_rex_formats(predicted_rex, true_rex):
    predictions_ready = []
    true_rex_ready = []
    for tmp_pred, tmp_true in zip(predicted_rex, true_rex):
        # True rex mod
        out_true = []
        if tmp_true:
            for rel_tmp_true in tmp_true:
                out_true.append(edit_true_rex(inp_true_rex=rel_tmp_true))
        true_rex_ready.append(out_true)
        # Pred rex mod
        out_pred = []
        if tmp_pred:
            for rel_tmp_pred in tmp_pred:
                out_pred.append(edit_pred_rex(inp_pred_rex=rel_tmp_pred))
        predictions_ready.append(out_pred)
    return predictions_ready, true_rex_ready


def edit_pred_rex(inp_pred_rex):
    left_ent = keep_only_rel(inp_pred_rex['left'])
    right_ent = keep_only_rel(inp_pred_rex['right'])
    out_rel = dict(head_span=left_ent, child_span=right_ent, label=inp_pred_rex['label'])
    return out_rel


def edit_true_rex(inp_true_rex):
    left_ent, right_ent = arrange_relationship(inp_rel=inp_true_rex)
    left_ent = keep_only_rel(left_ent)
    right_ent = keep_only_rel(right_ent)
    out_rel = dict(head_span=left_ent, child_span=right_ent, label=inp_true_rex['label'])
    return out_rel


def keep_only_rel(inp_ent):
    return dict(start=inp_ent['start'], end=inp_ent['end'], label=inp_ent['label'])


def encode_rels(inp_rels):
    out_encoded = dict()
    for rel in inp_rels:
        rel_identifier = int(
            str(rel['child_span']['end']) + str(rel['child_span']['start']) + str(rel['head_span']['end']) + str(
                rel['head_span']['start']))
        out_encoded[rel_identifier] = rel['label']
    return out_encoded


if __name__ == "__main__":
    typer.run(main)
