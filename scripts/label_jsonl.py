import os
from pathlib import Path
import typer
from transformers import BertTokenizerFast
from pkrex.models.bertpkrex import load_pretrained_model
from pkrex.annotation_preproc import clean_instance_span
from pkrex.utils import read_jsonl, write_jsonl
from pkrex.models.utils import predict_pl_bert_rex
import spacy
from tqdm import tqdm


def main(
        model_checkpoint: Path = typer.Option(
            default="results/checkpoints/rex-augmented-128-clean-epoch=0017-cval_f1=0.87.ckpt",
            help="Path to the input model"),
        predict_file_path: Path = typer.Option(default="data/annotations/P1/to_annotate/dev200-500.jsonl",
                                               help="Path to the jsonl file to predict relations from"),
        out_file_path: Path = typer.Option(default="data/annotations/P1/to_annotate/dev200-500-preannotated.jsonl",
                                           help="output file path"),
        batch_size: int = typer.Option(default=16, help="Batch size"),

        gpu: bool = typer.Option(default=True, help="Whether to use GPU for inference"),

        n_workers: int = typer.Option(default=12, help="Number of workers to use for the dataloader"),
        debug: float = typer.Option(default=False)
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if debug:
        gpu = False
        n_workers = 1

    pl_model = load_pretrained_model(model_checkpoint_path=model_checkpoint, gpu=gpu)
    tokenizer = BertTokenizerFast.from_pretrained(pl_model.bert.name_or_path)

    # ============= 2. Load corpus  ============================ #
    predict_sentences = [x for x in read_jsonl(file_path=predict_file_path)]
    if debug:
        predict_sentences = predict_sentences[0:32]

    texts_to_predict = [sentence["text"] for sentence in predict_sentences]
    predicted_entities, predicted_rex = predict_pl_bert_rex(inp_texts=texts_to_predict, inp_model=pl_model,
                                                            inp_tokenizer=tokenizer, batch_size=batch_size,
                                                            n_workers=n_workers)
    predicted_entities_offsets = [clean_instance_span(x) for x in predicted_entities]

    assert len(predicted_entities_offsets) == len(predicted_rex) == len(predict_sentences)
    out_annotations = []
    for orig_sent, tmp_ents, tmp_rex, in tqdm(zip(predict_sentences, predicted_entities_offsets, predicted_rex)):
        rels = []
        ents = []
        if tmp_rex:
            ents, rels = clean_rels(inp_rels=tmp_rex, inp_text=orig_sent['text'])
        out_s = dict(metadata=orig_sent['metadata'],
                     text=orig_sent['text'],
                     spans=ents,
                     relations=rels,
                     meta=orig_sent['meta'],
                     sentence_hash=orig_sent['sentence_hash'])

        out_annotations.append(out_s)
    write_jsonl(file_path=out_file_path, lines=out_annotations)

    a = 1


def clean_rels(inp_rels, inp_text):
    out_spans = []
    out_rels = []
    spacy_nlp = spacy.load("data/models/tokenizers/super-tokenizer")
    doc = spacy_nlp(inp_text)
    tok_ofs = [(tok.idx, tok.idx + len(tok)) for tok in doc]
    for r in inp_rels:
        if r['label'] != "NO_RELATION":
            tmp_r = dict(label=r['label'])
            lspan = dict(start=r['left']['start'], end=r['left']['end'], label=r['left']['label'])
            rspan = dict(start=r['right']['start'], end=r['right']['end'], label=r['right']['label'])
            if lspan not in out_spans:
                out_spans.append(lspan)
            if rspan not in out_spans:
                out_spans.append(rspan)
            head, child = sort_sp_order(left_sp=lspan, right_sp=rspan, label=r['label'])
            # tmp_r['head'] = 1
            # tmp_r['child'] = 2
            token_id_head = span_to_tok_start(inp_spacy_tok_offsets=tok_ofs, inp_span=head)
            token_id_child = span_to_tok_start(inp_spacy_tok_offsets=tok_ofs, inp_span=child)
            if token_id_head and token_id_child:
                tmp_r['head'] = token_id_head
                tmp_r['child'] = token_id_child
                tmp_r['head_span'] = head
                tmp_r['child_span'] = child
            out_rels.append(tmp_r)

    return out_spans, out_rels


def span_to_tok_start(inp_spacy_tok_offsets, inp_span):
    for i, (s, _) in enumerate(inp_spacy_tok_offsets):
        if s == inp_span['start']:
            return i
    return None


def sort_sp_order(left_sp, right_sp, label):
    if label == "C_VAL":
        if left_sp['label'] == "PK":
            return left_sp, right_sp
        else:
            print(f"carefuk=l, left: {left_sp}, right: {right_sp} label: {label}")
            # assert right_sp['label'] == "PK"
            return right_sp, left_sp
    if label == "RELATED":
        if right_sp['label'] in ["VALUE", "RANGE"]:
            return left_sp, right_sp
        else:
            assert left_sp['label'] in ["VALUE", "RANGE"]
            return right_sp, left_sp
    if label == "D_VAL":
        return right_sp, left_sp

    raise ValueError(f"Relation with weird label {label}")


if __name__ == "__main__":
    typer.run(main)
