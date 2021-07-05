import os

import typer
from tqdm import tqdm
from pkrex.models.utils import align_tokens_and_annotations_bilou
from pkrex.utils import read_jsonl, write_jsonl
from transformers import BertTokenizerFast


def main(
        #
        # microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
        bert_model_name: str = typer.Option(default="dmis-lab/biobert-v1.1",
                                            help="Path to the input model"),
        prodigy_annotated_dir: str = typer.Option(default="data/annotations/P1/ready",
                                                  help="Path to the jsonl file of the annotated "
                                                       "dataset using prodigy.ner recipe"),

        output_dir: str = typer.Option(default="data/pubmedbert_tokenized/",
                                       help="Output path")
):
    files_to_transform = [x for x in os.listdir(prodigy_annotated_dir) if ".jsonl" in x]
    for prodigy_annotated_file in files_to_transform:
        inp_file_path = os.path.join(prodigy_annotated_dir, prodigy_annotated_file)
        prodigy_annotations = list(read_jsonl(inp_file_path))
        tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
        out_annotations = []
        for i, example in tqdm(enumerate(prodigy_annotations)):
            text = example['text']
            annotations = example['spans']
            tokenized_text = tokenizer(text, return_offsets_mapping=True)
            new_labels_bilou, new_labels_bio, entity_tokens = align_tokens_and_annotations_bilou(
                tokenized=tokenized_text[0],
                annotations=annotations,
                example=example
            )

            assert len(tokenized_text[0]) == len(new_labels_bilou) == len(new_labels_bio)
            # visualise_alignment(inp_tokens=tokenized_text[0].tokens, aligned_labels=new_labels_bio)

            sentence_ready = get_ready(bert_tokens=tokenized_text[0], bio_tags=new_labels_bio,
                                       bilou_tags=new_labels_bilou,
                                       bert_entity_tokens=entity_tokens, original_text=text)

            sentence_ready["metadata"] = example["metadata"] if "metadata" in example.keys() else dict()
            out_annotations.append(sentence_ready)

        out_path = os.path.join(output_dir, prodigy_annotated_file)
        print(f"Writing to {out_path}")
        write_jsonl(file_path=out_path, lines=out_annotations)


def get_ready(bert_tokens, bio_tags, bilou_tags, bert_entity_tokens, original_text):
    prodigy_sentence = dict(text=original_text)
    tokens = []
    for i, (tok_text, tok_id, ch_offsets) in enumerate(zip(bert_tokens.tokens, bert_tokens.ids, bert_tokens.offsets)):
        tokens.append(dict(text=tok_text, id=i, start=ch_offsets[0], end=ch_offsets[1], tokenizer_id=tok_id))
    prodigy_sentence["tokens"] = tokens
    prodigy_sentence["spans"] = bert_entity_tokens
    prodigy_sentence["bio_tags"] = bio_tags
    prodigy_sentence["bilou_tags"] = bilou_tags
    return prodigy_sentence


def visualise_alignment(inp_tokens, aligned_labels):
    for token, label in zip(inp_tokens, aligned_labels):
        print(token, "-", label)


if __name__ == '__main__':
    typer.run(main)
