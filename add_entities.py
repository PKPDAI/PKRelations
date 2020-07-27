import spacy
from prodigy.util import read_jsonl, write_jsonl
import os
from tqdm import tqdm
import requests


def tag_them(sentences, entities, nlp, use_bern=True):
    sentences_out = []
    for sentence in tqdm(sentences):
        sentence["spans"] = []
        if not use_bern:
            # Use Scispacy
            doc = nlp(sentence["text"])
            for entity in doc.ents:
                if entity.label_ in entities:
                    sentence["spans"].append(dict(start=entity.start_char, end=entity.end_char, label=entity.label_))
        else:
            # Use BERN
            mapper = dict(drug='CHEMICAL', disease='DISEASE', species='SPECIES')
            try:
                result = tag_biobert(sentence["text"])
                for i, entity in enumerate(result["denotations"]):
                    if entity['obj'] in ['drug', 'species', 'disease']:
                        sentence["spans"].append(
                            dict(start=entity['span']['begin'], end=entity['span']['end'], label=mapper[entity['obj']],
                                 ids=entity['id']))
            except:
                # If BERN didn't work, use Scispacy
                doc = nlp(sentence["text"])
                for entity in doc.ents:
                    if entity.label_ in entities:
                        sentence["spans"].append(
                            dict(start=entity.start_char, end=entity.end_char, label=entity.label_))
        sentences_out.append(sentence)
    return sentences_out


def tag_biobert(text, url="https://bern.korea.ac.kr/plain"):
    return requests.post(url, data={'sample_text': text}).json()


if __name__ == '__main__':
    base_dir = os.path.join("data", "all_sentences", "selected", "nocontext", "ready")
    file_train = "training.jsonl"
    out_path = os.path.join(base_dir, "training_tagged.jsonl")

    # Load model and sentences
    train_sentences = list(read_jsonl(os.path.join(base_dir, file_train)))
    nlp_drugs = spacy.load("en_ner_bc5cdr_md")
    out_sentences = tag_them(sentences=train_sentences, entities=["CHEMICAL", "DISEASE", "SPECIES"], nlp=nlp_drugs,
                             use_bern=True)
