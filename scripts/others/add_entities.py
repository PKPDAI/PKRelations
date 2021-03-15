import spacy
from prodigy.util import read_jsonl
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


def tag_additional(sentences, nlp_model):
    texts = [sentence['text'] for sentence in sentences]
    docs = list(nlp_model.pipe(texts))
    assert len(texts) == len(docs)
    for sentence, doc in zip(sentences, docs):
        a = 1

    return sentences


if __name__ == '__main__':
    base_dir = os.path.join("../../data", "raw", "selected", "nocontext", "ready")
    #   file_train = "training.jsonl"
    #   file_test = "test.jsonl"
    out_path1 = os.path.join(base_dir, "training_tagged.jsonl")
    out_path2 = os.path.join(base_dir, "test_tagged.jsonl")
    # Load model and sentences
    nlp_extra = spacy.load(os.path.join("../../data", "pk_ner_supertok"))
    #   train_sentences = list(read_jsonl(os.path.join(base_dir, file_train)))
    train_sentences = list(read_jsonl(out_path1))
    #   nlp_drugs = spacy.load("en_ner_bc5cdr_md")
    #   out_sentences1 = tag_them(sentences=train_sentences, entities=["CHEMICAL", "DISEASE", "SPECIES"], nlp=nlp_drugs,
    #                             use_bern=True)
    out_sentences1 = tag_additional(train_sentences, nlp_extra)
    #  write_jsonl(out_path1, out_sentences1)

    #   test_sentences = list(read_jsonl(os.path.join(base_dir, file_test)))
    test_sentences = list(read_jsonl(out_path2))
    # out_sentences2 = tag_them(sentences=test_sentences, entities=["CHEMICAL", "DISEASE", "SPECIES"], nlp=nlp_drugs,
    #                          use_bern=True)
    out_sentences2 = tag_additional(test_sentences, nlp_extra)
# write_jsonl(out_path2, out_sentences2)
