from typing import Dict, List
import requests
from spacy.language import Language
import json


def request_bern(text, url="https://bern.korea.ac.kr/plain"):
    req = requests.post(url, data={'sample_text': text}, timeout=120)
    return req.json()


def get_drugs_diseases_species(inp_sentence: str, inp_entities: List, inp_model: Language,
                               use_local_bern: bool) -> List[Dict]:
    mapper = dict(drug='CHEMICAL', disease='DISEASE', species='SPECIES', gene='GENE', mutation='MUTATION')
    bern_entities = ['drug', 'species', 'disease', 'gene', 'mutation']
    out_spans = []
    if isinstance(inp_sentence, str) and inp_sentence != "":
        try:
            # result = request_bern(inp_sentence)
            if use_local_bern:
                body_data = {'param': json.dumps({"text": inp_sentence})}
                req = requests.post("http://0.0.0.0:8888", data=body_data, timeout=120)
            else:
                req = requests.post("https://bern.korea.ac.kr/plain", data={'sample_text': inp_sentence}, timeout=120)

            if req.text:
                try:
                    result = req.json()
                # ...
                except ValueError:
                    result = {"denotations": []}
            else:
                result = {"denotations": []}
            # no JSON returne

            print("Using BERN...")
            for i, entity in enumerate(result["denotations"]):
                if entity['obj'] in bern_entities:
                    out_spans.append(dict(start=entity['span']['begin'], end=entity['span']['end'],
                                          label=mapper[entity['obj']], ids=entity['id']))
        except ValueError:
            print("Using scispaCy")
            # If BERN didn't work, use scispaCy
            doc = inp_model(inp_sentence)
            for entity in doc.ents:
                if entity.label_ in inp_entities:
                    out_spans.append(dict(start=entity.start_char, end=entity.end_char, label=entity.label_))

    return out_spans
