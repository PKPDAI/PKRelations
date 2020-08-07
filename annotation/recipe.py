import prodigy
from prodigy.components.loaders import JSONL
from prodigy.util import combine_models, split_string
from typing import List, Optional
from prodigy.components.preprocess import add_tokens
import spacy 

@prodigy.recipe(
    "myrel",
    dataset=("The dataset to save to", "positional", None, str),
    spacy_model=("The base model", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    label=("One or more comma-separated labels", "option", "l", split_string),
    wrap=("optional wrapping", "flag", "w", bool),
    span_label=("One or more comma-separated span-labels", "option", "sl", split_string),
    patterns=("Optional match patterns", "option", "p", str),
    disable_patterns=("The disable patterns as a JSONL file", "option", "dpt", str),
    add_ents=("Add entities", "flag", "R", bool),

)
def myrel(
    dataset: str,
    spacy_model: str,
    source: str,
    label: Optional[List[str]] = None,
    wrap: bool = False,
    span_label: Optional[List[str]] = None,
    patterns: Optional[str] = None,
    disable_patterns: Optional[str] = None,
    add_ents: bool = False,
  
    ):
    
    stream = JSONL(source)
    nlp = spacy.load(spacy_model)
    stream = add_tokens(nlp, stream)
    

    return {
        "dataset": dataset,   
        "view_id": "relations",  
        "stream": stream,
        "config": {  
            "lang": nlp.lang,
            "labels": label,  
        }
    }

