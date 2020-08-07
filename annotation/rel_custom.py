from prodigy.util import get_labels, split_string
from typing import List, Optional, Union, Iterable, Dict, Any
import prodigy
from prodigy.recipes.rel import manual as rel_manual


@prodigy.recipe(
    "custom.rel.manual",
    dataset=("Dataset to save annotations to", "positional", None, str),
    spacy_model=("Loadable spaCy model or blank:lang (e.g. blank:en)", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    label=("Comma-separated relation label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    span_label=("Comma-separated span label(s) to annotate or text file with one label per line", "option", "sl", get_labels),
    patterns=("Patterns file for defining custom spans to be added", "option", "pt", str),
    disable_patterns=("Patterns file for defining tokens to disable (make unselectable)", "option", "dpt", str),
    add_ents=("Add entities predicted by the model", "flag", "AE", bool),
    add_nps=("Add noun phrases (if noun chunks rules are available), based on tagger and parser", "flag", "AN"),
    wrap=("Wrap lines in the UI by default (instead of showing tokens in one row)", "flag", "W", bool),
    exclude=("Comma-separated list of dataset IDs whose annotations to exclude", "option", "e", split_string),
    hide_arrow_heads=("Hide the arrow heads visually", "option", "HA", bool),
)
def custom_rel_manual(dataset: str,
                      spacy_model: str,
                      source: Union[str, Iterable[dict]] = "-",
                      loader: Optional[str] = None,
                      label: Optional[List[str]] = None,
                      span_label: Optional[List[str]] = None,
                      exclude: Optional[List[str]] = None,
                      patterns: Optional[Union[str, List]] = None,
                      disable_patterns: Optional[Union[str, List]] = None,
                      add_ents: bool = False,
                      add_nps: bool = False,
                      wrap: bool = False,
                      hide_arrow_heads: bool = False):
    components = rel_manual(
        dataset=dataset,
        spacy_model=spacy_model,
        source=source,
        loader=loader,
        label=label,
        span_label=span_label,
        exclude=exclude,
        patterns=patterns,
        disable_patterns=disable_patterns,
        add_ents=add_ents,
        add_nps=add_nps,
        wrap=wrap,
        hide_arrow_heads=hide_arrow_heads, )
    # Add callback to the components returned by the recipe
    components["validate_answer"] = validate_answer
    return components


def validate_answer(eg):
    for relation in eg['relations']:
        head_label = relation['head_span']['label']
        child_label = relation['child_span']['label']
        rel_type = relation['label']
        assert (head_label, child_label) in mapit[rel_type]

    selected = eg.get("accept", [])
    print(eg['relations'])
    # assert 1 <= len(selected) <= 3, "Select at least 1 but not more than 3 options."
    print("HIIIIIIIIII")





mapit = {
    "C_VAL": [("PK", "VALUE")],
    "C_MIN": [("PK", "VALUE")],
    "C_MAX": [("PK", "VALUE")],
    "D_VAL": [("VALUE", "VALUE")],
    "D_MIN": [("VALUE", "VALUE")],
    "D_MAX": [("VALUE", "VALUE")],
    "DOSAGE": [("VALUE", "VALUE")],
    "COMPLEMENT": [("COVARIATES", "COVARIATES")],
    "RELATED": [("TYPE_MEAS", "VALUE"), ("COMPARATIVE", "VALUE"), ("SPECIES", "VALUE"), ("CHEMICAL", "VALUE"),
                ("DISEASES", "VALUE"), ("UNITS", "VALUE")]
}