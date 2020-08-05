import spacy
from typing import List
from spacy.lang import char_classes
from spacy.symbols import ORTH
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
from scispacy.consts import ABBREVIATIONS
import re
from spacy.tokens import Span


def combined_rule_prefixes() -> List[str]:
    """Helper function that returns the prefix pattern for the tokenizer.
       It is a helper function to accomodate spacy tests that only test
       prefixes.
    """
    # add lookahead assertions for brackets (may not work properly for unbalanced brackets)
    prefix_punct = char_classes.PUNCT.replace("|", " ")
    #   prefix_punct = prefix_punct.replace(r"\(", r"\((?![^\(\s]+\)\S+)")
    #   prefix_punct = prefix_punct.replace(r"\[", r"\[(?![^\[\s]+\]\S+)")
    #   prefix_punct = prefix_punct.replace(r"\{", r"\{(?![^\{\s]+\}\S+)")

    prefixes = (
            ["(", "/", ")", "[", "]",
                       "§",
             "%",
             "=",
             r"\+",
             r"[^a-zA-Z\d\s:]"]  # all non alphanum
            + char_classes.split_chars(prefix_punct)
            + char_classes.LIST_ELLIPSES
            + char_classes.LIST_QUOTES
            + char_classes.LIST_CURRENCY
            + char_classes.LIST_ICONS
    )
    return prefixes


def add_special_cases(inp_list=["i.p.", "i.v.", "i.p.", "i.n.", "i.a.", "i.m.", "i.c.v.", "i.t.", "i.d.",
                                "u.i.d.",
                                "b.i.d.", "b.w.",
                                "t.i.d.",
                                "q.i.d.", "q.h.s.", "q.d.", "q.t.t.", "q.d.", "q.w.",
                                "o.d.", "o.s.", "o.u.",
                                "a.c.",
                                "p.o.", "p.c.", "p.v.", "p.i.",
                                "s.l.", "s.c.", "s.d.",
                                "t.i.d.",
                                "H.p",
                                "d.w."], nlp_model=None):
    for exception in inp_list:
        raw_exception = exception
        special_case = [{ORTH: exception}]
        nlp_model.tokenizer.add_special_case(raw_exception, special_case)
        upper_exception = exception.upper()
        special_case = [{ORTH: upper_exception}]
        nlp_model.tokenizer.add_special_case(upper_exception, special_case)
    return nlp_model


def combined_rule_infixes():
    # remove the first hyphen to prevent tokenization of the normal hyphen
    hyphens = char_classes.HYPHENS.replace("-|", "", 1)
    infixes = (
            char_classes.LIST_ELLIPSES
            + char_classes.LIST_ICONS
            + ["[", "]",
               r"(?<=[0-9])([a-zA-Z]+|[^a-zA-Z\d\s:,\.]+)",  # digit + non digit
               r"×",  # added this special x character to tokenize it separately
               # r"(?<=[0-9])[+\-\*^](?=[0-9-])", ######################################## OWN Modification
               r"(?<=[0-9])[+\-\*^](?=[0-9-])",
               r"(?<=[{al}])\.(?=[{au}])".format(
                   al=char_classes.ALPHA_LOWER, au=char_classes.ALPHA_UPPER
               ),
               r"[^a-zA-Z\d\s:\.,]",  # All non alphanumerics are infixes
               r"(?<=[0-9])[a-zA-Z+]([^a-zA-Z\d\s:])",  # Separate digit + alpha + non-alphanum
               r"(?<=[{a}]),(?=[{a}])".format(a=char_classes.ALPHA),
               r'(?<=[{a}])[?";:=,.]*(?:{h})(?=[{a}])'.format(
                   a=char_classes.ALPHA, h=hyphens
               ),
               # removed / to prevent tokenization of /
               r'(?<=[{a}"])[:<>=](?=[{a}])'.format(a=char_classes.ALPHA),
               ]
    )
    return infixes


def combined_rule_suffixes():
    # add the last apostrophe
    quotes = char_classes.LIST_QUOTES.copy() + ["’"]

    # add lookbehind assertions for brackets (may not work properly for unbalanced brackets)
    suffix_punct = char_classes.PUNCT.replace("|", " ")
    # These lookbehinds are commented out because they are variable width lookbehinds, and as of spacy 2.1,
    # spacy uses the re package instead of the regex package. The re package does not support variable width
    # lookbehinds. Hacking spacy internals to allow us to use the regex package is doable, but would require
    # creating our own instance of the language class, with our own Tokenizer class, with the from_bytes method
    # using the regex package instead of the re package
    # suffix_punct = suffix_punct.replace(r"\)", r"(?<!\S+\([^\)\s]+)\)")
    # suffix_punct = suffix_punct.replace(r"\]", r"(?<!\S+\[[^\]\s]+)\]")
    # suffix_punct = suffix_punct.replace(r"\}", r"(?<!\S+\{[^\}\s]+)\}")

    suffixes = (
            char_classes.split_chars(suffix_punct)
            + char_classes.LIST_ELLIPSES
            + quotes
            + char_classes.LIST_ICONS
            + ["'s", "'S", "’s", "’S", "’s", "’S"]
            + [
                "[", "]",
                ",",
                r"[^a-zA-Z\d\s:]",  # all non-alphanum
                r"(?<=[0-9])([a-zA-Z]+|[^a-zA-Z\d\s:])",
                # r"(?<=[0-9])\D+", # digit + any non digit (handling unit separation)
                r"(?<=[0-9])\+",
                r"(?<=°[FfCcKk])\.",
                r"(?<=[0-9])(?:{})".format(char_classes.CURRENCY),
                # this is another place where we used a variable width lookbehind
                # so now things like 'H3g' will be tokenized as ['H3', 'g']
                # previously the lookbehind was (^[0-9]+)
                r"(?<=[0-9])(?:{u})".format(u=char_classes.UNITS),
                r"(?<=[0-9{}{}(?:{})])\.".format(
                    char_classes.ALPHA_LOWER, r"%²\-\)\]\+", "|".join(quotes)
                ),
                # add |\d to split off the period of a sentence that ends with 1D.
                r"(?<=[{a}|\d][{a}])\.".format(a=char_classes.ALPHA_UPPER),
            ]
    )

    return suffixes


def combined_rule_tokenizer():
    """Creates a custom tokenizer on top of spaCy's default tokenizer. The
       intended use of this function is to replace the tokenizer in a spaCy
       pipeline like so:
            nlp = spacy.load("some_spacy_model")
            nlp.tokenizer = combined_rule_tokenizer(nlp)
       @param nlp: a loaded spaCy model
    """

    infixes = combined_rule_infixes()

    prefixes = combined_rule_prefixes()

    suffixes = combined_rule_suffixes()

    infix_re = compile_infix_regex(infixes)
    prefix_re = compile_prefix_regex(prefixes)
    suffix_re = compile_suffix_regex(suffixes)

    # Update exclusions to include these abbreviations so the period is not split off
    exclusions = {
        abbreviation: [{ORTH: abbreviation}] for abbreviation in ABBREVIATIONS
    }
    return prefix_re, infix_re, suffix_re, exclusions


if __name__ == '__main__':
    nlp = spacy.load("en_core_sci_lg")
    text2 = "Model-dependent  (AUC(0-30min/L-1*2)) mean terminal half-life of 12.0h"

    print("\n============ ORIGINAL TOKENIZER ============\n")
    tok_exp = nlp.tokenizer.explain(text2)
    for t in tok_exp:
        print(t[1], "\t", t[0])

    prefix_re, infix_re, suffix_re, exclusions = combined_rule_tokenizer()
    token_match_re = re.compile(r"(?<=[a-zA-Z])\-\d")
    # nlp.tokenizer.token_match = token_match_re.search
    nlp.tokenizer.prefix_search = prefix_re.search
    nlp.tokenizer.infix_finditer = infix_re.finditer
    nlp.tokenizer.suffix_search = suffix_re.search
    nlp = add_special_cases(nlp_model=nlp)
    tok_exp = nlp.tokenizer.explain(text2)
    print("\n============ NEW TOKENIZER ============\n")
    for t in tok_exp:
        print(t[1], "\t", t[0])
    text3 = 'The accumulation ratio was calculated as the mean ratio of AUC0–τ,ss to AUC0–τ (single dose) which were ' \
            '45.151 87,7 546 4.3 and 15.6h/l-1, and the population fluctuation ratio was 3333.4 [AUC was 3443]. ' \
            'The influence of age, sex, race, total protein, and BMI was assessed for the volumes of distribution ' \
            '[apparent central volume of distribution (V2/F), apparent peripheral volume of distribution (V3/F). ' \
            'The exposures [maximum plasma concentration (Cmax)...]'
    tok_exp = nlp.tokenizer.explain(text3)
    print("\n============ NEW TOKENIZER ============\n")
    for t in tok_exp:
        print(t[1], "\t", t[0])

    nlp.to_disk('data/sci_lg_supertokenizer')

