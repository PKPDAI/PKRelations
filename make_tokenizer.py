import spacy
from typing import List
from spacy.lang import char_classes
from spacy.symbols import ORTH
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
from scispacy.consts import ABBREVIATIONS


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
            ["§",
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
            + [
                r"(?<=[0-9])([a-zA-Z]+|[^a-zA-Z\d\s:,\.]+)",  # digit + non digit
                r"×",  # added this special x character to tokenize it separately
                # r"(?<=[0-9])[+\-\*^](?=[0-9-])", ######################################## OWN Modification
                r"(?<=[0-9])[+\-\*^](?=[0-9-])",
                r"(?<=[{al}])\.(?=[{au}])".format(
                    al=char_classes.ALPHA_LOWER, au=char_classes.ALPHA_UPPER
                ),
                r"[^a-zA-Z\d\s:]",  # All non alphanumerics are infixes
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
    nlp.tokenizer.prefix_search = prefix_re.search
    nlp.tokenizer.infix_finditer = infix_re.finditer
    nlp.tokenizer.suffix_search = suffix_re.search
    nlp = add_special_cases(nlp_model=nlp)
    tok_exp = nlp.tokenizer.explain(text2)
    print("\n============ NEW TOKENIZER ============\n")
    for t in tok_exp:
        print(t[1], "\t", t[0])
    text3 = 'The accumulation ratio was calculated as the ratio of AUC0–τ,ss to AUC0–τ (single dose), and the fluctuation ratio was calculated by (Cmax,ss − Cmin,ss)/Cavg, where Cavg is the average steady-state drug concentration during multiple dosing, which is calculated as AUC0–τ,ss/τ, where τ is the dosing interval (6 hours). The clearance was 452.234 and 3,2'
    tok_exp = nlp.tokenizer.explain(text3)
    print("\n============ NEW TOKENIZER ============\n")
    for t in tok_exp:
        print(t[1], "\t", t[0])

    nlp.to_disk('sci_lg_supertokenizer')
# text3 = "times higher following i.p. injection of 95 mg kg-1 (0.25 mmol kg-1) of the prodrug as compared with " \
#         "administration via the oral and i.v. routes, respectively. After i.v. injection , peak levels of t "
# tok_exp = nlp.tokenizer.explain(text3)
# for t in tok_exp:
#    print(t[1], "\t", t[0])

#

# tok_exp = nlp.tokenizer.explain(text3)
# for t in tok_exp:
#    print(t[1], "\t", t[0])

# for sent in nlp(text3).sents:
#    print(sent)


# letsee = {"text":"The accumulation ratio was calculated as the ratio of AUC0\u2013\u03c4,ss to AUC0\u2013\u03c4 (single dose), and the fluctuation ratio was calculated by (Cmax,ss \u2212 Cmin,ss)/Cavg, where Cavg is the average steady-state drug concentration during multiple dosing, which is calculated as AUC0\u2013\u03c4,ss/\u03c4, where \u03c4 is the dosing interval (6 hours).","spans":[{"start":45,"end":60,"token_start":7,"token_end":9,"label":"PK"},{"start":61,"end":63,"token_start":11,"token_end":11,"label":"PK"},{"start":67,"end":73,"token_start":13,"token_end":13,"label":"PK"},{"start":134,"end":141,"token_start":27,"token_end":29,"label":"PK"},{"start":144,"end":157,"token_start":31,"token_end":33,"label":"PK"},{"start":165,"end":169,"token_start":36,"token_end":36,"label":"PK"},{"start":177,"end":216,"token_start":39,"token_end":42,"label":"PK"},{"start":264,"end":270,"token_start":51,"token_end":51,"label":"PK"},{"start":271,"end":275,"token_start":53,"token_end":53,"label":"PK"}],"_input_hash":-1570795161,"_task_hash":306860192,"tokens":[{"text":"The","start":0,"end":3,"id":0},{"text":"accumulation","start":4,"end":16,"id":1},{"text":"ratio","start":17,"end":22,"id":2},{"text":"was","start":23,"end":26,"id":3},{"text":"calculated","start":27,"end":37,"id":4},{"text":"as","start":38,"end":40,"id":5},{"text":"the","start":41,"end":44,"id":6},{"text":"ratio","start":45,"end":50,"id":7},{"text":"of","start":51,"end":53,"id":8},{"text":"AUC0\u2013\u03c4","start":54,"end":60,"id":9},{"text":",","start":60,"end":61,"id":10},{"text":"ss","start":61,"end":63,"id":11},{"text":"to","start":64,"end":66,"id":12},{"text":"AUC0\u2013\u03c4","start":67,"end":73,"id":13},{"text":"(","start":74,"end":75,"id":14},{"text":"single","start":75,"end":81,"id":15},{"text":"dose","start":82,"end":86,"id":16},{"text":")","start":86,"end":87,"id":17},{"text":",","start":87,"end":88,"id":18},{"text":"and","start":89,"end":92,"id":19},{"text":"the","start":93,"end":96,"id":20},{"text":"fluctuation","start":97,"end":108,"id":21},{"text":"ratio","start":109,"end":114,"id":22},{"text":"was","start":115,"end":118,"id":23},{"text":"calculated","start":119,"end":129,"id":24},{"text":"by","start":130,"end":132,"id":25},{"text":"(","start":133,"end":134,"id":26},{"text":"Cmax","start":134,"end":138,"id":27},{"text":",","start":138,"end":139,"id":28},{"text":"ss","start":139,"end":141,"id":29},{"text":"\u2212","start":142,"end":143,"id":30},{"text":"Cmin","start":144,"end":148,"id":31},{"text":",","start":148,"end":149,"id":32},{"text":"ss)/Cavg","start":149,"end":157,"id":33},{"text":",","start":157,"end":158,"id":34},{"text":"where","start":159,"end":164,"id":35},{"text":"Cavg","start":165,"end":169,"id":36},{"text":"is","start":170,"end":172,"id":37},{"text":"the","start":173,"end":176,"id":38},{"text":"average","start":177,"end":184,"id":39},{"text":"steady-state","start":185,"end":197,"id":40},{"text":"drug","start":198,"end":202,"id":41},{"text":"concentration","start":203,"end":216,"id":42},{"text":"during","start":217,"end":223,"id":43},{"text":"multiple","start":224,"end":232,"id":44},{"text":"dosing","start":233,"end":239,"id":45},{"text":",","start":239,"end":240,"id":46},{"text":"which","start":241,"end":246,"id":47},{"text":"is","start":247,"end":249,"id":48},{"text":"calculated","start":250,"end":260,"id":49},{"text":"as","start":261,"end":263,"id":50},{"text":"AUC0\u2013\u03c4","start":264,"end":270,"id":51},{"text":",","start":270,"end":271,"id":52},{"text":"ss/\u03c4","start":271,"end":275,"id":53},{"text":",","start":275,"end":276,"id":54},{"text":"where","start":277,"end":282,"id":55},{"text":"\u03c4","start":283,"end":284,"id":56},{"text":"is","start":285,"end":287,"id":57},{"text":"the","start":288,"end":291,"id":58},{"text":"dosing","start":292,"end":298,"id":59},{"text":"interval","start":299,"end":307,"id":60},{"text":"(","start":308,"end":309,"id":61},{"text":"6","start":309,"end":310,"id":62},{"text":"hours","start":311,"end":316,"id":63},{"text":")","start":316,"end":317,"id":64},{"text":".","start":317,"end":318,"id":65}],"_session_id":None,"_view_id":"review","answer":"accept","sessions":["training_ferran2"],"versions":[{"text":"The accumulation ratio was calculated as the ratio of AUC0\u2013\u03c4,ss to AUC0\u2013\u03c4 (single dose), and the fluctuation ratio was calculated by (Cmax,ss \u2212 Cmin,ss)/Cavg, where Cavg is the average steady-state drug concentration during multiple dosing, which is calculated as AUC0\u2013\u03c4,ss/\u03c4, where \u03c4 is the dosing interval (6 hours).","spans":[{"start":45,"end":60,"token_start":7,"token_end":9,"label":"PK"},{"start":61,"end":63,"token_start":11,"token_end":11,"label":"PK"},{"start":67,"end":73,"token_start":13,"token_end":13,"label":"PK"},{"start":134,"end":141,"token_start":27,"token_end":29,"label":"PK"},{"start":144,"end":157,"token_start":31,"token_end":33,"label":"PK"},{"start":165,"end":169,"token_start":36,"token_end":36,"label":"PK"},{"start":177,"end":216,"token_start":39,"token_end":42,"label":"PK"},{"start":264,"end":270,"token_start":51,"token_end":51,"label":"PK"},{"start":271,"end":275,"token_start":53,"token_end":53,"label":"PK"}],"_input_hash":-1570795161,"_task_hash":306860192,"tokens":[{"text":"The","start":0,"end":3,"id":0},{"text":"accumulation","start":4,"end":16,"id":1},{"text":"ratio","start":17,"end":22,"id":2},{"text":"was","start":23,"end":26,"id":3},{"text":"calculated","start":27,"end":37,"id":4},{"text":"as","start":38,"end":40,"id":5},{"text":"the","start":41,"end":44,"id":6},{"text":"ratio","start":45,"end":50,"id":7},{"text":"of","start":51,"end":53,"id":8},{"text":"AUC0\u2013\u03c4","start":54,"end":60,"id":9},{"text":",","start":60,"end":61,"id":10},{"text":"ss","start":61,"end":63,"id":11},{"text":"to","start":64,"end":66,"id":12},{"text":"AUC0\u2013\u03c4","start":67,"end":73,"id":13},{"text":"(","start":74,"end":75,"id":14},{"text":"single","start":75,"end":81,"id":15},{"text":"dose","start":82,"end":86,"id":16},{"text":")","start":86,"end":87,"id":17},{"text":",","start":87,"end":88,"id":18},{"text":"and","start":89,"end":92,"id":19},{"text":"the","start":93,"end":96,"id":20},{"text":"fluctuation","start":97,"end":108,"id":21},{"text":"ratio","start":109,"end":114,"id":22},{"text":"was","start":115,"end":118,"id":23},{"text":"calculated","start":119,"end":129,"id":24},{"text":"by","start":130,"end":132,"id":25},{"text":"(","start":133,"end":134,"id":26},{"text":"Cmax","start":134,"end":138,"id":27},{"text":",","start":138,"end":139,"id":28},{"text":"ss","start":139,"end":141,"id":29},{"text":"\u2212","start":142,"end":143,"id":30},{"text":"Cmin","start":144,"end":148,"id":31},{"text":",","start":148,"end":149,"id":32},{"text":"ss)/Cavg","start":149,"end":157,"id":33},{"text":",","start":157,"end":158,"id":34},{"text":"where","start":159,"end":164,"id":35},{"text":"Cavg","start":165,"end":169,"id":36},{"text":"is","start":170,"end":172,"id":37},{"text":"the","start":173,"end":176,"id":38},{"text":"average","start":177,"end":184,"id":39},{"text":"steady-state","start":185,"end":197,"id":40},{"text":"drug","start":198,"end":202,"id":41},{"text":"concentration","start":203,"end":216,"id":42},{"text":"during","start":217,"end":223,"id":43},{"text":"multiple","start":224,"end":232,"id":44},{"text":"dosing","start":233,"end":239,"id":45},{"text":",","start":239,"end":240,"id":46},{"text":"which","start":241,"end":246,"id":47},{"text":"is","start":247,"end":249,"id":48},{"text":"calculated","start":250,"end":260,"id":49},{"text":"as","start":261,"end":263,"id":50},{"text":"AUC0\u2013\u03c4","start":264,"end":270,"id":51},{"text":",","start":270,"end":271,"id":52},{"text":"ss/\u03c4","start":271,"end":275,"id":53},{"text":",","start":275,"end":276,"id":54},{"text":"where","start":277,"end":282,"id":55},{"text":"\u03c4","start":283,"end":284,"id":56},{"text":"is","start":285,"end":287,"id":57},{"text":"the","start":288,"end":291,"id":58},{"text":"dosing","start":292,"end":298,"id":59},{"text":"interval","start":299,"end":307,"id":60},{"text":"(","start":308,"end":309,"id":61},{"text":"6","start":309,"end":310,"id":62},{"text":"hours","start":311,"end":316,"id":63},{"text":")","start":316,"end":317,"id":64},{"text":".","start":317,"end":318,"id":65}],"_session_id":"training_ferran2","_view_id":"ner_manual","answer":"accept","sessions":["training_ferran2"],"default":True},{"text":"The accumulation ratio was calculated as the ratio of AUC0\u2013\u03c4,ss to AUC0\u2013\u03c4 (single dose), and the fluctuation ratio was calculated by (Cmax,ss \u2212 Cmin,ss)/Cavg, where Cavg is the average steady-state drug concentration during multiple dosing, which is calculated as AUC0\u2013\u03c4,ss/\u03c4, where \u03c4 is the dosing interval (6 hours).","spans":[{"start":45,"end":60,"token_start":7,"token_end":9,"label":"PK"},{"start":67,"end":73,"token_start":13,"token_end":13,"label":"PK"},{"start":134,"end":141,"token_start":27,"token_end":29,"label":"PK"},{"start":144,"end":157,"token_start":31,"token_end":33,"label":"PK"},{"start":264,"end":270,"token_start":51,"token_end":51,"label":"PK"}],"_input_hash":-1570795161,"_task_hash":349682656,"tokens":[{"text":"The","start":0,"end":3,"id":0},{"text":"accumulation","start":4,"end":16,"id":1},{"text":"ratio","start":17,"end":22,"id":2},{"text":"was","start":23,"end":26,"id":3},{"text":"calculated","start":27,"end":37,"id":4},{"text":"as","start":38,"end":40,"id":5},{"text":"the","start":41,"end":44,"id":6},{"text":"ratio","start":45,"end":50,"id":7},{"text":"of","start":51,"end":53,"id":8},{"text":"AUC0\u2013\u03c4","start":54,"end":60,"id":9},{"text":",","start":60,"end":61,"id":10},{"text":"ss","start":61,"end":63,"id":11},{"text":"to","start":64,"end":66,"id":12},{"text":"AUC0\u2013\u03c4","start":67,"end":73,"id":13},{"text":"(","start":74,"end":75,"id":14},{"text":"single","start":75,"end":81,"id":15},{"text":"dose","start":82,"end":86,"id":16},{"text":")","start":86,"end":87,"id":17},{"text":",","start":87,"end":88,"id":18},{"text":"and","start":89,"end":92,"id":19},{"text":"the","start":93,"end":96,"id":20},{"text":"fluctuation","start":97,"end":108,"id":21},{"text":"ratio","start":109,"end":114,"id":22},{"text":"was","start":115,"end":118,"id":23},{"text":"calculated","start":119,"end":129,"id":24},{"text":"by","start":130,"end":132,"id":25},{"text":"(","start":133,"end":134,"id":26},{"text":"Cmax","start":134,"end":138,"id":27},{"text":",","start":138,"end":139,"id":28},{"text":"ss","start":139,"end":141,"id":29},{"text":"\u2212","start":142,"end":143,"id":30},{"text":"Cmin","start":144,"end":148,"id":31},{"text":",","start":148,"end":149,"id":32},{"text":"ss)/Cavg","start":149,"end":157,"id":33},{"text":",","start":157,"end":158,"id":34},{"text":"where","start":159,"end":164,"id":35},{"text":"Cavg","start":165,"end":169,"id":36},{"text":"is","start":170,"end":172,"id":37},{"text":"the","start":173,"end":176,"id":38},{"text":"average","start":177,"end":184,"id":39},{"text":"steady-state","start":185,"end":197,"id":40},{"text":"drug","start":198,"end":202,"id":41},{"text":"concentration","start":203,"end":216,"id":42},{"text":"during","start":217,"end":223,"id":43},{"text":"multiple","start":224,"end":232,"id":44},{"text":"dosing","start":233,"end":239,"id":45},{"text":",","start":239,"end":240,"id":46},{"text":"which","start":241,"end":246,"id":47},{"text":"is","start":247,"end":249,"id":48},{"text":"calculated","start":250,"end":260,"id":49},{"text":"as","start":261,"end":263,"id":50},{"text":"AUC0\u2013\u03c4","start":264,"end":270,"id":51},{"text":",","start":270,"end":271,"id":52},{"text":"ss/\u03c4","start":271,"end":275,"id":53},{"text":",","start":275,"end":276,"id":54},{"text":"where","start":277,"end":282,"id":55},{"text":"\u03c4","start":283,"end":284,"id":56},{"text":"is","start":285,"end":287,"id":57},{"text":"the","start":288,"end":291,"id":58},{"text":"dosing","start":292,"end":298,"id":59},{"text":"interval","start":299,"end":307,"id":60},{"text":"(","start":308,"end":309,"id":61},{"text":"6","start":309,"end":310,"id":62},{"text":"hours","start":311,"end":316,"id":63},{"text":")","start":316,"end":317,"id":64},{"text":".","start":317,"end":318,"id":65}],"_session_id":"training_simon","_view_id":"ner_manual","answer":"accept","sessions":["training_simon"],"default":False}],"view_id":"ner_manual"}
#
# for pepito in letsee['spans']:
#    print(letsee['text'][pepito['start']:pepito['end']])
#
# for token in nlp(letsee['text']):
#	print(token)
