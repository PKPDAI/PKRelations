import os
from bs4 import BeautifulSoup
import csv
from nltk.stem import PorterStemmer
import spacy
from spacy.tokens import Span
from spacy import displacy
from tqdm import tqdm


def filter_spans(spans):
    # Filter a sequence of spans so they don't contain overlaps
    # For spaCy 2.1.4+: this function is available as spacy.util.filter_spans()
    get_sort_key = lambda span: (span.end - span.start, -span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        # Check for end - 1 here because boundaries are inclusive
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
        seen_tokens.update(range(span.start, span.end))
    result = sorted(result, key=lambda span: span.start)
    return result


def get_relations(inp_sentence):
    doc_pk = nlp_pk(inp_sentence)
    doc_ch = nlp_ch(inp_sentence)
    for chemical in doc_ch.ents:
        ch_ent = Span(doc_pk, chemical.start, chemical.end, label="CHEMICAL")
        try:
            doc_pk.ents = list(doc_pk.ents) + [ch_ent]
        except:
            doc_pk.ents = list(doc_pk.ents)

    # Merge entities and noun chunks into one token
    spans = list(doc_pk.ents)  # + list(doc_pk.noun_chunks)
    spans = filter_spans(spans)
    with doc_pk.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)

    verbs = [verb for verb in doc_pk if verb.pos_ == "VERB"]
    # entities = doc_pk.ents
    relations = []

    # We extact relations of the type: (inducer_chemical,relational_verb,pk_parameter,subject_chemical)

    # 1. find the inducer chemical that will be related to the verb
    for verb in verbs:
        subject_chemical = []
        inducer_chemical = []
        relational_verb = verb
        pk_parameters = []

        for children in verb.children:
            if children.ent_type_ == "CHEMICAL":
                if children.dep_ in ["nsubj", "dobj"]:
                    inducer_chemical = children

        if not inducer_chemical:
            for children in verb.children:
                if children.ent_type_ == "CHEMICAL":
                    if children.dep_ == "nmod":
                        for minichildren in children.children:
                            if minichildren.dep_ == "case" and minichildren.pos_ == "ADP" and minichildren.text in ["by", "through"]:  # by//through etc
                                # check that there is no parameter between the verb and the chemical
                                inducer_chemical = children

        if not inducer_chemical:
            for children in verb.children:
                if children.dep_ == 'nmod':
                    for minichildren in children.children:
                        if minichildren.ent_type_ == "CHEMICAL":
                            inducer_chemical = minichildren

        if not inducer_chemical:
            if verb.head.pos_ == "VERB" and verb.dep_ == 'conj':
                for children in verb.head.children:
                    if children.ent_type_ == "CHEMICAL":
                        if children.dep_ in ["nsubj", "dobj"]:
                            inducer_chemical = children
                if not inducer_chemical:
                    for children in verb.head.children:
                        if children.ent_type_ == "CHEMICAL":
                            if children.dep_ == "nmod":
                                for minichildren in children.children:
                                    if minichildren.dep_ == "case" and minichildren.pos_ == "ADP" and minichildren.text in [
                                        "by",
                                        "through"]:  # by//through etc
                                        # check that there is no parameter between the verb and the chemical
                                        inducer_chemical = children

                                        # 1. ENDED UP WITH INDUCER, NOW LET'S LOOK AT THE RELATED PARAMETER/S

        for children in verb.children:
            if children.ent_type_ == "PK":
                if children.dep_ in ["dobj", "nmod", "nsubjpass", "nsubj", "conj"]:
                    pk_parameters.append(children)

        if not pk_parameters:
            if verb.i > 0:
                if doc_pk[verb.i - 1].ent_type_ == "PK":
                    pk_parameters.append(doc_pk[verb.i - 1])
                else:
                    if doc_pk[verb.i - 1].is_punct:
                        if doc_pk[verb.i - 2].ent_type_ == "PK":
                            pk_parameters.append(doc_pk[verb.i - 2])

            if not pk_parameters:
                try:
                    if verb.i < len(doc_pk):
                        if doc_pk[verb.i + 1].ent_type_ == "PK":
                            pk_parameters.append(doc_pk[verb.i + 1])
                except:
                    pk_parameters = []
        # 3. ENDED UP WITH PARAMETER, LET'S FIND THE ORIGINAL CHEMICAL

        for pk_parameter in pk_parameters:
            for children in pk_parameter.children:
                if children.ent_type_ == "CHEMICAL" and children.dep_ in ["nmod", "poss"]:
                    subject_chemical = children

        if not subject_chemical:
            for children in verb.children:
                if children.ent_type_ == "CHEMICAL" and children.dep_ == "nmod":
                    if inducer_chemical:
                        if not children.text == inducer_chemical.text:
                            subject_chemical = children
                    else:
                        subject_chemical = children

        if not subject_chemical:
            for children in verb.children:
                if children.dep_ == "nmod":
                    for minichildren in children.children:
                        if minichildren.ent_type_ == "CHEMICAL" and minichildren.dep_ in ["amod", "case"]:
                            subject_chemical = minichildren

        intermediate_param = []
        if not subject_chemical:
            for children in verb.children:
                if children.pos_ == "VERB" and children.dep_ == "conj":  # case in which there is an additional verb
                    # between the original verb and the drug mention
                    for minichildren in children.children:
                        if minichildren.ent_type_ == "CHEMICAL" and minichildren.dep_ in ["amod", "case"]:
                            subject_chemical = minichildren
                        else:
                            if minichildren.ent_type_ == "PK" and minichildren.dep_ in ["dobj", "nmod", "nsubjpass",
                                                                                        "nsubj",
                                                                                        "conj"] and not intermediate_param:
                                intermediate_param = minichildren

        if not subject_chemical and intermediate_param:
            for children in intermediate_param.children:
                if children.ent_type_ == "CHEMICAL" and children.dep_ == "nmod":
                    subject_chemical = children
            if not subject_chemical:
                for children in intermediate_param.children:
                    if children.dep_ == "nmod":
                        for minichildren in children.children:
                            if minichildren.ent_type_ == "CHEMICAL" and minichildren.dep_ in ["amod", "case"]:
                                subject_chemical = minichildren

        # LAST CHECKS FOR WHEN THERE IS ONLY ONE MISSING CHEMICAL
        if pk_parameters:
            for pk_parameter in pk_parameters:
                potential_rel = (inducer_chemical, relational_verb, pk_parameter, subject_chemical)
        else:
            potential_rel = (inducer_chemical, relational_verb, [], subject_chemical)

        presents = [True if token else False for token in potential_rel]
        if sum(presents) == 3:
            if not presents[0]:
                if potential_rel[1].head.pos_ == "VERB":
                    new_verb = potential_rel[1].head
                    for children in new_verb.children:
                        if children.ent_type_ == "CHEMICAL":
                            if children.dep_ in ["nsubj", "dobj"]:
                                inducer_chemical = children

                    if not inducer_chemical:
                        for children in new_verb.children:
                            if children.ent_type_ == "CHEMICAL":
                                if children.dep_ == "nmod":
                                    inducer_chemical = children

                    if not inducer_chemical:
                        for children in new_verb.children:
                            if children.dep_ == 'nmod':
                                for minichildren in children.children:  # if  sum(presents)==3 and presents[0] ==
                                    # False or presents[-1] == False:
                                    if minichildren.ent_type_ == "CHEMICAL":  # if
                                        inducer_chemical = minichildren
            if not presents[-1]:  # @TODO:CAREFUL, if it didn't find a subject then assign the previous if present
                if len(relations) > 0 and len(list(set([x.text for x in doc_ch.ents]))) == 2:
                    for ch in list(set([cents.text for cents in doc_ch.ents])):
                        if inducer_chemical:
                            if not ch == inducer_chemical.text:
                                subject_chemical = ch

        for pk_parameter in pk_parameters:
            out_rel = (inducer_chemical, relational_verb, pk_parameter, subject_chemical)
            relations.append(out_rel)

    return relations


inp_file_path = os.path.join("data", "pkddi", "invivo_train.xml")

with open(inp_file_path) as infile:
    soup = BeautifulSoup(infile)

ps = PorterStemmer()
sentences_rel = list(
    [(element.text.replace("\n", ""), element['ddi'], element['sem']) for element in soup.findAll("annotation")])
change_voc = list(set([ps.stem(mention.text) for element in soup.findAll("annotation") for mention in
                       element.findAll("cons", {"sem": "G_CHANGE"})]))

# nlp = spacy.load("en_core_sci_lg")
nlp_pk = spacy.load("data/scispacy_ner")
nlp_ch = spacy.load("en_ner_bc5cdr_md")


# doc = nlp(sentences_rel[323][0])

def analyse_all(sentence):
    rels = get_relations(sentence)
    print("============== Original sentence ========================")
    print(sentence)
    print("============== Model extracted ========================")
    for rel in rels:
        #print(rel[0], "|", rel[1], "| the |", rel[2], "| of |", rel[3])
        print(rel[0], "|", rel[1],  rel[2],  rel[3])
    print("\n")


def print_relations(sentence):
    rels = get_relations(sentence)

    for rel in rels:
        presents = [True if token else False for token in rel]
        if sum(presents) == 4:
            print("============== Original sentence ========================")
            print(sentence)
            print("============== Model extracted ========================")
            print(rel[0], "|", rel[1], "| the |", rel[2], "| of |", rel[3])
            print("\n")


analyse_all(sentences_rel[323][0])
analyse_all(sentences_rel[324][0])
analyse_all(sentences_rel[4][0])
analyse_all(sentences_rel[5][0])
analyse_all(sentences_rel[8][0])
analyse_all("Midazolam increased the AUC of amoxicillin")
analyse_all("Midazolam's AUC was increased by amoxicillin")
analyse_all("Midazolam's AUC was increased through amoxicillin administration")
analyse_all("The clearance of mitoxantrone and etopside was decreased by 64% and 60%, respectively, when combined "
            "with valspodar")
analyse_all("The bioavailability of oral ondansetron was reduced from 60% to 40% (P<.01) by rifampin")

for x in tqdm(sentences_rel):
    print_relations(x[0])

# TODO: INCLUDE NON IN FRONT OF VERB!
# TODO: INCLUDE MORE THAN 2 CHEMICALS MENTIONED
# TODO: TAKE INTO ACCOUNT WHEN MORE THAN 2 CHEMICALS ARE MENTIONED
