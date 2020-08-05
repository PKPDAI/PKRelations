import os
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
import spacy
from spacy.tokens import Span
from spacy import displacy
from tqdm import tqdm
from spacy.util import filter_spans


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
    relations = []

    # We extact relations of the type: (inducer_chemicals,relational_verb,pk_parameters,subject_chemicals)

    # 1. find the inducer chemical that will be related to the verb
    for verb in verbs:

        # ======== 1. Add negation term if present ======================================= #
        relational_verb = verb
        # TODO: Add a condition depending on whether the verb is part of a list of key modifier verbs
        if doc_pk[verb.i - 1].text == "not":
            relational_verb = Span(doc_pk, verb.i - 1, verb.i + 1)
        else:
            if doc_pk[verb.i - 1].pos_ == 'ADV' and doc_pk[verb.i - 2].text == "not":  # e.g. "not significantly affect"
                relational_verb = Span(doc_pk, verb.i - 2, verb.i + 1)
                print("========= HEY EXCEPTIONAL NEGATION: ", relational_verb)

        # ======== 2. Find inducer chemical if present ============================================ #
        """Currently this process is not handling multiple inducers"""
        inducer_chemical = find_inducer(inp_verb=verb)

        # ======== 3. Find all the parameters related to the verb ========================= #
        pk_parameters = find_parameters(inp_verb=verb, inp_doc=doc_pk)

        # ======== 4. Find subject chemicals if present ============================================ #

        subject_chemicals = find_subject(inp_parameters=pk_parameters, inp_verb=verb, inp_inducer=inducer_chemical)

        # ======== 5. Final tricks for when there is something missing ============================================ #

        potential_rel = (inducer_chemical, relational_verb, pk_parameters, subject_chemicals)
        presents = [True if token else False for token in potential_rel]
        if sum(presents) == 3:
            inducer_chemical, relational_verb, pk_parameters, subject_chemicals = last_tricks(inp_presents=presents,
                                                                                              inp_potential=potential_rel,
                                                                                              inp_doc_ch=doc_ch,
                                                                                              inp_relations=relations)

        out_rel = (inducer_chemical, relational_verb, pk_parameters, subject_chemicals)
        relations.append(out_rel)

    return relations


def last_tricks(inp_presents, inp_potential, inp_doc_ch, inp_relations):
    inducer_chemical, relational_verb, pk_parameters, subject_chemicals = inp_potential
    if not inp_presents[0]:
        if inp_potential[1].head.pos_ == "VERB":
            new_verb = inp_potential[1].head
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
                        for minichildren in children.children:  # if  sum(inp_presents)==3 and inp_presents[0] ==
                            # False or inp_presents[-1] == False:
                            if minichildren.ent_type_ == "CHEMICAL":  # if
                                inducer_chemical = minichildren
    if not inp_presents[-1]:  # @TODO:CAREFUL, if it didn't find a subject then assign the previous if present (not
        # very nice rule)
        if len(inp_relations) > 0 and len(list(set([ch_ent.text for ch_ent in inp_doc_ch.ents]))) == 2:
            for ch in list(set([cents.text for cents in inp_doc_ch.ents])):
                if inducer_chemical:
                    if not ch == inducer_chemical.text:
                        subject_chemicals = [ch]
    return inducer_chemical, relational_verb, pk_parameters, subject_chemicals


def find_subject(inp_parameters, inp_verb, inp_inducer):
    subject_chemical = []
    for pk_parameter in inp_parameters:
        for children in pk_parameter.children:
            if children.ent_type_ == "CHEMICAL" and children.dep_ in ["nmod", "poss"]:
                subject_chemical = children

    if not subject_chemical:
        for children in inp_verb.children:
            if children.ent_type_ == "CHEMICAL" and children.dep_ == "nmod":
                if inp_inducer:
                    if not children.text == inp_inducer.text:
                        subject_chemical = children
                else:
                    subject_chemical = children

    if not subject_chemical:
        for children in inp_verb.children:
            if children.dep_ == "nmod":
                for minichildren in children.children:
                    if minichildren.ent_type_ == "CHEMICAL" and minichildren.dep_ in ["amod", "case"]:
                        subject_chemical = minichildren

    intermediate_param = []
    if not subject_chemical:
        for children in inp_verb.children:
            if children.pos_ == "inp_verb" and children.dep_ == "conj":  # case in which there is an additional inp_verb
                # between the original inp_verb and the drug mention
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
    if subject_chemical:
        subject_chemicals = [subject_chemical]
        for children in subject_chemical.children:
            if children.ent_type_ == "CHEMICAL" and children.dep_ == "conj":
                subject_chemicals.append(children)
        return subject_chemicals
    else:
        return []


def find_parameters(inp_verb, inp_doc):
    pk_parameters = []
    for children in inp_verb.children:
        if children.ent_type_ == "PK":
            if children.dep_ in ["dobj", "nmod", "nsubjpass", "nsubj", "conj"]:
                pk_parameters.append(children)

    if not pk_parameters:
        if inp_verb.i > 0:
            if inp_doc[inp_verb.i - 1].ent_type_ == "PK":
                pk_parameters.append(inp_doc[inp_verb.i - 1])
            else:
                if inp_doc[inp_verb.i - 1].is_punct:
                    if inp_doc[inp_verb.i - 2].ent_type_ == "PK":
                        pk_parameters.append(inp_doc[inp_verb.i - 2])

        if not pk_parameters:
            try:
                if inp_verb.i < len(inp_doc):
                    if inp_doc[inp_verb.i + 1].ent_type_ == "PK":
                        pk_parameters.append(inp_doc[inp_verb.i + 1])
            except:
                print("Some problem with verb location of the following (line 164):", inp_doc)
                pk_parameters = []
    return pk_parameters


def find_inducer(inp_verb):
    inducer_chemical = []

    for children in inp_verb.children:
        if children.ent_type_ == "CHEMICAL":
            if children.dep_ in ["nsubj", "dobj"]:
                inducer_chemical = children

    if not inducer_chemical:
        for children in inp_verb.children:
            if children.ent_type_ == "CHEMICAL":
                if children.dep_ == "nmod":
                    for minichildren in children.children:
                        if minichildren.dep_ == "case" and minichildren.pos_ == "ADP" and \
                                minichildren.text in ["by", "through"]:  # by//through etc
                            # check that there is no parameter between the verb and the chemical
                            inducer_chemical = children

    if not inducer_chemical:
        for children in inp_verb.children:
            if children.dep_ == 'nmod':
                for minichildren in children.children:
                    if minichildren.ent_type_ == "CHEMICAL":
                        inducer_chemical = minichildren

    if not inducer_chemical:
        if inp_verb.head.pos_ == "VERB" and inp_verb.dep_ == 'conj':
            for children in inp_verb.head.children:
                if children.ent_type_ == "CHEMICAL":
                    if children.dep_ in ["nsubj", "dobj"]:
                        inducer_chemical = children
            if not inducer_chemical:
                for children in inp_verb.head.children:
                    if children.ent_type_ == "CHEMICAL":
                        if children.dep_ == "nmod":
                            for minichildren in children.children:
                                if minichildren.dep_ == "case" and minichildren.pos_ == "ADP" and \
                                        minichildren.text in ["by", "through"]:  # by//through etc
                                    # check that there is no parameter between the verb and the chemical
                                    inducer_chemical = children

    if not inducer_chemical:
        for children in inp_verb.children:
            if children.pos_ == "VERB" and children.dep_ in ["advcl"]:  # TODO: Maybe include only when it's
                # "combined with" or similar
                # Case in which the inducer appears at the end and there is a transitory verb, e.g. The clearance
                # of mitoxantrone and etopside was decreased by 64% and 60%, respectively, when combined with
                # valspodar
                for minichildren in children.children:
                    if minichildren.ent_type_ == "CHEMICAL" and minichildren.dep_ in ["nmod"]:
                        inducer_chemical = minichildren
    return inducer_chemical


# doc = nlp(sentences_rel[323][0])

def analyse_all(sentence):
    rels = get_relations(sentence)
    print("============== Original sentence ========================")
    print(sentence)
    print("============== Model extracted ========================")
    for rel in rels:
        # print(rel[0], "|", rel[1], "| the |", rel[2], "| of |", rel[3])
        print(rel[0], "|", rel[1], "|", rel[2], "|", rel[3])
    print("\n")


def get_sdp_path(doc, subj, obj, lca_matrix):
    lca = lca_matrix[subj, obj]

    current_node = doc[subj]
    subj_path = [current_node]
    if lca != -1:
        if lca != subj:
            while current_node.head.i != lca:
                current_node = current_node.head
                subj_path.append(current_node)
            subj_path.append(current_node.head)

    current_node = doc[obj]
    obj_path = [current_node]
    if lca != -1:
        if lca != obj:
            while current_node.head.i != lca:
                current_node = current_node.head
                obj_path.append(current_node)
            obj_path.append(current_node.head)

    return subj_path + obj_path[::-1][1:]


def print_relations(sentence):
    rels = get_relations(sentence)

    for rel in rels:
        presents = [True if token else False for token in rel]
        if sum(presents) == 4:
            print("============== Original sentence ========================")
            print(sentence)
            print("============== Model extracted ========================")
            print(rel[0], "|", rel[1], "|", rel[2], "|", rel[3])
            print("\n")


def visualize_text(text):
    doc_pk = nlp_pk(text)
    doc_ch = nlp_ch(text)
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
    spacy.displacy.serve(doc_pk, style='dep')


if __name__ == '__main__':
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

    analyse_all("In healthy male volunteers, mean Cmax and AUC(0-10 h) of cabergoline increased to a similar degree "
                "during co-administration of clarithromycin.")

    analyse_all("The clearance of mitoxantrone and etopside was decreased by 64% and 60%, respectively, when combined "
                "with valspodar")
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
    analyse_all("The volume of distribution of racemic primaquine was decreased by a median (95% CI) of 22.0% ("
                "2.24%-39.9%), 24.0% (15.0%-31.5%) and 25.7% (20.3%-31.1%) when co-administered with chloroquine, "
                "dihydroartemisinin/piperaquine and pyronaridine/artesunate, respectively. ")

    for x in tqdm(sentences_rel):
        print_relations(x[0])

inducer_example = "The volume of distribution of racemic primaquine was decreased by a median (95% CI) of 22.0% (" \
                  "2.24%-39.9%), 24.0% (15.0%-31.5%) and 25.7% (20.3%-31.1%) when co-administered with chloroquine, " \
                  "dihydroartemisinin/piperaquine and pyronaridine/artesunate, respectively. "
