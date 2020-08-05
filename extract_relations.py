import os
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
import spacy
from spacy.tokens import Span
from spacy import displacy
from tqdm import tqdm
from spacy.util import filter_spans
import itertools


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

    chemicals = [entity for entity in doc_pk if entity.ent_type_ == "CHEMICAL"]
    verbs = [verb for verb in doc_pk if verb.pos_ == "VERB"]
    parameters = [pk for pk in doc_pk if pk.ent_type_ == "PK"]

    if len(chemicals) >= 2 and parameters and verbs:  # first condition
        lca_matrix = doc_pk.get_lca_matrix()
        ch_combos = get_combinations(chemicals)
        for combo in ch_combos:
            print("== Comparing", combo[0], "with", combo[1], "==")
            sdp = get_sdp_path(doc_pk, combo[0].i, combo[1].i, lca_matrix)
            print(sdp)

        relations = []
        for verb in verbs:
            subject_chemical = []
            inducer_chemical = []
            relational_verb = verb
            pk_parameters = []

    else:
        return []


def visualize_text(text):
    options = {"compact": True, "font": "Source Sans Pro"}
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
    spacy.displacy.serve(doc_pk, style='dep', options=options)


def get_combinations(
        ch_list):  # get all the possible combinations within elements in the list including swaped versions
    new_list_tuple = list(itertools.combinations(ch_list, 2))
    for temp_tuple in itertools.combinations(ch_list, 2):
        new_list_tuple.append((temp_tuple[1], temp_tuple[0]))  # add swapped version
    return new_list_tuple


if __name__ == '__main__':
    inp_file_path = os.path.join("data", "pkddi", "invivo_train.xml")

    with open(inp_file_path) as infile:
        soup = BeautifulSoup(infile)

    sentences_rel = list(
        [(element.text.replace("\n", ""), element['ddi'], element['sem']) for element in soup.findAll("annotation")])
    # ps = PorterStemmer()
    # change_voc = list(set([ps.stem(mention.text) for element in soup.findAll("annotation") for mention in
    #                       element.findAll("cons", {"sem": "G_CHANGE"})]))

    # nlp = spacy.load("en_core_sci_lg")
    nlp_pk = spacy.load("data/scispacy_ner")
    nlp_ch = spacy.load("en_ner_bc5cdr_md")

    # analyse_all(sentences_rel[323][0])
    # analyse_all(sentences_rel[324][0])
    # analyse_all(sentences_rel[4][0])
    get_relations(sentences_rel[5][0])
    # analyse_all(sentences_rel[8][0])
    get_relations("Midazolam increased the AUC of amoxicillin")
    get_relations("The bioavailability and AUC of oral ondansetron was reduced from 60% to 40% (P<.01) by rifampin")

#  analyse_all("Midazolam's AUC was increased by amoxicillin")
#  analyse_all("Midazolam's AUC was increased through amoxicillin administration")
#  analyse_all("The clearance of mitoxantrone and etopside was decreased by 64% and 60%, respectively, when combined "
#              "with valspodar")
#  analyse_all("The bioavailability and AUC of oral ondansetron was reduced from 60% to 40% (P<.01) by rifampin")
#  analyse_all("The volume of distribution of racemic primaquine was decreased by a median (95% CI) of 22.0% ("
#              "2.24%-39.9%), 24.0% (15.0%-31.5%) and 25.7% (20.3%-31.1%) when co-administered with chloroquine, "
#              "dihydroartemisinin/piperaquine and pyronaridine/artesunate, respectively. ")


## Challenges
["Verapamil increased the Cmax of simvastatin acid 3.4-fold (p < 0.001) and the AUC(0-24) 2.8-fold (p < 0.001).",
 "On average, itraconazole increased the peak plasma concentration (Cmax) of felodipine nearly eightfold (p < 0.001), "
 "the areas under the felodipine concentration-time curve  [AUC(0-32) and AUC(0-infinity)] about sixfold (p < 0.001), "
 "and the elimination half-life twofold (p < 0.05).", "In contrast, erythromycin did not significantly affect the "
                                                      "AUC(0-24) or half-life of either losartan or E3174. "
 ]