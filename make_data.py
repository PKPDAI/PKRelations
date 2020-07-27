"""This script makes all the data needed to subset potential sentences for relation extraction"""

from prodigy.util import read_jsonl, write_jsonl
import os
import spacy
import re
import string
from tqdm import tqdm
import ujson


def contains_digit(sentence):
    return any(map(str.isdigit, sentence))


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def check_to_keep(inp_sections):
    keepit = True
    for inp_section in inp_sections:
        inp_section = re.sub(r'\d+', '', inp_section)
        inp_section = inp_section.translate(inp_section.maketrans('', '', string.punctuation))
        inp_section = inp_section.lower().strip()
        if "materials and methods" in inp_section or "patients and methods" in inp_section or "study design" in inp_section or "material and method" in inp_section or "methods and materials" in inp_section:
            keepit = False
        else:
            if "methods" == inp_section:
                keepit = False
    return keepit


def filter_pmc(inp_path, inp_model, output_path, include_context=False):
    all_sentences = list(read_jsonl(inp_path))
    if not include_context:
        tmp_sentences = [sentence for sentence in tqdm(all_sentences) if
                         contains_digit(sentence['text']) and check_to_keep(
                             sentence['metadata']['sections'])]

        texts = [cleanhtml(sentence['text']) for sentence in tmp_sentences]
        print("Starting processing ", len(texts), " documents")
        out_sentences = [sentence for sentence, doc in tqdm(zip(tmp_sentences, inp_model.pipe(texts, n_process=12))) if
                         doc.ents]
        write_jsonl(output_path, out_sentences)
    else:
        texts = [cleanhtml(sentence['text']) for sentence in all_sentences]
        all_chunks = []
        chunk = []
        previous_relevant = False
        pmid = all_sentences[0]['metadata']['pmid']

        for i, (sentence, doc) in tqdm(enumerate(zip(all_sentences, inp_model.pipe(texts, n_process=12)))):
            if pmid == sentence['metadata']['pmid']:
                # ==========  SAME PMID =============
                if contains_digit(sentence['text']) and len(sentence['text']) > 5 and doc.ents:
                    # ==========  CURRENT SENTENCE IS RELEVANT =============
                    sentence['metadata']['relevant'] = True  # add field
                    if previous_relevant:
                        # === Previous sentence is relevant and this one as well
                        chunk.append(sentence)  # append current to chunk (the previous was done before)
                    else:
                        # === Previous sentence is not relevant but this one yes
                        # append the previous one and this one too
                        chunk.append(all_sentences[i - 1])  # append previous
                        chunk.append(sentence)
                    if len(all_sentences) - 1 == i:  # case in which it's the last one in the file and relevant
                        all_chunks.append(chunk)  # update chunk !!
                        chunk = []
                    previous_relevant = True  # reset for next iteration

                else:
                    # ========== CURRENT SENTENCE IS NOT RELEVANT =============
                    sentence['metadata']['relevant'] = False
                    if previous_relevant:
                        chunk.append(sentence)  # if the previous was relevant also add this one
                        all_chunks.append(chunk)  # update chunk !!
                    chunk = []  # reset chunk
                    previous_relevant = False  # reset for next iteration
            else:
                # ==========  NEW PMID =============
                if previous_relevant:
                    all_chunks.append(chunk)
                chunk = []  # reset any previous chunk
                if contains_digit(sentence['text']) and len(sentence['text']) > 5 and doc.ents:
                    # ==========  CURRENT SENTENCE IS RELEVANT =============
                    sentence['metadata']['relevant'] = True  # add field
                    chunk.append(sentence)  # append current to chunk (the previous doesn't exist)
                    previous_relevant = True
                    if len(all_sentences) - 1 == i:  # case in which it's the last one in the file and relevant
                        # (very odd if so)
                        all_chunks.append(chunk)  # update chunk !!
                else:
                    # ========== CURRENT SENTENCE IS NOT RELEVANT ==========
                    sentence['metadata']['relevant'] = False
                    previous_relevant = False
            # reset PMID
            pmid = sentence['metadata']['pmid']

        # Flatten chunks and write

        with open(output_path, 'w', encoding='utf-8') as file:
            for i, chk in enumerate(all_chunks):
                for sentence in chk:
                    sentence['metadata']['chunkn'] = int(str(sentence['metadata']['pmid']) + str(i))
                    towrite = ujson.dumps(sentence, escape_forward_slashes=False) + '\n'
                    file.write(towrite)
        #        out_sentences.append(sentence)
        # write_jsonl(output_path, out_sentences)
        # === END INCLUDE PREVIOUS/SUBSEQUENT SENTENCE ===#


def filter_pmid(inp_path, inp_model, output_path, include_context=True):
    """
    :param include_context: whether to include the sentence before/after the sentence
    :param inp_model: NER model
    :param output_path: path of the output dir
    :type inp_path: string input path
    """

    # tmp_sentences = [sentence for file in tqdm(os.listdir(inp_path)) for sentence in
    #                  list(read_jsonl(os.path.join(inp_path, file)))]

    file: object
    for file in tqdm(os.listdir(inp_path)):

        print("Extracting sentences from jsonl")
        output_path_tmp = output_path.split("/")
        output_path_tmp[-1] = file
        output_path_tmp = "/".join(output_path_tmp)
        tmp_sentences = [sentence for sentence in list(read_jsonl(os.path.join(inp_path, file))) if
                         not sentence['metadata']['istitle']]
        print("Number of sentences to filter:", len(tmp_sentences))

        if include_context:
            # === INCLUDE PREVIOUS/SUBSEQUENT SENTENCE ===#

            texts = [cleanhtml(sentence['text']) for sentence in tmp_sentences]
            all_chunks = []
            chunk = []
            previous_relevant = False
            pmid = tmp_sentences[0]['metadata']['pmid']

            for i, (sentence, doc) in enumerate(zip(tmp_sentences, inp_model.pipe(texts, n_process=12))):
                if pmid == sentence['metadata']['pmid']:
                    # ==========  SAME PMID =============
                    if contains_digit(sentence['text']) and len(sentence['text']) > 5 and doc.ents:
                        # ==========  CURRENT SENTENCE IS RELEVANT =============
                        sentence['metadata']['relevant'] = True  # add field
                        if previous_relevant:
                            # === Previous sentence is relevant and this one as well
                            chunk.append(sentence)  # append current to chunk (the previous was done before)
                        else:
                            # === Previous sentence is not relevant but this one yes
                            # append the previous one and this one too
                            chunk.append(tmp_sentences[i - 1])  # append previous
                            chunk.append(sentence)
                        if len(tmp_sentences) - 1 == i:  # case in which it's the last one in the file and relevant
                            all_chunks.append(chunk)  # update chunk !!
                            chunk = []
                        previous_relevant = True  # reset for next iteration

                    else:
                        # ========== CURRENT SENTENCE IS NOT RELEVANT =============
                        sentence['metadata']['relevant'] = False
                        if previous_relevant:
                            chunk.append(sentence)  # if the previous was relevant also add this one
                            all_chunks.append(chunk)  # update chunk !!
                        chunk = []  # reset chunk
                        previous_relevant = False  # reset for next iteration
                else:
                    # ==========  NEW PMID =============
                    if previous_relevant:
                        all_chunks.append(chunk)
                    chunk = []  # reset any previous chunk
                    if contains_digit(sentence['text']) and len(sentence['text']) > 5 and doc.ents:
                        # ==========  CURRENT SENTENCE IS RELEVANT =============
                        sentence['metadata']['relevant'] = True  # add field
                        chunk.append(sentence)  # append current to chunk (the previous doesn't exist)
                        previous_relevant = True
                        if len(tmp_sentences) - 1 == i:  # case in which it's the last one in the file and relevant
                            # (very odd if so)
                            all_chunks.append(chunk)  # update chunk !!
                    else:
                        # ========== CURRENT SENTENCE IS NOT RELEVANT ==========
                        sentence['metadata']['relevant'] = False
                        previous_relevant = False
                # reset PMID
                pmid = sentence['metadata']['pmid']

            # Flatten chunks and write
            with open(output_path, 'w', encoding='utf-8') as file:
                for i, chk in enumerate(all_chunks):
                    for sentence in chk:
                        sentence['metadata']['chunkn'] = int(str(sentence['metadata']['pmid']) + str(i))
                        towrite = ujson.dumps(sentence, escape_forward_slashes=False) + '\n'
                        file.write(towrite)
            # === END INCLUDE PREVIOUS/SUBSEQUENT SENTENCE ===#
        else:
            tmp_sentences = [sentence for sentence in tmp_sentences if
                             contains_digit(sentence['text']) and len(sentence['text']) > 5]
            print("Number of sentences to process:", len(tmp_sentences))
            texts = [cleanhtml(sentence['text']) for sentence in tmp_sentences]
            print("Starting processing ", len(texts), " documents")
            out_sentences = [sentence for sentence, doc in zip(tmp_sentences, inp_model.pipe(texts, n_process=12)) if
                             doc.ents]
            write_jsonl(output_path_tmp, out_sentences)


if __name__ == "__main__":
    path_model = os.path.join("data", "scispacy_ner")

    path_pmid = os.path.join("/home/ferran/Dropbox/PKEmbeddings/data/parsed_sentences/nottokenized/pmids")
    path_pmc = os.path.join("/home/ferran/Dropbox/PKRelations/data/all_sentences/raw/all_sentences.jsonl")

    out_path_pmc_context = os.path.join("data", "all_sentences", "selected", "context", "pmc",
                                        "all_selected_context.jsonl")
    out_path_pmid_context = os.path.join("data", "all_sentences", "selected", "context", "pmid",
                                         "all_selected_context.jsonl")

    out_path_pmc_nocontext = os.path.join("data", "all_sentences", "selected", "nocontext", "pmc",
                                          "all_selected_nocontext.jsonl")
    out_path_pmid_nocontext = os.path.join("data", "all_sentences", "selected", "nocontext", "pmid",
                                           "all_selected_nocontext.jsonl")

    nlp = spacy.load(path_model)
    filter_pmc(inp_path=path_pmc, inp_model=nlp, output_path=out_path_pmc_context, include_context=True)
    filter_pmid(inp_path=path_pmid, inp_model=nlp, output_path=out_path_pmid_context, include_context=True)
    filter_pmc(inp_path=path_pmc, inp_model=nlp, output_path=out_path_pmc_nocontext, include_context=False)
    filter_pmid(inp_path=path_pmid, inp_model=nlp, output_path=out_path_pmid_nocontext, include_context=False)
