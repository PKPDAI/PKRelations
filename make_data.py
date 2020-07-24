"""This script makes all the data needed to subset potential sentences for relation extraction"""

from prodigy.util import read_jsonl, write_jsonl
import os
import spacy
import re
import string
from tqdm import tqdm


def contains_digit(sentence):
    return any(map(str.isdigit, sentence))


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


def filter_pmc(inp_path, inp_model, output_path):
    all_sentences = list(read_jsonl(inp_path))

    tmp_sentences = [sentence for sentence in tqdm(all_sentences) if
                     contains_digit(sentence['text']) and check_to_keep(
                         sentence['metadata']['sections'])]

    texts = [sentence['text'] for sentence in tmp_sentences]
    print("Starting processing ", len(texts), " documents")
    out_sentences = [sentence for sentence, doc in tqdm(zip(tmp_sentences, inp_model.pipe(texts, n_process=12))) if
                     doc.ents]
    write_jsonl(output_path, out_sentences)
    # return out_sentences


def filter_pmid(inp_path, inp_model, output_path):
    """
    :param inp_model:
    :param output_path:
    :type inp_path: string input path
    """

    # tmp_sentences = [sentence for file in tqdm(os.listdir(inp_path)) for sentence in
    #                  list(read_jsonl(os.path.join(inp_path, file)))]

    for file in tqdm(os.listdir(inp_path)):

        print("Extracting sentences from jsonl")
        output_path_tmp = output_path.split("/")
        output_path_tmp[-1] = file
        output_path_tmp = "/".join(output_path_tmp)
        tmp_sentences = [sentence for sentence in list(read_jsonl(os.path.join(inp_path, file)))]
        print("Number of sentences to filter:", len(tmp_sentences))

        all_chunks = []
        chunk = []
        previous_relevant = False
        pmid = tmp_sentences[0]['metadata']['pmid']

        for i, sentence in enumerate(tmp_sentences):
            if pmid == sentence['metadata']['pmid']:
                # ==========  SAME PMID =============
                if contains_digit(sentence['text']) and len(sentence['text']) > 5 and nlp(sentence['text']).ents:
                    # ==========  CURRENT SENTENCE IS RELEVANT =============
                    sentence['relevant'] = True  # add field
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
                    sentence['relevant'] = False
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
                if contains_digit(sentence['text']) and len(sentence['text']) > 5:
                    if nlp(sentence['text']).ents:
                        # ==========  CURRENT SENTENCE IS RELEVANT =============
                        sentence['relevant'] = True  # add field
                        chunk.append(sentence)  # append current to chunk (the previous doesn't exist)
                        previous_relevant = True
                        if len(tmp_sentences) - 1 == i:  # case in which it's the last one in the file and relevant
                            # (very odd if so)
                            all_chunks.append(chunk)  # update chunk !!
                    else:
                        # ========== CURRENT SENTENCE IS NOT RELEVANT ==========
                        sentence['relevant'] = False
                        previous_relevant = False
                else:
                    # ========== CURRENT SENTENCE IS NOT RELEVANT ==========
                    sentence['relevant'] = False
                    previous_relevant = False
            # reset PMID
            pmid = sentence['metadata']['pmid']

        # Flatten chunks and write
        out_sentences = []
        for i, chk in enumerate(all_chunks):
            for sentence in chk:
                sentence['chunkn'] = int(str(sentence['pmid']) + str(i))
                out_sentences.append(sentence)
        write_jsonl(output_path_tmp, out_sentences)

      # tmp_sentences = [sentence for sentence in tmp_sentences if
      #                  contains_digit(sentence['text']) and len(sentence['text']) > 5]
      # print("Number of sentences to process:", len(tmp_sentences))
      # texts = [sentence['text'] for sentence in tmp_sentences]
      # print("Starting processing ", len(texts), " documents")
      # out_sentences = [sentence for sentence, doc in zip(tmp_sentences, inp_model.pipe(texts, n_process=12)) if
      #                  doc.ents]
      # write_jsonl(output_path_tmp, out_sentences)


if __name__ == "__main__":
    path_model = os.path.join("data", "scispacy_ner")
    nlp = spacy.load(path_model)
    path_pmid = os.path.join("/home/ferran/Dropbox/PKEmbeddings/data/parsed_sentences/nottokenized/pmids")
    path_pmc = os.path.join("/home/ferran/Dropbox/PKRelations/data/all_sentences/raw/all_sentences.jsonl")
    out_path_pmc = os.path.join("data", "all_sentences", "selected", "pmc", "all_selected.jsonl")
    out_path_pmid = os.path.join("data", "all_sentences", "selected", "pmid", "all_selected.jsonl")
    # filter_pmc(path_pmc, nlp, out_path_pmc)
    filter_pmid(path_pmid, nlp, out_path_pmid)
