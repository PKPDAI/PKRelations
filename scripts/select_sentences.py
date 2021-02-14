"""This script makes all the data needed to subset potential sentences for relation extraction"""
import argparse
from prodigy.util import read_jsonl, write_jsonl
import os
import spacy
from tqdm import tqdm
import ujson
from pkrex.utils import contains_digit, check_to_keep, clean_html, get_output_tmp_file_path, has_pk


def filter_pmc(inp_path, inp_model, output_path, include_context=True):
    all_sentences = list(read_jsonl(inp_path))
    if not include_context:
        tmp_sentences = [sentence for sentence in tqdm(all_sentences) if
                         contains_digit(sentence['text']) and check_to_keep(
                             sentence['metadata']['sections'])]

        texts = [clean_html(sentence['text']) for sentence in tmp_sentences]
        print("Starting processing ", len(texts), " documents")
        out_sentences = [sentence for sentence, doc in tqdm(zip(tmp_sentences, inp_model.pipe(texts, n_process=12))) if
                         doc.ents]
        write_jsonl(output_path, out_sentences)
    else:
        texts = [clean_html(sentence['text']) for sentence in all_sentences]
        all_chunks = []
        chunk = []
        previous_relevant = False
        prev_pmid = all_sentences[0]['metadata']['pmid']

        for i, (sentence, doc) in tqdm(enumerate(zip(all_sentences, inp_model.pipe(texts, n_process=12)))):
            if prev_pmid == sentence['metadata']['pmid']:
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
            prev_pmid = sentence['metadata']['pmid']

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
    :param include_context: whether to include the sentence before/after the central one
    :param inp_model: NER model
    :param output_path: path of the output dir
    :param inp_path: string input path
    """

    print('Entities considered by the NER model: ', inp_model.entity.labels)

    for current_file in tqdm(os.listdir(inp_path)):
        print("Extracting sentences from jsonl")
        output_path_tmp = get_output_tmp_file_path(out_path=output_path, inp_file_name=current_file)
        tmp_sentences = [sentence for sentence in read_jsonl(os.path.join(inp_path, current_file)) if
                         not sentence['metadata']['istitle']]
        print("Number of sentences to filter:", len(tmp_sentences))

        if include_context:
            # === INCLUDE PREVIOUS/SUBSEQUENT SENTENCE === #
            texts = [clean_html(sentence['text']) for sentence in tmp_sentences]
            all_chunks = []
            chunk = []
            previous_relevant = False
            prev_pmid = tmp_sentences[0]['metadata']['pmid']
            assert len(texts) == len(tmp_sentences)
            for i, (sentence, doc) in enumerate(zip(tmp_sentences, inp_model.pipe(texts, n_process=12))):
                if prev_pmid == sentence['metadata']['pmid']:
                    # ==========  SAME PMID =============
                    if contains_digit(sentence['text']) and len(sentence['text']) > 5 and has_pk(doc):
                        # ==========  CURRENT SENTENCE IS RELEVANT =============
                        sentence['metadata']['relevant'] = True  # add field
                        for x in doc.ents:
                            print(x.label_)
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
                prev_pmid = sentence['metadata']['pmid']

            # Flatten chunks and write
            with open(output_path_tmp, 'w', encoding='utf-8') as filename:
                for i, chk in enumerate(all_chunks):
                    for sentence in chk:
                        sentence['metadata']['chunkn'] = int(str(sentence['metadata']['pmid']) + str(i))
                        towrite = ujson.dumps(sentence, escape_forward_slashes=False) + '\n'
                        filename.write(towrite)
            # === END INCLUDE PREVIOUS/SUBSEQUENT SENTENCE ===#
        else:
            tmp_sentences = [sentence for sentence in tmp_sentences if
                             contains_digit(sentence['text']) and len(sentence['text']) > 5]
            print("Number of sentences to process:", len(tmp_sentences))
            texts = [clean_html(sentence['text']) for sentence in tmp_sentences]
            print("Starting processing ", len(texts), " documents")
            out_sentences = [sentence for sentence, doc in zip(tmp_sentences, inp_model.pipe(texts, n_process=12)) if
                             doc.ents]
            write_jsonl(output_path_tmp, out_sentences)


def run(path_model: str, path_pmid: str, path_pmc: str, out_dir: str, include_context: bool):
    spacy_model = spacy.load(path_model)

    if include_context:
        ctx_string = 'context'
    else:
        ctx_string = 'nocontext'

    out_pmid_path = os.path.join(out_dir, ctx_string, 'pmid', 'all_selected_' + ctx_string + '.jsonl')
    out_pmc_path = os.path.join(out_dir, ctx_string, 'pmc', 'all_selected_' + ctx_string + '.jsonl')

    filter_pmid(inp_path=path_pmid, inp_model=spacy_model, output_path=out_pmid_path,
                include_context=include_context)
    filter_pmc(inp_path=path_pmc, inp_model=spacy_model, output_path=out_pmc_path,
               include_context=include_context)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-model", type=str, help="Path to spaCy NER model for PK parameter mentions.",
                        default='../data/models/pk_ner_supertok'
                        )
    parser.add_argument("--path-pmid", type=str, help="Path to the directory with all the PK relevant files in xml",
                        default='../data/all_sentences/pmids'
                        )
    parser.add_argument("--path-pmc", type=str, help="Path to the file with all the PMC sentences",
                        default='../data/all_semtemces/raw/all_sentences.jsonl'
                        )

    parser.add_argument("--out-dir", type=str, help="Path to the output directory.",
                        default='../data/all_sentences/selected'
                        )
    parser.add_argument("--include-context", type=bool, help="Whether to include contextual sentences.",
                        default=True
                        )
    args = parser.parse_args()
    run(path_model=args.path_model, path_pmid=args.path_pmid, path_pmc=args.path_pmc, out_dir=args.out_dir,
        include_context=args.include_context)


if __name__ == '__main__':
    main()
    """
    path_model = os.path.join("data", "pk_ner_supertok")

    path_pmid = os.path.join("/home/ferran/Dropbox/PKRelations/data/all_sentences/pmids")
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
    #
    filter_pmid(inp_path=path_pmid, inp_model=nlp, output_path=out_path_pmid_context, include_context=True)
    filter_pmc(inp_path=path_pmc, inp_model=nlp, output_path=out_path_pmc_context, include_context=True)
    """

    # filter_pmc(inp_path=path_pmc, inp_model=nlp, output_path=out_path_pmc_nocontext, include_context=False)
    # filter_pmid(inp_path=path_pmid, inp_model=nlp, output_path=out_path_pmid_nocontext, include_context=False)
