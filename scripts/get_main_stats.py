import typer
from pkrex.utils import read_jsonl
from collections import Counter
import matplotlib.pyplot as plt


def main(
        input_file: str = typer.Option(default="data/pubmedbert_tokenized/train-all-reviewed-clean-4.jsonl",
                                       help="File to perform checks on"),

):
    dataset = list(read_jsonl(input_file))
    print(f"Number of sentences in {input_file.split('/')[-1].upper()} set:\n{len(dataset)}\n")
    ents = [span['label'] for annot in dataset for span in annot['spans'] if span]
    rels = [rel['label'] for annot in dataset for rel in annot['relations'] if rel]
    entity_count = Counter(ents).most_common()
    rel_count = Counter(rels).most_common()
    print(10 * "=", "ENTITY STATS", 10 * "=")
    print(f"We have {len(entity_count)} entity types")
    i = 0
    for en, c in entity_count:
        en_lengths = [t['token_end'] - t['token_start'] + 1 for annot in dataset for t in annot['spans']
                      if t and t['label'] == en]
        plt.figure(i)
        plt.title(f'Number of tokens per entity type {en}')
        plt.hist(en_lengths, bins=20)
        print(
            f"Entity type: {en}; Number of mentions {c}; Average # tokens: {round(sum(en_lengths) / len(en_lengths), 2)}")
        i += 1

    print("\n", 10 * "=", "RELATION STATS", 10 * "=")
    print(f"We have {len(rel_count)} relation classes")
    for r, c in rel_count:
        print(f"Number of {r} relations: {c}")
        rels_distance = []
        for annot in dataset:
            if annot['relations']:
                for rel in annot['relations']:
                    if rel['label'] == r:
                        hs_start = rel['head_span']['token_start']
                        hs_end = rel['head_span']['token_end']
                        cs_start = rel['child_span']['token_start']
                        cs_end = rel['child_span']['token_end']
                        if cs_start > hs_end:
                            dist = cs_start - hs_end + 1
                        else:
                            assert hs_start > cs_end
                            dist = hs_start - cs_end + 1
                        # print(hs_start-cs_start)
                        rels_distance.append(dist)
        plt.figure(i)
        plt.title(f'Number tokens between entities of relation type {r}')
        plt.hist(rels_distance, bins=20)
        print(
            f"Relation {c}; Average distance: {round(sum(rels_distance) / len(rels_distance), 2)} tokens appart")
        i += 1

    plt.show()


if __name__ == "__main__":
    typer.run(main)
