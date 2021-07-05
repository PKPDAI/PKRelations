
import typer
from prodigy.util import read_jsonl, write_jsonl
import os
from pkrex.annotation_preproc import check_and_fix_p1
from pkrex.utils import get_blob, add_annotator_meta


def main(
        azure_file_name: str = typer.Option(default="test-500-800-output.jsonl",
                                            help="Annotated file that we want to review"),
        out_dir: str = typer.Option(default="data/annotations/P1/to_review/",
                                    help="Dir to write files"),
        run_checks: bool = typer.Option(default=False,
                                        help="Whether to run checks on the input file and fix them if present. "
                                             "Checks and fixes include:"
                                             "(1) D_VAL relations pointing from the a central value to the dval")
):
    blob = get_blob(inp_blob_name=azure_file_name)

    file_path = os.path.join(out_dir, "temp_annotations.jsonl")
    with open(file_path, "wb") as f:
        f.write(blob.download_blob().readall())

    annotations = list(read_jsonl(file_path))

    os.remove(file_path)
    output_file_name = azure_file_name.replace("output.jsonl", "to-review.jsonl")
    out_file_path = file_path.replace("temp_annotations.jsonl", output_file_name)

    annotations_ready = [add_annotator_meta(inp_dict=annotation,
                                            base_dataset_name=azure_file_name.replace("-output.jsonl", ""))
                         for annotation in annotations]

    annotator_names = [x["_session_id"] for x in annotations_ready]
    uq_annotators = set(annotator_names)
    for annot_name in uq_annotators:
        an_tmp = len([x for x in annotator_names if x == annot_name])
        print(f"Annotator {annot_name} annotated {an_tmp} samples")

    if run_checks:
        annotations_ready = check_and_fix_p1(annotations_ready)

    write_jsonl(location=out_file_path, lines=annotations_ready)

    reviewed_dataset_name = output_file_name.replace("-to-review.jsonl", "-reviewed")

    print("Run the following to review the dataset:")
    print("PRODIGY_ALLOWED_SESSIONS=ferran PRODIGY_PORT=8001 prodigy custom.rel.manual {} "
          "data/models/tokenizers/rex-tokenizer {} --label C_VAL,"
          "D_VAL,RELATED --wrap --span-label UNITS,PK,TYPE_MEAS,COMPARE,RANGE,VALUE  --wrap -F "
          "recipes/rel_custom.py".format(reviewed_dataset_name, out_file_path.replace("../", "")))
    print("Number of examples in the dataset: {}".format(len(annotations_ready)))


if __name__ == '__main__':
    typer.run(main)
