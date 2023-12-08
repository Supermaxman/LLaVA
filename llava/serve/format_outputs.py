import json
import argparse


def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                yield ex


def write_jsonl(path, examples):
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def main(
    pred_path: str,
    output_path: str,
):
    examples = []
    for ex in read_jsonl(pred_path):
        ex_id, f_id = ex["id"].split("-")
        examples.append(
            {
                "post_id": ex_id,
                "f_id": f_id,
                "content": ex["response"],
            }
        )

    write_jsonl(output_path, examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    main(
        args.pred_path,
        args.output_path,
    )
