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
    data_path: str,
    frame_path: str,
    output_path: str,
):
    stance_values = ["Accept", "Reject", "No Stance"]
    frames = json.load(open(frame_path))

    examples = []
    for ex in read_jsonl(data_path):
        for f_id, f_stance in ex["labels"].items():
            if f_stance == "Not Relevant":
                continue
            f_text = frames[f_id]["text"]
            text = ex["text"]
            ex_id = ex["id"]
            for stance in stance_values:
                examples.append(
                    {
                        "id": f"{ex_id}-{f_id}-{stance}",
                        "text": text,
                        "frame": f_text,
                        "images": ex["images"],
                        "stance": stance,
                    }
                )

    write_jsonl(output_path, examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--frame_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    main(
        args.data_path,
        args.frame_path,
        args.output_path,
    )
