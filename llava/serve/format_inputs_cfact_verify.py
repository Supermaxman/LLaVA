from collections import defaultdict
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
    cfact_path: str,
    data_path: str,
    frame_path: str,
    output_path: str,
):
    stance_values = ["Accept", "Reject", "No Stance"]
    frames = json.load(open(frame_path))

    cfacts = defaultdict(lambda: defaultdict(list))
    for cf in read_jsonl(cfact_path):
        post_id, f_id, stance = cf["id"].split("-")
        content = cf["response"].replace("</s>", "").strip()
        cfacts[post_id][f_id].append(
            {
                "stance": stance,
                "content": content,
            }
        )
    examples = []
    for ex in read_jsonl(data_path):
        for f_id, f_stance in ex["labels"].items():
            if f_stance == "Not Relevant":
                continue
            f_text = frames[f_id]["text"]
            text = ex["text"]
            ex_id = ex["id"]

            pf_cfacts = sorted(
                cfacts[ex_id][f_id], key=lambda x: stance_values.index(x["stance"])
            )
            accept_rationale, reject_rationale, no_stance_rationale = pf_cfacts

            examples.append(
                {
                    "id": f"{ex_id}-{f_id}",
                    "text": text,
                    "frame": f_text,
                    "images": ex["images"],
                    "accept_rationale": accept_rationale,
                    "reject_rationale": reject_rationale,
                    "no_stance_rationale": no_stance_rationale,
                }
            )

    write_jsonl(output_path, examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfact_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--frame_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    main(
        args.cfact_path,
        args.data_path,
        args.frame_path,
        args.output_path,
    )
