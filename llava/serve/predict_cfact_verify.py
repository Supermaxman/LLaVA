import argparse

import yaml
import torch
import os

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from tqdm import tqdm
import json

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import dataclasses


@dataclasses.dataclass
class ChatCompletionConfig:
    max_tokens: int
    temperature: float
    top_p: float
    system_prompt: str
    user_prompt: str


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


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config = ChatCompletionConfig(**config)
    # Model
    disable_torch_init()
    if args.model_name is None:
        model_name = get_model_name_from_path(args.model_path)
    else:
        model_name = args.model_name
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        args.load_bf16,
        device=args.device,
        lora_path=args.lora_path,
    )

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    data = list(read_jsonl(args.input_path))
    seen_ids = set()
    if os.path.exists(args.output_path):
        for p in read_jsonl(args.output_path):
            seen_ids.add(p["id"])

    for ex in tqdm(data):
        if ex["id"] in seen_ids:
            continue
        conv = conv_templates[args.conv_mode].copy()
        image_path = ex["images"][0]
        try:
            image = load_image(os.path.join(args.images_path, image_path))
            # Similar operation in model_worker.py
            image_tensor = process_images([image], image_processor, model.config)
            if type(image_tensor) is list:
                image_tensor = [
                    image.to(
                        model.device,
                        dtype=torch.bfloat16 if args.load_bf16 else torch.float16,
                    )
                    for image in image_tensor
                ]
            else:
                image_tensor = image_tensor.to(
                    model.device,
                    dtype=torch.bfloat16 if args.load_bf16 else torch.float16,
                )
            user_prompt = config.user_prompt.format(
                post=ex["text"],
                frame=ex["frame"],
                accept_rationale=ex["accept_rationale"],
                reject_rationale=ex["reject_rationale"],
                no_stance_rationale=ex["no_stance_rationale"],
            )
            inp = f"{config.system_prompt}\n{user_prompt}"

            if image is not None:
                # first message
                if model.config.mm_use_im_start_end:
                    inp = (
                        DEFAULT_IM_START_TOKEN
                        + DEFAULT_IMAGE_TOKEN
                        + DEFAULT_IM_END_TOKEN
                        + "\n"
                        + inp
                    )
                else:
                    inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
                conv.append_message(conv.roles[0], inp)
                image = None
            else:
                # later messages
                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .to(model.device)
            )
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    inputs=input_ids,
                    do_sample=True,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    max_new_tokens=config.max_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    images=image_tensor,
                )

        except Exception as e:
            print(e)
            continue
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()

        with open(args.output_path, "a") as f:
            f.write(json.dumps({"id": ex["id"], "response": outputs}) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument(
        "--input_path", type=str, required=True, help="input path for .jsonl file"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="output path for .jsonl file"
    )
    parser.add_argument(
        "--images_path", type=str, required=True, help="input path for images"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--load-bf16", action="store_true")
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    args = parser.parse_args()
    main(args)
