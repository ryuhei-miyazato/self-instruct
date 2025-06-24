import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
# os.environ["TORCH_COMPILE_DISABLE"] = "1"

import json
import random
import re
import string
import tqdm
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from rouge_score import rouge_scorer
# from gpt3_api import make_requests as make_gpt3_requests

# add
import torch, re
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


random.seed(42)

def initialize_models(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def encode_prompt(prompt_instructions, tokenizer, classification=False):
    prompt = "あなたはタスク設計の専門家です。与えられた一連のタスクを参考に、形式を揃えて次に来るべきタスクを提案してください。\n"
    # prompt = ""
    for idx, instruction in enumerate(prompt_instructions):
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        prompt += f"{idx+1}. {instruction}\n"
    prompt += f"{len(prompt_instructions) + 1}."

    # ChatGPT形式のmessagesを生成
    messages = [
        # {"role": "system", "content": "あなたはタスク設計の専門家です。与えられた一連のタスクを参考に、形式を揃えて次に来るべきタスクを提案してください。"},
        {"role": "user", "content": prompt}
        # {"role": "system", "content": prompt}
    ]
    formatted_promt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True, 
        tokenize=False
    )
    return formatted_promt

def instruction_generation(
    formatted_prompt, 
    model, 
    tokenizer, 
    max_generation_tokens=512,
    temperature=0.5
):

    with torch.inference_mode():
        # ProcessorまたはTokenizerでテキストのみをまとめて前処理
        inputs = tokenizer(
            formatted_prompt,       # List[str]
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_generation_tokens ,
            do_sample=True,
            temperature=temperature,
            return_dict_in_generate=False,
    )
    
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, prompt_len:]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    del inputs, outputs, generated_ids   # 明示的に削除
    torch.cuda.empty_cache()
    
    return responses

def sample_machine_instructions(machine_instructions, similarities, n):
    """Sample n machine instructions from a list of machine instructions."""
    return random.sample(machine_instructions, min(n, len(machine_instructions)))


def find_word_in_string(w, s):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search(s)


def post_process_response(response):
    if response is None:
        return []
    raw_instructions = re.findall(r'\d+\.\s*(.+)', response)
    instructions = []
    for inst in raw_instructions:
        inst = re.sub(r"\s+", " ", inst).strip()
        if inst == "":
            continue      
        # filter out too short or too long instructions
        if len(inst) <= 3 or len(inst) > 150:
            continue
        instructions.append(inst)
    return instructions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        default="data/gpt3_generations/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        required=True,
        default="data/seed_tasks.jsonl",
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--num_instructions_to_generate",
        type=int,
        default=100,
        help="th",
    )
    parser.add_argument(
        "--use_clf_seed_tasks_only",
        action="store_true",
        help="If specified, we will only use the classification seed tasks to prompt new instructions. This will lead to more classification instructions.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="davinci",
        help="The engine to use."
    )
    parser.add_argument(
        "--num_prompt_instructions",
        type=int,
        default=4,
        help="The number of instructions to use in the prompt."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=8,
        help="The number of requests to send to GPT3 at a time."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="The API key to use. If not specified, the key will be read from the environment variable OPENAI_API_KEY."
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="The organization to use. If not specified, the default organization id will be used."
    )
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen3-30B-A3B",
        type=str,
        help="The organization to use. If not specified, the default organization id will be used."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, tokenizer = initialize_models(args.model_name)
    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
    
    if args.use_clf_seed_tasks_only:
        seed_tasks = [t for t in seed_tasks if t["is_classification"]]
        
    seed_instructions = [t["instruction"] for t in seed_tasks]
    print(f"Loaded {len(seed_instructions)} human-written seed instructions")
    
    os.makedirs(args.batch_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instructions = []
    if os.path.exists(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl")):
        with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "r", encoding="utf-8") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                machine_instructions.append(instruction_info["instruction"])
                request_idx = instruction_info["request_idx"] + 1
        print(f"Loaded {len(machine_instructions)} machine-generated instructions")

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    
    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=args.num_instructions_to_generate)
    if machine_instructions:
        progress_bar.update(len(machine_instructions))

    with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "a", encoding="utf-8") as fout:
        while len(machine_instructions) < args.num_instructions_to_generate:
            batch_inputs = []
            for _ in range(args.request_batch_size):
                # sample machine instructions from the pool
                prompt_instructions = sample_machine_instructions(
                    machine_instructions, 
                    similarities=None,
                    n=2)
                # sample human instructions from the pool
                prompt_instructions += random.sample(seed_instructions, args.num_prompt_instructions - len(prompt_instructions))
                random.shuffle(prompt_instructions)
                prompt = encode_prompt(prompt_instructions,tokenizer,kclassification=args.use_clf_seed_tasks_only)
                batch_inputs.append(prompt)
            
            results = instruction_generation(batch_inputs, model, tokenizer)
            
            instructions = []
            all_metadata = []
            for result in results:
                new_instructions = post_process_response(result)
                instructions += new_instructions
                all_metadata += [result] * len(new_instructions)
                

            for inst, metadata in zip(instructions, all_metadata):
                # with Pool(4) as p:
                #     rouge_scores = p.map(partial(scorer.score, inst), seed_instructions + machine_instructions)
                # rouge_scores = [score["rougeL"].fmeasure for score in rouge_scores]
                # # rouge_scores = [scorer.score(inst, e_inst)["rougeL"].fmeasure for e_inst in human_instructions + machine_instructions]
                # if max(rouge_scores) > 0.7:
                #     continue
                # all_instructions = seed_instructions + machine_instructions
                # most_similar_instructions = {
                #         all_instructions[i] : rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
                #     }
                most_similar_instructions = {}
                
                machine_instructions.append(inst)
                fout.write(json.dumps({
                    "instruction": inst,
                    "most_similar": most_similar_instructions,
                    "avg_similarity_score": 0.0,
                    "metadata": metadata,
                    "request_idx": request_idx
                }, ensure_ascii=False) + "\n")
                progress_bar.update(1)
            request_idx += 1
