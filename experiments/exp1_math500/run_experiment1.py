"""
Experiment 1: When does the correct answer first appear during LLaDA
diffusion decoding, and what happens after it appears? (MATH-500 version)

Usage:
    python run_experiment1.py --n_problems 50 --steps 64 --gen_length 256 \
        --block_length 64 --out results_math500.jsonl

Each line of the output file is one JSON record, matching the schema from
the experiment design doc:

    {
      "problem_id": ...,
      "ground_truth": ...,
      "total_steps": 64,
      "first_correct_step": 23,        # or null if never correct
      "final_correct": true,
      "correct_at_first_arrival": true,
      "changed_after_arrival": false,
      "post_arrival_steps": 41,
      "post_arrival_fraction": 0.640625,
      "trajectory": ["W","W", ..., "C", "C", ...]   # per-step correctness
    }

Run this on a machine with a CUDA GPU with enough memory for an 8B model in
bf16 (~16GB+ VRAM). Start with a small --n_problems (5-10) to sanity check
before running the full 500.
"""

import argparse
import json
import time

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from llada_step_trace import generate_with_step_trace
from record_utils import build_record

MASK_ID = 126336


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_problems", type=int, default=50, help="how many MATH-500 problems to run")
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--steps", type=int, default=64, help="total diffusion steps")
    ap.add_argument("--gen_length", type=int, default=256, help="generated answer length in tokens")
    ap.add_argument("--block_length", type=int, default=64, help="semi-autoregressive block size; must divide gen_length and steps")
    ap.add_argument("--out", type=str, default="results_math500.jsonl")
    ap.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device found, this will be extremely slow for an 8B model.")

    print(f"Loading {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()

    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    assert tokenizer.pad_token_id != MASK_ID

    print("Loading MATH-500 ...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    end_idx = min(args.start_idx + args.n_problems, len(ds))
    subset = ds.select(range(args.start_idx, end_idx))

    n_written = 0
    with open(args.out, "w") as f_out:
        for row in subset:
            problem_id = row.get("unique_id", row.get("id", n_written))
            problem_text = row["problem"]
            ground_truth = row["answer"]

            messages = [{"role": "user", "content": problem_text}]
            prompt_str = tokenizer.apply_chat_template(
                [messages[0]], add_generation_prompt=True, tokenize=False
            )
            encoded = tokenizer(prompt_str, add_special_tokens=False, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            t0 = time.time()
            _, step_texts = generate_with_step_trace(
                model,
                tokenizer,
                input_ids,
                attention_mask=attention_mask,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=0.0,
                cfg_scale=0.0,
                remasking="low_confidence",
                mask_id=MASK_ID,
            )
            dt = time.time() - t0

            record = build_record(problem_id, ground_truth, args.steps, step_texts)
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()
            n_written += 1

            print(
                f"[{n_written}/{len(subset)}] id={problem_id} "
                f"first_correct_step={record['first_correct_step']} "
                f"final_correct={record['final_correct']} "
                f"({dt:.1f}s)"
            )

    print(f"Wrote {n_written} records to {args.out}")


if __name__ == "__main__":
    main()
