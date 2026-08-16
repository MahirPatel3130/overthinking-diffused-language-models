"""One-off debug script: prints the raw decoded text (not just C/W) for
specific dataset indices, so we can see WHY extraction failed -- wrong
answer, truncation, or a format extract_boxed doesn't handle."""
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from llada_step_trace import generate_with_step_trace
from math_utils import extract_boxed, is_equiv

MASK_ID = 126336
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True,
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0},
).eval()

ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

for idx in [1, 2]:  # the two problems that stayed wrong
    row = ds[idx]
    print("=" * 70)
    print(f"idx={idx} id={row.get('unique_id')}")
    print(f"PROBLEM: {row['problem'][:200]}")
    print(f"GROUND TRUTH: {row['answer']}")

    prompt_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["problem"]}], add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(prompt_str, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    _, step_texts = generate_with_step_trace(
        model, tokenizer, input_ids, attention_mask=attention_mask,
        steps=64, gen_length=256, block_length=64,
        temperature=0.0, cfg_scale=0.0, remasking="low_confidence", mask_id=MASK_ID,
    )
    final_text = step_texts[-1]
    print(f"\nFINAL DECODED TEXT (last 500 chars):\n{final_text[-500:]}")
    pred = extract_boxed(final_text)
    print(f"\nEXTRACTED: {pred!r}")
    print(f"IS_EQUIV to gold: {is_equiv(pred, row['answer']) if pred else 'N/A (no boxed found)'}")
