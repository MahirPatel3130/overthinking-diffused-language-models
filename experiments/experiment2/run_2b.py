"""
Experiment 2B: does letting the model reconsider a committed answer help or hurt?

Paired runs on the same problems with identical seed/config:
  Run A = normal diffusion (byte-identical decoding to experiment 1)
  Run B = diffusion + remasking of the committed final-answer tokens

Config is pinned to experiment1_gsm8k.ipynb. Do not change SCIENTIFIC_CONFIG
values without also rerunning experiment 1, or the runs stop being comparable.

Usage:
    CUDA_VISIBLE_DEVICES=6 python experiment2/run_2b.py --stage smoke
    CUDA_VISIBLE_DEVICES=6 python experiment2/run_2b.py --stage full
    python experiment2/run_2b.py --analyze-only
"""

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import random
import re
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extractor import extract_answer, parse_ground_truth  # noqa: E402

# ---------------------------------------------------------------- config
# Every value below is copied from experiment1_gsm8k.ipynb.
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
MODEL_REVISION = "6059b30"
DATASET_ID = "gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "test"
SEED = 20250814
NUM_STEPS = 64
GENERATION_LENGTH = 256
BLOCK_LENGTH = 256
TEMPERATURE = 0.0
CFG_SCALE = 0.0
REMASKING = "low_confidence"
MASK_TOKEN_ID = 126336
SMOKE_SIZE = 20
EXPECTED_TEST_SIZE = 1319
EXTRACTOR_VERSION = "1"

PROMPT_TEMPLATE = (
    "Solve this grade-school math problem carefully. Show concise reasoning. "
    "End with exactly: Final answer: <number>\n\nProblem: {question}"
)

# 2B-specific intervention parameters (these DO belong in the config hash).
INTERVENTION = "remask_final_answer_span"
INTERVENTION_TRIGGER = "step_after_answer_commit_region_mask_free"
MAX_INTERVENTIONS = 1

SCIENTIFIC_CONFIG = {
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "dataset_id": DATASET_ID,
    "dataset_config": DATASET_CONFIG,
    "dataset_split": DATASET_SPLIT,
    "seed": SEED,
    "num_steps": NUM_STEPS,
    "generation_length": GENERATION_LENGTH,
    "block_length": BLOCK_LENGTH,
    "temperature": TEMPERATURE,
    "cfg_scale": CFG_SCALE,
    "remasking": REMASKING,
    "mask_token_id": MASK_TOKEN_ID,
    "extractor_version": EXTRACTOR_VERSION,
    "intervention": INTERVENTION,
    "intervention_trigger": INTERVENTION_TRIGGER,
    "max_interventions": MAX_INTERVENTIONS,
}
CONFIG_JSON = json.dumps(SCIENTIFIC_CONFIG, sort_keys=True, separators=(",", ":"))
CONFIG_HASH = hashlib.sha256(CONFIG_JSON.encode()).hexdigest()[:12]

PROJECT_ROOT = Path(os.getenv("EXP2B_PROJECT_ROOT", Path.cwd())).resolve()
RUN_ROOT = PROJECT_ROOT / "artifacts" / "experiment2b_gsm8k"
CACHE_ROOT = PROJECT_ROOT / "artifacts" / "cache"
PAIR_DIR = RUN_ROOT / "pairs"
RESULT_DIR = RUN_ROOT / "results"
FAILURE_DIR = RUN_ROOT / "failures"

# The explicit "Final answer: N" form we intervene on. Kept deliberately
# narrower than the extractor's fallbacks: we only remask when the model has
# actually committed an explicit final answer, never a stray number.
FINAL_ANSWER_RE = re.compile(
    r"(?is)final\s+answer\s*[:=]?\s*"
    r"([-+]?\s*(?:\$\s*)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
)


def log(message: str) -> None:
    stamp = dt.datetime.now().strftime("%H:%M:%S")
    print(f"{stamp} | {message}", flush=True)


# ---------------------------------------------------------------- io helpers
def pair_path(problem_id: int) -> Path:
    return PAIR_DIR / f"pair_{problem_id:04d}.json.gz"


def atomic_write_json_gz(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
    os.replace(temporary, path)


def load_pair(problem_id: int) -> Optional[Dict[str, Any]]:
    path = pair_path(problem_id)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("config_hash") != CONFIG_HASH:
            return None
        if payload.get("problem_id") != problem_id:
            return None
        if "run_a" not in payload or "run_b" not in payload:
            return None
        return payload
    except (OSError, EOFError, json.JSONDecodeError, KeyError, TypeError):
        return None


def append_failure(row: Dict[str, Any]) -> None:
    with (FAILURE_DIR / "failures.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------- schedule
def transfer_schedule(mask_count: int, steps: int) -> List[int]:
    """Even split of `mask_count` unmaskings across `steps`, remainder first.

    Experiment 1 precomputes this once for the whole block. Remasking adds
    tokens back after that point, so we recompute over the REMAINING steps
    whenever we intervene; otherwise the schedule under-counts and the run
    ends with masks still in the sequence.
    """
    if steps <= 0:
        return []
    base = mask_count // steps
    remainder = mask_count % steps
    return [base + (1 if i < remainder else 0) for i in range(steps)]


# ---------------------------------------------------------------- answer span
def token_offset_map(
    tokenizer, token_ids: List[int]
) -> Tuple[str, List[Tuple[int, int, int]]]:
    """Decode committed tokens one at a time and keep char spans per position.

    Returns the concatenated text and a list of (char_start, char_end, position).
    Masked positions contribute nothing, mirroring skip_special_tokens=True on
    the full sequence. Individual-token decoding can differ from whole-sequence
    decoding on whitespace, which is why the intervention log records the exact
    text that was remasked so it can be checked by hand.
    """
    pieces: List[Tuple[int, int, int]] = []
    text_parts: List[str] = []
    cursor = 0
    for position, token_id in enumerate(token_ids):
        if token_id == MASK_TOKEN_ID:
            continue
        piece = tokenizer.decode([token_id], skip_special_tokens=True)
        if not piece:
            continue
        text_parts.append(piece)
        pieces.append((cursor, cursor + len(piece), position))
        cursor += len(piece)
    return "".join(text_parts), pieces


def locate_final_answer_positions(
    tokenizer, token_ids: List[int]
) -> Optional[Dict[str, Any]]:
    """Token positions holding the number in a committed 'Final answer: N'.

    Returns None when no explicit final answer is committed yet, or when the
    number's own tokens are not all committed (partial commit -> not ready).
    """
    text, pieces = token_offset_map(tokenizer, token_ids)
    matches = list(FINAL_ANSWER_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    span_start, span_end = match.span(1)
    positions = [
        position
        for (char_start, char_end, position) in pieces
        if char_start < span_end and char_end > span_start
    ]
    if not positions:
        return None
    # Every token overlapping the number must be committed (they are, by
    # construction of the offset map) and contiguous in the sequence.
    positions = sorted(set(positions))
    last = max(positions)
    # The number is only genuinely committed when nothing can still be inserted
    # into or around it. The offset map skips masked slots, so "Final answer: 0"
    # can look contiguous while masked positions sit between the marker and the
    # digit. Require the whole region from the marker through one token past the
    # number to be mask-free.
    marker_start = match.span(0)[0]
    marker_positions = [
        position
        for (char_start, char_end, position) in pieces
        if char_start < span_end and char_end > marker_start
    ]
    if not marker_positions:
        return None
    region_start = min(marker_positions)
    region_end = min(last + 2, len(token_ids))
    if any(token_ids[i] == MASK_TOKEN_ID for i in range(region_start, region_end)):
        return None
    if region_end <= last + 1:
        return None
    if positions != list(range(min(positions), last + 1)):
        return None
    return {
        "positions": positions,
        "answer_text": match.group(1).strip(),
        "matched_text": text[max(0, span_start - 40): span_end + 10],
    }


# ---------------------------------------------------------------- generation
@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    do_remask: bool,
) -> Dict[str, Any]:
    """One diffusion run. Identical to experiment 1 when do_remask=False.

    Single block (BLOCK_LENGTH == GENERATION_LENGTH), greedy argmax,
    low-confidence remasking selection, batch size 1.
    """
    device = prompt_ids.device
    prompt_width = prompt_ids.shape[1]
    total_width = prompt_width + GENERATION_LENGTH

    tokens = torch.full(
        (1, total_width), MASK_TOKEN_ID, dtype=torch.long, device=device
    )
    tokens[:, :prompt_width] = prompt_ids
    full_attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones((1, GENERATION_LENGTH), dtype=attention_mask.dtype, device=device),
        ],
        dim=-1,
    )

    assert GENERATION_LENGTH % BLOCK_LENGTH == 0
    assert BLOCK_LENGTH == GENERATION_LENGTH, "single-block config, per experiment 1"

    schedule = transfer_schedule(GENERATION_LENGTH, NUM_STEPS)
    step_records: List[Dict[str, Any]] = []
    intervention: Optional[Dict[str, Any]] = None
    interventions_done = 0
    step = 0

    while step < NUM_STEPS:
        mask_index = tokens.eq(MASK_TOKEN_ID)
        candidate_mask = mask_index.clone()
        candidate_mask[:, :prompt_width] = False

        logits = model(tokens, attention_mask=full_attention_mask).logits
        predicted = torch.argmax(logits, dim=-1)
        probabilities = F.softmax(logits.to(torch.float32), dim=-1)
        predicted_probability = torch.gather(
            probabilities, -1, predicted.unsqueeze(-1)
        ).squeeze(-1)
        predicted = torch.where(mask_index, predicted, tokens)
        confidence = torch.where(candidate_mask, predicted_probability, -torch.inf)

        count = min(int(schedule[step]), int(candidate_mask.sum().item()))
        transfer_index = torch.zeros_like(tokens, dtype=torch.bool)
        if count:
            chosen = torch.topk(confidence[0], k=count).indices
            transfer_index[0, chosen] = True
        tokens[transfer_index] = predicted[transfer_index]

        committed_now = (
            (transfer_index[0].nonzero().flatten() - prompt_width).tolist()
        )
        committed_confidence = {
            int(position): float(predicted_probability[0, position + prompt_width])
            for position in committed_now
            if position >= 0
        }

        generated = tokens[0, prompt_width:].detach().cpu()
        generated_ids = generated.tolist()
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
        parsed = extract_answer(decoded)

        step += 1
        step_records.append(
            {
                "step": step,
                "remaining_mask_count": int(generated.eq(MASK_TOKEN_ID).sum().item()),
                "decoded_text": decoded,
                "token_ids": generated_ids,
                "committed_positions": sorted(int(p) for p in committed_now if p >= 0),
                "committed_confidence": committed_confidence,
                "extraction_status": parsed["status"],
                "extraction_method": parsed["method"],
                "extracted_answer": parsed["answer"],
            }
        )

        del logits, probabilities, predicted_probability, confidence

        # --- intervention: remask the committed final-answer tokens once
        if do_remask and interventions_done < MAX_INTERVENTIONS and step < NUM_STEPS:
            located = locate_final_answer_positions(tokenizer, generated_ids)
            if located is not None:
                positions = located["positions"]
                absolute = [p + prompt_width for p in positions]
                pre_confidence = {}
                for record in reversed(step_records):
                    for position in positions:
                        key = str(position)
                        if key in pre_confidence:
                            continue
                        if position in record["committed_confidence"]:
                            pre_confidence[key] = record["committed_confidence"][position]
                        elif str(position) in record["committed_confidence"]:
                            pre_confidence[key] = record["committed_confidence"][str(position)]
                tokens[0, absolute] = MASK_TOKEN_ID
                interventions_done += 1
                remaining_masks = int(tokens[0, prompt_width:].eq(MASK_TOKEN_ID).sum().item())
                remaining_steps = NUM_STEPS - step
                if remaining_steps <= 0:
                    # No budget left to re-predict: extend rather than leave
                    # masks in the output, and record that we did.
                    extra = max(1, len(positions))
                    schedule = schedule + transfer_schedule(remaining_masks, extra)
                else:
                    schedule = schedule[:step] + transfer_schedule(
                        remaining_masks, remaining_steps
                    )
                intervention = {
                    "trigger_step": step,
                    "remasked_positions": positions,
                    "pre_intervention_answer": located["answer_text"],
                    "context_text": located["matched_text"],
                    "commit_time_confidence": pre_confidence,
                    "steps_extended": max(0, len(schedule) - NUM_STEPS),
                }

    final_ids = tokens[0, prompt_width:].detach().cpu().tolist()
    final_text = tokenizer.decode(final_ids, skip_special_tokens=True)
    return {
        "steps": step_records,
        "final_text": final_text,
        "final_extraction": extract_answer(final_text),
        "remaining_masks": int(
            sum(1 for token_id in final_ids if token_id == MASK_TOKEN_ID)
        ),
        "intervention": intervention,
    }


# ---------------------------------------------------------------- driver
def make_prompt(tokenizer, question: str) -> str:
    message = {"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}
    return tokenizer.apply_chat_template(
        [message], add_generation_prompt=True, tokenize=False
    )


def run_problem(model, tokenizer, row: Dict[str, Any]) -> Dict[str, Any]:
    prompt = make_prompt(tokenizer, row["question"])
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    device = next(model.parameters()).device
    prompt_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    ground_truth = parse_ground_truth(row["answer"])

    started = time.perf_counter()
    seed_everything(SEED)
    run_a = generate(model, tokenizer, prompt_ids, attention_mask, do_remask=False)
    seed_everything(SEED)
    run_b = generate(model, tokenizer, prompt_ids, attention_mask, do_remask=True)
    elapsed = time.perf_counter() - started

    def outcome(run: Dict[str, Any]) -> Optional[bool]:
        parsed = run["final_extraction"]
        if parsed["status"] != "ok":
            return None
        return parsed["answer"] == ground_truth

    return {
        "config_hash": CONFIG_HASH,
        "scientific_config": SCIENTIFIC_CONFIG,
        "problem_id": row["problem_id"],
        "question": row["question"],
        "ground_truth": ground_truth,
        "prompt": prompt,
        "elapsed_seconds": elapsed,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_a": run_a,
        "run_b": run_b,
        "outcome_a": outcome(run_a),
        "outcome_b": outcome(run_b),
        "intervened": run_b["intervention"] is not None,
    }


def paired_summary() -> Dict[str, Any]:
    rows = []
    for path in sorted(PAIR_DIR.glob("pair_*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        intervention = payload["run_b"].get("intervention") or {}
        confidences = list((intervention.get("commit_time_confidence") or {}).values())
        rows.append(
            {
                "problem_id": payload["problem_id"],
                "ground_truth": payload["ground_truth"],
                "outcome_a": payload["outcome_a"],
                "outcome_b": payload["outcome_b"],
                "intervened": payload["intervened"],
                "trigger_step": intervention.get("trigger_step"),
                "pre_intervention_answer": intervention.get("pre_intervention_answer"),
                "answer_a": payload["run_a"]["final_extraction"]["answer"],
                "answer_b": payload["run_b"]["final_extraction"]["answer"],
                "answer_changed": payload["run_a"]["final_extraction"]["answer"]
                != payload["run_b"]["final_extraction"]["answer"],
                "min_commit_confidence": min(confidences) if confidences else None,
                "mean_commit_confidence": (
                    sum(confidences) / len(confidences) if confidences else None
                ),
            }
        )

    def transition(row: Dict[str, Any]) -> str:
        a, b = row["outcome_a"], row["outcome_b"]
        if a is None or b is None:
            return "unparsed"
        if a and b:
            return "C->C"
        if a and not b:
            return "C->W (remasking hurt)"
        if not a and b:
            return "W->C (remasking helped)"
        return "W->W"

    for row in rows:
        row["transition"] = transition(row)

    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["transition"]] = counts.get(row["transition"], 0) + 1

    intervened = [row for row in rows if row["intervened"]]
    summary = {
        "n_problems": len(rows),
        "n_intervened": len(intervened),
        "accuracy_normal": (
            sum(1 for row in rows if row["outcome_a"] is True) / len(rows)
            if rows
            else None
        ),
        "accuracy_remasked": (
            sum(1 for row in rows if row["outcome_b"] is True) / len(rows)
            if rows
            else None
        ),
        "transition_counts": counts,
        "answer_changed_rate_among_intervened": (
            sum(1 for row in intervened if row["answer_changed"]) / len(intervened)
            if intervened
            else None
        ),
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "paired_summary.json").write_text(json.dumps(summary, indent=2))
    try:
        import pandas as pd

        pd.DataFrame(rows).sort_values("problem_id").to_csv(
            RESULT_DIR / "paired_table.csv", index=False
        )
    except ImportError:
        pass
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    for directory in [PAIR_DIR, RESULT_DIR, FAILURE_DIR, CACHE_ROOT]:
        directory.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        print(json.dumps(paired_summary(), indent=2))
        return

    log(f"config_hash={CONFIG_HASH} stage={args.stage}")
    log(f"cuda_visible_devices={os.getenv('CUDA_VISIBLE_DEVICES', 'not set')}")
    assert torch.cuda.is_available(), "no GPU visible"

    local_model_path = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=CACHE_ROOT / "huggingface",
        )
    )
    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        revision="main",
        cache_dir=str(CACHE_ROOT / "datasets"),
    )
    assert len(dataset) == EXPECTED_TEST_SIZE, (len(dataset), EXPECTED_TEST_SIZE)

    # Same seeded permutation experiment 1 uses, so smoke ids line up exactly.
    rng = np.random.default_rng(SEED)
    smoke_ids = sorted(rng.permutation(EXPECTED_TEST_SIZE)[:SMOKE_SIZE].tolist())
    selected = smoke_ids if args.stage == "smoke" else list(range(EXPECTED_TEST_SIZE))
    if args.limit:
        selected = selected[: args.limit]

    from transformers import AutoModel, AutoTokenizer

    seed_everything(SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        str(local_model_path), trust_remote_code=True, local_files_only=True
    )
    model = (
        AutoModel.from_pretrained(
            str(local_model_path),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        .to("cuda")
        .eval()
    )

    pending = [pid for pid in selected if load_pair(pid) is None]
    log(f"selected={len(selected)} pending={len(pending)}")

    for position, problem_id in enumerate(pending, start=1):
        try:
            source = dataset[problem_id]
            payload = run_problem(
                model,
                tokenizer,
                {
                    "problem_id": problem_id,
                    "question": source["question"],
                    "answer": source["answer"],
                },
            )
            atomic_write_json_gz(pair_path(problem_id), payload)
            log(
                f"{position}/{len(pending)} id={problem_id} "
                f"normal={payload['outcome_a']} remasked={payload['outcome_b']} "
                f"intervened={payload['intervened']} "
                f"{payload['elapsed_seconds']:.1f}s"
            )
        except Exception as exc:  # noqa: BLE001
            log(f"FAILED id={problem_id}: {type(exc).__name__}: {exc}")
            append_failure(
                {
                    "problem_id": problem_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
            )
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()

    print(json.dumps(paired_summary(), indent=2))


if __name__ == "__main__":
    main()
