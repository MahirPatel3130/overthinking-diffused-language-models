import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extractor import ANY_NUMBER, canonical_decimal


TRANSITION_ORDER = [
    ("W", "C"),
    ("C", "W"),
    ("C", "N"),
    ("W", "W"),
    ("C", "C"),
]


def _state_for_step(step: Dict[str, Any]) -> str:
    if step.get("extraction_status") == "ok" and step.get("correct") is True:
        return "C"
    if step.get("extraction_status") == "ok" and step.get("correct") is False:
        return "W"
    return "N"


def _iter_artifacts(trajectory_dir: Path) -> Iterable[Path]:
    if not trajectory_dir.exists():
        return []
    return sorted(trajectory_dir.glob("problem_*.json.gz"))


def _load_artifact(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _canonical_numbers_in_text(text: str) -> List[str]:
    numbers: List[str] = []
    for match in ANY_NUMBER.finditer(text):
        raw = re.sub(r"\s+", "", match.group(0))
        try:
            numbers.append(canonical_decimal(raw))
        except Exception:
            continue
    return numbers


def classify_flip_cause(prev_text: str, curr_text: str, prev_answer: Optional[str], curr_answer: Optional[str]) -> str:
    if prev_answer in (None, "") or curr_answer in (None, ""):
        return "other"

    prev_text_norm = prev_text or ""
    curr_text_norm = curr_text or ""
    prev_answer_norm = str(prev_answer)
    curr_answer_norm = str(curr_answer)

    if prev_answer_norm == curr_answer_norm:
        return "other"

    # A number was committed digit-by-digit in-place.
    if (
        prev_answer_norm in {curr_answer_norm[: len(prev_answer_norm)], curr_answer_norm[-len(prev_answer_norm):]}
        or curr_answer_norm.startswith(prev_answer_norm)
        or curr_answer_norm.endswith(prev_answer_norm)
        or prev_answer_norm.startswith(curr_answer_norm)
        or prev_answer_norm.endswith(curr_answer_norm)
    ):
        if prev_answer_norm in curr_text_norm or curr_answer_norm in prev_text_norm:
            return "partial_commit"

    # The original answer is still present in the new text, but a later number now wins.
    if prev_answer_norm in curr_text_norm:
        current_numbers = _canonical_numbers_in_text(curr_text_norm)
        if len(current_numbers) >= 2 and curr_answer_norm == current_numbers[-1]:
            return "extractor_jump"

    return "other"


def analyze_problem(payload: Dict[str, Any]) -> Dict[str, Any]:
    steps = payload.get("steps", [])
    transitions = Counter()
    flip_records: List[Dict[str, Any]] = []
    flip_counts_by_cause = Counter()
    n_to_anything = 0

    for idx in range(len(steps) - 1):
        prev = steps[idx]
        curr = steps[idx + 1]
        prev_state = _state_for_step(prev)
        curr_state = _state_for_step(curr)
        transitions[(prev_state, curr_state)] += 1
        if prev_state == "N":
            n_to_anything += 1

        if {prev_state, curr_state} == {"C", "W"}:
            cause = classify_flip_cause(
                str(prev.get("decoded_text", "")),
                str(curr.get("decoded_text", "")),
                prev.get("extracted_answer"),
                curr.get("extracted_answer"),
            )
            flip_counts_by_cause[cause] += 1
            flip_records.append({
                "problem_id": payload.get("problem_id"),
                "prev_step": prev.get("step"),
                "curr_step": curr.get("step"),
                "prev_state": prev_state,
                "curr_state": curr_state,
                "prev_answer": prev.get("extracted_answer"),
                "curr_answer": curr.get("extracted_answer"),
                "prev_text": prev.get("decoded_text"),
                "curr_text": curr.get("decoded_text"),
                "cause": cause,
            })

    ever_correct = any(step.get("correct") is True for step in steps)
    first_correct_step = None
    for step in steps:
        if step.get("correct") is True:
            first_correct_step = step.get("step")
            break

    final_correct = steps[-1].get("correct") is True if steps else False
    if not ever_correct:
        category = "never_correct"
        temporary_regression_type = None
    elif first_correct_step is None:
        category = "never_correct"
        temporary_regression_type = None
    else:
        later = [step for step in steps if step.get("step", 0) > first_correct_step]
        changed_after_arrival = any(_state_for_step(step) != "C" for step in later)
        if not changed_after_arrival:
            category = "stable_correct"
            temporary_regression_type = None
        elif final_correct:
            category = "temporary_regression"
            temporary_regression_type = "artifact_only" if flip_counts_by_cause["other"] == 0 else "semantic"
        else:
            category = "ended_incorrect"
            temporary_regression_type = None

    transition_summary = {
        "W_to_C": transitions.get(("W", "C"), 0),
        "C_to_W": transitions.get(("C", "W"), 0),
        "C_to_N": transitions.get(("C", "N"), 0),
        "W_to_W": transitions.get(("W", "W"), 0),
        "C_to_C": transitions.get(("C", "C"), 0),
        "N_to_anything": n_to_anything,
    }
    transition_summary.update({f"{src}_{dst}": transitions.get((src, dst), 0) for src, dst in [("N", "C"), ("N", "W"), ("N", "N")]})

    problem_summary = {
        "problem_id": payload.get("problem_id"),
        "ground_truth": payload.get("ground_truth"),
        "first_correct_step": first_correct_step,
        "ever_correct": ever_correct,
        "n_flips": sum(1 for record in flip_records),
        "n_partial_commit_flips": flip_counts_by_cause.get("partial_commit", 0),
        "n_extractor_jump_flips": flip_counts_by_cause.get("extractor_jump", 0),
        "n_other_flips": flip_counts_by_cause.get("other", 0),
        "trajectory_category": category,
        "temporary_regression_type": temporary_regression_type,
        "transition_counts": transition_summary,
        "flip_records": flip_records,
    }
    return problem_summary


def _write_summary(output_dir: Path, summary_table: Dict[str, Any], problem_summary: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame([summary_table])
    parameters = summary_df.columns.tolist()
    summary_df = summary_df[parameters]
    summary_df.to_csv(output_dir / "aggregate_summary.csv", index=False)
    (output_dir / "aggregate_summary.json").write_text(json.dumps(summary_table, indent=2, sort_keys=True), encoding="utf-8")

    problem_df = pd.DataFrame(problem_summary)
    if not problem_df.empty:
        problem_df = problem_df.sort_values("problem_id").reset_index(drop=True)
        problem_df.to_csv(output_dir / "problem_summary.csv", index=False)
        problem_df.to_json(output_dir / "problem_summary.json", orient="records", indent=2)


def _plot_flip_breakdown(output_dir: Path, flip_counts: Counter) -> None:
    labels = ["partial_commit", "extractor_jump", "other"]
    values = [flip_counts.get(label, 0) for label in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color=["#4C78A8", "#F58518", "#E45756"])
    ax.set_title("Flip cause breakdown")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "flip_cause_breakdown.png", dpi=200)
    plt.close(fig)


def _plot_transition_counts(output_dir: Path, transition_counts: Counter) -> None:
    labels = ["W→C", "C→W", "C→N", "W→W", "C→C", "N→anything"]
    values = [
        transition_counts.get(("W", "C"), 0),
        transition_counts.get(("C", "W"), 0),
        transition_counts.get(("C", "N"), 0),
        transition_counts.get(("W", "W"), 0),
        transition_counts.get(("C", "C"), 0),
        sum(transition_counts.get(("N", state), 0) for state in ["C", "W", "N"]),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values, color=["#54A24B", "#EECA3B", "#B279A2", "#F58518", "#4C78A8", "#E45756"])
    ax.set_title("Transition counts")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "transition_counts.png", dpi=200)
    plt.close(fig)


def analyze_trajectories(trajectory_dir: str, output_dir: str | None = None) -> Dict[str, Any]:
    root_dir = Path(trajectory_dir)
    result_dir = Path(output_dir) if output_dir else root_dir / "results"

    problem_summaries: List[Dict[str, Any]] = []
    overall_transition_counts = Counter()
    flip_counts = Counter()
    per_problem_transition_counts: List[Dict[str, Any]] = []

    for artifact_path in _iter_artifacts(root_dir):
        payload = _load_artifact(artifact_path)
        problem_summary = analyze_problem(payload)
        problem_summaries.append(problem_summary)
        per_problem_transition_counts.append({
            "problem_id": payload.get("problem_id"),
            **problem_summary["transition_counts"],
        })

        for (src, dst), count in Counter({
            ("W", "C"): problem_summary["transition_counts"]["W_to_C"],
            ("C", "W"): problem_summary["transition_counts"]["C_to_W"],
            ("C", "N"): problem_summary["transition_counts"]["C_to_N"],
            ("W", "W"): problem_summary["transition_counts"]["W_to_W"],
            ("C", "C"): problem_summary["transition_counts"]["C_to_C"],
            ("N", "C"): problem_summary["transition_counts"].get("N_to_C", 0),
            ("N", "W"): problem_summary["transition_counts"].get("N_to_W", 0),
            ("N", "N"): problem_summary["transition_counts"].get("N_to_N", 0),
        }).items():
            overall_transition_counts[(src, dst)] += count

        flip_counts["partial_commit"] += problem_summary["n_partial_commit_flips"]
        flip_counts["extractor_jump"] += problem_summary["n_extractor_jump_flips"]
        flip_counts["other"] += problem_summary["n_other_flips"]

    problem_df = pd.DataFrame(problem_summaries).sort_values("problem_id").reset_index(drop=True)
    ever_correct = int(problem_df["ever_correct"].sum()) if not problem_df.empty else 0
    n_problems = int(len(problem_df))
    ever_correct_pct = (ever_correct / n_problems * 100.0) if n_problems else 0.0
    ever_correct_df = problem_df[problem_df["ever_correct"]].copy() if not problem_df.empty else pd.DataFrame()
    stayed_correct = int((ever_correct_df["trajectory_category"] == "stable_correct").sum()) if not ever_correct_df.empty else 0
    temporarily_wrong = int((ever_correct_df["trajectory_category"] == "temporary_regression").sum()) if not ever_correct_df.empty else 0
    ended_wrong = int((ever_correct_df["trajectory_category"] == "ended_incorrect").sum()) if not ever_correct_df.empty else 0

    summary_table = {
        "n_problems": n_problems,
        "n_ever_correct": ever_correct,
        "ever_correct_pct": ever_correct_pct,
        "n_stayed_correct": stayed_correct,
        "stayed_correct_pct": (stayed_correct / ever_correct * 100.0) if ever_correct else 0.0,
        "n_temporarily_wrong": temporarily_wrong,
        "temporarily_wrong_pct": (temporarily_wrong / ever_correct * 100.0) if ever_correct else 0.0,
        "n_ended_wrong": ended_wrong,
        "ended_wrong_pct": (ended_wrong / ever_correct * 100.0) if ever_correct else 0.0,
        "partial_commit_flips": flip_counts.get("partial_commit", 0),
        "extractor_jump_flips": flip_counts.get("extractor_jump", 0),
        "other_flips": flip_counts.get("other", 0),
    }

    _write_summary(result_dir, summary_table, problem_summaries)
    _plot_flip_breakdown(result_dir, flip_counts)
    _plot_transition_counts(result_dir, overall_transition_counts)

    return {
        "trajectory_dir": str(root_dir),
        "output_dir": str(result_dir),
        "problem_summary": problem_summaries,
        "summary_table": summary_table,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze experiment1 trajectory transitions.")
    parser.add_argument("--trajectory-dir", required=True, help="Directory containing problem_*.json.gz trajectory artifacts")
    parser.add_argument("--output-dir", default="experiment2/results", help="Directory for summary csv/json and plots")
    args = parser.parse_args()
    analyze_trajectories(args.trajectory_dir, args.output_dir)


if __name__ == "__main__":
    main()
