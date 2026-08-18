"""
Experiment 2B follow-up: what if we only remasked when the model was unsure?

OFFLINE SIMULATION. This script never loads a model and never touches a GPU. It
reruns nothing; it reuses the paired artifacts written by run_2b.py.

------------------------------------------------------------------------------
WHY THE SIMULATION IS VALID  (the whole basis of this script -- check it here)
------------------------------------------------------------------------------
run_2b.py ran two decodes per problem from the same seed and config:

    run_a = normal diffusion, no intervention
    run_b = identical decoding, plus ONE remasking of the committed
            "Final answer: N" tokens at the first step where that span is
            fully committed and mask-free

A confidence gate is a decision made AT the trigger step, using only
`commit_time_confidence` -- the softmax probability the model assigned to each
answer token at the moment it committed it. Those probabilities are a property
of the decoding prefix, which is byte-identical in run_a and run_b up to the
trigger step (same seed, same config, same greedy argmax, and the intervention
is the first thing that differs). So the gate statistic is already known before
the branch, and gating changes only WHICH of two already-observed decodes we get:

  * gate BLOCKS the intervention  -> no remasking happens -> the run is exactly
    run_a, and the outcome is exactly `outcome_a`. Nothing is counterfactual
    here: an ungated normal run is literally what run_a is.
  * gate ALLOWS the intervention  -> the run is exactly run_b (the gate changes
    no other decoding decision), so the outcome is exactly `outcome_b`.
  * problem never triggered an intervention (`intervened == False`) -> there was
    nothing to gate, so the outcome is `outcome_a` under EVERY threshold, and
    run_a and run_b are the same decode anyway.

Therefore accuracy under any threshold t is computable in closed form from the
pairs we already have, with no re-decoding:

    outcome(t) = outcome_b  if (intervened and stat < t) else outcome_a

The one thing this does NOT simulate: a gate that fires more than once, or that
changes the trigger rule. MAX_INTERVENTIONS is 1 in run_2b.py, and the trigger
is fixed, so every policy in this sweep is "the same single intervention,
sometimes skipped". Any richer policy would need a real rerun.

Gate direction: we intervene when the model was UNCONFIDENT, i.e. `stat < t`.
So t = 0.0 intervenes on nobody (equals the normal baseline) and t = 1.0
intervenes on essentially everybody (equals the remasked baseline).

Usage:
    python experiments/experiment2/confidence_gate_sweep.py
    python experiments/experiment2/confidence_gate_sweep.py --stat mean
    python experiments/experiment2/confidence_gate_sweep.py --pairs-dir PATH
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_PAIRS_DIR = Path("artifacts/experiment2b_gsm8k/pairs")
DEFAULT_OUTPUT_DIR = Path("experiments/experiment2/results_gate")
THRESHOLD_STEP = 0.05

# Row labels for the two reference policies. They are not thresholds: they are
# the two ends of the sweep, computed directly rather than through the gate, so
# they reproduce run_2b.py's reported numbers exactly.
NEVER_LABEL = "never (baseline: normal)"
NO_GATE_LABEL = "no gate (baseline: remasked)"


# ---------------------------------------------------------------- statistics
def mcnemar_exact_p(n_helped: int, n_hurt: int) -> float:
    """Two-sided exact McNemar test on the discordant pairs.

    Under H0 a discordant pair is equally likely to fall either way, so the
    smaller count is Binomial(n_helped + n_hurt, 0.5). The two-sided p-value is
    twice the lower tail (capped at 1.0). Exact, not the chi-square
    approximation -- the discordant counts here are small enough that the
    approximation would be doing real work.
    """
    n = n_helped + n_hurt
    if n == 0:
        return 1.0
    k = min(n_helped, n_hurt)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def accuracy(outcomes: List[Optional[bool]]) -> float:
    """Fraction of problems answered correctly.

    `None` means the extractor could not parse a final answer; run_2b.py counts
    those as not-correct in the denominator, and so do we, or the baselines
    would not reproduce.
    """
    if not outcomes:
        return float("nan")
    return sum(1 for outcome in outcomes if outcome is True) / len(outcomes)


# ---------------------------------------------------------------- loading
def gate_statistic(confidences: Dict[str, float], stat: str) -> Optional[float]:
    values = [float(v) for v in confidences.values()]
    if not values:
        return None
    return min(values) if stat == "min" else sum(values) / len(values)


def load_records(pairs_dir: Path, stat: str) -> List[Dict[str, Any]]:
    paths = sorted(pairs_dir.glob("pair_*.json.gz"))
    if not paths:
        raise SystemExit(f"no pair_*.json.gz artifacts under {pairs_dir}")

    records: List[Dict[str, Any]] = []
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        intervention = payload["run_b"].get("intervention")
        intervened = intervention is not None
        confidences = (intervention or {}).get("commit_time_confidence") or {}
        records.append(
            {
                "problem_id": int(payload["problem_id"]),
                "outcome_a": payload["outcome_a"],
                "outcome_b": payload["outcome_b"],
                "intervened": intervened,
                "trigger_step": (intervention or {}).get("trigger_step"),
                "n_remasked": len((intervention or {}).get("remasked_positions") or []),
                "gate_stat": gate_statistic(confidences, stat) if intervened else None,
            }
        )
    return records


# ---------------------------------------------------------------- the gate
def gated_outcomes(records: List[Dict[str, Any]], threshold: float) -> List[Optional[bool]]:
    """Outcome each problem would have had under a `stat < threshold` gate.

    A problem with no gate statistic (intervened but no commit-time confidence
    was recorded) cannot be judged by the gate, so the gate blocks -- it falls
    back to the normal run. There are none of these in the current artifacts;
    the count is reported so a future rerun cannot hide them.
    """
    outcomes: List[Optional[bool]] = []
    for record in records:
        allow = (
            record["intervened"]
            and record["gate_stat"] is not None
            and record["gate_stat"] < threshold
        )
        outcomes.append(record["outcome_b"] if allow else record["outcome_a"])
    return outcomes


def evaluate(
    records: List[Dict[str, Any]],
    label: str,
    threshold: Optional[float],
    baseline_accuracy: float,
    policy: str = "gate",
) -> Dict[str, Any]:
    """Score one policy against the normal-run baseline on `records`."""
    if policy == "never":
        outcomes = [record["outcome_a"] for record in records]
        n_intervened = 0
    elif policy == "no_gate":
        outcomes = [record["outcome_b"] for record in records]
        n_intervened = sum(1 for record in records if record["intervened"])
    else:
        outcomes = gated_outcomes(records, float(threshold))
        n_intervened = sum(
            1
            for record in records
            if record["intervened"]
            and record["gate_stat"] is not None
            and record["gate_stat"] < float(threshold)
        )

    # Helped / hurt are always measured against the normal run: a discordant
    # pair is a problem where this policy and plain decoding disagree.
    n_helped = sum(
        1
        for record, outcome in zip(records, outcomes)
        if record["outcome_a"] is not True and outcome is True
    )
    n_hurt = sum(
        1
        for record, outcome in zip(records, outcomes)
        if record["outcome_a"] is True and outcome is not True
    )
    accuracy_value = accuracy(outcomes)
    return {
        "policy": policy,
        "label": label,
        "threshold": threshold,
        "n_problems": len(records),
        "n_intervened": n_intervened,
        "n_correct": sum(1 for outcome in outcomes if outcome is True),
        "accuracy": accuracy_value,
        "gain_pp": (accuracy_value - baseline_accuracy) * 100.0,
        "n_helped": n_helped,
        "n_hurt": n_hurt,
        "mcnemar_p": mcnemar_exact_p(n_helped, n_hurt),
    }


def sweep(records: List[Dict[str, Any]], thresholds: np.ndarray) -> pd.DataFrame:
    baseline = accuracy([record["outcome_a"] for record in records])
    rows = [evaluate(records, NEVER_LABEL, None, baseline, policy="never")]
    for threshold in thresholds:
        rows.append(
            evaluate(records, f"{threshold:.2f}", float(threshold), baseline, policy="gate")
        )
    rows.append(evaluate(records, NO_GATE_LABEL, None, baseline, policy="no_gate"))
    return pd.DataFrame(rows)


def best_gate_row(frame: pd.DataFrame) -> pd.Series:
    """Highest accuracy gain among genuine gate thresholds.

    Ties break toward fewer interventions, then toward the lower threshold: if
    two gates score the same, prefer the one that meddles less.
    """
    gates = frame[frame["policy"] == "gate"]
    ordered = gates.sort_values(
        ["gain_pp", "n_intervened", "threshold"], ascending=[False, True, True]
    )
    return ordered.iloc[0]


# ---------------------------------------------------------------- reporting
def markdown_table(frame: pd.DataFrame) -> str:
    header = (
        "| threshold | n_intervened | accuracy | gain (pp) | helped | hurt | McNemar p |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['label']} | {int(row['n_intervened'])} | {row['accuracy'] * 100:.2f}% | "
            f"{row['gain_pp']:+.2f} | {int(row['n_helped'])} | {int(row['n_hurt'])} | "
            f"{row['mcnemar_p']:.4f} |"
        )
    return "\n".join(lines)


def make_figure(frame: pd.DataFrame, output_path: Path, stat: str, baselines: Dict[str, float]) -> None:
    gates = frame[frame["policy"] == "gate"]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        gates["threshold"], gates["accuracy"] * 100, color="#4C78A8",
        marker="o", markersize=4, label="accuracy under gate",
    )
    axis.axhline(
        baselines["normal"] * 100, color="#E45756", linestyle="--", linewidth=1,
        label=f"normal baseline ({baselines['normal'] * 100:.2f}%)",
    )
    axis.axhline(
        baselines["remasked"] * 100, color="#54A24B", linestyle="--", linewidth=1,
        label=f"always remask ({baselines['remasked'] * 100:.2f}%)",
    )
    axis.set_xlabel(f"confidence gate threshold (intervene when {stat} commit confidence < t)")
    axis.set_ylabel("accuracy (%)")
    axis.set_title(f"Confidence-gated remasking sweep ({stat} commit-time confidence)")

    secondary = axis.twinx()
    secondary.plot(
        gates["threshold"], gates["n_intervened"], color="#F58518",
        marker="s", markersize=3, linestyle=":", label="n_intervened",
    )
    secondary.set_ylabel("problems intervened on")

    handles, labels = axis.get_legend_handles_labels()
    handles2, labels2 = secondary.get_legend_handles_labels()
    axis.legend(handles + handles2, labels + labels2, loc="lower right", fontsize=9)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def sanity_checks(frame: pd.DataFrame, records: List[Dict[str, Any]]) -> List[str]:
    lines = []
    never = frame[frame["policy"] == "never"].iloc[0]
    no_gate = frame[frame["policy"] == "no_gate"].iloc[0]
    lines.append(
        f"never-intervene row reproduces normal accuracy: {never['accuracy'] * 100:.2f}%"
    )
    lines.append(
        f"no-gate row reproduces remasked accuracy:       {no_gate['accuracy'] * 100:.2f}%"
    )

    gates = frame[frame["policy"] == "gate"].sort_values("threshold")
    counts = gates["n_intervened"].tolist()
    monotone = all(earlier <= later for earlier, later in zip(counts, counts[1:]))
    lines.append(
        "n_intervened monotonically non-decreasing in threshold: "
        f"{monotone} ({counts[0]} at t=0.00 -> {counts[-1]} at t=1.00)"
    )

    ungated = sum(1 for record in records if record["intervened"])
    missing = sum(
        1 for record in records if record["intervened"] and record["gate_stat"] is None
    )
    lines.append(
        f"intervened problems in artifacts: {ungated}; "
        f"of those, missing a gate statistic: {missing}"
    )
    return lines


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline confidence-gate sweep over experiment 2B pairs (no model, no GPU)."
    )
    parser.add_argument("--pairs-dir", type=Path, default=DEFAULT_PAIRS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--stat", choices=["min", "mean"], default="min",
        help="gate on the min (default) or mean commit-time confidence of the remasked tokens",
    )
    args = parser.parse_args()

    records = load_records(args.pairs_dir, args.stat)
    thresholds = np.round(np.arange(0.0, 1.0 + THRESHOLD_STEP / 2, THRESHOLD_STEP), 2)

    frame = sweep(records, thresholds)
    baselines = {
        "normal": float(frame[frame["policy"] == "never"].iloc[0]["accuracy"]),
        "remasked": float(frame[frame["policy"] == "no_gate"].iloc[0]["accuracy"]),
    }
    best = best_gate_row(frame)

    # ---- honest split: choose on the even half, report on the odd half.
    even = [record for record in records if record["problem_id"] % 2 == 0]
    odd = [record for record in records if record["problem_id"] % 2 == 1]
    even_baseline = accuracy([record["outcome_a"] for record in even])
    odd_baseline = accuracy([record["outcome_a"] for record in odd])
    even_frame = sweep(even, thresholds)
    even_best = best_gate_row(even_frame)
    selected_threshold = float(even_best["threshold"])
    held_out = evaluate(
        odd, f"{selected_threshold:.2f}", selected_threshold, odd_baseline, policy="gate"
    )
    held_out_no_gate = evaluate(odd, NO_GATE_LABEL, None, odd_baseline, policy="no_gate")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "sweep.csv", index=False)
    make_figure(frame, output_dir / "gate_sweep.png", args.stat, baselines)

    summary = {
        "pairs_dir": str(args.pairs_dir),
        "stat": args.stat,
        "n_problems": len(records),
        "n_intervened_ungated": sum(1 for record in records if record["intervened"]),
        "threshold_grid": [float(t) for t in thresholds],
        "baselines": {
            "normal_accuracy": baselines["normal"],
            "remasked_accuracy": baselines["remasked"],
            "remasked_gain_pp": (baselines["remasked"] - baselines["normal"]) * 100.0,
            "remasked_n_helped": int(frame[frame["policy"] == "no_gate"].iloc[0]["n_helped"]),
            "remasked_n_hurt": int(frame[frame["policy"] == "no_gate"].iloc[0]["n_hurt"]),
            "remasked_mcnemar_p": float(frame[frame["policy"] == "no_gate"].iloc[0]["mcnemar_p"]),
        },
        "best_threshold_in_sample": {
            "threshold": float(best["threshold"]),
            "accuracy": float(best["accuracy"]),
            "gain_pp": float(best["gain_pp"]),
            "n_intervened": int(best["n_intervened"]),
            "n_helped": int(best["n_helped"]),
            "n_hurt": int(best["n_hurt"]),
            "mcnemar_p": float(best["mcnemar_p"]),
            "WARNING": (
                "This threshold was selected on the same 1319 problems it is scored on. "
                "The gain is optimistically biased and is NOT a validated result."
            ),
        },
        "held_out_split": {
            "scheme": "problem_id % 2 == 0 selects the threshold; % 2 == 1 evaluates it",
            "n_select": len(even),
            "n_eval": len(odd),
            "threshold_selected_on_even": selected_threshold,
            "gain_pp_on_even": float(even_best["gain_pp"]),
            "accuracy_on_odd": held_out["accuracy"],
            "odd_baseline_accuracy": odd_baseline,
            "gain_pp_on_odd": held_out["gain_pp"],
            "n_intervened_on_odd": held_out["n_intervened"],
            "n_helped_on_odd": held_out["n_helped"],
            "n_hurt_on_odd": held_out["n_hurt"],
            "mcnemar_p_on_odd": held_out["mcnemar_p"],
            "ungated_gain_pp_on_odd": held_out_no_gate["gain_pp"],
            "gate_beats_ungated_on_odd": bool(
                held_out["gain_pp"] > held_out_no_gate["gain_pp"]
            ),
        },
    }
    (output_dir / "sweep.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- stdout
    print(f"\nloaded {len(records)} pairs from {args.pairs_dir}  (gate stat: {args.stat})\n")
    print(markdown_table(frame))

    print("\n### Sanity checks")
    for line in sanity_checks(frame, records):
        print(f"- {line}")

    print("\n### Best threshold (in-sample)")
    print(
        f"- t = {best['threshold']:.2f}: accuracy {best['accuracy'] * 100:.2f}% "
        f"({best['gain_pp']:+.2f} pp over normal), intervened on {int(best['n_intervened'])}, "
        f"helped {int(best['n_helped'])} / hurt {int(best['n_hurt'])}, "
        f"McNemar exact p = {best['mcnemar_p']:.4f}"
    )
    print(
        "! WARNING: this threshold was chosen by scanning the same 1319 problems it is\n"
        "! scored on. That is selection on test data and the gain above is biased upward.\n"
        "! It must be validated on a held-out split or a different dataset before it goes\n"
        "! in a paper. The honest split below is the number to quote."
    )

    print("\n### Held-out split (choose on even problem_ids, report on odd)")
    print(
        f"- selected t = {selected_threshold:.2f} on the even half "
        f"(n={len(even)}, gain there {even_best['gain_pp']:+.2f} pp)"
    )
    print(
        f"- on the odd half (n={len(odd)}): accuracy {held_out['accuracy'] * 100:.2f}% "
        f"vs baseline {odd_baseline * 100:.2f}% = {held_out['gain_pp']:+.2f} pp, "
        f"helped {held_out['n_helped']} / hurt {held_out['n_hurt']}, "
        f"McNemar exact p = {held_out['mcnemar_p']:.4f}"
    )
    print(
        f"- ungated remasking on the same odd half: {held_out_no_gate['gain_pp']:+.2f} pp "
        f"(helped {held_out_no_gate['n_helped']} / hurt {held_out_no_gate['n_hurt']}, "
        f"p = {held_out_no_gate['mcnemar_p']:.4f})"
    )
    if held_out["gain_pp"] < held_out_no_gate["gain_pp"]:
        print(
            "- VERDICT: the tuned gate does WORSE on held-out data than just always\n"
            "  remasking. The gate is not carrying its weight; do not claim it helps."
        )
    elif held_out["gain_pp"] <= 0:
        print(
            "- VERDICT: the tuned gate gives no held-out gain over plain decoding.\n"
            "  Do not claim it helps."
        )
    else:
        print(
            "- VERDICT: the tuned gate beats both plain decoding and ungated remasking\n"
            "  on held-out problems, but on one split only -- treat as suggestive."
        )

    print(f"\nwrote {output_dir}/sweep.csv, sweep.json, gate_sweep.png\n")


if __name__ == "__main__":
    main()
