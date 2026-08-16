"""
Aggregate + visualize results_math500.jsonl produced by run_experiment1.py.

Usage:
    python analyze_experiment1.py --in results_math500.jsonl --out_dir figs/

Produces:
    - summary printed to stdout (mean/median first_correct_step, regression rates)
    - <out_dir>/first_correct_step_histogram.png
    - <out_dir>/results_table.csv   (the per-problem table from the design doc)
"""

import argparse
import json
import os

import pandas as pd
import matplotlib.pyplot as plt


def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="figs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = load_records(args.in_path)
    df = pd.DataFrame(records)

    n_total = len(df)
    ever_correct = df[df["first_correct_step"].notna()]
    n_ever_correct = len(ever_correct)

    print(f"Total problems: {n_total}")
    print(f"Ever correct (answer appeared at some step): {n_ever_correct} "
          f"({100 * n_ever_correct / n_total:.1f}%)")

    if n_ever_correct > 0:
        print(f"Mean first_correct_step: {ever_correct['first_correct_step'].mean():.2f}")
        print(f"Median first_correct_step: {ever_correct['first_correct_step'].median():.2f}")
        print(f"Mean post_arrival_fraction: {ever_correct['post_arrival_fraction'].mean():.3f}")

        final_correct_bool = ever_correct["final_correct"].astype(bool)
        changed_bool = ever_correct["changed_after_arrival"].astype(bool)

        stayed_correct = (final_correct_bool & ~changed_bool).sum()
        temporarily_wrong_but_recovered = (final_correct_bool & changed_bool).sum()
        ended_wrong = (~final_correct_bool).sum()
        assert stayed_correct + temporarily_wrong_but_recovered + ended_wrong == n_ever_correct, (
            "category counts don't sum to n_ever_correct -- something is still wrong"
        )

        print("\nAmong problems where the correct answer appeared before the final step:")
        print(f"  stayed correct:              {stayed_correct} ({100 * stayed_correct / n_ever_correct:.1f}%)")
        print(f"  temporarily wrong, recovered: {temporarily_wrong_but_recovered} ({100 * temporarily_wrong_but_recovered / n_ever_correct:.1f}%)")
        print(f"  ended incorrect:              {ended_wrong} ({100 * ended_wrong / n_ever_correct:.1f}%)")

        plt.figure(figsize=(7, 4))
        plt.hist(ever_correct["first_correct_step"], bins=20, edgecolor="black")
        plt.xlabel("First correct step")
        plt.ylabel("Number of problems")
        plt.title("When does the correct answer first appear? (MATH-500, LLaDA-8B-Instruct)")
        plt.tight_layout()
        hist_path = os.path.join(args.out_dir, "first_correct_step_histogram.png")
        plt.savefig(hist_path, dpi=150)
        print(f"\nSaved histogram to {hist_path}")

    table = df.copy()
    table["Changed After?"] = table["changed_after_arrival"].map({True: "Yes", False: "No"}).fillna("—")
    table["Post-Arrival %"] = table["post_arrival_fraction"].apply(
        lambda x: f"{100 * x:.0f}%" if pd.notna(x) else "—"
    )
    table["First Correct"] = table["first_correct_step"].apply(lambda x: int(x) if pd.notna(x) else "Never")
    table["Final Correct"] = table["final_correct"].map({True: "Yes", False: "No"})
    out_table = table[["problem_id", "ground_truth", "First Correct", "Final Correct", "Changed After?", "Post-Arrival %"]]
    out_table.columns = ["ID", "GT", "First Correct", "Final Correct", "Changed After?", "Post-Arrival %"]
    csv_path = os.path.join(args.out_dir, "results_table.csv")
    out_table.to_csv(csv_path, index=False)
    print(f"Saved per-problem table to {csv_path}")


if __name__ == "__main__":
    main()
