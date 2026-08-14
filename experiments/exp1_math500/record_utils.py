"""
Turns a list of per-step decoded strings into the per-problem JSON record
described in the experiment design doc. Kept dependency-free (just
math_utils) so it can be unit-tested without torch/transformers installed.
"""

from math_utils import extract_boxed, is_equiv


def build_record(problem_id, ground_truth, total_steps, step_texts):
    trajectory = []
    first_correct_step = None
    for step_idx, text in enumerate(step_texts, start=1):  # 1-indexed steps
        pred = extract_boxed(text)
        correct = pred is not None and is_equiv(pred, ground_truth)
        trajectory.append("C" if correct else "W")
        if correct and first_correct_step is None:
            first_correct_step = step_idx

    final_correct = trajectory[-1] == "C" if trajectory else False

    if first_correct_step is None:
        return {
            "problem_id": problem_id,
            "ground_truth": ground_truth,
            "total_steps": total_steps,
            "first_correct_step": None,
            "final_correct": False,
            "correct_at_first_arrival": None,
            "changed_after_arrival": None,
            "post_arrival_steps": None,
            "post_arrival_fraction": None,
            "trajectory": trajectory,
        }

    after = trajectory[first_correct_step:]  # steps strictly after arrival
    changed_after_arrival = any(s == "W" for s in after)
    post_arrival_steps = total_steps - first_correct_step
    post_arrival_fraction = post_arrival_steps / total_steps

    return {
        "problem_id": problem_id,
        "ground_truth": ground_truth,
        "total_steps": total_steps,
        "first_correct_step": first_correct_step,
        "final_correct": final_correct,
        "correct_at_first_arrival": True,
        "changed_after_arrival": changed_after_arrival,
        "post_arrival_steps": post_arrival_steps,
        "post_arrival_fraction": post_arrival_fraction,
        "trajectory": trajectory,
    }
