import gzip
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CONFIG_HASH = "fixture-config-hash"


def _step(
    step: int,
    *,
    decoded_text: str,
    extraction_status: str,
    extraction_method: Optional[str],
    extracted_answer: Optional[str],
    ground_truth: str,
    remaining_mask_count: int = 0,
) -> Dict[str, object]:
    correct = None
    if extraction_status == "ok" and extracted_answer is not None:
        correct = extracted_answer == ground_truth
    return {
        "step": step,
        "remaining_mask_count": remaining_mask_count,
        "decoded_text": decoded_text,
        "extraction_status": extraction_status,
        "extraction_method": extraction_method,
        "extracted_answer": extracted_answer,
        "correct": correct,
    }


def _write_problem(path: Path, problem_id: int, ground_truth: str, steps: Iterable[Dict[str, object]]) -> None:
    payload = {
        "config_hash": CONFIG_HASH,
        "problem_id": problem_id,
        "ground_truth": ground_truth,
        "steps": list(steps),
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)


def make_fixtures(base_dir: str | Path) -> Dict[int, str]:
    target = Path(base_dir)
    target.mkdir(parents=True, exist_ok=True)

    case_map: Dict[int, str] = {}
    cases: List[Tuple[int, str, List[Dict[str, object]]]] = []

    def add_case(problem_id: int, label: str, ground_truth: str, steps: List[Dict[str, object]]) -> None:
        case_map[problem_id] = label
        cases.append((problem_id, ground_truth, steps))

    # 1. stable correct
    gt = "42"
    stable_steps = []
    for step in range(1, 65):
        if step < 8:
            text = f"We compute 41 and then 0. Final answer: 41"
            answer = "41"
            status = "ok"
            method = "explicit"
        else:
            text = f"Final answer: {gt}"
            answer = gt
            status = "ok"
            method = "explicit"
        stable_steps.append(_step(step, decoded_text=text, extraction_status=status, extraction_method=method, extracted_answer=answer, ground_truth=gt, remaining_mask_count=max(0, 64 - step)))
    add_case(1001, "stable_correct", gt, stable_steps)

    # 2. partial-commit fake flip (C->W then recover)
    gt = "128"
    partial_steps = []
    for step in range(1, 65):
        if step in {10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64}:
            text = f"Final answer: {gt}"
            answer = gt
            status = "ok"
            method = "explicit"
        elif step == 11:
            text = f"Final answer: 12{gt[2:]}"
            answer = "12"
            status = "ok"
            method = "last_number"
        elif step == 12:
            text = f"Final answer: {gt}"
            answer = gt
            status = "ok"
            method = "explicit"
        else:
            text = f"Reasoning: {step}"
            answer = None
            status = "no_number"
            method = None
        partial_steps.append(_step(step, decoded_text=text, extraction_status=status, extraction_method=method, extracted_answer=answer, ground_truth=gt, remaining_mask_count=max(0, 64 - step)))
    add_case(1002, "partial_commit_fake_flip", gt, partial_steps)

    # 3. extractor-jump flip (old number remains, new later number wins)
    gt = "75"
    jump_steps = []
    for step in range(1, 65):
        if step in {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64}:
            text = f"The answer is {gt}."
            answer = gt
            status = "ok"
            method = "explicit"
        elif step == 8:
            text = f"The answer is 75. Another candidate 12 appears later."
            answer = "12"
            status = "ok"
            method = "last_number"
        else:
            text = f"The answer is {gt}."
            answer = gt
            status = "ok"
            method = "explicit"
        jump_steps.append(_step(step, decoded_text=text, extraction_status=status, extraction_method=method, extracted_answer=answer, ground_truth=gt, remaining_mask_count=max(0, 64 - step)))
    add_case(1003, "extractor_jump_flip", gt, jump_steps)

    # 4. C->N parse failure
    gt = "99"
    failure_steps = []
    for step in range(1, 65):
        if step < 15:
            text = f"Final answer: {gt}"
            answer = gt
            status = "ok"
            method = "explicit"
        else:
            text = "The reasoning keeps going without a final numeric answer."
            answer = None
            status = "no_number"
            method = None
        failure_steps.append(_step(step, decoded_text=text, extraction_status=status, extraction_method=method, extracted_answer=answer, ground_truth=gt, remaining_mask_count=max(0, 64 - step)))
    add_case(1004, "c_to_n_parse_failure", gt, failure_steps)

    # 5. never correct
    gt = "7"
    never_steps = []
    for step in range(1, 65):
        text = f"A numeric guess of 6 appears here."
        answer = "6"
        status = "ok"
        method = "last_number"
        never_steps.append(_step(step, decoded_text=text, extraction_status=status, extraction_method=method, extracted_answer=answer, ground_truth=gt, remaining_mask_count=max(0, 64 - step)))
    add_case(1005, "never_correct", gt, never_steps)

    # 6. recovery after regression (W->C with artifact-only reasons)
    gt = "24"
    recovery_steps = []
    for step in range(1, 65):
        if step == 12:
            text = f"Final answer: 25"
            answer = "25"
            status = "ok"
            method = "last_number"
        elif step in {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64}:
            text = f"Final answer: {gt}."
            answer = gt
            status = "ok"
            method = "explicit"
        else:
            text = f"Final answer: {gt}"
            answer = gt
            status = "ok"
            method = "explicit"
        recovery_steps.append(_step(step, decoded_text=text, extraction_status=status, extraction_method=method, extracted_answer=answer, ground_truth=gt, remaining_mask_count=max(0, 64 - step)))
    add_case(1006, "recovery_after_regression", gt, recovery_steps)

    # 7. semantic regression (other cause)
    gt = "30"
    semantic_steps = []
    for step in range(1, 65):
        if step == 14:
            text = "The answer is 31."
            answer = "31"
            status = "ok"
            method = "explicit"
        elif step == 15:
            text = "The answer is 29."
            answer = "29"
            status = "ok"
            method = "explicit"
        elif step < 14:
            text = f"Final answer: {gt}"
            answer = gt
            status = "ok"
            method = "explicit"
        else:
            text = f"Final answer: {gt}"
            answer = gt
            status = "ok"
            method = "explicit"
        semantic_steps.append(_step(step, decoded_text=text, extraction_status=status, extraction_method=method, extracted_answer=answer, ground_truth=gt, remaining_mask_count=max(0, 64 - step)))
    add_case(1007, "semantic_regression", gt, semantic_steps)

    # 8. W->C recovery from no-number to correct
    gt = "50"
    w_to_c_steps = []
    for step in range(1, 65):
        if step < 10:
            text = f"Final answer: {gt}"
            answer = gt
            status = "ok"
            method = "explicit"
        elif step < 20:
            text = "Reasoning continues without a final answer."
            answer = None
            status = "no_number"
            method = None
        else:
            text = f"Final answer: {gt}"
            answer = gt
            status = "ok"
            method = "explicit"
        w_to_c_steps.append(_step(step, decoded_text=text, extraction_status=status, extraction_method=method, extracted_answer=answer, ground_truth=gt, remaining_mask_count=max(0, 64 - step)))
    add_case(1008, "w_to_c_recovery", gt, w_to_c_steps)

    # 9. repeated wrong, no correct
    gt = "17"
    wrong_repeated_steps = []
    for step in range(1, 65):
        text = "The final answer is 16."
        answer = "16"
        status = "ok"
        method = "last_number"
        wrong_repeated_steps.append(_step(step, decoded_text=text, extraction_status=status, extraction_method=method, extracted_answer=answer, ground_truth=gt, remaining_mask_count=max(0, 64 - step)))
    add_case(1009, "repeated_wrong", gt, wrong_repeated_steps)

    # 10. no extractable answer throughout
    gt = "3"
    no_extract_steps = []
    for step in range(1, 65):
        text = "We are still reasoning through the arithmetic."
        answer = None
        status = "no_number"
        method = None
        no_extract_steps.append(_step(step, decoded_text=text, extraction_status=status, extraction_method=method, extracted_answer=answer, ground_truth=gt, remaining_mask_count=max(0, 64 - step)))
    add_case(1010, "no_extractable_answer", gt, no_extract_steps)

    for problem_id, ground_truth, steps in cases:
        _write_problem(target / f"problem_{problem_id:04d}.json.gz", problem_id, ground_truth, steps)

    return case_map


if __name__ == "__main__":
    import tempfile

    base = Path(tempfile.mkdtemp(prefix="experiment2-fixtures-"))
    out = make_fixtures(base)
    print(sorted(out))
