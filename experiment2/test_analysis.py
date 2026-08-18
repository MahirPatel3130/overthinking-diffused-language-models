import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment2.make_fixtures import make_fixtures
from experiment2.analyze_transitions import analyze_trajectories


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="exp2-fixtures-") as tmpdir:
        temp_path = Path(tmpdir)
        case_map = make_fixtures(temp_path)
        result_dir = temp_path / "results"

        summary = analyze_trajectories(str(temp_path), str(result_dir))

        problem_ids = {int(row["problem_id"]) for row in summary["problem_summary"]}
        assert set(case_map) == problem_ids

        problem_rows = summary["problem_summary"]
        row_by_id = {int(row["problem_id"]): row for row in problem_rows}

        assert row_by_id[1001]["trajectory_category"] == "stable_correct"
        assert row_by_id[1002]["trajectory_category"] == "temporary_regression"
        assert row_by_id[1003]["trajectory_category"] == "temporary_regression"
        assert row_by_id[1004]["trajectory_category"] == "ended_incorrect"
        assert row_by_id[1005]["trajectory_category"] == "never_correct"
        assert row_by_id[1006]["trajectory_category"] == "temporary_regression"
        assert row_by_id[1007]["trajectory_category"] == "temporary_regression"
        assert row_by_id[1008]["trajectory_category"] == "temporary_regression"
        assert row_by_id[1009]["trajectory_category"] == "never_correct"
        assert row_by_id[1010]["trajectory_category"] == "never_correct"

        assert summary["summary_table"]["ever_correct_pct"] > 0
        assert summary["summary_table"]["stayed_correct_pct"] >= 0
        assert (result_dir / "aggregate_summary.csv").exists()
        assert (result_dir / "aggregate_summary.json").exists()
        assert (result_dir / "flip_cause_breakdown.png").exists()
        assert (result_dir / "transition_counts.png").exists()

        print(json.dumps({
            "n_problems": len(problem_rows),
            "summary_table": summary["summary_table"],
        }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
