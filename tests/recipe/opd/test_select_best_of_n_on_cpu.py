import sys

import numpy as np
import pandas as pd

from recipe.opd.run.opsd_best_of_n import select_best_of_n


def test_select_best_of_n_drops_rows_when_all_candidates_fail(tmp_path, monkeypatch):
    input_path = tmp_path / "candidates.parquet"
    output_path = tmp_path / "selected.parquet"
    dataset = pd.DataFrame(
        {
            "responses": [
                ["row0-c0", "row0-c1", "row0-c2", "row0-c3"],
                ["row1-c0", "row1-c1", "row1-c2", "row1-c3"],
                ["row2-c0", "row2-c1", "row2-c2", "row2-c3"],
            ],
            "extra_info": [
                {"index": 0},
                {"index": 1},
                {"index": 2},
            ],
        }
    )
    dataset.to_parquet(input_path, index=False)

    passing_responses = {"row0-c2", "row2-c0"}

    def fake_score_rows(candidate_records, workers, progress_interval):
        del workers, progress_interval
        return np.asarray(
            [
                1.0 if responses[0] in passing_responses else 0.0
                for responses in candidate_records["responses"]
            ]
        )

    monkeypatch.setattr(select_best_of_n, "_score_rows_parallel", fake_score_rows)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_best_of_n",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--n",
            "4",
            "--workers",
            "1",
        ],
    )

    select_best_of_n.main()

    selected = pd.read_parquet(output_path)
    assert len(selected) == 2
    assert selected["responses"].tolist() == [["row0-c2"], ["row2-c0"]]
    assert [info["index"] for info in selected["extra_info"]] == [0, 2]
    assert [info["best_of_n_selected_index"] for info in selected["extra_info"]] == [2, 0]
    assert all(info["best_of_n_any_correct"] for info in selected["extra_info"])
