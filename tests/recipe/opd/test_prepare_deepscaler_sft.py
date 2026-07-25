import pandas as pd
import pytest

from recipe.opd.run.sft_deepscaler.prepare_deepscaler_sft import build_sft_dataset


def test_uses_only_nonempty_solution_as_assistant_target():
    source = pd.DataFrame(
        [
            {"problem": "p0", "answer": "wrong target 0", "solution": "reasoning 0"},
            {"problem": "p1", "answer": "answer-only row", "solution": ""},
            {"problem": "p2", "answer": "answer-only row", "solution": "   "},
            {"problem": "p3", "answer": "wrong target 3", "solution": "reasoning 3"},
        ]
    )

    output, stats = build_sft_dataset(source)

    assert stats == {
        "input_rows": 4,
        "kept_nonempty_solution_rows": 2,
        "dropped_empty_solution_rows": 2,
    }
    assert [messages[1]["content"] for messages in output["messages"]] == [
        "reasoning 0",
        "reasoning 3",
    ]
    assert [info["answer"] for info in output["extra_info"]] == [
        "wrong target 0",
        "wrong target 3",
    ]
    assert all(info["target_source"] == "solution" for info in output["extra_info"])


def test_never_falls_back_to_answer_when_solution_column_is_missing():
    source = pd.DataFrame([{"problem": "p0", "answer": "answer-only"}])

    with pytest.raises(ValueError, match="must not fall back"):
        build_sft_dataset(source)
