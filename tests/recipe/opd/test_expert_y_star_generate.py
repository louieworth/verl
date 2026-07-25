import pandas as pd
import pytest

from recipe.opd.generation.expert_y_star_generate import (
    attach_expert_responses,
    select_rows,
)


def test_expert_solution_is_used_verbatim_as_response():
    source = pd.DataFrame(
        {
            "prompt": [[{"role": "user", "content": "p0"}]],
            "extra_info": [{"expert_cot": "reasoning then answer", "answer": "answer-only"}],
        }
    )

    output = attach_expert_responses(source)

    assert output.iloc[0]["responses"] == ["reasoning then answer"]
    assert output.iloc[0]["extra_info"]["trajectory_source"] == "expert_solution"
    assert output.iloc[0]["extra_info"]["answer"] == "answer-only"


def test_empty_expert_solution_is_rejected_without_answer_fallback():
    source = pd.DataFrame(
        {
            "prompt": [[{"role": "user", "content": "p0"}]],
            "extra_info": [{"expert_cot": "", "answer": "answer-only"}],
        }
    )

    with pytest.raises(ValueError, match="non-empty"):
        attach_expert_responses(source)


def test_precomputed_trajectory_can_be_sliced_for_multistep_training():
    source = pd.DataFrame(
        {
            "extra_info": [
                {"expert_cot": f"solution-{index}"}
                for index in range(5)
            ],
            "responses": [[f"solution-{index}"] for index in range(5)],
        }
    )

    selected = select_rows(source, start_index=2, num_samples=2)
    output = attach_expert_responses(selected)

    assert output["responses"].tolist() == [["solution-2"], ["solution-3"]]
