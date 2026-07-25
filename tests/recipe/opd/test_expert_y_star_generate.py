import pandas as pd
import pytest

from recipe.opd.generation.expert_y_star_generate import (
    attach_expert_responses,
    select_rows,
)
from recipe.opd.generation.y_o_prepare import make_map_fn_stage1


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


def test_empty_expert_solution_falls_back_to_answer():
    source = pd.DataFrame(
        {
            "prompt": [[{"role": "user", "content": "p0"}]],
            "extra_info": [{"expert_cot": "", "answer": "answer-only"}],
        }
    )

    output = attach_expert_responses(source)

    assert output.iloc[0]["responses"] == ["answer-only"]
    assert output.iloc[0]["extra_info"]["expert_solution_source"] == "answer"
    assert output.iloc[0]["extra_info"]["trajectory_source"] == "expert_answer"


def test_row_without_solution_or_answer_is_rejected():
    source = pd.DataFrame(
        {
            "prompt": [[{"role": "user", "content": "p0"}]],
            "extra_info": [{"expert_cot": "", "answer": ""}],
        }
    )

    with pytest.raises(ValueError, match="solution or answer"):
        attach_expert_responses(source)


def test_stage1_prefers_solution_over_answer_for_y_star():
    output = make_map_fn_stage1()(
        {
            "problem": "problem",
            "solution": "full chain of thought",
            "answer": "final answer",
        },
        0,
    )

    assert output["extra_info"]["expert_cot"] == "full chain of thought"
    assert output["extra_info"]["expert_solution_source"] == "solution"


def test_stage1_falls_back_to_answer_for_y_star():
    output = make_map_fn_stage1()(
        {
            "problem": "problem",
            "solution": " ",
            "answer": "final answer",
        },
        0,
    )

    assert output["extra_info"]["expert_cot"] == "final answer"
    assert output["extra_info"]["expert_solution_source"] == "answer"


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
