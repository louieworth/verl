from __future__ import annotations

import pandas as pd

from recipe.opd.generation.teacher_y_t_prepare import (
    CONDITIONING_TAG,
    build_teacher_prompt_frame,
    slice_frame,
    validate_alignment,
)


def _stage1_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "data_source": "deepscaleR",
                "prompt": [{"role": "user", "content": "Problem 0"}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": "2"},
                "extra_info": {
                    "index": 0,
                    "problem": "What is 1+1?",
                    "expert_cot": "Adding gives \\\\boxed{2}.",
                    "answer": "2",
                },
            },
            {
                "data_source": "deepscaleR",
                "prompt": [{"role": "user", "content": "Problem 1"}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": "4"},
                "extra_info": {
                    "index": 1,
                    "problem": "What is 2+2?",
                    "expert_cot": "Adding gives \\\\boxed{4}.",
                    "answer": "4",
                },
            },
        ]
    )


def test_build_teacher_prompt_frame_conditions_on_expert_without_initial_response():
    output = build_teacher_prompt_frame(_stage1_frame())
    prompt = output.iloc[0]["prompt"][0]["content"]

    assert "What is 1+1?" in prompt
    assert "Adding gives \\\\boxed{2}." in prompt
    assert "Given the expert solution below" in prompt
    assert "**Your Initial Solution:**" not in prompt
    assert output.iloc[0]["extra_info"]["trajectory_conditioning"] == CONDITIONING_TAG


def test_prepared_prompt_slice_aligns_with_stage1_chunk(tmp_path):
    stage1 = _stage1_frame()
    prepared = build_teacher_prompt_frame(stage1)
    teacher_slice = slice_frame(prepared, start_index=1, num_samples=1)
    alignment = slice_frame(stage1, start_index=1, num_samples=1)
    alignment_path = tmp_path / "alignment.parquet"
    alignment.to_parquet(alignment_path, index=False)

    validate_alignment(teacher_slice, str(alignment_path))
