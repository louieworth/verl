from __future__ import annotations

import asyncio
import argparse
import json

import datasets
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

from recipe.opd.base_completion import BaseCompletionSFTDataset, render_plain_prompt
from recipe.opd.dataset import prepare_experiment_data
from recipe.opd.dataset.data_utils import KLTrainingDataset, build_student_prompt
from recipe.opd.dataset.prepare_experiment_data import (
    CODE_GRPO_MAX_TEST_CASES,
    CODE_GRPO_REWARD_CONTRACT_VERSION,
    CONTAMINATION_HASH_VERSION,
    load_canonical_math_eval_hashes,
    normalize_problem_for_contamination,
    normalized_problem_hash,
    pad_rows,
    render_example,
    select_deepcoder_test_cases,
)
from recipe.opd.generation.y_r_prepare import make_map_fn
from verl.trainer.main_generation_server import generate_per_replica, render_base_completion_prompt


class _Tokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(1, len(text) + 1))

    def decode(self, token_ids, skip_special_tokens=True):
        return "x" * len(token_ids)


def _write_eval_problem(eval_root, dataset_name, problem):
    output = eval_root / dataset_name / f"{dataset_name}_test.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([{"problem": problem}]), output)
    return output


def test_plain_prompt_never_adds_chat_roles():
    assert render_plain_prompt([{"role": "user", "content": "  solve x  "}]) == "solve x\n"
    assert render_base_completion_prompt([{"role": "user", "content": "  solve x  "}]) == "solve x\n"
    with pytest.raises(ValueError, match="unsupported roles"):
        render_plain_prompt([{"role": "system", "content": "system"}])


def test_completion_endpoint_drops_chat_only_request_fields(monkeypatch):
    captured = []

    async def fake_submit(request_index, server_address, **request):
        captured.append(request)
        return request_index, "ok"

    monkeypatch.setenv("VERL_FORCE_BASE_COMPLETION", "true")
    monkeypatch.setattr("verl.trainer.main_generation_server.submit_indexed_request", fake_submit)
    results = asyncio.run(
        generate_per_replica(
            "localhost:1",
            "Qwen/Qwen3-1.7B-Base",
            1,
            {
                "max_tokens": 16,
                "chat_template": "chat-only",
                "add_generation_prompt": False,
            },
            [[{"role": "user", "content": "solve"}]],
            max_concurrency=1,
        )
    )

    assert results == ["ok"]
    assert captured == [
        {
            "model": "Qwen/Qwen3-1.7B-Base",
            "max_tokens": 16,
            "prompt": "solve\n",
            "_base_completion": True,
        }
    ]


def test_rollout_sft_and_kl_use_the_same_completion_prefix(tmp_path):
    tokenizer = _Tokenizer()
    expected_prefix = tokenizer.encode(render_plain_prompt("abc"), add_special_tokens=False)

    parquet = tmp_path / "sft.parquet"
    pd.DataFrame([{"prompt": "abc", "response": "xy"}]).to_parquet(parquet)
    config = OmegaConf.create({"max_length": 16, "pad_mode": "no_padding", "truncation": "error"})
    sft_sample = BaseCompletionSFTDataset(str(parquet), tokenizer, config)[0]
    assert sft_sample["input_ids"][: len(expected_prefix)].tolist() == expected_prefix

    kl_dataset = object.__new__(KLTrainingDataset)
    kl_dataset.tokenizer = tokenizer
    kl_dataset.prompt_truncation = False
    kl_dataset.max_length = 16
    input_ids, _, prompt_len = kl_dataset._build_sequence("abc", "xy")
    assert prompt_len == len(expected_prefix)
    assert input_ids[:prompt_len].tolist() == expected_prefix


def test_base_sft_masks_prompt_and_keeps_response(tmp_path):
    parquet = tmp_path / "sft.parquet"
    pd.DataFrame([{"prompt": "abc", "response": "xy"}]).to_parquet(parquet)
    config = OmegaConf.create({"max_length": 16, "pad_mode": "no_padding", "truncation": "error"})

    sample = BaseCompletionSFTDataset(str(parquet), _Tokenizer(), config)[0]

    # The adapter inserts one newline between prompt and response and one EOS.
    assert sample["input_ids"].numel() == 7
    assert sample["loss_mask"].tolist() == [0, 0, 0, 0, 1, 1, 1]
    assert "attention_mask" not in sample


def test_clean_taco_schema_and_deterministic_batch_padding():
    problem, prompt, solution, metadata = render_example(
        "code",
        {
            "problem": "Add two integers.",
            "solution": "print(sum(map(int, input().split())))",
            "input_output": '{"inputs":["1 2\\n"],"outputs":["3\\n"]}',
            "difficulty": "EASY",
            "source": "unit",
        },
    )
    assert problem == "Add two integers."
    assert prompt == build_student_prompt(problem, task="code")
    assert "Return only raw Python source code" in prompt
    assert "Markdown fences" in prompt
    assert solution.startswith("print")
    assert metadata["reward"]

    rows = [{"extra_info": {"index": 3}, "prompt": prompt}]
    assert pad_rows(rows, multiple=4, source_size=10) == 3
    assert len(rows) == 4
    assert [row["extra_info"]["index"] for row in rows[1:]] == [10, 11, 12]
    assert all(row["extra_info"]["is_batch_padding"] for row in rows[1:])


def test_deepcoder_test_selection_uses_15_longest_inputs_stably():
    inputs = ["x" * length for length in [3, 8, 8, 1, 20, 7, 6, 5, 4, 2, 9, 10, 11, 12, 13, 14, 15]]
    outputs = [f"answer-{index}" for index in range(len(inputs))]

    payload, original_count, selected_count = select_deepcoder_test_cases(
        {"fn_name": "solve", "inputs": inputs, "outputs": outputs}
    )
    selected = json.loads(payload)

    expected_indices = sorted(range(len(inputs)), key=lambda index: (-len(inputs[index]), index))[:15]
    assert original_count == 17
    assert selected_count == CODE_GRPO_MAX_TEST_CASES == 15
    assert selected["fn_name"] == "solve"
    assert selected["inputs"] == [inputs[index] for index in expected_indices]
    assert selected["outputs"] == [outputs[index] for index in expected_indices]


def test_code_prepare_externalizes_tests_and_writes_zstd_artifacts(tmp_path, monkeypatch):
    source = datasets.Dataset.from_list(
        [
            {
                "question": "Read one integer and print it.",
                "starter_code": "",
                "solutions": '["print(int(input()))"]',
                "input_output": json.dumps(
                    {"inputs": ["7\n"] * 100, "outputs": ["7\n"] * 100}
                ),
                "difficulty": "EASY",
                "source": "unit",
            }
        ]
    )
    output_dir = tmp_path / "taco"
    args = argparse.Namespace(
        task="code",
        input="",
        split="train",
        output_dir=str(output_dir),
        cache_dir=str(tmp_path / "cache"),
        model_path="unit-tokenizer",
        max_prompt_length=2048,
        max_response_length=16384,
        pad_to_multiple=4,
        min_prompt_coverage=0.95,
        max_samples=None,
        eval_root=str(tmp_path / "unused-eval"),
        overwrite=True,
    )
    monkeypatch.setattr(prepare_experiment_data, "parse_args", lambda: args)
    monkeypatch.setattr(prepare_experiment_data, "load_source", lambda *unused: source)
    monkeypatch.setattr(
        prepare_experiment_data.AutoTokenizer,
        "from_pretrained",
        lambda *unused, **unused_kwargs: _Tokenizer(),
    )

    prepare_experiment_data.main()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["code_distill_artifact_version"] == "taco_tests_externalized_v1"
    assert manifest["code_grpo_reward_contract_version"] == CODE_GRPO_REWARD_CONTRACT_VERSION
    assert manifest["code_grpo_reward_type"] == "binary_all_selected_tests"
    assert manifest["code_grpo_max_test_cases"] == 15
    assert manifest["parquet_compression"] == "zstd"
    assert set(manifest["artifacts"]) == {"grpo", "sft", "distill"}
    assert set(manifest["output_row_fingerprints"]) == {"grpo", "sft", "distill"}

    grpo_rows = pq.read_table(output_dir / "train_grpo.parquet").to_pylist()
    distill_rows = pq.read_table(output_dir / "train_distill.parquet").to_pylist()
    assert len(grpo_rows) == len(distill_rows) == 4
    selected_tests = json.loads(grpo_rows[0]["reward_model"]["ground_truth"])
    assert len(selected_tests["inputs"]) == len(selected_tests["outputs"]) == 15
    assert grpo_rows[0]["extra_info"]["original_test_case_count"] == 100
    assert grpo_rows[0]["extra_info"]["reward_test_case_count"] == 15
    assert all(row["reward_model"] == {"ground_truth": "", "style": "distill_only"} for row in distill_rows)
    for filename in ("train_grpo.parquet", "train_sft.parquet", "train_distill.parquet"):
        parquet = pq.ParquetFile(output_dir / filename)
        codecs = {
            parquet.metadata.row_group(group).column(column).compression
            for group in range(parquet.num_row_groups)
            for column in range(parquet.metadata.num_columns)
        }
        assert codecs == {"ZSTD"}


def test_contamination_hash_removes_nfkc_whitespace_and_latex_spacing():
    formatted = "Ｆind\u3000$x$ \\quad plus \\, $y$~now."
    compact = "find$x$plus$y$now."

    assert normalize_problem_for_contamination(formatted) == compact
    assert normalized_problem_hash(formatted) == normalized_problem_hash(compact)
    assert normalized_problem_hash(compact) != normalized_problem_hash(compact.rstrip("."))


def test_math_prepare_drops_eval_overlap_from_grpo_and_sft_and_pins_eval_inputs(
    tmp_path, monkeypatch
):
    eval_root = tmp_path / "eval"
    contaminated_problem = "Ｆind\u3000$x$ \\quad plus \\, $y$~now."
    _write_eval_problem(eval_root, "aime25", "An unrelated AIME25 problem.")
    _write_eval_problem(eval_root, "aime26", "find$x$plus$y$now.")
    hmmt_path = _write_eval_problem(eval_root, "hmmt26", "An unrelated HMMT problem.")
    _write_eval_problem(eval_root, "amobench", "An unrelated AMO problem.")

    hashes, summary = load_canonical_math_eval_hashes(eval_root)
    assert normalized_problem_hash(contaminated_problem) in hashes
    assert [item["dataset"] for item in summary["datasets"]] == [
        "aime25",
        "aime26",
        "hmmt26",
        "amobench",
    ]
    assert all(len(item["parquet_sha256"]) == 64 for item in summary["datasets"])
    assert len(summary["fingerprint"]) == 64

    source = datasets.Dataset.from_list(
        [
            {
                "problem": contaminated_problem,
                "solution": "The answer is \\boxed{1}.",
                "Answer": "1",
                "source": "unit-overlap",
            },
            {
                "problem": "A retained training problem.",
                "solution": "The answer is \\boxed{2}.",
                "Answer": "2",
                "source": "unit-clean",
            },
        ]
    )
    output_dir = tmp_path / "prepared"
    args = argparse.Namespace(
        task="math",
        input="",
        split="train",
        output_dir=str(output_dir),
        cache_dir=str(tmp_path / "cache"),
        model_path="unit-tokenizer",
        max_prompt_length=2048,
        max_response_length=16384,
        pad_to_multiple=4,
        min_prompt_coverage=0.95,
        max_samples=None,
        eval_root=str(eval_root),
        overwrite=True,
    )
    monkeypatch.setattr(prepare_experiment_data, "parse_args", lambda: args)
    monkeypatch.setattr(prepare_experiment_data, "load_source", lambda *unused: source)
    monkeypatch.setattr(
        prepare_experiment_data.AutoTokenizer,
        "from_pretrained",
        lambda *unused, **unused_kwargs: _Tokenizer(),
    )

    prepare_experiment_data.main()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["contamination_hash_version"] == CONTAMINATION_HASH_VERSION
    assert manifest["dropped_contaminated_rows"] == 1
    assert manifest["dropped_contamination_details"] == [
        {
            "source_index": 0,
            "source": "unit-overlap",
            "sha256": normalized_problem_hash(contaminated_problem),
            "eval_matches": [
                {
                    "dataset": "aime26",
                    "eval_index": 0,
                    "sha256": normalized_problem_hash(contaminated_problem),
                }
            ],
        }
    ]
    assert manifest["retained_grpo_rows"] == manifest["retained_sft_rows"] == 1
    assert manifest["grpo_padding_rows"] == manifest["sft_padding_rows"] == 3
    assert manifest["grpo_rows"] == manifest["sft_rows"] == 4
    assert len(manifest["prepared_dataset_fingerprint"]) == 64
    assert set(manifest["output_row_fingerprints"]) == {"grpo", "sft"}

    for filename in ("train_grpo.parquet", "train_sft.parquet"):
        rows = pq.read_table(output_dir / filename).to_pylist()
        assert len(rows) == 4
        assert all(row["extra_info"]["problem"] == "A retained training problem." for row in rows)
        assert all(row["extra_info"]["index"] != 0 for row in rows)

    # An unchanged eval suite is compatible with the existing manifest.
    args.overwrite = False
    prepare_experiment_data.main()

    # Byte/content changes to any canonical eval parquet invalidate reuse.
    _write_eval_problem(eval_root, "hmmt26", "A changed HMMT problem.")
    assert hmmt_path.is_file()
    with pytest.raises(RuntimeError, match="contamination_eval"):
        prepare_experiment_data.main()


def test_srd_prompt_is_capped_without_dropping_rewrite_instructions():
    tokenizer = _Tokenizer()
    example = {
        "data_source": "unit",
        "ability": "math",
        "responses": ["student draft " * 20],
        "extra_info": {
            "problem": "Solve x.",
            "expert_cot": "expert proof " * 10,
        },
        "reward_model": {"ground_truth": "1"},
    }
    mapped = make_map_fn(
        "opsd",
        "math",
        tokenizer=tokenizer,
        max_prompt_tokens=700,
    )(example)
    prompt = mapped["prompt"][0]["content"]
    assert len(tokenizer.encode(prompt + "\n", add_special_tokens=False)) <= 700
    assert "Output ONLY the rewritten solution" in prompt
    assert "truncated to fit the native context" in prompt

    mapped["responses"] = ["rewritten answer"]
    kl_dataset = object.__new__(KLTrainingDataset)
    kl_dataset.tokenizer = tokenizer
    kl_dataset.kl_type = "forward"
    kl_dataset.max_length = 2048
    kl_dataset.use_initial_response = True
    kl_dataset.prompt_truncation = False
    kl_dataset.log_difficulty_buckets = False
    kl_dataset.distill_mode = "opsd"
    kl_dataset.task = "math"
    kl_dataset.corrected_responses = None
    kl_dataset.data = [mapped]
    item = kl_dataset._prepare_forward_kl_item(0)
    first_loss_position = int(item["teacher_loss_mask"].nonzero()[0])
    assert first_loss_position == len(tokenizer.encode(prompt + "\n", add_special_tokens=False)) - 1
