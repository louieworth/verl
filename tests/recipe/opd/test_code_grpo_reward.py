from __future__ import annotations

import json

from recipe.opd.run.grpo import code_reward


def _tests(count: int = 3) -> str:
    return json.dumps(
        {
            "inputs": [f"input-{index}" for index in range(count)],
            "outputs": [f"output-{index}" for index in range(count)],
        }
    )


def test_binary_reward_requires_every_selected_test(monkeypatch):
    monkeypatch.setattr(
        code_reward,
        "apps_check_correctness",
        lambda **unused: ([True, True, True], [{}]),
    )
    assert code_reward.compute_binary_execution_reward("print('ok')", _tests()) == 1.0

    monkeypatch.setattr(
        code_reward,
        "apps_check_correctness",
        lambda **unused: ([True, False], [{}]),
    )
    assert code_reward.compute_binary_execution_reward("print('ok')", _tests()) == 0.0


def test_binary_reward_maps_execution_errors_and_invalid_suites_to_zero(monkeypatch):
    def fail(**unused):
        raise TimeoutError("executor timeout")

    monkeypatch.setattr(code_reward, "apps_check_correctness", fail)
    assert code_reward.compute_binary_execution_reward("while True: pass", _tests()) == 0.0
    assert code_reward.compute_binary_execution_reward("pass", "not-json") == 0.0
    assert code_reward.compute_binary_execution_reward(
        "pass", json.dumps({"inputs": ["x"] * 16, "outputs": ["y"] * 16})
    ) == 0.0


def test_reward_accepts_raw_source_and_legacy_python_fence(monkeypatch):
    generations = []
    timeouts = []

    def capture(**kwargs):
        generations.append(kwargs["generation"])
        timeouts.append(kwargs["timeout"])
        return [True], [{}]

    monkeypatch.setattr(code_reward, "apps_check_correctness", capture)
    assert code_reward.compute_score("BAAI/TACO", "print(1)", _tests(1)) == 1.0
    assert code_reward.compute_score(
        "BAAI/TACO", "reasoning\n```python\nprint(2)\n```", _tests(1)
    ) == 1.0
    assert generations == ["print(1)", "print(2)"]
    assert timeouts == [10, 10]
