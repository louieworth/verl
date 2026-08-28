from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

from recipe.math_evaluation.run_eval_with_wandb import (
    PROGRESS_PREFIX,
    EvalProgressTracker,
    parse_progress_line,
    stream_command,
)


ROOT = Path(__file__).parents[3]
BENCHMARK = ROOT / "recipe/math_evaluation/benchmark_kl_model.sh"
EVAL_UTILS = ROOT / "recipe/math_evaluation/eval_utils.py"


def test_progress_lines_are_streamed_to_terminal_without_chart_metrics():
    event = {
        "phase": "sample_complete",
        "dataset": "aime25",
        "dataset_index": 1,
        "dataset_total": 4,
        "sample_index": 3,
        "sample_total": 16,
        "overall_fraction": 3 / 64,
    }
    code = (
        "import json\n"
        f"print({PROGRESS_PREFIX!r} + json.dumps({event!r}), flush=True)\n"
        "print('terminal traceback context', flush=True)\n"
    )
    progress = EvalProgressTracker(global_step=15, milestone_fraction=0.25)
    terminal = io.StringIO()
    with contextlib.redirect_stdout(terminal):
        return_code = stream_command([sys.executable, "-u", "-c", code], progress)

    assert return_code == 0
    assert "terminal traceback context" in terminal.getvalue()
    assert PROGRESS_PREFIX in terminal.getvalue()
    assert progress.current_dataset == "aime25"
    assert progress.current_phase == "sample_complete"


def test_progress_parser_tolerates_prefixes_and_failure_is_visible():
    event = parse_progress_line(
        "worker-0 " + PROGRESS_PREFIX + json.dumps({"phase": "dataset_started", "dataset": "aime26"})
    )
    assert event == {"phase": "dataset_started", "dataset": "aime26"}

    progress = EvalProgressTracker(global_step=29, milestone_fraction=0.5)
    progress.observe(event)
    terminal = io.StringIO()
    with contextlib.redirect_stdout(terminal):
        progress.fail(17, "RuntimeError: vLLM failed")
    failure = parse_progress_line(terminal.getvalue().strip())
    assert failure["phase"] == "failed"
    assert failure["exit_code"] == 17
    assert failure["error"] == "RuntimeError: vLLM failed"
    assert failure["dataset"] == "aime26"


def test_math_benchmark_wraps_terminal_and_avoids_duplicate_wandb_init():
    benchmark = BENCHMARK.read_text(encoding="utf-8")
    eval_utils = EVAL_UTILS.read_text(encoding="utf-8")

    assert 'exec "${PYTHON_BIN}" "$SCRIPT_DIR/run_eval_with_wandb.py"' in benchmark
    assert '[ "${EVAL_WANDB_WRAPPED:-0}" != 1 ]' in benchmark
    wrapper = (ROOT / "recipe/math_evaluation/run_eval_with_wandb.py").read_text(encoding="utf-8")
    assert 'define_metric("eval/progress/' not in wrapper
    assert '"eval/progress/' not in wrapper
    assert '"sample_started"' in eval_utils
    assert '"sample_complete"' in eval_utils
    assert '"dataset_complete"' in eval_utils
