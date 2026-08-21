#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import time as real_time
from pathlib import Path
import importlib.util
import sys
import types

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "recipe" / "opd" / "generation" / "skd_vllm_y_o_generate.py"

# The helpers tested here are pure Python. Stub heavy runtime-only imports so
# this regression test runs in CPU-only pytest environments without torch/vLLM.
_STUBBED_MODULE_NAMES = ("vllm", "pandas", "tqdm", "transformers")
_MISSING_MODULE = object()
_ORIGINAL_MODULES = {
    name: sys.modules.get(name, _MISSING_MODULE) for name in _STUBBED_MODULE_NAMES
}

vllm_stub = types.ModuleType("vllm")


class DummyLLM:
    pass


class DummySamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


vllm_stub.LLM = DummyLLM
vllm_stub.SamplingParams = DummySamplingParams
sys.modules["vllm"] = vllm_stub

pandas_stub = types.ModuleType("pandas")
pandas_stub.DataFrame = object
sys.modules["pandas"] = pandas_stub

tqdm_stub = types.ModuleType("tqdm")
tqdm_stub.tqdm = lambda iterable=None, **kwargs: iterable if iterable is not None else []
sys.modules["tqdm"] = tqdm_stub

transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoTokenizer = object
sys.modules["transformers"] = transformers_stub

spec = importlib.util.spec_from_file_location("skd_vllm_y_o_generate_for_test", MODULE_PATH)
skd_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = skd_module
try:
    spec.loader.exec_module(skd_module)
finally:
    # Do not leak lightweight stubs into unrelated tests collected in the same
    # interpreter. The loaded module keeps the object references it needs.
    for module_name, original_module in _ORIGINAL_MODULES.items():
        if original_module is _MISSING_MODULE:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original_module

RowState = skd_module.RowState
_one_pos_accepts = skd_module._one_pos_accepts
_run_batch = skd_module._run_batch


@dataclass
class LogprobEntry:
    rank: int
    logprob: float


class FakeCompletion:
    def __init__(self, token_id: int):
        self.token_ids = [token_id]
        self.logprobs = [{token_id: LogprobEntry(rank=1, logprob=0.0)}]


class FakeOutput:
    def __init__(self, token_id: int):
        self.outputs = [FakeCompletion(token_id)]
        self.prompt_logprobs = None


class FakeLLM:
    def __init__(self, token_id: int):
        self.token_id = token_id
        self.batch_sizes: list[int] = []

    def generate(self, prompts, params, use_tqdm=False):
        self.batch_sizes.append(len(prompts))
        return [FakeOutput(self.token_id) for _ in prompts]


class SequenceCompletion:
    def __init__(self, token_ids, logprobs=None):
        self.token_ids = list(token_ids)
        self.logprobs = logprobs or []


class SequenceOutput:
    def __init__(self, token_ids, logprobs=None, prompt_logprobs=None):
        self.outputs = [SequenceCompletion(token_ids, logprobs=logprobs)]
        self.prompt_logprobs = prompt_logprobs


class SequenceStudentLLM:
    def __init__(self, token_ids):
        self.token_ids = list(token_ids)
        self.calls = []

    def generate(self, prompts, params, use_tqdm=False):
        self.calls.append((list(prompts), dict(params)))
        max_tokens = int(params["max_tokens"])
        return [SequenceOutput(self.token_ids[:max_tokens]) for _ in prompts]


class GammaTeacherLLM:
    def __init__(self, accepted_tokens: set[int], replacement_token: int = 7):
        self.accepted_tokens = {int(token) for token in accepted_tokens}
        self.replacement_token = int(replacement_token)
        self.calls = []

    def generate(self, prompts, params, use_tqdm=False):
        params = dict(params)
        self.calls.append((list(prompts), params))
        if params.get("prompt_logprobs") is not None:
            outputs = []
            for prompt in prompts:
                prompt_ids = list(prompt["prompt_token_ids"])
                prompt_logprobs = []
                for token_id in prompt_ids:
                    if int(token_id) in self.accepted_tokens:
                        prompt_logprobs.append({int(token_id): LogprobEntry(rank=1, logprob=0.0)})
                    else:
                        prompt_logprobs.append({-1: LogprobEntry(rank=1, logprob=0.0)})
                outputs.append(SequenceOutput([0], prompt_logprobs=prompt_logprobs))
            return outputs

        logprobs = []
        if params.get("logprobs") is not None:
            logprobs = [{self.replacement_token: LogprobEntry(rank=1, logprob=0.0)}]
        return [SequenceOutput([self.replacement_token], logprobs=logprobs) for _ in prompts]


class TimedFakeLLM(FakeLLM):
    def __init__(self, token_id: int, delay: float):
        super().__init__(token_id)
        self.delay = delay
        self.intervals: list[tuple[float, float]] = []

    def generate(self, prompts, params, use_tqdm=False):
        start = real_time.perf_counter()
        real_time.sleep(self.delay)
        outputs = super().generate(prompts, params, use_tqdm=use_tqdm)
        self.intervals.append((start, real_time.perf_counter()))
        return outputs


class FakeProgress:
    def __init__(self):
        self.count = 0
        self.postfixes = []

    def update(self, n: int):
        self.count += n

    def set_postfix(self, value, refresh=True):
        self.postfixes.append(value)


def test_one_pos_accepts_handles_normal_logprobs():
    assert _one_pos_accepts(
        {1: LogprobEntry(rank=1, logprob=-3.0), 2: LogprobEntry(rank=2, logprob=-0.1)},
        2,
        top_k=25,
        top_p=0.95,
    )
    assert not _one_pos_accepts(
        {1: LogprobEntry(rank=1, logprob=-0.01), 2: LogprobEntry(rank=2, logprob=-5.0)},
        2,
        top_k=25,
        top_p=0.95,
    )


def test_one_pos_accepts_handles_raw_scores_without_overflow():
    assert _one_pos_accepts(
        {1: LogprobEntry(rank=1, logprob=1000.0), 2: LogprobEntry(rank=2, logprob=999.0)},
        2,
        top_k=25,
        top_p=0.95,
    )
    assert not _one_pos_accepts(
        {1: LogprobEntry(rank=1, logprob=1000.0), 2: LogprobEntry(rank=2, logprob=0.0)},
        2,
        top_k=25,
        top_p=0.95,
    )


def test_one_pos_accepts_handles_inf_without_overflow():
    assert not _one_pos_accepts(
        {1: LogprobEntry(rank=1, logprob=float("inf")), 2: LogprobEntry(rank=2, logprob=0.0)},
        2,
        top_k=25,
        top_p=0.95,
    )
    assert _one_pos_accepts(
        {
            1: LogprobEntry(rank=1, logprob=float("inf")),
            2: LogprobEntry(rank=2, logprob=float("inf")),
        },
        2,
        top_k=25,
        top_p=0.95,
    )


def test_run_batch_refills_rolling_window():
    pending = iter(range(2, 6))
    completed = []

    def refill():
        try:
            row = next(pending)
        except StopIteration:
            return None
        return RowState(prompt_ids=[row], generated=[], row_idx=row)

    def complete(state: RowState):
        completed.append((state.row_idx, list(state.generated)))

    student = FakeLLM(token_id=1)
    teacher = FakeLLM(token_id=1)
    progress = FakeProgress()

    _run_batch(
        states=[
            RowState(prompt_ids=[0], generated=[], row_idx=0),
            RowState(prompt_ids=[1], generated=[], row_idx=1),
        ],
        tokenizer=None,
        student_llm=student,
        teacher_llm=teacher,
        eos_ids=set(),
        max_tokens=1,
        gamma=1,
        top_k=25,
        student_temperature=0.6,
        student_top_p=0.95,
        teacher_temperature=0.6,
        teacher_top_p=0.95,
        adaptive_gamma=False,
        adaptive_gamma_low_accept=0.2,
        adaptive_gamma_high_accept=0.6,
        adaptive_gamma_ema_alpha=0.2,
        parallel_gamma1=False,
        target_rollout_seconds=0.0,
        target_rollout_safety=0.85,
        target_rollout_warmup_seconds=300.0,
        total_rows_for_budget=6,
        progress=progress,
        on_state_complete=complete,
        refill_state=refill,
    )

    assert [row for row, _ in completed] == list(range(6))
    assert student.batch_sizes == [2, 2, 2]
    assert teacher.batch_sizes == [2, 2, 2]
    assert progress.count == 6



def test_run_batch_budget_cap_truncates_rows():
    completed = []
    student = FakeLLM(token_id=1)
    teacher = FakeLLM(token_id=1)
    progress = FakeProgress()

    class FakeTime:
        def __init__(self):
            self.calls = 0

        def monotonic(self):
            self.calls += 1
            return 0.0 if self.calls == 1 else 1000.0

    original_time = skd_module.time.monotonic
    skd_module.time.monotonic = FakeTime().monotonic
    try:
        _run_batch(
            states=[
                RowState(prompt_ids=[0], generated=[], row_idx=0),
                RowState(prompt_ids=[1], generated=[], row_idx=1),
            ],
            tokenizer=None,
            student_llm=student,
            teacher_llm=teacher,
            eos_ids=set(),
            max_tokens=5,
            gamma=1,
            top_k=25,
            student_temperature=0.6,
            student_top_p=0.95,
            teacher_temperature=0.6,
            teacher_top_p=0.95,
            adaptive_gamma=False,
            adaptive_gamma_low_accept=0.2,
            adaptive_gamma_high_accept=0.6,
            adaptive_gamma_ema_alpha=0.2,
            parallel_gamma1=False,
            target_rollout_seconds=10.0,
            target_rollout_safety=1.0,
            target_rollout_warmup_seconds=0.0,
            total_rows_for_budget=100,
            progress=progress,
            on_state_complete=completed.append,
            refill_state=None,
        )
    finally:
        skd_module.time.monotonic = original_time

    assert [state.row_idx for state in completed] == [0, 1]
    assert all(state.effective_max_tokens == 1 for state in completed)
    assert all(state.truncated_by_budget for state in completed)
    assert progress.count == 2

def test_run_batch_parallel_gamma1_overlaps_separate_engines():
    completed = []
    student = TimedFakeLLM(token_id=1, delay=0.05)
    teacher = TimedFakeLLM(token_id=1, delay=0.05)

    _run_batch(
        states=[RowState(prompt_ids=[0], generated=[], row_idx=0)],
        tokenizer=None,
        student_llm=student,
        teacher_llm=teacher,
        eos_ids=set(),
        max_tokens=1,
        gamma=1,
        top_k=25,
        student_temperature=0.6,
        student_top_p=0.95,
        teacher_temperature=0.6,
        teacher_top_p=0.95,
        adaptive_gamma=False,
        adaptive_gamma_low_accept=0.2,
        adaptive_gamma_high_accept=0.6,
        adaptive_gamma_ema_alpha=0.2,
        parallel_gamma1=True,
        target_rollout_seconds=0.0,
        target_rollout_safety=0.85,
        target_rollout_warmup_seconds=300.0,
        total_rows_for_budget=1,
        progress=None,
        on_state_complete=completed.append,
        refill_state=None,
    )

    assert [state.row_idx for state in completed] == [0]
    s0, s1 = student.intervals[0]
    t0, t1 = teacher.intervals[0]
    assert s0 < t1 and t0 < s1


def test_run_batch_parallel_gamma1_keeps_shared_engine_serial():
    completed = []
    shared = TimedFakeLLM(token_id=1, delay=0.01)

    _run_batch(
        states=[RowState(prompt_ids=[0], generated=[], row_idx=0)],
        tokenizer=None,
        student_llm=shared,
        teacher_llm=shared,
        eos_ids=set(),
        max_tokens=1,
        gamma=1,
        top_k=25,
        student_temperature=0.6,
        student_top_p=0.95,
        teacher_temperature=0.6,
        teacher_top_p=0.95,
        adaptive_gamma=False,
        adaptive_gamma_low_accept=0.2,
        adaptive_gamma_high_accept=0.6,
        adaptive_gamma_ema_alpha=0.2,
        parallel_gamma1=True,
        target_rollout_seconds=0.0,
        target_rollout_safety=0.85,
        target_rollout_warmup_seconds=300.0,
        total_rows_for_budget=1,
        progress=None,
        on_state_complete=completed.append,
        refill_state=None,
    )

    assert [state.row_idx for state in completed] == [0]
    assert len(shared.intervals) == 2
    assert shared.intervals[0][1] <= shared.intervals[1][0]



def test_run_batch_gamma5_accepts_multiple_student_tokens():
    completed = []
    student = SequenceStudentLLM([1, 2, 3, 4, 5])
    teacher = GammaTeacherLLM({1, 2, 3, 4, 5}, replacement_token=7)
    progress = FakeProgress()

    _run_batch(
        states=[RowState(prompt_ids=[10, 11], generated=[], row_idx=0)],
        tokenizer=None,
        student_llm=student,
        teacher_llm=teacher,
        eos_ids={5},
        max_tokens=6,
        gamma=5,
        top_k=25,
        top_p=1.0,
        student_temperature=0.6,
        student_top_p=0.95,
        teacher_temperature=0.6,
        teacher_top_p=0.95,
        adaptive_gamma=False,
        adaptive_gamma_low_accept=0.2,
        adaptive_gamma_high_accept=0.6,
        adaptive_gamma_ema_alpha=0.2,
        parallel_gamma1=True,
        target_rollout_seconds=0.0,
        target_rollout_safety=0.85,
        target_rollout_warmup_seconds=300.0,
        total_rows_for_budget=1,
        progress=progress,
        on_state_complete=completed.append,
        refill_state=None,
    )

    assert [state.generated for state in completed] == [[1, 2, 3, 4, 5]]
    assert student.calls[0][1]["max_tokens"] == 5
    assert len(teacher.calls) == 1
    verify_prompts, verify_params = teacher.calls[0]
    assert verify_params["prompt_logprobs"] == 25
    assert verify_prompts[0]["prompt_token_ids"] == [10, 11, 1, 2, 3, 4, 5]
    assert progress.count == 1


def test_run_batch_gamma5_rejects_and_discards_remaining_proposal():
    completed = []
    student = SequenceStudentLLM([1, 99, 3, 4, 5])
    teacher = GammaTeacherLLM({1}, replacement_token=7)

    _run_batch(
        states=[RowState(prompt_ids=[10, 11], generated=[], row_idx=0)],
        tokenizer=None,
        student_llm=student,
        teacher_llm=teacher,
        eos_ids={7},
        max_tokens=6,
        gamma=5,
        top_k=25,
        top_p=1.0,
        student_temperature=0.6,
        student_top_p=0.95,
        teacher_temperature=0.6,
        teacher_top_p=0.95,
        adaptive_gamma=False,
        adaptive_gamma_low_accept=0.2,
        adaptive_gamma_high_accept=0.6,
        adaptive_gamma_ema_alpha=0.2,
        parallel_gamma1=True,
        target_rollout_seconds=0.0,
        target_rollout_safety=0.85,
        target_rollout_warmup_seconds=300.0,
        total_rows_for_budget=1,
        progress=None,
        on_state_complete=completed.append,
        refill_state=None,
    )

    assert [state.generated for state in completed] == [[1, 7]]
    assert len(teacher.calls) == 2
    verify_prompts, verify_params = teacher.calls[0]
    replacement_prompts, replacement_params = teacher.calls[1]
    assert verify_params["prompt_logprobs"] == 25
    assert verify_prompts[0]["prompt_token_ids"] == [10, 11, 1, 99, 3, 4, 5]
    assert "prompt_logprobs" not in replacement_params
    assert replacement_params["logprobs"] is None
    assert replacement_prompts[0]["prompt_token_ids"] == [10, 11, 1]


def test_run_batch_gamma5_caps_student_request_near_response_limit():
    completed = []
    student = SequenceStudentLLM([1, 2, 3, 4, 5])
    teacher = GammaTeacherLLM({1}, replacement_token=1)

    _run_batch(
        states=[RowState(prompt_ids=[10, 11], generated=[], row_idx=0)],
        tokenizer=None,
        student_llm=student,
        teacher_llm=teacher,
        eos_ids=set(),
        max_tokens=1,
        gamma=5,
        top_k=25,
        top_p=1.0,
        student_temperature=0.6,
        student_top_p=0.95,
        teacher_temperature=0.6,
        teacher_top_p=0.95,
        adaptive_gamma=False,
        adaptive_gamma_low_accept=0.2,
        adaptive_gamma_high_accept=0.6,
        adaptive_gamma_ema_alpha=0.2,
        parallel_gamma1=True,
        target_rollout_seconds=0.0,
        target_rollout_safety=0.85,
        target_rollout_warmup_seconds=300.0,
        total_rows_for_budget=1,
        progress=None,
        on_state_complete=completed.append,
        refill_state=None,
    )

    assert [state.generated for state in completed] == [[1]]
    assert student.calls[0][1]["max_tokens"] == 1
