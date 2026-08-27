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
_STUBBED_MODULE_NAMES = (
    "vllm",
    "pandas",
    "tqdm",
    "transformers",
    "recipe.opd.base_completion",
)
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

base_completion_stub = types.ModuleType("recipe.opd.base_completion")


def _render_plain_prompt_for_test(messages):
    if isinstance(messages, str):
        text = messages
    else:
        text = "\n".join(message["content"] for message in messages)
    return text.strip() + "\n"


base_completion_stub.render_plain_prompt = _render_plain_prompt_for_test
sys.modules["recipe.opd.base_completion"] = base_completion_stub

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
_build_teacher_prompt_ids = skd_module._build_teacher_prompt_ids
_llm_worker_main = skd_module._llm_worker_main
_load_lora_rank = skd_module._load_lora_rank
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
        self.prompts: list[list[dict]] = []

    def generate(self, prompts, params, use_tqdm=False):
        self.batch_sizes.append(len(prompts))
        self.prompts.append(list(prompts))
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


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


def test_load_lora_rank_requires_complete_adapter(tmp_path):
    adapter = tmp_path / "lora_adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 64}', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")

    assert _load_lora_rank(str(adapter)) == 64
    assert _load_lora_rank("") == 0

    (adapter / "adapter_model.safetensors").unlink()
    try:
        _load_lora_rank(str(adapter))
    except FileNotFoundError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("an incomplete rolling adapter must fail closed")


def test_llm_worker_applies_rolling_student_lora(monkeypatch):
    captured = {"init": None, "generate": None}

    class RecordingLLM:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def generate(self, prompts, params, **kwargs):
            captured["generate"] = (prompts, params, kwargs)
            return []

    class RecordingLoRARequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class RequestQueue:
        def __init__(self):
            self.items = iter([(7, [], {}), None])

        def get(self):
            return next(self.items)

    class ResponseQueue:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    runtime_vllm = types.ModuleType("vllm")
    runtime_vllm.LLM = RecordingLLM
    runtime_vllm.SamplingParams = DummySamplingParams
    runtime_lora = types.ModuleType("vllm.lora")
    runtime_lora_request = types.ModuleType("vllm.lora.request")
    runtime_lora_request.LoRARequest = RecordingLoRARequest
    monkeypatch.setitem(sys.modules, "vllm", runtime_vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", runtime_lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", runtime_lora_request)

    responses = ResponseQueue()
    _llm_worker_main(
        RequestQueue(),
        responses,
        model_path="base-model",
        tokenizer_path="base-model",
        gpus="0",
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        max_logprobs=25,
        max_num_seqs=8,
        max_num_batched_tokens=4096,
        seed=20,
        lora_adapter_path="/tmp/rolling/lora_adapter",
        lora_rank=64,
    )

    assert captured["init"]["model"] == "base-model"
    assert captured["init"]["enable_lora"] is True
    assert captured["init"]["max_lora_rank"] == 64
    request = captured["generate"][2]["lora_request"]
    assert request.kwargs["lora_path"] == "/tmp/rolling/lora_adapter"
    assert responses.items[0] == ("ready", None)
    assert responses.items[1] == (7, [])


def test_opsd_teacher_prompt_reads_expert_solution(monkeypatch):
    data_utils_stub = types.ModuleType("recipe.opd.dataset.data_utils")
    data_utils_stub.build_teacher_prompt = lambda problem, expert, **_kwargs: (
        f"problem={problem}\nexpert={expert}"
    )
    prepare_stub = types.ModuleType("recipe.opd.generation.y_r_prepare")
    prepare_stub._fit_rewrite_prompt = (
        lambda render, expert, initial, _tokenizer, _cap, *_args: render(expert, initial)
    )
    monkeypatch.setitem(sys.modules, "recipe.opd.dataset.data_utils", data_utils_stub)
    monkeypatch.setitem(sys.modules, "recipe.opd.generation.y_r_prepare", prepare_stub)

    tokenizer = CharacterTokenizer()
    teacher_ids = _build_teacher_prompt_ids(
        tokenizer,
        {"extra_info": {"problem": "P", "expert_cot": "Y_STAR"}},
        [1, 2],
        distill_mode="opsd",
        task="code",
        teacher_prompt_length=128,
    )

    assert tokenizer.decode(teacher_ids) == "problem=P\nexpert=Y_STAR\n"
    assert teacher_ids != [1, 2]


def test_opsd_teacher_prompt_rejects_missing_expert_solution():
    try:
        _build_teacher_prompt_ids(
            CharacterTokenizer(),
            {"extra_info": {"problem": "P", "expert_cot": ""}},
            [1, 2],
            distill_mode="opsd",
            task="code",
            teacher_prompt_length=128,
        )
    except ValueError as exc:
        assert "expert_cot" in str(exc)
    else:
        raise AssertionError("missing OPSD y* must fail closed")


def test_opd_nonthinking_teacher_uses_plain_completion(monkeypatch):
    data_utils_stub = types.ModuleType("recipe.opd.dataset.data_utils")
    data_utils_stub.build_teacher_prompt = lambda problem, _expert, **_kwargs: f"solve={problem}"
    prepare_stub = types.ModuleType("recipe.opd.generation.y_r_prepare")
    prepare_stub._fit_rewrite_prompt = (
        lambda render, expert, initial, _tokenizer, _cap, *_args: render(expert, initial)
    )
    monkeypatch.setitem(sys.modules, "recipe.opd.dataset.data_utils", data_utils_stub)
    monkeypatch.setitem(sys.modules, "recipe.opd.generation.y_r_prepare", prepare_stub)

    teacher_ids = _build_teacher_prompt_ids(
        CharacterTokenizer(),
        {"extra_info": {"problem": "P", "expert_cot": "must-not-be-read"}},
        [10, 11],
        distill_mode="opd",
        task="code",
        teacher_prompt_length=128,
    )
    assert CharacterTokenizer().decode(teacher_ids) == "solve=P\n"
    assert teacher_ids != [10, 11]


def test_opd_thinking_teacher_uses_chat_prefix(monkeypatch):
    data_utils_stub = types.ModuleType("recipe.opd.dataset.data_utils")
    data_utils_stub.build_teacher_prompt = lambda problem, _expert, **_kwargs: f"solve={problem}"
    data_utils_stub.build_teacher_chat_prompt_ids = (
        lambda tokenizer, prompt, **_kwargs: tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    )
    prepare_stub = types.ModuleType("recipe.opd.generation.y_r_prepare")
    prepare_stub._fit_rewrite_prompt = (
        lambda render, expert, initial, _tokenizer, _cap, *_args: render(expert, initial)
    )
    monkeypatch.setitem(sys.modules, "recipe.opd.dataset.data_utils", data_utils_stub)
    monkeypatch.setitem(sys.modules, "recipe.opd.generation.y_r_prepare", prepare_stub)

    class ThinkingTokenizer(CharacterTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": True,
            }
            return [900, *self.encode(messages[0]["content"]), 901]

    teacher_ids = _build_teacher_prompt_ids(
        ThinkingTokenizer(),
        {"extra_info": {"problem": "P", "expert_cot": "must-not-be-read"}},
        [10, 11],
        distill_mode="opd",
        task="math",
        teacher_prompt_length=128,
        teacher_enable_thinking=True,
    )

    assert teacher_ids[0] == 900
    assert teacher_ids[-1] == 901
    assert teacher_ids != [10, 11]


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


def test_run_batch_gamma1_uses_distinct_opsd_teacher_prefix():
    completed = []
    student = FakeLLM(token_id=1)
    teacher = FakeLLM(token_id=1)

    _run_batch(
        states=[
            RowState(
                prompt_ids=[10, 11],
                teacher_prompt_ids=[20, 21, 22],
                generated=[],
                row_idx=0,
            )
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
        total_rows_for_budget=1,
        progress=None,
        on_state_complete=completed.append,
        refill_state=None,
    )

    assert student.prompts[0][0]["prompt_token_ids"] == [10, 11]
    assert teacher.prompts[0][0]["prompt_token_ids"] == [20, 21, 22]
    assert [state.generated for state in completed] == [[1]]



def test_run_batch_gamma5_accepts_multiple_student_tokens():
    completed = []
    student = SequenceStudentLLM([1, 2, 3, 4, 5])
    teacher = GammaTeacherLLM({1, 2, 3, 4, 5}, replacement_token=7)
    progress = FakeProgress()

    _run_batch(
        states=[
            RowState(
                prompt_ids=[10, 11],
                teacher_prompt_ids=[20, 21, 22],
                generated=[],
                row_idx=0,
            )
        ],
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
    assert student.calls[0][0][0]["prompt_token_ids"] == [10, 11]
    assert verify_prompts[0]["prompt_token_ids"] == [20, 21, 22, 1, 2, 3, 4, 5]
    assert progress.count == 1


def test_run_batch_gamma5_rejects_and_discards_remaining_proposal():
    completed = []
    student = SequenceStudentLLM([1, 99, 3, 4, 5])
    teacher = GammaTeacherLLM({1}, replacement_token=7)

    _run_batch(
        states=[
            RowState(
                prompt_ids=[10, 11],
                teacher_prompt_ids=[20, 21, 22],
                generated=[],
                row_idx=0,
            )
        ],
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
    assert verify_prompts[0]["prompt_token_ids"] == [20, 21, 22, 1, 99, 3, 4, 5]
    assert "prompt_logprobs" not in replacement_params
    assert replacement_params["logprobs"] is None
    assert replacement_prompts[0]["prompt_token_ids"] == [20, 21, 22, 1]


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
