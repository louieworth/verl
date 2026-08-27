#!/usr/bin/env python3
"""Generate OPD/OPSD y_o responses with SKD using vLLM workers.

The output schema intentionally matches verl.trainer.main_generation_server:
it reads a stage1 prompt parquet and writes the same rows with a `responses`
column containing one generated string per row.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time
import traceback
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from recipe.opd.base_completion import render_plain_prompt


@dataclass
class RowState:
    prompt_ids: list[int]
    generated: list[int]
    row_idx: int
    # Student always drafts from x. OPSD teacher verification uses x+y*,
    # while OPD leaves this unset and uses the exact student prefix.
    teacher_prompt_ids: list[int] | None = None
    effective_max_tokens: int = 0
    truncated_by_budget: bool = False


def _gpu_count(gpus: str) -> int:
    return len([part for part in str(gpus).split(',') if part.strip()])


def _load_lora_rank(adapter_path: str) -> int:
    if not adapter_path:
        return 0
    config_path = Path(adapter_path) / "adapter_config.json"
    weights_path = Path(adapter_path) / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"incomplete student LoRA adapter: {adapter_path}")
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)
    rank = int(config.get("r") or 0)
    if rank <= 0:
        raise ValueError(f"invalid LoRA rank in {config_path}: {rank}")
    return rank


def _normalize_chat(chat: Any) -> list[dict[str, str]]:
    if hasattr(chat, 'tolist'):
        chat = chat.tolist()
    if isinstance(chat, str):
        return [{"role": "user", "content": chat}]
    return list(chat)


def _prompt_token_ids(tokenizer, chat: Any, prompt_length: int) -> list[int]:
    prompt_ids = tokenizer.encode(
        render_plain_prompt(_normalize_chat(chat)),
        add_special_tokens=False,
    )
    if len(prompt_ids) > prompt_length:
        raise ValueError(
            f"Base completion prompt has {len(prompt_ids)} tokens, exceeding cap {prompt_length}"
        )
    return prompt_ids


def _teacher_prefix_ids(state: RowState) -> list[int]:
    return state.teacher_prompt_ids if state.teacher_prompt_ids is not None else state.prompt_ids


def _build_teacher_prompt_ids(
    tokenizer,
    row: dict[str, Any],
    student_prompt_ids: list[int],
    *,
    distill_mode: str,
    task: str,
    teacher_prompt_length: int,
    teacher_enable_thinking: bool = False,
) -> list[int]:
    """Build the SKD teacher prefix under the canonical OPD/OPSD contract."""
    if distill_mode not in {"opd", "opsd"}:
        raise ValueError(f"unsupported distill_mode={distill_mode!r}")

    extra_info = row.get("extra_info")
    if hasattr(extra_info, "as_py"):
        extra_info = extra_info.as_py()
    if not isinstance(extra_info, dict):
        raise ValueError("SKD requires extra_info.problem")
    problem = str(extra_info.get("problem") or "").strip()
    expert_solution = str(extra_info.get("expert_cot") or "").strip()
    if not problem:
        raise ValueError("SKD requires non-empty extra_info.problem")
    if distill_mode == "opsd" and not expert_solution:
        raise ValueError("OPSD SKD requires non-empty extra_info.expert_cot (y*)")

    # Lazy imports keep the rollout primitives usable in lightweight CPU tests.
    from recipe.opd.dataset.data_utils import build_teacher_prompt
    from recipe.opd.generation.y_r_prepare import _fit_rewrite_prompt

    def render_prompt(expert_text: str, _initial_response: str) -> str:
        return build_teacher_prompt(
            problem,
            expert_text,
            use_initial_response=False,
            distill_mode=distill_mode,
            task=task,
        )

    teacher_prompt = _fit_rewrite_prompt(
        render_prompt,
        expert_solution,
        "",
        tokenizer,
        teacher_prompt_length,
        teacher_enable_thinking,
        distill_mode == "opd" and teacher_enable_thinking,
    )
    if distill_mode == "opd" and teacher_enable_thinking:
        from recipe.opd.dataset.data_utils import build_teacher_chat_prompt_ids

        prompt_ids = build_teacher_chat_prompt_ids(
            tokenizer,
            teacher_prompt,
            enable_thinking=teacher_enable_thinking,
        )
        if len(prompt_ids) > teacher_prompt_length:
            raise ValueError(
                f"OPD teacher chat prompt has {len(prompt_ids)} tokens, "
                f"exceeding cap {teacher_prompt_length}"
            )
        return prompt_ids
    return _prompt_token_ids(
        tokenizer,
        [{"role": "user", "content": teacher_prompt}],
        teacher_prompt_length,
    )


def _decode_response(tokenizer, token_ids: list[int], eos_ids: set[int]) -> str:
    end = len(token_ids)
    for i, token_id in enumerate(token_ids):
        if int(token_id) in eos_ids:
            end = i
            break
    return tokenizer.decode(token_ids[:end], skip_special_tokens=True)


def _one_pos_accepts(logprobs, token_id: int, top_k: int, top_p: float) -> bool:
    if not logprobs:
        return False
    entry = logprobs.get(int(token_id))
    if entry is None:
        return False
    if top_k and top_k > 0:
        rank = getattr(entry, 'rank', None)
        if rank is None or int(rank) > int(top_k):
            return False
    if top_p is None or top_p <= 0.0 or top_p >= 1.0:
        return True

    inf_token_ids = []
    finite_scores = []
    for other_id, other in logprobs.items():
        score = float(getattr(other, 'logprob', float('-inf')))
        if math.isinf(score) and score > 0:
            inf_token_ids.append(int(other_id))
        elif math.isfinite(score):
            finite_scores.append((int(other_id), score))
    if inf_token_ids:
        return int(token_id) in inf_token_ids
    if not finite_scores:
        return False

    finite_scores.sort(key=lambda item: item[1], reverse=True)
    max_score = finite_scores[0][1]
    denom = sum(math.exp(score - max_score) for _, score in finite_scores)
    if denom <= 0.0:
        return False
    mass = 0.0
    for other_id, score in finite_scores:
        prob = math.exp(score - max_score) / denom
        if other_id == int(token_id):
            return True
        mass += prob
        if mass >= top_p:
            return False
    return False


def _sampling_params(max_tokens: int, temperature: float, top_p: float,
                     logprobs: int | None = None,
                     prompt_logprobs: int | None = None) -> dict[str, Any]:
    params = {
        "max_tokens": max_tokens,
        "min_tokens": 1,
        "temperature": temperature,
        "top_p": top_p,
        "logprobs": logprobs,
        "detokenize": False,
        "skip_special_tokens": False,
    }
    if prompt_logprobs is not None:
        params["prompt_logprobs"] = prompt_logprobs
    return params


def _convert_logprobs_list(logprobs_list) -> list[dict[int, dict[str, Any]] | None]:
    converted_list = []
    for one_pos in logprobs_list or []:
        if not one_pos:
            converted_list.append(None)
            continue
        converted = {}
        for token_id, entry in one_pos.items():
            converted[int(token_id)] = {
                "rank": getattr(entry, "rank", None),
                "logprob": float(getattr(entry, "logprob", float("-inf"))),
            }
        converted_list.append(converted)
    return converted_list


def _simplify_outputs(outputs, prompt_logprobs_tail: int | None = None,
                      prompts: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    simplified = []
    for output_index, output in enumerate(outputs):
        completion = output.outputs[0] if output.outputs else SimpleNamespace(token_ids=[], logprobs=[])
        prompt_logprobs = getattr(output, "prompt_logprobs", None)
        prompt_logprobs_offset = 0
        if prompt_logprobs_tail is not None and prompt_logprobs_tail > 0 and prompt_logprobs:
            tail = int(prompt_logprobs_tail)
            prompt_len = None
            if prompts is not None:
                prompt_len = len(prompts[output_index].get("prompt_token_ids") or [])
            if (
                prompt_len is not None
                and len(prompt_logprobs) <= tail + 1
                and prompt_logprobs[0] is None
            ):
                # vLLM tail patch returns [None] + proposal-tail rows.
                prompt_logprobs_offset = max(0, prompt_len - tail - 1)
            else:
                # Vanilla vLLM returns full-prompt rows; keep only proposal tail.
                prompt_logprobs_offset = max(0, len(prompt_logprobs) - tail)
                prompt_logprobs = prompt_logprobs[prompt_logprobs_offset:]
        simplified.append({
            "token_ids": [int(tok) for tok in list(completion.token_ids)],
            "logprobs": _convert_logprobs_list(completion.logprobs),
            "prompt_logprobs": _convert_logprobs_list(prompt_logprobs),
            "prompt_logprobs_offset": prompt_logprobs_offset,
        })
    return simplified


def _restore_output(row: dict[str, Any]):
    def restore_logprobs(serialized):
        restored = []
        for one_pos in serialized or []:
            if not one_pos:
                restored.append(None)
                continue
            restored.append({
                int(token_id): SimpleNamespace(rank=value.get("rank"), logprob=value.get("logprob"))
                for token_id, value in one_pos.items()
            })
        return restored

    return SimpleNamespace(
        outputs=[
            SimpleNamespace(
                token_ids=row.get("token_ids") or [],
                logprobs=restore_logprobs(row.get("logprobs")),
            )
        ],
        prompt_logprobs=restore_logprobs(row.get("prompt_logprobs")),
        prompt_logprobs_offset=int(row.get("prompt_logprobs_offset") or 0),
    )


def _llm_worker_main(request_q, response_q, *, model_path: str, tokenizer_path: str,
                     gpus: str, tensor_parallel_size: int, dtype: str,
                     gpu_memory_utilization: float, max_model_len: int,
                     max_logprobs: int, max_num_seqs: int,
                     max_num_batched_tokens: int, seed: int,
                     lora_adapter_path: str = "", lora_rank: int = 0) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from vllm import LLM, SamplingParams

        if (
            os.environ.get("VLLM_SKD_USE_RUNTIME_MONKEYPATCH") == "1"
            and os.environ.get("VLLM_SKD_PROMPT_LOGPROBS_TAIL")
        ):
            from recipe.opd.generation.vllm_skd_prompt_logprobs_patch import apply_patch as apply_vllm_skd_patch
            if apply_vllm_skd_patch():
                print(
                    "Applied runtime vLLM SKD prompt-logprobs tail monkeypatch "
                    f"tail={os.environ.get('VLLM_SKD_PROMPT_LOGPROBS_TAIL')}",
                    flush=True,
                )

        kwargs = {
            "model": model_path,
            "tokenizer": tokenizer_path,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "trust_remote_code": True,
            "seed": seed,
            "max_logprobs": max_logprobs,
            "enable_prefix_caching": True,
            "disable_log_stats": True,
        }
        if max_num_seqs > 0:
            kwargs["max_num_seqs"] = max_num_seqs
        if max_num_batched_tokens > 0:
            kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        lora_request = None
        if lora_adapter_path:
            from vllm.lora.request import LoRARequest

            kwargs.update(
                enable_lora=True,
                max_loras=1,
                max_lora_rank=lora_rank,
            )
            lora_request = LoRARequest(
                lora_name="pipeline_student",
                lora_int_id=1,
                lora_path=lora_adapter_path,
            )
        llm = LLM(**kwargs)
        response_q.put(("ready", None))
        while True:
            request = request_q.get()
            if request is None:
                break
            request_id, prompts, params = request
            try:
                params = dict(params)
                prompt_logprobs_tail = params.pop("_prompt_logprobs_tail", None)
                outputs = llm.generate(
                    prompts,
                    SamplingParams(**params),
                    use_tqdm=False,
                    lora_request=lora_request,
                )
                response_q.put((request_id, _simplify_outputs(outputs, prompt_logprobs_tail, prompts)))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                response_q.put((request_id, {"error": repr(exc), "traceback": traceback.format_exc()}))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        response_q.put(("error", {"error": repr(exc), "traceback": traceback.format_exc()}))


class RemoteVLLM:
    def __init__(self, *, model_path: str, tokenizer_path: str, gpus: str,
                 tensor_parallel_size: int, dtype: str,
                 gpu_memory_utilization: float, max_model_len: int,
                 max_logprobs: int, max_num_seqs: int,
                 max_num_batched_tokens: int, seed: int,
                 lora_adapter_path: str = "", lora_rank: int = 0):
        ctx = mp.get_context("spawn")
        self._request_q = ctx.Queue(maxsize=2)
        self._response_q = ctx.Queue(maxsize=2)
        self._counter = itertools.count(1)
        self._process = ctx.Process(
            target=_llm_worker_main,
            kwargs={
                "request_q": self._request_q,
                "response_q": self._response_q,
                "model_path": model_path,
                "tokenizer_path": tokenizer_path,
                "gpus": gpus,
                "tensor_parallel_size": tensor_parallel_size,
                "dtype": dtype,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_len": max_model_len,
                "max_logprobs": max_logprobs,
                "max_num_seqs": max_num_seqs,
                "max_num_batched_tokens": max_num_batched_tokens,
                "seed": seed,
                "lora_adapter_path": lora_adapter_path,
                "lora_rank": lora_rank,
            },
        )
        self._process.start()
        kind, payload = self._response_q.get()
        if kind == "error":
            raise RuntimeError(payload["traceback"])
        if kind != "ready":
            raise RuntimeError(f"unexpected worker startup message: {kind!r} {payload!r}")

    def generate(self, prompts, params, use_tqdm: bool = False):  # pylint: disable=unused-argument
        request_id = next(self._counter)
        if hasattr(params, "kwargs"):
            params = params.kwargs
        self._request_q.put((request_id, prompts, dict(params)))
        while True:
            kind, payload = self._response_q.get()
            if kind == request_id:
                if isinstance(payload, dict) and "error" in payload:
                    raise RuntimeError(payload["traceback"])
                return [_restore_output(row) for row in payload]
            if kind == "error":
                raise RuntimeError(payload["traceback"])

    def close(self):
        if getattr(self, "_process", None) is None:
            return
        try:
            self._request_q.put(None)
        except Exception:
            pass
        self._process.join(timeout=30)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=10)
        self._process = None

    def __del__(self):
        self.close()


def _first_token(output, eos_fallback: int) -> int:
    token_ids = list(output.outputs[0].token_ids)
    return int(token_ids[0]) if token_ids else int(eos_fallback)


def _first_logprobs(output):
    logprobs = output.outputs[0].logprobs
    if not logprobs:
        return None
    return logprobs[0]


def _token_ids(output) -> list[int]:
    return [int(token_id) for token_id in list(output.outputs[0].token_ids)]


def _prompt_logprobs_at(output, position: int):
    prompt_logprobs = getattr(output, "prompt_logprobs", None)
    prompt_logprobs_offset = int(getattr(output, "prompt_logprobs_offset", 0) or 0)
    index = position - prompt_logprobs_offset
    if not prompt_logprobs or index < 0 or index >= len(prompt_logprobs):
        return None
    return prompt_logprobs[index]


def _run_batch(*, states: list[RowState], tokenizer, student_llm, teacher_llm,
               eos_ids: set[int], max_tokens: int, gamma: int, top_k: int,
               student_temperature: float, student_top_p: float,
               teacher_temperature: float, teacher_top_p: float,
               adaptive_gamma: bool, adaptive_gamma_low_accept: float,
               adaptive_gamma_high_accept: float,
               adaptive_gamma_ema_alpha: float, parallel_gamma1: bool,
               target_rollout_seconds: float, target_rollout_safety: float,
               target_rollout_warmup_seconds: float, total_rows_for_budget: int,
               progress, on_state_complete, refill_state=None, top_p: float = 1.0,
               pipeline_lanes: int = 1):
    del tokenizer, adaptive_gamma, adaptive_gamma_low_accept, adaptive_gamma_high_accept
    del adaptive_gamma_ema_alpha, target_rollout_safety
    del target_rollout_warmup_seconds, total_rows_for_budget
    if gamma < 1:
        raise ValueError(f"SKD gamma must be >= 1, got {gamma}")

    active = list(states)
    start_time = time.monotonic()
    decisions = 0
    accepted_count = 0
    corrected_count = 0
    eos_fallback = next(iter(eos_ids)) if eos_ids else 0
    student_params = _sampling_params(max(1, gamma), student_temperature, student_top_p, None)
    teacher_logprobs = int(top_k) if top_k and top_k > 0 else None
    if teacher_logprobs is None:
        raise ValueError("SKD top-k teacher verification requires --top_k > 0")
    teacher_params = _sampling_params(1, teacher_temperature, teacher_top_p, teacher_logprobs)
    teacher_verify_params = _sampling_params(
        1,
        teacher_temperature,
        teacher_top_p,
        None,
        prompt_logprobs=teacher_logprobs,
    )
    teacher_verify_params["_prompt_logprobs_tail"] = gamma
    teacher_replacement_params = _sampling_params(1, teacher_temperature, teacher_top_p, None)
    parallel_student_teacher = bool(gamma == 1 and parallel_gamma1 and student_llm is not teacher_llm)
    pipeline_lanes = max(1, int(pipeline_lanes or 1))
    pipelined_gamma = bool(gamma > 1 and pipeline_lanes > 1 and student_llm is not teacher_llm)
    executor = ThreadPoolExecutor(max_workers=2) if parallel_student_teacher else None
    pipeline_executor = ThreadPoolExecutor(max_workers=1) if pipelined_gamma else None

    for state in active:
        state.effective_max_tokens = max_tokens

    def apply_budget(state: RowState) -> None:
        if target_rollout_seconds and target_rollout_seconds > 0:
            elapsed = time.monotonic() - start_time
            if elapsed >= target_rollout_seconds and len(state.generated) < state.effective_max_tokens:
                state.effective_max_tokens = max(1, len(state.generated))
                state.truncated_by_budget = True

    def finish_state(state: RowState, next_active: list[RowState]) -> None:
        on_state_complete(state)
        if progress is not None:
            progress.update(1)
        if refill_state is not None:
            refill = refill_state()
            if refill is not None:
                refill.effective_max_tokens = max_tokens
                next_active.append(refill)

    def append_token(state: RowState, token_id: int) -> bool:
        state.generated.append(int(token_id))
        apply_budget(state)
        return int(token_id) in eos_ids or len(state.generated) >= state.effective_max_tokens

    def set_progress(active_count: int, parallel_flag: bool) -> None:
        if progress is not None:
            progress.set_postfix({
                "active": active_count,
                "gamma": gamma,
                "tokens": decisions,
                "accept": f"{accepted_count / max(1, decisions):.2f}",
                "corr": corrected_count,
                "parallel": int(parallel_flag),
            }, refresh=False)

    def generate_gamma1(student_prompts, teacher_prompts):
        if executor is None:
            return (
                student_llm.generate(student_prompts, student_params, use_tqdm=False),
                teacher_llm.generate(teacher_prompts, teacher_params, use_tqdm=False),
            )
        student_future = executor.submit(
            student_llm.generate, student_prompts, student_params, use_tqdm=False
        )
        teacher_future = executor.submit(
            teacher_llm.generate, teacher_prompts, teacher_params, use_tqdm=False
        )
        return student_future.result(), teacher_future.result()

    def make_proposal_bundle(lane_active: list[RowState]) -> dict[str, Any] | None:
        if not lane_active:
            return None
        student_prompts = [
            {"prompt_token_ids": state.prompt_ids + state.generated}
            for state in lane_active
        ]
        proposal_limits = []
        student_groups: dict[int, list[tuple[int, dict[str, Any]]]] = {}
        for index, (state, prompt) in enumerate(zip(lane_active, student_prompts)):
            remaining = max(1, state.effective_max_tokens - len(state.generated))
            proposal_limit = min(gamma, remaining)
            if proposal_limit > 1 and proposal_limit == remaining:
                proposal_limit -= 1
            proposal_limits.append(proposal_limit)
            student_groups.setdefault(proposal_limit, []).append((index, prompt))

        student_outputs: list[Any | None] = [None] * len(lane_active)
        for proposal_limit, group in student_groups.items():
            group_outputs = student_llm.generate(
                [prompt for _, prompt in group],
                _sampling_params(proposal_limit, student_temperature, student_top_p, None),
                use_tqdm=False,
            )
            for (index, _), student_output in zip(group, group_outputs):
                student_outputs[index] = student_output

        records = []
        single_teacher_prompts = []
        multi_teacher_prompts = []
        for state, _student_prompt, proposal_limit, student_output in zip(
            lane_active, student_prompts, proposal_limits, student_outputs
        ):
            if student_output is None:
                raise RuntimeError("missing SKD student proposal output")
            teacher_prefix = _teacher_prefix_ids(state) + state.generated
            proposed = _token_ids(student_output)[:proposal_limit] or [eos_fallback]
            if len(proposed) == 1:
                records.append({"kind": "single", "state": state, "proposed": proposed})
                single_teacher_prompts.append({"prompt_token_ids": teacher_prefix})
            else:
                records.append({
                    "kind": "multi",
                    "state": state,
                    "teacher_prefix_len": len(teacher_prefix),
                    "proposed": proposed,
                })
                multi_teacher_prompts.append({"prompt_token_ids": teacher_prefix + proposed})
        return {
            "records": records,
            "single_teacher_prompts": single_teacher_prompts,
            "multi_teacher_prompts": multi_teacher_prompts,
        }

    def run_teacher(bundle: dict[str, Any] | None):
        if bundle is None:
            return [], []
        single_prompts = bundle["single_teacher_prompts"]
        multi_prompts = bundle["multi_teacher_prompts"]
        single_outputs = (
            teacher_llm.generate(single_prompts, teacher_params, use_tqdm=False)
            if single_prompts else []
        )
        multi_outputs = (
            teacher_llm.generate(multi_prompts, teacher_verify_params, use_tqdm=False)
            if multi_prompts else []
        )
        return single_outputs, multi_outputs

    def process_teacher(bundle: dict[str, Any] | None, teacher_result) -> list[RowState]:
        nonlocal decisions, accepted_count, corrected_count
        next_active: list[RowState] = []
        if bundle is None:
            return next_active
        single_outputs = iter(teacher_result[0])
        multi_outputs = iter(teacher_result[1])
        pending_replacements: list[RowState] = []

        for record in bundle["records"]:
            state = record["state"]
            if record["kind"] == "single":
                teacher_output = next(single_outputs)
                proposed = int(record["proposed"][0])
                replacement = _first_token(teacher_output, proposed)
                accepted = _one_pos_accepts(_first_logprobs(teacher_output), proposed, top_k=top_k, top_p=top_p)
                chosen = proposed if accepted else replacement
                decisions += 1
                accepted_count += int(accepted)
                corrected_count += int(not accepted)
                if append_token(state, chosen):
                    finish_state(state, next_active)
                else:
                    next_active.append(state)
                continue

            teacher_output = next(multi_outputs)
            teacher_prefix_len = int(record["teacher_prefix_len"])
            rejected = False
            finished = False
            for offset, proposed in enumerate(record["proposed"]):
                accepted = _one_pos_accepts(
                    _prompt_logprobs_at(teacher_output, teacher_prefix_len + offset),
                    int(proposed),
                    top_k=top_k,
                    top_p=top_p,
                )
                decisions += 1
                accepted_count += int(accepted)
                corrected_count += int(not accepted)
                if accepted:
                    if append_token(state, int(proposed)):
                        finish_state(state, next_active)
                        finished = True
                        break
                else:
                    pending_replacements.append(state)
                    rejected = True
                    break
            if not rejected and not finished:
                next_active.append(state)

        if pending_replacements:
            replacement_prompts = [
                {"prompt_token_ids": _teacher_prefix_ids(state) + state.generated}
                for state in pending_replacements
            ]
            replacement_outputs = teacher_llm.generate(
                replacement_prompts,
                teacher_replacement_params,
                use_tqdm=False,
            )
            for state, teacher_output in zip(pending_replacements, replacement_outputs):
                replacement = _first_token(teacher_output, eos_fallback)
                if append_token(state, replacement):
                    finish_state(state, next_active)
                else:
                    next_active.append(state)
        return next_active

    try:
        if pipelined_gamma:
            lane_size = len(active)
            lanes: list[list[RowState]] = [active]
            for _ in range(1, pipeline_lanes):
                lane: list[RowState] = []
                while refill_state is not None and len(lane) < lane_size:
                    refill = refill_state()
                    if refill is None:
                        break
                    refill.effective_max_tokens = max_tokens
                    lane.append(refill)
                if lane:
                    lanes.append(lane)
            bundles: list[dict[str, Any] | None] = [None for _ in lanes]
            lane_index = 0
            while any(lanes) or any(bundle is not None for bundle in bundles):
                current = lane_index % len(lanes)
                if not lanes[current] and bundles[current] is None:
                    lane_index += 1
                    continue
                if bundles[current] is None:
                    bundles[current] = make_proposal_bundle(lanes[current])
                bundle = bundles[current]
                bundles[current] = None
                assert pipeline_executor is not None
                teacher_future = pipeline_executor.submit(run_teacher, bundle)

                next_lane_index = (current + 1) % len(lanes)
                if next_lane_index != current and lanes[next_lane_index] and bundles[next_lane_index] is None:
                    bundles[next_lane_index] = make_proposal_bundle(lanes[next_lane_index])

                lanes[current] = process_teacher(bundle, teacher_future.result())
                set_progress(sum(len(lane) for lane in lanes), True)
                lane_index += 1
            return

        while active:
            student_prompts = [
                {"prompt_token_ids": state.prompt_ids + state.generated}
                for state in active
            ]
            teacher_prompts = [
                {"prompt_token_ids": _teacher_prefix_ids(state) + state.generated}
                for state in active
            ]
            next_active: list[RowState] = []

            if gamma == 1:
                student_outputs, teacher_outputs = generate_gamma1(
                    student_prompts, teacher_prompts
                )
                for state, student_output, teacher_output in zip(active, student_outputs, teacher_outputs):
                    proposed = _first_token(student_output, eos_fallback)
                    replacement = _first_token(teacher_output, proposed)
                    accepted = _one_pos_accepts(_first_logprobs(teacher_output), proposed, top_k=top_k, top_p=top_p)
                    chosen = proposed if accepted else replacement
                    decisions += 1
                    accepted_count += int(accepted)
                    corrected_count += int(not accepted)
                    if append_token(state, chosen):
                        finish_state(state, next_active)
                    else:
                        next_active.append(state)
                active = next_active
                set_progress(len(active), parallel_student_teacher)
                continue

            bundle = make_proposal_bundle(active)
            active = process_teacher(bundle, run_teacher(bundle))
            set_progress(len(active), False)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if pipeline_executor is not None:
            pipeline_executor.shutdown(wait=True)

def _make_remote_llm(model_path: str, tokenizer_path: str, gpus: str, tp: int,
                     dtype: str, gpu_memory_utilization: float,
                     max_model_len: int, max_logprobs: int,
                     max_num_seqs: int, max_num_batched_tokens: int,
                     seed: int, lora_adapter_path: str = "",
                     lora_rank: int = 0) -> RemoteVLLM:
    if tp <= 0:
        tp = _gpu_count(gpus)
    return RemoteVLLM(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        gpus=gpus,
        tensor_parallel_size=tp,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_logprobs=max_logprobs,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        seed=seed,
        lora_adapter_path=lora_adapter_path,
        lora_rank=lora_rank,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate y_o with SKD top-k/top-p accept using vLLM")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--student_model_path", required=True)
    parser.add_argument("--student_lora_adapter_path", default="")
    parser.add_argument("--teacher_model_path", default="")
    parser.add_argument("--tokenizer_path", default="")
    parser.add_argument("--teacher_tokenizer_path", default="")
    parser.add_argument(
        "--teacher_enable_thinking",
        type=lambda value: str(value).lower() in {"1", "true", "yes", "y", "on"},
        default=False,
    )
    parser.add_argument("--distill_mode", choices=["opd", "opsd"], default="opd")
    parser.add_argument("--task", choices=["math", "code"], required=True)
    parser.add_argument("--max_tokens", type=int, required=True)
    parser.add_argument("--prompt_length", type=int, required=True)
    parser.add_argument("--teacher_prompt_length", type=int, required=True)
    parser.add_argument("--max_model_len", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--student_temperature", type=float, default=0.6)
    parser.add_argument("--student_top_p", type=float, default=0.95)
    parser.add_argument("--teacher_temperature", type=float, default=0.6)
    parser.add_argument("--teacher_top_p", type=float, default=0.95)
    parser.add_argument("--student_gpus", default=os.environ.get("SKD_STUDENT_GPUS", "0,1,2,3"))
    parser.add_argument("--teacher_gpus", default=os.environ.get("SKD_TEACHER_GPUS", "4,5,6,7"))
    parser.add_argument("--shared_gpus", default=os.environ.get("SKD_SHARED_GPUS", "0,1,2,3,4,5,6,7"))
    parser.add_argument("--student_tp", type=int, default=int(os.environ.get("SKD_STUDENT_TP", "0")))
    parser.add_argument("--teacher_tp", type=int, default=int(os.environ.get("SKD_TEACHER_TP", "0")))
    parser.add_argument("--dtype", default=os.environ.get("SKD_VLLM_DTYPE", "bfloat16"))
    parser.add_argument("--gpu_memory_utilization", type=float, default=float(os.environ.get("SKD_VLLM_GPU_MEMORY_UTILIZATION", "0.85")))
    parser.add_argument("--student_gpu_memory_utilization", type=float, default=float(os.environ.get("SKD_STUDENT_VLLM_GPU_MEMORY_UTILIZATION", "0")))
    parser.add_argument("--teacher_gpu_memory_utilization", type=float, default=float(os.environ.get("SKD_TEACHER_VLLM_GPU_MEMORY_UTILIZATION", "0")))
    parser.add_argument("--max_num_seqs", type=int, default=int(os.environ.get("SKD_VLLM_MAX_NUM_SEQS", "0")))
    parser.add_argument("--student_max_num_seqs", type=int, default=int(os.environ.get("SKD_STUDENT_VLLM_MAX_NUM_SEQS", "0")))
    parser.add_argument("--teacher_max_num_seqs", type=int, default=int(os.environ.get("SKD_TEACHER_VLLM_MAX_NUM_SEQS", "0")))
    parser.add_argument("--max_num_batched_tokens", type=int, default=int(os.environ.get("SKD_VLLM_MAX_NUM_BATCHED_TOKENS", "0")))
    parser.add_argument("--student_max_num_batched_tokens", type=int, default=int(os.environ.get("SKD_STUDENT_VLLM_MAX_NUM_BATCHED_TOKENS", "0")))
    parser.add_argument("--teacher_max_num_batched_tokens", type=int, default=int(os.environ.get("SKD_TEACHER_VLLM_MAX_NUM_BATCHED_TOKENS", "0")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SKD_VLLM_SEED", "20")))
    parser.add_argument("--share_engine_if_same", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add_argument("--parallel_student_teacher", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=str(os.environ.get("SKD_PARALLEL_STUDENT_TEACHER", "true")).lower() in {"1", "true", "yes", "y"})
    parser.add_argument("--pipeline_lanes", type=int, default=int(os.environ.get("SKD_PIPELINE_LANES", "1")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.distill_mode == "opsd" and args.teacher_enable_thinking:
        raise ValueError("OPSD uses a Base self-teacher and cannot enable teacher thinking mode")
    teacher_model_path = args.teacher_model_path or args.student_model_path
    tokenizer_path = args.tokenizer_path or args.student_model_path
    teacher_tokenizer_path = args.teacher_tokenizer_path or teacher_model_path
    student_lora_rank = _load_lora_rank(args.student_lora_adapter_path)
    share_engine = (
        args.share_engine_if_same
        and not args.student_lora_adapter_path
        and teacher_model_path == args.student_model_path
    )

    dataset = pd.read_parquet(args.input)
    chats = dataset[args.prompt_key].tolist()
    rows = dataset.to_dict(orient="records")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    teacher_tokenizer = (
        tokenizer
        if teacher_tokenizer_path == tokenizer_path
        else AutoTokenizer.from_pretrained(teacher_tokenizer_path, trust_remote_code=True)
    )
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    if tokenizer.pad_token_id is not None:
        eos_ids.add(int(tokenizer.pad_token_id))

    states = []
    for i, (chat, row) in enumerate(zip(chats, rows)):
        student_prompt_ids = _prompt_token_ids(tokenizer, chat, args.prompt_length)
        teacher_prompt_ids = _build_teacher_prompt_ids(
            teacher_tokenizer,
            row,
            student_prompt_ids,
            distill_mode=args.distill_mode,
            task=args.task,
            teacher_prompt_length=args.teacher_prompt_length,
            teacher_enable_thinking=args.teacher_enable_thinking,
        )
        required_context = max(len(student_prompt_ids), len(teacher_prompt_ids)) + args.max_tokens
        if required_context > args.max_model_len:
            raise ValueError(
                f"row {i} requires {required_context} tokens for SKD rollout, "
                f"exceeding max_model_len={args.max_model_len}"
            )
        states.append(
            RowState(
                prompt_ids=student_prompt_ids,
                teacher_prompt_ids=teacher_prompt_ids,
                generated=[],
                row_idx=i,
            )
        )
    responses: list[str | None] = [None] * len(states)

    max_logprobs = max(20, int(args.top_k or 0))
    local_vllm_patch = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "local_vllm_patch"))
    if os.path.isdir(os.path.join(local_vllm_patch, "vllm")):
        if local_vllm_patch not in sys.path:
            sys.path.insert(0, local_vllm_patch)
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        if not current_pythonpath.split(":") or current_pythonpath.split(":")[0] != local_vllm_patch:
            os.environ["PYTHONPATH"] = f"{local_vllm_patch}:{current_pythonpath}" if current_pythonpath else local_vllm_patch
    if args.top_k and args.top_k > 0:
        os.environ["VLLM_SKD_PROMPT_LOGPROBS_TAIL"] = str(max(1, int(args.gamma)))
    student_gpu_memory_utilization = args.student_gpu_memory_utilization or args.gpu_memory_utilization
    teacher_gpu_memory_utilization = args.teacher_gpu_memory_utilization or args.gpu_memory_utilization
    student_max_num_seqs = args.student_max_num_seqs or args.max_num_seqs
    teacher_max_num_seqs = args.teacher_max_num_seqs or args.max_num_seqs
    student_max_num_batched_tokens = args.student_max_num_batched_tokens or args.max_num_batched_tokens
    teacher_max_num_batched_tokens = args.teacher_max_num_batched_tokens or args.max_num_batched_tokens
    print(
        "SKD vLLM y_o rollout: "
        f"rows={len(states)} task={args.task} distill_mode={args.distill_mode} "
        f"max_tokens={args.max_tokens} prompt_length={args.prompt_length} "
        f"teacher_prompt_length={args.teacher_prompt_length} "
        f"teacher_enable_thinking={args.teacher_enable_thinking} "
        f"max_model_len={args.max_model_len} batch={args.batch_size} gamma={args.gamma} "
        f"student={args.student_model_path} teacher={teacher_model_path} "
        f"student_lora_adapter={args.student_lora_adapter_path or 'none'} "
        f"student_mem={student_gpu_memory_utilization} teacher_mem={teacher_gpu_memory_utilization} "
        f"student_max_num_seqs={student_max_num_seqs} teacher_max_num_seqs={teacher_max_num_seqs} "
        f"share_engine={share_engine} parallel_student_teacher={args.parallel_student_teacher and not share_engine} "
        f"pipeline_lanes={args.pipeline_lanes} "
        f"vllm_skd_prompt_logprobs_tail={os.environ.get('VLLM_SKD_PROMPT_LOGPROBS_TAIL', '')} "
        f"local_vllm_patch={local_vllm_patch if os.path.isdir(os.path.join(local_vllm_patch, 'vllm')) else ''}",
        flush=True,
    )

    student_llm = None
    teacher_llm = None
    try:
        if share_engine:
            student_llm = _make_remote_llm(
                args.student_model_path, tokenizer_path, args.shared_gpus,
                args.student_tp or _gpu_count(args.shared_gpus), args.dtype,
                student_gpu_memory_utilization, args.max_model_len, max_logprobs,
                student_max_num_seqs, student_max_num_batched_tokens, args.seed,
                args.student_lora_adapter_path, student_lora_rank,
            )
            teacher_llm = student_llm
        else:
            student_llm = _make_remote_llm(
                args.student_model_path, tokenizer_path, args.student_gpus,
                args.student_tp or _gpu_count(args.student_gpus), args.dtype,
                student_gpu_memory_utilization, args.max_model_len, max_logprobs,
                student_max_num_seqs, student_max_num_batched_tokens, args.seed,
                args.student_lora_adapter_path, student_lora_rank,
            )
            teacher_llm = _make_remote_llm(
                teacher_model_path, teacher_tokenizer_path, args.teacher_gpus,
                args.teacher_tp or _gpu_count(args.teacher_gpus), args.dtype,
                teacher_gpu_memory_utilization, args.max_model_len, max_logprobs,
                teacher_max_num_seqs, teacher_max_num_batched_tokens, args.seed + 1,
            )

        pending = iter(states)
        active = [state for _, state in zip(range(args.batch_size), pending)]
        progress = tqdm(total=len(states), desc="SKD vLLM y_o rollout")

        def refill_state():
            return next(pending, None)

        def complete_state(state: RowState):
            responses[state.row_idx] = _decode_response(tokenizer, state.generated, eos_ids)

        _run_batch(
            states=active,
            tokenizer=tokenizer,
            student_llm=student_llm,
            teacher_llm=teacher_llm,
            eos_ids=eos_ids,
            max_tokens=args.max_tokens,
            gamma=args.gamma,
            top_k=args.top_k,
            top_p=args.top_p,
            student_temperature=args.student_temperature,
            student_top_p=args.student_top_p,
            teacher_temperature=args.teacher_temperature,
            teacher_top_p=args.teacher_top_p,
            adaptive_gamma=False,
            adaptive_gamma_low_accept=0.2,
            adaptive_gamma_high_accept=0.6,
            adaptive_gamma_ema_alpha=0.2,
            parallel_gamma1=args.parallel_student_teacher,
            pipeline_lanes=args.pipeline_lanes,
            target_rollout_seconds=0.0,
            target_rollout_safety=0.85,
            target_rollout_warmup_seconds=300.0,
            total_rows_for_budget=len(states),
            progress=progress,
            on_state_complete=complete_state,
            refill_state=refill_state,
        )
        progress.close()
    finally:
        if teacher_llm is not None and teacher_llm is not student_llm:
            teacher_llm.close()
        if student_llm is not None:
            student_llm.close()

    missing = [i for i, response in enumerate(responses) if response is None]
    if missing:
        raise RuntimeError(f"missing SKD responses for rows: {missing[:10]}")
    dataset["responses"] = [[response] for response in responses]
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(args.output)
    manifest = {
        "mode": "skd_vllm_y_o",
        "distill_mode": args.distill_mode,
        "task": args.task,
        "teacher_prompt_contract": (
            "opsd_x_y_star_v1"
            if args.distill_mode == "opsd"
            else (
                "opd_x_only_qwen3_thinking_v1"
                if args.teacher_enable_thinking
                else "opd_x_only_plain_completion_v1"
            )
        ),
        "teacher_enable_thinking": args.teacher_enable_thinking,
        "input": args.input,
        "output": args.output,
        "rows": len(dataset),
        "student_model_path": args.student_model_path,
        "student_lora_adapter_path": args.student_lora_adapter_path,
        "teacher_model_path": teacher_model_path,
        "share_engine": share_engine,
        "parallel_student_teacher": bool(args.parallel_student_teacher and not share_engine),
        "max_tokens": args.max_tokens,
        "prompt_length": args.prompt_length,
        "teacher_prompt_length": args.teacher_prompt_length,
        "max_observed_teacher_prompt_tokens": max(
            (len(_teacher_prefix_ids(state)) for state in states), default=0
        ),
        "max_model_len": args.max_model_len,
        "gamma": args.gamma,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "accept_top_k": args.top_k,
        "accept_top_p": args.top_p,
        "student_temperature": args.student_temperature,
        "student_top_p": args.student_top_p,
        "teacher_temperature": args.teacher_temperature,
        "teacher_top_p": args.teacher_top_p,
        "student_gpus": args.shared_gpus if share_engine else args.student_gpus,
        "teacher_gpus": args.shared_gpus if share_engine else args.teacher_gpus,
        "shared_gpus": args.shared_gpus if share_engine else None,
    }
    with open(f"{args.output}.skd_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Saved SKD y_o responses to {args.output}", flush=True)


if __name__ == "__main__":
    main()
