# y_o H100 Job 299303 Bottleneck Report

Date: 2026-05-19

## Run Summary

- Slurm job: `299303`
- Script: `recipe/gkd/megatron/run/sbatch/opd/multi_step_forward_y_o.sh`
- Final state: `CANCELLED by 3147326`
- Elapsed allocation time: `09:20:43`
- Nodes: `tg10701,tg11208`
- Allocated resources: `2` nodes, `8 x H100`, `96` CPU cores, `192G` host memory
- Approximate consumed allocation: `74.8` H100 GPU-hours, `18.7` node-hours, `897` CPU core-hours, `1794` GB host-memory-hours
- Main log: `/scratch/l/luli/openclaw/tmp/opd_megatron/299303/rerun_2node_teacher_train_share_shm_dense8_initial_sync_then_opt_sync.log`
- Teacher logs: `/scratch/l/luli/openclaw/tmp/opd_megatron/299303/teacher/worker.[0-3].log`

The job was cancelled manually to release resources. The final log tail shows termination from `scancel` (`Training exited with status 143`), not a new Python traceback.

## Active Configuration

- Training mode: `y_o`
- Scheduler: `two_step_off`
- Optimization mode: `multi_step`
- Teacher backend: `vllm_server`
- Teacher model: `Qwen3-8B`
- Student model: `Qwen3-1.7B`
- KL method: `full_vocab`
- `N_LOGPROBS=full_vocab`
- `MAX_PROMPT_LENGTH=1024`
- `MAX_RESPONSE_LENGTH=8192`
- `TEACHER_SEQ_LEN=9216`
- `EFFECTIVE_BATCH_SIZE=32`
- `TRAIN_BATCH_SIZE=4`
- `GRADIENT_ACCUMULATION_STEPS=8`
- `TEACHER_N_SERVER_WORKERS=4`
- `TEACHER_REQUEST_BATCH_SIZE=1`
- `AGENT_NUM_WORKERS=4`
- `ROLLOUT_MAX_NUM_BATCHED_TOKENS=20480`
- `STUDENT_ACTOR_MAX_TOKENS_PER_GPU=12288`

Physical layout for the latest run:

- `tg10701` GPU `0,1,2,3`: teacher vLLM replicas, `TP=1`, 4 replicas total
- `tg11208` GPU `0,1`: student rollout vLLM
- `tg11208` GPU `2,3`: Megatron actor

## Step Timing

Stable long-response region, steps `2-9`:

| Metric | Value |
|---|---:|
| Avg response length | `7373.6` tokens |
| Avg rollout generation | `59.3s` |
| Avg teacher knowledge | `267.0s` |
| Avg actor update | `6.1s` |
| Avg step wall time | `267.0s` |
| Avg actor MFU | `3.82%` |
| Peak actor GPU reserved, reported | `66.1 GB` |
| Peak task CPU memory, reported | `82.4 GB` |

All logged steps, steps `1-11`:

| Metric | Value |
|---|---:|
| Avg response length | `6872.5` tokens |
| Avg rollout generation | `55.3s` |
| Avg teacher knowledge | `240.3s` |
| Avg actor update | `6.3s` |
| Avg step wall time | `254.0s` |
| Avg actor MFU | `3.59%` |
| Peak actor GPU reserved, reported | `67.1 GB` |
| Peak task CPU memory, reported | `82.4 GB` |

Representative step table:

| Step | Avg resp | Rollout mean | Teacher | Actor update | Step wall | Optimizer stepped |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7928 | 63.7s | 259.3s | 12.2s | 413.2s | 0 |
| 2 | 6756 | 53.3s | 231.6s | 5.7s | 225.5s | 0 |
| 3 | 6803 | 53.8s | 223.2s | 6.4s | 225.6s | 0 |
| 4 | 7333 | 58.2s | 267.0s | 7.3s | 267.5s | 0 |
| 5 | 7345 | 58.9s | 270.7s | 5.7s | 268.2s | 0 |
| 6 | 7950 | 63.7s | 292.4s | 6.0s | 298.2s | 0 |
| 7 | 7010 | 55.8s | 251.5s | 5.8s | 251.0s | 0 |
| 8 | 8192 | 68.0s | 314.5s | 6.4s | 308.3s | 1 |
| 9 | 7600 | 63.0s | 285.2s | 5.8s | 291.3s | 0 |
| 10 | 3985 | 32.0s | 111.2s | 3.8s | 102.1s | 0 |
| 11 | 4695 | 37.7s | 136.6s | 4.2s | 143.4s | 0 |

Step 8 confirms the intended accumulation behavior: with `EFFECTIVE_BATCH_SIZE=32` and `TRAIN_BATCH_SIZE=4`, the optimizer steps at accumulation iteration 8. Step 9 then performs rollout weight sync after the optimizer step.

## Teacher Worker Timing

Aggregated from teacher worker logs:

| Timer | Count | Avg | Min | Max |
|---|---:|---:|---:|---:|
| `get_prompt_topk_logprobs` | 49 | `150.0s` | `11.9s` | `230.2s` |
| `serialize` | 49 | `1.83s` | `0.20s` | `3.94s` |
| `send` | 49 | `1.18s` | `0.14s` | `2.69s` |
| `deserialize` | 52 | `0.012s` | `0.002s` | `0.046s` |

This shows the bottleneck is server-side vLLM full-vocab logprob computation, not ZMQ send or serialization. Communication overhead is seconds-level; teacher full-vocab scoring is tens to hundreds of seconds per request.

## Bottleneck

The current bottleneck is remote teacher full-vocab logprob calculation:

`student rollout -> remote teacher vLLM full-vocab prompt_logprobs -> actor KL update`

For `TRAIN_BATCH_SIZE=4`, `TEACHER_N_SERVER_WORKERS=4`, and `TEACHER_REQUEST_BATCH_SIZE=1`, each batch becomes four single-sample teacher requests. The batch waits on the slowest teacher worker. With long `8192` token responses, the teacher path dominates the whole step.

The actor and rollout are not the limiting path:

- Rollout is around `59s` in stable long-response steps.
- Actor update is around `6s`.
- Teacher knowledge is around `267s`.
- Actor MFU is only around `3-4%`, because actor GPUs spend most wall time waiting for teacher output.

The data volume is also large. At the stable average response length, dense bf16 full-vocab teacher payload is roughly:

`4 samples x 7374 loss tokens x 151936 vocab x 2 bytes ~= 9.0 GB`

That estimate excludes Python object, serialization, and Ray/ZMQ overhead. The current compact-row payload avoids padding/prompt rows, but it still transfers dense `[loss_tokens, vocab]` distributions.

## Why Old recipe/opd Was Faster

The old `recipe/opd` full-vocab training did not have the same cost model:

- Teacher forward happened locally inside the training path, not through a remote vLLM server.
- Dense teacher logits were consumed locally by the loss, not serialized as a remote payload.
- KL was computed sample-by-sample and chunked across token positions.
- It avoided transporting multi-GB full-vocab distributions through ZMQ/Ray.

So "full_vocab" in the old recipe was exact full-vocab, but not remote full-vocab payload transfer.

## Optimization Directions

1. Exact full-vocab path: switch to local teacher loss.

   Use the existing `TEACHER_BACKEND=local_hf` path so actor workers load the teacher locally and compute KL inside the actor `logits_processor`. This avoids returning dense full-vocab teacher logprobs from a remote server. It preserves exact full-vocab semantics, but risks actor GPU memory pressure because the actor GPU must also host the teacher model.

   A more suitable layout for this path is likely:

   - node 0: rollout, 4 GPUs
   - node 1: actor plus local teacher loss, 4 GPUs

   The launch scripts need a guard so `TEACHER_BACKEND=local_hf` does not start or wait for the teacher vLLM server.

2. Approximate path: switch to top-k teacher payload.

   Use a finite `N_LOGPROBS`, for example `N_LOGPROBS=64` and `TOP_K=64`. This reduces teacher payload from `[tokens, vocab]` to `[tokens, 64]` and should drastically reduce teacher transfer/memory pressure. This changes the loss target and is no longer exact full-vocab KL.

3. GPU split alone is not enough.

   Moving from `4/2/2` to `6/1/1` will not address the single-request full-vocab cost unless `TRAIN_BATCH_SIZE` and teacher concurrency are also increased enough to use more than four teacher workers. It also reduces rollout/actor capacity. With the current `TRAIN_BATCH_SIZE=4`, more than four teacher replicas are mostly idle.

## Current Status

Job `299303` has been cancelled and resources have been released. No allocation is currently being held for this run.
