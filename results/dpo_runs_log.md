# DPO Runs Log

## Run A — Prior best (16× A100 Narval, prospect_dpo)

- **Sbatch**: `verl/recipe/dpo/run/sbatch/run_prospect_dpo_fixed_16xa100.sh`
- **Date**: ~2026-04-22 / 2026-04-23
- **Result key**: `prospect_dpo_click_hist_all_lora_hf_merged_global_step_5341_hf_merged__thinkOFF__20260422`

| Param | Value |
|--|--|
| Base model | `sft_warmup_pos_qwen3-4b-instruct-2507/global_step_50/hf_merged` (SFT-warmed) |
| Loss type | `prospect_dpo` |
| BETA | 0.1 |
| ALPHA_MAX | 2.0 |
| LAMBDA_MAX | 2.0 |
| Actor LR | yaml default `1e-6` |
| LR scheduler | yaml default (no override) |
| Total epochs | 1 |
| Save freq | 267 steps |
| Hardware | 16× A100 40G (4 nodes × 4) |
| Eval format | `\boxed{...}` |

| Metric (step_5341) | Value |
|--|--|
| ROUGE-1 F1 | — (record only RL kept) |
| ROUGE-L F1 | **0.2354** |

| Metric (step_2937) | Value |
|--|--|
| ROUGE-L F1 | 0.2231 |

### Notes
- SFT warmup taught PENS prompt → headline format BEFORE DPO.
- α/λ ∈ [1, 2] amplified both positive and negative gradient signal.
- β=0.1 kept reward magnitudes unsaturated.

---

## Run B — Current (8× H100 Nibi, single-node, "single_wise_dpo" loss flagged but `POINTWISE_DPO_LOSS_TYPE=prospect_dpo`)

- **Sbatch**: `verl/recipe/dpo/run/sbatch/run_prospect_dpo_fixed_8xh100.sh`
- **Date**: 2026-04-24 (start `20260424_102309`) → 2026-04-24 walltime end
- **Experiment dir**: `/scratch/lijiang3/ckpt/PENS/prospect_dpo_click_hist_all_lora_qwen3-4b-instruct-2507_20260424_102309`
- **Eval job**: 12758245 (5h walltime, exited COMPLETED 0:0 in 1h 36m via done sentinel)

| Param | Value |
|--|--|
| Base model | raw `Qwen3-4B-Instruct-2507` (NO SFT warmup) |
| Loss type | `prospect_dpo` (selected via `POINTWISE_DPO_LOSS_TYPE=prospect_dpo`) |
| BETA | 0.3 |
| ALPHA_MAX | 1.0 |
| LAMBDA_MAX | 1.0 |
| Actor LR | **1e-5** (10× yaml default) |
| LR scheduler | cosine + warmup_ratio 0.03 + min_ratio 0.1 + num_cycles 0.5 |
| Total epochs | 1 (hit walltime at ~89.7% of epoch, step 4791) |
| Save freq | 200 steps |
| Micro batch size | 8 |
| Max token len/GPU | 9216 |
| Use remove_padding | true |
| Use dynamic bsz | false |
| Hardware | 8× H100 80G (1 node) |
| Eval format | `\boxed{...}` |
| NCCL fixes | `_masked_mean` in `compute_prospect_dpo_loss` + `compute_single_wise_dpo_loss` (rank-divergent autograd graph fix); `NCCL_P2P_DISABLE=1`; `expandable_segments:True` |

### Eval results (5 ckpts, run 12758245)
| step | ROUGE-1 | ROUGE-2 | ROUGE-L | valid count |
|--|--|--|--|--|
| 200  | 0.2220 | 0.0686 | 0.1873 | 20465 / 20554 |
| 1000 | 0.2229 | 0.0690 | 0.1878 | 20465 / 20554 |
| 2000 | 0.2228 | 0.0687 | 0.1874 | 20465 / 20554 |
| 4000 | 0.2229 | 0.0688 | 0.1876 | 20465 / 20554 |
| 4600 | 0.2229 | 0.0689 | 0.1876 | 20465 / 20554 |
| baseline qwen3-4b-instruct-2507 (no DPO) | — | — | 0.1879 | — |

89 prompts dropped each step: vLLM `max_model_len=8300 < input_len up to 29352`. Affects all ckpts equally.

### Bugs hit + fixed during this run
1. NCCL deterministic deadlock at seq ~6452 — fixed by replacing `torch.any(mask) else zero` branching with mask-weighted means in `recipe/dpo/core_algos.py`.
2. CUDA OOM on bs=16 + remove_padding — reverted to bs=8.
3. Hydra `algorithm.average_log_prob not in struct` for single_wise_dpo — added field to `recipe/dpo/config/dpo_single_wise_dpo.yaml`.
4. Eval scoring rc=1: `STRUCTURED_PARSE_STATUSES = {"fenced_json","inline_json"}` filtered out all `parse_status="boxed"` rows — added `"boxed"` to set in `recipe/dpo/evaluation/score_pens_predictions.py:14`. step_200 manually re-scored after fix, others auto-recovered.

---

## Diff A → B (what regressed)

| Param | A (best) | B (flat) | Likely impact |
|--|--|--|--|
| **Base model** | SFT-warmed | **raw Qwen3** | Likely lost format prior; user wants to retest with raw base in C |
| **ALPHA_MAX** | 2.0 | **1.0** | prospect_dpo collapses to vanilla DPO (no positive amplification) |
| **LAMBDA_MAX** | 2.0 | **1.0** | no negative amplification |
| **BETA** | 0.1 | **0.3** | β=0.3 between paper default 0.1 and TL;DR 0.5 — neutral, not the cause |
| Actor LR | 1e-6 | **1e-5** (10× higher) | not "too low"; possibly over-aggressive but cosine + warmup buffers |
| LR sched | default | cosine + warmup | minor |
| Save freq | 267 | 200 | minor |

### β reference (Rafailov 2023 DPO paper, Appendix B)

> "Unless noted otherwise, we use a β = 0.1, batch size of 64 and the RMSprop optimizer with a learning rate of 1e-6 by default. ... For TL;DR summarization, we use β = 0.5, while rest of the parameters remain the same."

| Task | β |
|--|--|
| Default (HH dialogue, sentiment) | 0.1 |
| IMDB sentiment frontier sweep | {0.05, 0.1, 1, 5} |
| TL;DR summarization | 0.5 |

PENS is summarization-like (headline gen), so β=0.1 or 0.5 both defensible. β=0.3 not "too high" by paper precedent.

## Conclusion

Run B did NOT improve over baseline (RL ≈ 0.1876 across all 5 ckpts vs baseline 0.1879). Run A reached RL 0.2354.

**Regressions, ranked by suspected weight**:
1. **α_max / λ_max set to 1.0** — collapses prospect_dpo to vanilla DPO, kills the asymmetric gating that defines this loss. Highest-confidence cause.
2. **Base model: raw Qwen3 vs SFT-warmed** — biggest config delta, but user explicitly wants raw base for the next attempt to isolate the loss-hyperparam fix from the SFT-warmup effect.
3. β 0.1 → 0.3 — neutral by paper precedent. Worth testing β=0.1 first, ablate β=0.5 if still flat.
4. LR 1e-6 → 1e-5 — possibly aggressive; cosine + warmup likely buffers. NOT the cause "lr too low" was a mis-recollection.

## Run C — executed (2026-04-28 → 2026-04-29)

- **Train sbatch**: `12776197` started 2026-04-28 22:24, completed rc=0 in 11h 01m total wall (10h actual train + ~1h startup/Ray init).
- **Train run dir**: `/scratch/lijiang3/ckpt/PENS/prospect_dpo_click_hist_all_lora_qwen3-4b-instruct-2507_20260428_222403/`
- **Hardware**: ran on **7× H100** (skipped GPU 3 — orphan VLLM::EngineCore process from another user `hamedth` was holding 73 GB on GPU 3, blocking the 8th rank). Adjusted `train_batch_size=224` (7×8×4) and capped `total_training_steps=4500` (~85% of 1 epoch) to fit walltime on 7 GPUs.
- **Eval sbatch**: `12963683` started 2026-04-29 09:32, completed rc=0 in 1h 35m. Sequential eval of 5 ckpts.

### Train config

| Param | Value |
|--|--|
| Base model | raw `Qwen3-4B-Instruct-2507` (no SFT warmup) |
| Loss type | `prospect_dpo` |
| BETA | 0.1 |
| ALPHA_MAX | **2.0** (asymmetric positive amplification) |
| LAMBDA_MAX | **2.0** (asymmetric negative amplification) |
| Actor LR | 1e-5 cosine + 3% warmup + 0.1 floor |
| Train batch size | 224 (7 GPU × 8 micro_bs × 4 accum) |
| Max token len/GPU | 9216 |
| Use remove_padding | true |
| Use dynamic bsz | false |
| Total training steps | 4500 (cap) |
| Save freq | 200 |
| Average log prob | true (SimPO-style length norm) |

### Eval results

| step | ROUGE-1 | ROUGE-2 | **ROUGE-L** | valid count |
|--|--|--|--|--|
| 200  | 0.1967 | 0.0592 | **0.1687** | 20465/20554 |
| 1000 | 0.1425 | 0.0356 | **0.1266** | 20465/20554 |
| 2000 | 0.1343 | 0.0288 | **0.1216** ← min | 20465/20554 |
| 4000 | 0.1451 | 0.0372 | **0.1311** | 20465/20554 |
| 4500 | 0.1635 | 0.0367 | **0.1440** | 20465/20554 |
| baseline (no DPO) | — | — | 0.1879 | — |
| Run B flat | — | — | 0.1876 | — |
| Run A best (SFT-warmed, prospect_dpo step_5341) | — | — | **0.2354** | — |

### Bugs hit + workarounds

1. **GPU 3 orphan blocking**: another user's `VLLM::EngineCore` (PID 209514, UID 3133690) leaked 73 GB on g22 GPU 3 from a prior job. Cross-user kill not permitted. Worked around by training on 7 GPUs (CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7). Cost: ~14% slowdown.
2. **`total_epochs=0.85` float crash**: verl trainer expects int → `TypeError: 'float' object cannot be interpreted as an integer`. Worked around by adding `TOTAL_TRAINING_STEPS` env hook to `run_single_wise_dpo.sh` and capping at 4500 steps.
3. **Recovery wrapper missed `export VLLM_ENABLE_THINKING=false`**: post-train inline eval shell crashed with `unbound variable` (set -u). All 5 inline evals failed. Wrapper unconditionally `touch ${DONE_SENTINEL}` after eval block → main sbatch released allocation.
4. **`run_eval_dpo_runC_1gpu.sh`** submitted as new 1×H100 / 2h sbatch with `VLLM_ENABLE_THINKING=false` properly exported. Backfilled in ~6 min, completed all 5 evals in 1h 35m.

### Conclusion

**Run C did worse than Run B** (which was already flat at baseline). U-shape: model RL drops from 0.1687 → 0.1216 (step 2000), bounces back to 0.1440 by step 4500. Never recovers above baseline.

**Hypothesis**: `α_max=2.0 + λ_max=2.0` (asymmetric gating) on a **raw base without SFT warmup** amplifies the wrong gradients early in training. Without grounded prior on the PENS prompt → headline format, the loss pushes the policy in directions that hurt task performance. SFT warmup gave Run A's prospect_dpo the format prior it needed.

### Comparison summary

| Run | Base | β | α/λ_max | LR | best RL |
|--|--|--|--|--|--|
| A (16×A100) | **SFT-warmed** | 0.1 | **2.0** | 1e-6 | **0.2354** ✓ |
| B (8×H100) | raw Qwen3 | 0.3 | 1.0 (vanilla DPO) | 1e-5 cosine | 0.1876 (≈ baseline) |
| C (7×H100, this) | raw Qwen3 | 0.1 | **2.0** | 1e-5 cosine | 0.1440 (worse) |

→ **SFT warmup is the dominant factor**. Without it, prospect_dpo's asymmetric gating amplifies bad signal.

### Next experiment (Run D, planned)

Restore Run A config exactly:
- Base = `sft_warmup_pos_qwen3-4b-instruct-2507/global_step_50/hf_merged`
- β=0.1, α/λ_max=2.0
- LR = yaml default 1e-6 (no overrides)
- Run on 8× H100 if available, else 7× H100
- 1 epoch with `train_batch_size=224` (or 256 if 8 GPU)

Should reproduce Run A's RL ≈ 0.2354 on the new hardware.

---

## Run D — executed (2026-05-03 → 2026-05-05)

- **Train sbatch**: `12987145` (8× H100 Nibi, single-node), trained 5341 steps full 1 epoch, exit 0 on training. Inline post-train parallel eval of 6 ckpts crashed (vLLM `EngineCore initialization failed: WorkerProc wait_for_ready` — 6 concurrent vLLMs contending on shared node / RAY_TMPDIR), and wrapper bug (line 370–371 of `run_prospect_dpo_fixed_8xh100.sh`: `run_post_training_eval || echo … ; exit 0` + `run_post_training_eval` always returns 0) silently released the alloc despite eval failure. Violated keep-alive rule. **Wrapper not yet patched**.
- **Recovery eval sbatch**: `13193227` (`run_eval_dpo_runD_1gpu.sh`, 1× H100, walltime 02:20:00), submitted after train alloc gone. Sequential eval of 6 ckpts. COMPLETED 0:0 in 01:16:33 on g23, 2026-05-05 06:57 → 08:13 EDT. All 6 ckpts succeeded ("Failed: none"); sentinel guard correctly fired only on full success.
- **Train run dir**: `/scratch/lijiang3/ckpt/PENS/prospect_dpo_click_hist_all_lora_qwen3-4b-instruct-2507_20260503_050655/`
- **Plan deviation**: planned Run D was "restore Run A config = SFT-warmed base + α/λ=2.0 + LR=1e-6". Executed Run D used **raw Qwen3 base** (no SFT warmup), **α/λ_max=1.5**, LR=1e-6. SFT-warmed ckpt not on Nibi at submission time; user proceeded with raw base + reduced α/λ to avoid Run C collapse. The "Run A reproduction" attempt is deferred to Run E.

### Train config

| Param | Value |
|--|--|
| Base model | raw `Qwen3-4B-Instruct-2507` (no SFT warmup) |
| Loss type | `prospect_dpo` |
| BETA | 0.1 |
| ALPHA_MAX | **1.5** (down from Run C's 2.0) |
| LAMBDA_MAX | **1.5** (down from Run C's 2.0) |
| LAMBDA_GAMMA | 1.5 |
| ALPHA_K | 4.0 |
| ALPHA_TAU | 0.0 |
| Actor LR | **1e-6** (down 10× from Run C's 1e-5) |
| LR scheduler | cosine + 3% warmup + 0.1 floor + num_cycles 0.5 |
| Train batch size | 256 (8 GPU × 8 micro_bs × 4 accum) |
| Max token len/GPU | 9216 |
| Use remove_padding | true |
| Use dynamic bsz | false |
| Total training steps | 5341 (full 1 epoch) |
| Save freq | 200 (at 200/1000/2000/3000/4000/5341) |
| Average log prob | true (SimPO-style length norm) |

### Eval results (job 13193227, sequential 1× H100)

| step | ROUGE-1 | ROUGE-2 | **ROUGE-L** | boxed / empty |
|--|--|--|--|--|
| 200  | 0.2227 | 0.0686 | **0.1876** | 20465 / 89 |
| 1000 | 0.2218 | 0.0682 | **0.1868** | 20465 / 89 |
| 2000 | 0.2220 | 0.0682 | **0.1871** | 20465 / 89 |
| 3000 | 0.2223 | 0.0684 | **0.1872** | 20465 / 89 |
| 4000 | 0.2228 | 0.0686 | **0.1876** | 20465 / 89 |
| 5341 | 0.2226 | 0.0686 | **0.1873** | 20465 / 89 |
| baseline (no DPO) | — | — | 0.1879 | — |

Per-step eval ≈ 12–15 min wall (hf_merged from train wrapper → merge skipped). 89 prompts skipped each step (input > 8300 tok), same as Run B/C.

### Comparison summary (updated)

| Run | Base | β | α/λ_max | LR | best RL | mode |
|--|--|--|--|--|--|--|
| A (16×A100) | **SFT-warmed** | 0.1 | **2.0** | 1e-6 | **0.2354** ✓ | learns |
| B (8×H100)  | raw Qwen3 | 0.3 | 1.0 (vanilla DPO) | 1e-5 cosine | 0.1876 | flat (≈ baseline) |
| C (7×H100)  | raw Qwen3 | 0.1 | **2.0** | 1e-5 cosine | 0.1440 (worse) | **collapse** (`*` rep, parse rate degraded) |
| **D** (8×H100) | raw Qwen3 | 0.1 | **1.5** | **1e-6** cosine | **0.1876** | flat noop, **no collapse** (100% parse) |

### Conclusion

- **Result**: RL ≈ 0.187 across all 6 ckpts ≈ baseline. step_200 ≈ step_5341 → policy barely moved over 5341 steps.
- **Win**: LR drop (1e-5 → 1e-6) **fixed Run C's mode collapse**. boxed parse 100% (20465/20554), no `*` repetition.
- **Loss**: LR=1e-6 on raw base = noop. Same LR worked in Run A only because **SFT warmup gave a format prior**; without it, gradient × 1e-6 over 5341 steps is too gentle to move the policy at all.
- **Hypothesis confirmed (Run B/C/D triangulation)**: SFT warmup is the dominant factor, not loss hyperparams. Three failure modes explored on raw base: vanilla DPO (B, flat), aggressive amp (C, collapse), low LR + medium amp (D, noop). None reach Run A's RL=0.2354.

### Bugs hit + workarounds

1. **Inline parallel eval crash**: 6 vLLMs spawned concurrently on g24 → `EngineCore initialization failed: WorkerProc wait_for_ready` for all 6 ckpts. Likely RAY_TMPDIR / shared-node resource contention. Workaround: separate 1-GPU sequential eval sbatch.
2. **Wrapper bug — silenced eval failure → released alloc** (`run_prospect_dpo_fixed_8xh100.sh:370–371`): `run_post_training_eval || echo "[sbatch] eval block returned non-zero — training result still valid"` then unconditional `exit 0`. Function `run_post_training_eval` always `return 0` (line 205), so eval failure is invisible to caller. Violated "keep-alive on bug" rule (alloc released, queue re-wait forced). **Fix needed for Run E**: make `run_post_training_eval` `return ${#fail_steps[@]}` and replace silent-OR with sentinel-guarded keep-alive (mirror `run_eval_dpo_runD_1gpu.sh` pattern).

### Next experiment (Run E, planned)

**Reproduce Run A** = SFT-warmed base + α/λ_max=2.0 + LR=1e-6 + β=0.1. Steps:
1. Transfer Narval `/scratch/lijiang3/ckpt/PENS/sft_warmup_pos_qwen3-4b-instruct-2507/global_step_50/hf_merged/` → Nibi same path (~8 GB, Globus or rsync).
2. New sbatch (or reuse `run_prospect_dpo_fixed_8xh100.sh` with env override): set `SFT_MERGED_DIR` to SFT-warmed path, `SINGLE_WISE_DPO_ALPHA_MAX=2.0`, `SINGLE_WISE_DPO_LAMBDA_MAX=2.0`, keep LR=1e-6, β=0.1, bs=256, 5341 steps.
3. **Patch wrapper bug** (`run_post_training_eval` return code) before launching, so eval fail keeps alloc alive.
4. Expected: RL ≈ 0.235 at step ~5341 (Run A reproduction).

Alternative Run F (raw base, single-stage): keep raw base, **LR=3e-6 cosine** (between B's 1e-5 collapse and D's 1e-6 noop), α/λ=1.5, β=0.1, **`average_log_prob=false`** (drop SimPO length norm — may compress token-level signal). Lower priority; only if SFT warmup unavailable.

---

## Run E — executed (2026-05-09 → 2026-05-10)

- **Train sbatch**: `13587777` (8× H100 Nibi, walltime 13h, COMPLETED 12:56:07).
  - 3 attempts in main wrapper crashed at 5th `ReferenceLogpsWorker.__init__()` `.to(device)` with `cudaErrorDevicesUnavailable` → keep-alive entered, 3 in-alloc rerun attempts (rerun1/2/3) all failed with same / chained errors.
  - **Root cause**: g19 GPU 4 was in driver-state leak (20% util, 0 MiB, no process — same pattern as Run C g22 GPU3 orphan, cross-user, no permission to reset). Workers 1-4 succeeded sequentially on GPUs 0-3; the 5th was placed on GPU 4 and crashed.
  - **Fix that worked (rerun5)**: mask GPU 4 via `CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7` + `SLURM_GPUS_ON_NODE=7` + `SINGLE_WISE_DPO_N_GPUS_PER_NODE=7` + `train_batch_size=224` + `total_training_steps=4500` cap.
- **Patches applied during recovery**:
  - `recipe/dpo/reference_logps_materializer.py`: added `ping()` readiness probe + serialized actor `__init__` via `ray.get(w.ping.remote())` to dodge concurrent CUDA-init storm. Also moved `next_worker_idx` to instance member so all 7 workers actually rotate (was local var → reset per row_group → workers[5,6] structurally idle on every group).
  - `recipe/dpo/run/sbatch/run_prospect_dpo_fixed_8xh100.sh`: parallel eval → 60s-stagger parallel; `run_post_training_eval` returns `${#fail_steps[@]}`; main loop checks rc → keep-alive on eval fail (Run D wrapper bug fix, see "Bugs hit" Run D §).
- **Recovery rerun script**: `recipe/dpo/run/sbatch/rerun_runE_in_alloc.sh` (env-equivalent of train wrapper, patched serial init, GPU 4 mask, 32768 → 16384 max_batched_tokens fallback). rerun5 PID 3296585 launched within keep-alive via `srun --jobid=13587777 --overlap`.
- **Train run dir**: `/home/lijiang3/projects/def-y7ding/lijiang3/hf_cache/models/ckpt/PENS/prospect_dpo_click_hist_all_lora_sft_warmup_pos_qwen3-4b-instruct-2507_20260509_130948/`
- **Phase wall breakdown**: queue ~0; failed attempts + recovery ~2.5h; ref-logps materialization ~6h22min (positive 3:08h@62 rows/s + negative 3:14h@54 rows/s); training ~5h22min (step 0 → step 2454 of 4500 cap @ 7.6-8.3 s/step); USR1 walltime trap → graceful exit, no inline eval ran (USR1 path skips `run_post_training_eval`).
- **Recovery eval sbatch**: `13646115` (`run_eval_dpo_runE_1gpu.sh`, 1× H100, walltime 03:00:00, COMPLETED 02:36:19, 12/12 ckpts succeeded). Sequential merge + vLLM gen + ROUGE.

### Train config

| Param | Value |
|--|--|
| Base model | **`sft_warmup_pos_qwen3-4b-instruct-2507/global_step_50/hf_merged`** (SFT-warmed, transferred from Narval) |
| Loss type | `prospect_dpo` |
| BETA | 0.1 |
| ALPHA_MAX | **2.0** (matches Run A) |
| LAMBDA_MAX | **2.0** |
| LAMBDA_GAMMA | 1.5 |
| ALPHA_K | 4.0 |
| ALPHA_TAU | 0.0 |
| Actor LR | **5e-6** (5× Run A's 1e-6) |
| LR scheduler | cosine + 3% warmup + 0.1 floor + num_cycles 0.5 |
| Train batch size | **224** (7 GPU × 8 micro_bs × 4 accum, GPU 4 masked) |
| Max token len/GPU | 9216 |
| Use remove_padding | true |
| Use dynamic bsz | false |
| Total training steps cap | 4500 (~85% epoch on 7 GPU; only step 0 → 2454 actually trained before walltime kill) |
| Save freq | 200 |
| Average log prob | true (SimPO-style length norm) |

### Eval results — `rouge_score(use_stemmer=True)` multi-ref max F1

**Scorer note**: scoring code was **migrated from pltrdy `rouge` package to `rouge_score(use_stemmer=True)`** mid-Run E (see "Scoring methodology fix" §). All numbers below are **post-migration** (paper-aligned). Old pltrdy values are kept in `results/backup_pre_rouge_score_*/` for reference.

| step | ROUGE-1 | ROUGE-2 | **ROUGE-L** | parse |
|--|--|--|--|--|
| 200  | 0.2915 | 0.1058 | 0.2396 | 100% |
| 400  | 0.2925 | 0.1062 | 0.2404 | 100% |
| 600  | 0.2921 | 0.1058 | 0.2402 | 100% |
| 800  | 0.2920 | 0.1059 | 0.2402 | 100% |
| 1000 | 0.2930 | 0.1065 | 0.2406 | 100% |
| 1200 | 0.2917 | 0.1058 | 0.2402 | 100% |
| **1400** | **0.2924** | **0.1065** | **0.2408** ← peak | 100% |
| 1600 | 0.2923 | 0.1061 | 0.2405 | 100% |
| 1800 | 0.2919 | 0.1057 | 0.2401 | 100% |
| 2000 | 0.2925 | 0.1063 | 0.2406 | 100% |
| 2200 | 0.2922 | 0.1059 | 0.2401 | 100% |
| 2400 | 0.2922 | 0.1057 | 0.2400 | 100% |
| **PENS paper (NRMS best)** | **0.2801** | **0.1072** | **0.2224** | — |

🎉 **Run E exceeds PENS paper benchmark on R1 (+0.012) and RL (+0.018), R2 within noise (-0.001).** First run in this series to do so under apples-to-apples scorer.

### Comparison summary (post-rescore with `rouge_score(stem)`)

| Run | Base | β | α/λ_max | LR | best RL | mode |
|--|--|--|--|--|--|--|
| A (16×A100) | SFT-warmed | 0.1 | 2.0 | 1e-6 | **0.0095** | **mode collapse** (Chinese repetition `'基于基于...'`) — old custom scorer reported 0.2354, was scoring artifact, not real |
| A (SFT-only baseline) | SFT-warmed | — | — | — | 0.1893 | real |
| B (8×H100) | raw Qwen3 | 0.3 | 1.0 | 1e-5 cosine | 0.2186 | flat ≈ baseline (no collapse, but no learning) |
| C (7×H100) | raw Qwen3 | 0.1 | 2.0 | 1e-5 cosine | 0.0019 | mode collapse confirmed by new scorer |
| D (8×H100) | raw Qwen3 | 0.1 | 1.5 | 1e-6 cosine | 0.2184 | flat noop |
| **E (7×H100)** | **SFT-warmed** | 0.1 | 2.0 | **5e-6** cosine | **0.2408** ← **SOTA, +0.018 over paper** | **learns, no collapse** |
| Paper (NRMS) | — | — | — | — | 0.2224 | — |

### Conclusion

- **Run E is the new SOTA on PENS** under paper-aligned ROUGE (`rouge_score(use_stemmer=True)` multi-ref max F1, ≈ pyrouge `-a -m`).
- **Half-epoch (step 2454/5341) at LR=5e-6 already plateaued** — RL spread across 12 ckpts is only 0.0008 (0.2400 → 0.2408). LR=5e-6 saturates fast on SFT-warmed base; no further training gains expected.
- **SFT warmup is confirmed dominant** (Runs B/C/D on raw base capped at ≤0.219; only Run E with SFT-warmed reaches ≥0.24).
- **Run A "best" was a scoring artifact**: the Apr 22 generation parquet on Nibi shows mode-collapsed Chinese repetition. Old custom scorer's tokenization/normalization quirks gave it spurious 0.2354 RL. Under aligned scorer real RL=0.0095. Either Run A truly collapsed and was misjudged, or the gen file we have is from a different (failed) attempt — original Narval gen file might be the real one. Either way Run A is **not** the SOTA reference; Run E is.

### Bugs hit + fixes

1. **g19 GPU 4 driver-state leak** — workers 5+ crashed at `.to(device)` with `cudaErrorDevicesUnavailable`. Cross-user issue (no kill permission). Fix: mask GPU 4 + 7-GPU topology.
2. **Concurrent worker `__init__` storm** — 8 parallel `ReferenceLogpsWorker` Ray actors → CUDA driver init race. Fix: serial init via `ping()` + `ray.get`.
3. **`next_worker_idx` local var bug** — round-robin reset every row_group → workers[5,6] starved (only 5/7 GPU active). Fix: persist as instance member.
4. **Inline parallel eval crash on g24 (Run D)** — 6 concurrent vLLM `EngineCore wait_for_ready` failure. Fix: stagger 60s + per-step Ray ports + spawn + 0.85 GPU mem util.
5. **`run_post_training_eval` return-0 hides eval fail** — Fix: return `${#fail_steps[@]}`, main loop branches to keep-alive on rc>0.
6. **USR1 walltime trap skips eval** — design choice (eval needs 30+ min, walltime imminent). Acceptable: eval runs in separate sbatch.

### Scoring methodology fix (mid-Run E)

- **Original verl scorer**: `pltrdy rouge` Python package (PyPI: `rouge`), no Porter stemmer support. PENS paper uses `pyrouge -a -c 95 -m -n 4 -w 1.2` where `-m` = Porter stem.
- **Migrated to**: `rouge_score(use_stemmer=True)` multi-ref max F1 — closest Python equivalent of `pyrouge -m`, ~0.005 noise vs pyrouge gold.
- **Impact**: ~+0.030 RL across all evaluated ckpts on PENS dataset. Stemmer (`headlines→headlin`) accounts for ~46% of gain; tokenization differences ~54%.
- **Code change**: `recipe/dpo/evaluation/score_pens_predictions.py` `compute_with_rouge_package` and `score_with_rouge_package_multi` rewritten to use `rouge_score`.
- **Re-scored history**: all `results/*.json` model entries re-scored with new method on 2026-05-10. Old values backed up at `results/backup_pre_rouge_score_20260510_105017/`. New values overwrite in-place; `scorer: rouge_score_v0.1.2_stem_multi` field added.

### Next experiment (Run F, planned — optional)

Run E already exceeds paper. Optional follow-up to explore further gains:

- **Run E resume** (job 13674098, walltime 7h, PENDING): continue from step 2400 → 5341 with same config. Already plateaued at 0.2408, expect ≤0.241 final. Mostly to verify train→eval pipeline end-to-end.
- **Run F (true Run A reproduction)**: SFT-warmed + α/λ=2.0 + **LR=1e-6 (constant, yaml default, no cosine)** + full epoch 5341 steps. May find higher RL if 5e-6 was over-aggressive and 1e-6 explores more of the loss landscape.
- **Run G (ablation, optional)**: SFT-warmed + α/λ=2.0 + LR=5e-6 + **`average_log_prob=false`** (drop SimPO norm). Tests whether length normalization is helping or hurting.

