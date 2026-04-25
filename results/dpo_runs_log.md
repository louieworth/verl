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

## Run C — re-submit (planned)

Same hardware (8× H100 single-node) and walltime trimmed to **14 h** (1-epoch 11 h + inline post-eval ~15 min + ~2.5 h margin). Keep raw base per user direction.

| Param | Value | Change vs Run B |
|--|--|--|
| Base model | raw `Qwen3-4B-Instruct-2507` | unchanged |
| BETA | **0.1** | 0.3 → 0.1 |
| ALPHA_MAX | **2.0** | 1.0 → 2.0 |
| LAMBDA_MAX | **2.0** | 1.0 → 2.0 |
| Actor LR / sched | 1e-5 cosine + 3% warmup + 0.1 floor | unchanged |
| Walltime | **14 h** | 24 h → 14 h |
| Inline post-eval | last ckpt only | unchanged |
| Multi-ckpt eval | separate sbatch after train | same workflow as Run B |
| All NCCL fixes / remove_padding=true / micro_bs=8 / max_token_len=9216 | unchanged | unchanged |

Run B `_20260424_102309` checkpoint dir (244 GB) deleted before submission.

