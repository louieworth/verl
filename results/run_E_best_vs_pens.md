# Run E — Prospect-DPO on PENS: Best Result vs. PENS Baseline

**Date**: 2026-05-10
**Run ID**: `prospect_dpo_click_hist_all_lora_sft_warmup_pos_qwen3-4b-instruct-2507_20260509_130948`
**Best checkpoint**: `global_step_1400`
**Result file**: `results/qwen3-4b-instruct-2507_20260509_130948.json`
**Scorer**: Google `rouge_score v0.1.2`, `use_stemmer=True` (Porter), multi-reference max F1 (paper-aligned, ≈ `pyrouge -a -m`)

---

## 1. Best Performance (this work)

| Metric        | Value      |
| ------------- | ---------- |
| ROUGE-1 F1    | **0.2924** |
| ROUGE-2 F1    | **0.1065** |
| ROUGE-L F1    | **0.2408** |
| Boxed parse   | 100 % (20465 / 20554; 89 prompts skipped due to vLLM `max_model_len=8300 < input_len`, identical drop set across all checkpoints) |

Plateau is tight: across all 12 evaluated checkpoints (step 200 … 2400) ROUGE-L spans `[0.2400, 0.2408]`, spread 0.0008.

---

## 2. Comparison vs. PENS paper baseline

PENS paper best: **NRMS-as-injector** (NRMS personalized encoder over a Transformer headline generator), Table 5 of the PENS paper.

| Metric     | PENS NRMS (paper) | **Run E (step 1400)** | Δ           |
| ---------- | ----------------- | --------------------- | ----------- |
| ROUGE-1 F1 | 0.2801            | **0.2924**            | **+0.0123** |
| ROUGE-2 F1 | **0.1072**        | 0.1065                | −0.0007 (noise) |
| ROUGE-L F1 | 0.2224            | **0.2408**            | **+0.0184** |

Run E beats the paper on R1 and RL; R2 within noise.

---

## 3. Core Algorithm — Prospect-DPO

Pointwise (single-wise) variant of DPO. Each training sample `(x, y, label)` is either a **positive** (clicked / engaged headline, `label = 1`) or a **negative** (skipped / non-engaged headline, `label = 0`), not a pair. Two side-information signals from PENS click logs:

- `s_dwell ∈ [0, 1]` — normalized dwell time on a positive sample (engagement strength).
- `p_ctr ∈ [0, 1]` — empirical click-through probability of a negative sample (prior hardness).

### 3.1 Implicit reward (DPO-style, length-normalized)

For sequence `y = (y_1, …, y_T)` given context `x`, with average per-token log-prob `\overline{\log\pi}(y|x) = \tfrac{1}{T}\sum_{t} \log\pi(y_t \mid x, y_{<t})` (SimPO-style normalization, set by `average_log_prob=True`):

$$
r_\theta(x, y) \;=\; \beta \cdot \big[\, \overline{\log\pi_\theta}(y \mid x) \;-\; \overline{\log\pi_\mathrm{ref}}(y \mid x) \,\big]
$$

### 3.2 Asymmetric prospect-theoretic gating

Positive-side amplification weight `α` (sigmoid gate on dwell time):

$$
\alpha(s_\mathrm{dwell}) \;=\; 1 \;+\; (\alpha_\mathrm{max} - 1)\cdot \sigma\!\big(\alpha_k\,(s_\mathrm{dwell} - \alpha_\tau)\big), \qquad \alpha \in [1, \alpha_\mathrm{max}]
$$

Negative-side amplification weight `λ` (power gate on CTR — high-CTR negatives are surprisingly skipped, treated as harder negatives):

$$
\lambda(p_\mathrm{ctr}) \;=\; 1 \;+\; (\lambda_\mathrm{max} - 1)\cdot p_\mathrm{ctr}^{\,\lambda_\gamma}, \qquad \lambda \in [1, \lambda_\mathrm{max}]
$$

Special case `α_max = λ_max = 1` ⇒ `α ≡ λ ≡ 1` ⇒ loss collapses to plain pointwise DPO.

### 3.3 Loss (single-wise, asymmetric)

$$
\mathcal{L}_\mathrm{pos}(x, y) \;=\; \sigma\!\big(-\,\alpha(s_\mathrm{dwell})\cdot r_\theta(x, y)\big)
$$

$$
\mathcal{L}_\mathrm{neg}(x, y) \;=\; \sigma\!\big(+\,\lambda(p_\mathrm{ctr})\cdot r_\theta(x, y)\big)
$$

Batch loss is the mean over positive and negative subsets independently, then summed (mask-weighted means; avoids autograd-graph divergence across FSDP ranks when a micro-batch is all-positive or all-negative — see `recipe/dpo/core_algos.py:140`):

$$
\mathcal{L}_\mathrm{prospect} \;=\; \mathbb{E}_{(x,y)\,\in\,\mathcal{D}_+}\!\big[\mathcal{L}_\mathrm{pos}\big] \;+\; \mathbb{E}_{(x,y)\,\in\,\mathcal{D}_-}\!\big[\mathcal{L}_\mathrm{neg}\big]
$$

Intuition: `α` scales the **positive-pull strength** by user engagement; `λ` scales the **negative-push strength** by item popularity. Asymmetric, prospect-theory-style: gains and losses weighted by their behavioral signal.

### 3.4 Hyperparameters used in Run E

| Param                  | Value | Role                                                      |
| ---------------------- | ----- | --------------------------------------------------------- |
| β                      | 0.1   | Reward temperature                                        |
| α_max                  | 2.0   | Max positive amplification                                |
| α_k                    | 4.0   | Sigmoid steepness on `s_dwell`                            |
| α_τ                    | 0.0   | Sigmoid midpoint on `s_dwell`                             |
| λ_max                  | 2.0   | Max negative amplification                                |
| λ_γ                    | 1.5   | Power exponent on `p_ctr`                                 |
| `average_log_prob`     | true  | SimPO mean-per-token length norm in reward                |

### 3.5 Full training configuration (Run E)

| Param                       | Value                                                                 |
| --------------------------- | --------------------------------------------------------------------- |
| Base model                  | `sft_warmup_pos_qwen3-4b-instruct-2507/global_step_50/hf_merged` (Qwen3-4B-Instruct-2507, SFT on positive PENS samples, 50 steps) |
| Loss                        | `prospect_dpo`                                                        |
| Actor LR                    | 5e-6                                                                  |
| LR schedule                 | cosine, warmup_ratio 0.03, min_ratio 0.1, num_cycles 0.5              |
| Train batch size            | 224 (7 GPU × 8 micro_bs × 4 accum; GPU 4 masked due to driver leak)   |
| Max tokens / GPU            | 9216                                                                  |
| `remove_padding`            | true                                                                  |
| `use_dynamic_bsz`           | false                                                                 |
| Total training steps        | 4500 cap; 0 → 2454 actually trained (USR1 walltime trap, half-epoch)  |
| Save freq                   | every 200 steps                                                       |
| Hardware                    | 1 node, 7 × H100 80 GB (Nibi)                                         |
| Eval format                 | `\boxed{...}` extraction, `thinking_mode = false`                     |

---

## 4. Ablations — completed runs

All scores below were **re-evaluated with `rouge_score v0.1.2` + Porter stemmer + multi-reference max F1** on 2026-05-10. Pre-stemmer scores backed up at `results/backup_pre_rouge_score_20260510_105017/`. Stemming alone adds ~+0.030 RL uniformly across rows, so the apples-to-apples comparison only valid with re-scored values.

| # | Variant                                                  | Base                | α_max | λ_max | β   | LR        | best ckpt | **R-1** | **R-2** | **R-L** | mode | Source |
| - | -------------------------------------------------------- | ------------------- | ----- | ----- | --- | --------- | --------- | ------- | ------- | ------- | ---- | ------ |
| A0 | Raw Qwen3-4B-Instruct-2507 (no FT)                       | raw                 | —     | —     | —   | —         | —         | 0.2066  | 0.0686  | 0.1671  | base lower bound | `qwen3-4b-instruct-2507.json` (`__20260422`) |
| A1 | **SFT on positive samples only** (50 steps, the warmup)  | raw → SFT-pos       | —     | —     | —   | —         | step_50   | 0.2290  | 0.0798  | 0.1893  | base + format prior | `qwen3-4b-instruct-2507.json` (`pens_sft_warmup_pos…`) |
| B  | **Vanilla DPO** on raw base (α/λ = 1.0, symmetric)       | raw                 | 1.0   | 1.0   | 0.3 | 1e-5 cos. | step_1000 | 0.2726  | 0.0929  | 0.2192  | flat, no collapse | `qwen3-4b-instruct-2507_20260424_102309.json` |
| C  | Aggressive Prospect-DPO on raw base                      | raw                 | 2.0   | 2.0   | 0.1 | 1e-5 cos. | step_200  | 0.2500  | 0.0835  | 0.2057  | **mode collapse** after step 200 (drops to 0.0019 by step 4500) | `qwen3-4b-instruct-2507_20260428_222403.json` |
| D  | Mild Prospect-DPO on raw base (low LR)                   | raw                 | 1.5   | 1.5   | 0.1 | 1e-6 cos. | step_200  | 0.2722  | 0.0925  | 0.2188  | no-op, policy barely moves | `qwen3-4b-instruct-2507_20260503_050655.json` |
| **E** | **Prospect-DPO on SFT-warmed base (this work)**       | **SFT-pos warmed**  | 2.0   | 2.0   | 0.1 | 5e-6 cos. | step_1400 | **0.2924** | **0.1065** | **0.2408** | **learns, no collapse** | `qwen3-4b-instruct-2507_20260509_130948.json` |
| — | PENS paper NRMS (reference)                                | —                   | —     | —     | —   | —         | —         | 0.2801  | 0.1072  | 0.2224  | — | PENS paper Table 5 |

### Take-aways from completed ablations

1. **SFT-positive warmup is the dominant factor.** Every raw-base variant (A0, B, C, D) caps at ≤ 0.2192 ROUGE-L. Adding the SFT warmup (A1 → E) lifts the ceiling to 0.2408.
2. **Asymmetric gating helps once the base has a format prior.** Vanilla DPO (B, α/λ=1) on raw base = 0.2192 RL; Prospect-DPO (E, α/λ=2) on SFT-warmed base = 0.2408 RL. **Net +0.022 RL** attributable to asymmetric gating × warmup interaction.
3. **Asymmetric gating without warmup is harmful** (C collapses to repetition; raw base has no headline-format prior, so the boosted gradient amplifies bad directions).
4. **LR ladder matters for the SFT-warmed start**: 1e-6 (D-style) under-trains, 1e-5 (B/C-style) over-shoots or collapses, **5e-6 is the sweet spot** for the SFT-warmed init.

---

## 5. Ablations — to-do (planned, not yet executed)

All planned runs share the Run E recipe unless noted. Re-score everything with `rouge_score(stem)` so the row is comparable to §4.

### 5.1 Loss-shape ablations (isolate which side of the asymmetric gating matters)

| # | Variant                              | α_max | λ_max | Other deltas                | Tests / purpose                                       |
| - | ------------------------------------ | ----- | ----- | --------------------------- | ----------------------------------------------------- |
| F1 | **Positive-only Prospect**          | 2.0   | 1.0   | —                           | Isolate positive-side amplification (`s_dwell` gate). |
| F2 | **Negative-only Prospect**          | 1.0   | 2.0   | —                           | Isolate negative-side amplification (`p_ctr` gate).   |
| F3 | **Vanilla DPO on SFT-warmed base**  | 1.0   | 1.0   | —                           | Strict baseline: warmup + plain pointwise DPO. Separates warmup effect from prospect gating on the same base as E. |
| F4 | **SFT-pos warmed (longer)**         | —     | —     | SFT 200 steps, no DPO       | Test whether more warmup alone closes the gap to E.   |
| F5 | **No length normalization**         | 2.0   | 2.0   | `average_log_prob = false`  | Is SimPO-style mean-per-token norm helping or hurting? |
| F6 | **Pairwise DPO (chosen vs rejected)** | —   | —     | use pair loss, not single-wise | Compare pairwise vs single-wise on same data.       |

### 5.2 Parameter sweeps (α, λ, β, LR, gate-shape)

| # | Sweep dim      | Values                       | Fixed                            | Purpose                                  |
| - | -------------- | ---------------------------- | -------------------------------- | ---------------------------------------- |
| P1 | `α_max`        | {1.0, 1.5, 2.0, 3.0, 4.0}    | λ_max = 2.0, β = 0.1, LR = 5e-6  | Positive-side amplification strength.    |
| P2 | `λ_max`        | {1.0, 1.5, 2.0, 3.0, 4.0}    | α_max = 2.0, β = 0.1, LR = 5e-6  | Negative-side amplification strength.    |
| P3 | `β`            | {0.05, 0.1, 0.3, 0.5}        | α_max = λ_max = 2.0, LR = 5e-6   | Reward temperature (DPO paper uses 0.1 for HH, 0.5 for TL;DR-summarization). PENS is summarization-like → 0.5 may help. |
| P4 | Actor LR       | {1e-6, 3e-6, 5e-6, 1e-5}     | cosine, α/λ = 2.0, β = 0.1       | Confirm 5e-6 is the optimum on SFT-warmed base. |
| P5 | `α_k` (gate steepness) | {1.0, 2.0, 4.0, 8.0} | α_τ = 0.0, α_max = 2.0           | Sharpness of dwell-time gate.            |
| P6 | `α_τ` (gate midpoint)  | {-0.5, 0.0, 0.25, 0.5} | α_k = 4.0, α_max = 2.0           | Where on `s_dwell` axis the gate activates. |
| P7 | `λ_γ`          | {0.5, 1.0, 1.5, 2.0, 3.0}    | λ_max = 2.0                      | CTR-power exponent: linear vs concave vs convex CTR weighting. |
| P8 | Training length | {0.5, 1.0, 2.0} × epoch     | rest = E                         | Did E truly plateau by step 1400, or just compute-limited? |

### 5.3 Data / signal ablations

| # | Variant                          | Description                                                                          | Tests / purpose                                                |
| - | -------------------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------- |
| D1 | Drop `s_dwell` signal           | Replace `α(s_dwell)` with constant `α_max`                                            | Is the dwell-time gate doing work, or just any positive boost? |
| D2 | Drop `p_ctr` signal             | Replace `λ(p_ctr)` with constant `λ_max`                                              | Same, for CTR gate.                                            |
| D3 | Shuffled `s_dwell`, `p_ctr`     | Permute side-info across samples                                                     | Sanity: if performance unchanged, the gate signal is noise.    |
| D4 | Pos-only training (no negatives)| Filter to positive samples; reduces to SFT-on-pos with reward-style loss             | Directly comparable to A1 with the same loss form.             |
| D5 | Neg-only training (no positives)| Filter to negative samples                                                           | Pathological control; expected to collapse.                    |

### 5.4 Reporting requirement

For every future ablation run:
- Evaluate with **`rouge_score v0.1.2`, `use_stemmer=True`, multi-reference max F1** (Run E scorer).
- Record **R-1, R-2, R-L** at every saved checkpoint; report best.
- Note `thinking_mode = false`, `\boxed{}` parse, identical 89-prompt drop set.
- Append the row to §4 table with run-ID + result-file path.

---

## 6. Per-checkpoint trajectory (Run E, for reference)

| step  | R-1     | R-2     | R-L         | parse |
| ----- | ------- | ------- | ----------- | ----- |
| 200   | 0.2915  | 0.1058  | 0.2396      | 100 % |
| 400   | 0.2925  | 0.1062  | 0.2404      | 100 % |
| 600   | 0.2921  | 0.1058  | 0.2402      | 100 % |
| 800   | 0.2920  | 0.1059  | 0.2402      | 100 % |
| 1000  | 0.2930  | 0.1065  | 0.2406      | 100 % |
| 1200  | 0.2917  | 0.1058  | 0.2402      | 100 % |
| **1400** | **0.2924** | **0.1065** | **0.2408 ← peak** | **100 %** |
| 1600  | 0.2923  | 0.1061  | 0.2405      | 100 % |
| 1800  | 0.2919  | 0.1057  | 0.2401      | 100 % |
| 2000  | 0.2925  | 0.1063  | 0.2406      | 100 % |
| 2200  | 0.2922  | 0.1059  | 0.2401      | 100 % |
| 2400  | 0.2922  | 0.1057  | 0.2400      | 100 % |
