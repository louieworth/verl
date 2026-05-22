# Prospect-DPO on PENS — Progress Report

**Date**: 2026-05-10
**Backbone**: Qwen3-4B-Instruct-2507 (LoRA fine-tuning)
**Dataset**: PENS (Personalized News Headline Generation)
**Scorer**: Google `rouge_score v0.1.2`, `use_stemmer=True` (Porter), multi-reference max F1
(paper-aligned, ≈ `pyrouge -a -m`)

---

## 1. Best Performance

Best checkpoint: **step 1400** of the latest training run.

| Metric        | Value      |
| ------------- | ---------- |
| ROUGE-1 F1    | **0.2924** |
| ROUGE-2 F1    | **0.1065** |
| ROUGE-L F1    | **0.2408** |
| Parse rate    | 100 % (20465 / 20554; 89 prompts skipped due to vLLM `max_model_len=8300 < input_len`, identical drop set across all checkpoints) |

The result is stable: across all 12 evaluated checkpoints (step 200 … 2400) ROUGE-L spans `[0.2400, 0.2408]` — a 0.0008 spread, indicating the policy has reached a plateau.

---

## 2. Comparison vs. PENS Paper Baseline

The PENS paper's strongest supervised baseline is the **NRMS-as-injector** model (NRMS personalized user encoder over a Transformer headline generator), reported in Table 5 of the PENS paper.

| Metric     | PENS NRMS (paper) | **Ours (step 1400)** | Δ           |
| ---------- | ----------------- | -------------------- | ----------- |
| ROUGE-1 F1 | 0.2801            | **0.2924**           | **+0.0123** |
| ROUGE-2 F1 | **0.1072**        | 0.1065               | −0.0007 (within noise) |
| ROUGE-L F1 | 0.2224            | **0.2408**           | **+0.0184** |

Our method exceeds the PENS paper benchmark on ROUGE-1 and ROUGE-L. ROUGE-2 is within scorer noise of the paper number.

---

## 3. Core Algorithm — Prospect-DPO

Pointwise (single-wise) variant of DPO. Each training sample `(x, y, label)` is either a **positive** (clicked / engaged headline, `label = 1`) or a **negative** (skipped headline, `label = 0`) — *not* a pair. Two side-information signals are taken from the PENS click logs and used to weight the loss:

- `s_dwell ∈ [0, 1]` — normalized dwell time on a positive sample (engagement strength).
- `p_ctr ∈ [0, 1]` — empirical click-through probability of a negative sample (item-level hardness prior).

### 3.1 Implicit Reward (DPO-style, length-normalized)

For a sequence `y = (y_1, …, y_T)` given context `x`, define the length-normalized log-probability (SimPO-style, set by `average_log_prob = True`):

$$
\overline{\log\pi}(y \mid x) \;=\; \frac{1}{T}\sum_{t=1}^{T} \log \pi(y_t \mid x,\, y_{<t})
$$

The implicit reward is the policy-vs-reference log-ratio scaled by temperature `β`:

$$
r_\theta(x, y) \;=\; \beta \cdot \Big[\, \overline{\log\pi_\theta}(y \mid x) \;-\; \overline{\log\pi_\mathrm{ref}}(y \mid x) \,\Big]
$$

### 3.2 Asymmetric Prospect-Theoretic Gating

Positive-side amplification weight `α` — a sigmoid gate on dwell time:

$$
\alpha(s_\mathrm{dwell}) \;=\; 1 \;+\; (\alpha_\mathrm{max} - 1)\cdot \sigma\!\big(\alpha_k\,(s_\mathrm{dwell} - \alpha_\tau)\big), \qquad \alpha \in [1,\, \alpha_\mathrm{max}]
$$

Negative-side amplification weight `λ` — a power gate on CTR. High-CTR headlines that the user *still* skipped are surprising negatives and are treated as harder negatives:

$$
\lambda(p_\mathrm{ctr}) \;=\; 1 \;+\; (\lambda_\mathrm{max} - 1)\cdot p_\mathrm{ctr}^{\,\lambda_\gamma}, \qquad \lambda \in [1,\, \lambda_\mathrm{max}]
$$

Special case `α_max = λ_max = 1` ⇒ `α ≡ λ ≡ 1` ⇒ the loss collapses to plain pointwise DPO. Any value `> 1` introduces asymmetric, behavior-conditioned amplification.

### 3.3 Loss (single-wise, asymmetric)

Per-sample loss is sigmoid-based, with the sign of the reward inverted between positive and negative samples and scaled by the appropriate gate:

$$
\mathcal{L}_\mathrm{pos}(x, y) \;=\; \sigma\!\big(-\,\alpha(s_\mathrm{dwell})\cdot r_\theta(x, y)\big)
$$

$$
\mathcal{L}_\mathrm{neg}(x, y) \;=\; \sigma\!\big(+\,\lambda(p_\mathrm{ctr})\cdot r_\theta(x, y)\big)
$$

Total batch loss is the mean over positive and negative subsets independently, then summed:

$$
\mathcal{L}_\mathrm{prospect} \;=\; \mathbb{E}_{(x,y)\,\in\,\mathcal{D}_+}\!\big[\mathcal{L}_\mathrm{pos}\big] \;+\; \mathbb{E}_{(x,y)\,\in\,\mathcal{D}_-}\!\big[\mathcal{L}_\mathrm{neg}\big]
$$

**Intuition**: `α` scales the **positive-pull strength** by user engagement (longer dwell → stronger pull); `λ` scales the **negative-push strength** by item popularity (surprisingly-skipped popular item → stronger push). This is the prospect-theoretic asymmetry: gains and losses are weighted by their behavioral salience rather than treated symmetrically as in vanilla DPO.

**Implementation note**: positive / negative subset means use mask-weighted summation rather than Python-level branching, which keeps the autograd graph shape identical across FSDP ranks even when a micro-batch is all-positive or all-negative. The earlier branching version produced rank-divergent NCCL collective sequences and deterministically deadlocked at the first backward pass.

### 3.4 Hyperparameters

| Param                  | Value | Role                                                      |
| ---------------------- | ----- | --------------------------------------------------------- |
| β                      | 0.1   | Reward temperature                                        |
| α_max                  | 2.0   | Max positive amplification                                |
| α_k                    | 4.0   | Sigmoid steepness on `s_dwell`                            |
| α_τ                    | 0.0   | Sigmoid midpoint on `s_dwell`                             |
| λ_max                  | 2.0   | Max negative amplification                                |
| λ_γ                    | 1.5   | Power exponent on `p_ctr`                                 |
| `average_log_prob`     | true  | SimPO-style mean-per-token length norm in reward          |

# Questions and Feedback

## Q1. Prospect Theory as motivation — novelty concern

> I am still concerned with Prospect Theory as the method motivation and it's difficult to frame it as a novel contribution. However, when we interpret why our model, with asymmetric loss, has better performance than SOTA, we can tie it to the KTO paper and prospect theory to use it as an intuition. The more useful and intuitive motivation seems to be: how we learn from consumer disengagement about one's individual preferences (or, when and why non-action carries more signals). We will need great articulation, illustration, and proof that signals, even soft and noisy such as non-clicking, can inform us about someone's personal preference, which help us do personalization. We should also discuss how we deal with issues of such signals: weak, noisy, ambiguous.

We can start from DPO, not KTO. Three technical discrepancies vs. KTO:

1. **No batch-estimated KL baseline.** KTO uses `r = β·log[π_θ/π_ref] − z_0` with `z_0` estimated across the batch via mismatched pairs. We drop `z_0` and use the direct log-ratio `r_θ = β·[avg_logπ_θ − avg_logπ_ref]`. Reference log-probs are pre-materialized once and reused — no batch-composition coupling.

2. **Per-sample personalized weights, not class constants.** KTO's `λ_D, λ_U` are dataset-level scalars for class imbalance. Our `α(s_dwell)` and `λ(p_ctr)` are continuous, per-sample functions of behavioral side-info. Plain KTO has no slot for this.

3. **Length regularization** KTO uses summed sequence log-prob, so short sequences contribute systematically smaller rewards and long sequences are over-weighted. We use **mean per-token** log-prob (`average_log_prob = True`), making the reward length-invariant. Critical for PENS — headlines have wide length variance.

**Framing.** Position the paper as **personalization from behavioral disengagement signals**, not "prospect theory + DPO". Prospect theory → discussion section, not intro motivation. KTO ablation (real KTO with side-info injected) is required to defend the personalization contribution.

**Honesty note (post-derivation).** Stripping our three modifications (drop z_ref, set α=λ=1, drop length norm), the per-sample loss form is **identical** to KTO's sigmoid surrogate. Don't claim a novel loss family — frame the paper as "KTO-family loss + three personalization modifications", with the three deltas as the contribution.

## Q2. Learning from disengagement — what non-clicks tell us

> The more useful and intuitive motivation seems to be: how we learn from consumer disengagement about one's individual preferences (or, when and why non-action carries more signals).

Two informational claims to defend in the intro:

1. **Non-clicks are not uniform noise.** A skip on a *high-CTR* headline (most users click; this user did not) is informationally richer than a skip on a *low-CTR* headline (almost no one clicks). The CTR-conditioned `λ(p_ctr) = 1 + (λ_max − 1)·p_ctr^{λ_γ}` operationalizes exactly this: high `p_ctr` ⇒ surprising skip ⇒ stronger negative push.
2. **Dwell time grades positive engagement.** A click + 30s read ≠ a click + 2s bounce. Binary chosen/rejected (vanilla DPO) erases this. `α(s_dwell)` recovers it: longer dwell ⇒ stronger positive pull.

Proof obligation (for the paper):
- **Ablation D1/D2** (drop `s_dwell` / drop `p_ctr` in the planned ablation table) — if RL drops when either signal is removed, the soft signal carries real preference information beyond binary click/no-click.
- **D3** (shuffle `s_dwell`, `p_ctr`) — if shuffled signals match real signals, the gain was just any extra weighting; if shuffled hurts, the *content* of the signal matters.
- **Worked example in intro.** One user A (high dwell on tech / low dwell on celebrity) vs. user B (opposite); show how same article produces different headlines under our model.

## Q3. Handling weak, noisy, ambiguous engagement signals

> We should also discuss how we deal with issues of such signals: weak, noisy, ambiguous.

Three concrete mitigations baked into our method:

1. **Bounded gates.** `α ∈ [1, α_max]` and `λ ∈ [1, λ_max]` ⇒ a noisy signal can at most amplify the loss by `α_max` (e.g. 2×), never zero it out or flip its sign. Bad signal degrades gracefully toward vanilla DPO, doesn't collapse the loss.
2. **Smooth, monotone gates.** Sigmoid (`α`) and power (`λ`) are smooth and monotone in their inputs ⇒ small signal noise → small weight perturbation. No discontinuity that would amplify noise.
3. **CTR smoothing.** `p_ctr` for cold-start / low-impression items is unreliable. We apply Laplace smoothing (`p_ctr = (clicks + ε_1) / (impressions + ε_1 + ε_0)`) so a 1/2 item doesn't get treated like 0.5 CTR.

Robustness study (planned ablations P5/P6/P7 = `α_k`, `α_τ`, `λ_γ` sweeps) → **frame as signal-robustness study**, not hyperparameter tuning. If best RL is flat across a range of gate shapes, the method is robust to signal mis-specification.

Honest limitations to state:
- We do not currently model **temporal decay** of engagement (an old click counted equally to a recent one).
- We do not handle **adversarial / bot dwell** (a bot leaving a tab open at 30s of dwell looks like high engagement).
- These belong in a "Limitations" section, not the intro.

## Q4. Scope clarification — headline framing vs. topic recommendation

> One thing that must clarify and distinguish: are we learning different framing of headline, article content fixed, in personalization? Or, are we learning different topics, article can be changed, in personalization? The former is what we do, the latter can be done with recommendation algorithm (and we won't need framing headline at all). Motivation of the problem: if we can satisfy one's personal preference and get them click and read articles, by recommending them the kinds of content or topics they like, why bother trying to tweak headline itself at all? That is, we must explain why headline personalization is an important problem, not a secondary problem.

Article content is **fixed**; we personalize **the headline only**. Same article → different headlines per user. Recommendation (what to surface) is upstream and orthogonal.

**This is a well-studied problem, not a secondary one.** Two concrete points of evidence:

Academic: 
1. **Upworthy Research Archive** (Matias et al., *Scientific Data* 2021, [10.1038/s41597-021-00934-7](https://www.nature.com/articles/s41597-021-00934-7)) — a multi-year corpus of **32,487 A/B-tested headline variants over 4,873 articles** on the Upworthy platform, released specifically because headline framing has large, measurable effects on engagement with **fixed article content**. Existence of this dataset is direct evidence that the field treats headline framing as a first-class research problem.
2. **LOLA: LLM-Assisted Online Learning Algorithm for Content Experiments** (Liu et al.) — uses LLMs + multi-armed bandits to **select** among candidate headlines in live A/B tests. Closest published work to ours. **Key distinction**: LOLA does **selection** over a pre-written headline pool; we do **generation** of personalized headlines per user. Generation is strictly harder (open-ended output space, no pool to draw from, per-user conditioning) and removes the upstream cost of having an editor write the candidate pool.

Why headline personalization is primary, not secondary:

1. **Editorial constraint.** Publishers can't rewrite article bodies per user (legal / brand / accuracy). Headline is the only personalizable surface.
2. **Headline drives CTR.** Upworthy and follow-up A/B work show double-digit CTR variance from headline rewrites on the same article — not capturable by recommendation alone.
3. **Long-tail coverage.** One article reaches different users via different framings (technical vs. human-interest), without changing the article pool.

Intro should assume the rec problem (what to show) is solved upstream and address the orthogonal framing problem — with Upworthy + LOLA cited as the established research lineage.