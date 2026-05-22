# TODO Experiments

Current best: **Run E**, Prospect-DPO on SFT-positive-warmed Qwen3-4B-Instruct, best at step 1400.

All new results should use the same evaluation setup:

- PENS personalized headline test set
- `rouge_score v0.1.2`
- Porter stemming
- multi-reference max F1
- thinking mode off

## 1. Required Baselines

These are our internal baselines. External paper baselines can be reported separately in the final comparison table.

| ID | Done | Experiment | Purpose | Evidence |
| --- | --- | --- | --- | --- |
| B0 | ✅ | Base model, no finetuning | Lower bound for Qwen3-4B-Instruct | `results/baselines_qwen3_4b.json` |
| B1 |  | SFT positive-only | Check how much gain comes from learning good clicked headlines only | - |
| B2 |  | SFT negative-only | Pathological/control baseline: train only on skipped/non-clicked headlines | - |
| B3 | ✅ | Vanilla DPO | Remove Prospect-DPO weighting; compare plain preference-style objective | `results/ablation.json`, run B |
| B4 | ✅ | Prospect-DPO, full method | Main method, current Run E setting | `results/ablation.json`, run E |
| B5 |  | GPT-5.5 frontier model | External frontier-model baseline with the same prompt and evaluator | - |

## 2. Method Ablations

Use the same SFT-positive-warmed base as Run E unless the ablation explicitly changes it.

| ID | Done | Experiment | Change | Evidence |
| --- | --- | --- | --- | --- |
| A1 |  | Positive weight only | `alpha_max > 1`, `lambda_max = 1` | - |
| A2 |  | Negative weight only | `alpha_max = 1`, `lambda_max > 1` | - |
| A3 | ✅ | No asymmetric weighting | `alpha_max = 1`, `lambda_max = 1` | `results/ablation.json`, run B |
| A4 | ✅ | Full asymmetric weighting | `alpha_max = 2`, `lambda_max = 2` | `results/ablation.json`, runs C/E |
| A5 |  | No length normalization | `average_log_prob = false` | - |
| A6 | ✅ | With length normalization | `average_log_prob = true` | `results/ablation.json`, runs B/C/D/E |

## 3. Thinking Mode Ablation

Run the same model/checkpoint with identical prompt and evaluator, changing only thinking mode.

| ID | Done | Experiment | Change | Evidence |
| --- | --- | --- | --- | --- |
| T1 | ✅ | Base model thinking off | `thinking_mode = false` | `results/baselines_qwen3_4b.json` |
| T2 |  | Base model thinking on | `thinking_mode = true` | - |
| T3 |  | Run E thinking off | `thinking_mode = false` | - |
| T4 |  | Run E thinking on | `thinking_mode = true` | - |

## 4. Parameter Sweeps

Keep checkpoint/eval cadence fixed so curves are comparable.

| Parameter | Values |
| --- | --- |
| `alpha_max` | `1.0`, `1.5`, `2.0`, `2.5` |
| `lambda_max` | `1.0`, `1.5`, `2.0`, `2.5` |
| `beta` | `0.05`, `0.1`, `0.3`, `0.5` |
| `average_log_prob` | `true`, `false` |

## 5. Reporting

For each run, record:

- best checkpoint by ROUGE-L
- ROUGE-1 / ROUGE-2 / ROUGE-L
- `count`, `joined_count`, `skipped_count`
- parse status counts
- key hyperparameters: base, loss, `beta`, `alpha_max`, `lambda_max`, `actor_lr`, length normalization

Do not compare runs with poor parse coverage directly against full-coverage runs.
