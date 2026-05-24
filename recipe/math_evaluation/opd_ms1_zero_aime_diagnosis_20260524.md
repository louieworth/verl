# OPD ms1 Zero-AIME Diagnosis

Date: 2026-05-24

## Scope

Target models:

- `ms1_y_o_no_lora_bad`:
  `/scratch/l/luli/jiangli/ckpt/OPD/Qwen3-1.7B/teacherQwen3-8B_y_o_kl_forward_full_vocab_clip0_vanilla_ms1_20260523-181523/epoch1/ms1/batch00001/hf_merged`
- `ms40_y_r_no_lora_final`:
  `/scratch/l/luli/jiangli/ckpt/OPD/Qwen3-1.7B/teacherQwen3-8B_y_r_kl_forward_full_vocab_clip0_refine_ms40_20260523-175757/final/hf_merged`
- Reference result:
  `/scratch/l/luli/jiangli/ckpt/OPD/Qwen3-1.7B/teacherQwen3-8B_y_o_kl_forward_full_vocab_clip0_vanilla_ms40_20260520-234705/epoch1/ms40/batch00040/hf_merged`

The reference result in `results/OPD/Qwen3-1.7B.json` is a LoRA run, not a
no-LoRA full-parameter baseline.

## Evidence Summary

| item | ms1 y_o no-LoRA | ms40 y_r no-LoRA final | ms40 y_o reference |
| --- | --- | --- | --- |
| prompt/y mode | y_o / y_o | y_r / y_r | y_o / y_o |
| use_lora | false | false | true |
| multi_step | 1 | 40 | 40 |
| chunk samples | 40245 | 1006 | 1006 |
| train_batch_size | 8 | 2 | 4 |
| gradient_accumulation_steps | 8 | 63 | 32 |
| gradient accumulation source | user | auto_one_update_per_chunk | auto_one_update_per_chunk |
| final saved step | global_step_79 | global_step_1 | global_step_1 |
| learning rate | 5e-6 | 2e-6 | 5e-6 |
| result availability | local full pass16 result exists | local full pass16 result exists | local result exists |

Relevant logs/configs:

- ms1 config:
  `/scratch/l/luli/src/verl/outputs/OPD/Qwen3-1.7B/teacherQwen3-8B_y_o_kl_forward_full_vocab_clip0_vanilla_ms1_20260523-181523/epoch1/ms1/batch00001/training_config.yaml`
- ms1 training log:
  `/scratch/l/luli/src/verl/outputs/OPD/Qwen3-1.7B/teacherQwen3-8B_y_o_kl_forward_full_vocab_clip0_vanilla_ms1_20260523-181523/epoch1/ms1/batch00001/logs/training_20260523_204005.log`
- y_r final config:
  `/scratch/l/luli/src/verl/outputs/OPD/Qwen3-1.7B/teacherQwen3-8B_y_r_kl_forward_full_vocab_clip0_refine_ms40_20260523-175757/epoch1/ms40/batch00040/training_config.yaml`
- y_r final training log:
  `/scratch/l/luli/src/verl/outputs/OPD/Qwen3-1.7B/teacherQwen3-8B_y_r_kl_forward_full_vocab_clip0_refine_ms40_20260523-175757/epoch1/ms40/batch00040/logs/training_20260524_141909.log`
- y_o reference config:
  `/scratch/l/luli/src/verl/outputs/OPD/Qwen3-1.7B/teacherQwen3-8B_y_o_kl_forward_full_vocab_clip0_vanilla_ms40_20260520-234705/epoch1/ms40/batch00040/training_config.yaml`

## Existing Diagnostic Runs

Diagnostics were written under:

`/scratch/l/luli/jiangli/eval_diag/ms1_zero_aime24/job303379-20260524-160010`

Observed:

- tokenizer/config/generation config are aligned across bad/ref/base models.
- sampled weight tensors are finite; no NaN/Inf was found in sampled tensors.
- HF smoke generation does not show an immediate EOS or empty-output failure.
- `simple_2_plus_2` is solved correctly by bad, y_r final, and base.
- On AIME smoke with `max_new_tokens=2048`, all three models often reason
  correctly but do not reach a final boxed answer before truncation.

This rules out a simple broken merge, tokenizer mismatch, or immediate decoding
collapse.

## Single AIME24 Long-Output Check

A minimal long-output check was run for the ms1 y_o no-LoRA checkpoint:

`/scratch/l/luli/jiangli/eval_diag/single_aime_check/ms1_y_o_bad/aime24_rows0_1_max8192_n4.json`

Settings:

- dataset: `/scratch/l/luli/jiangli/datasets/eval/aime24/aime24_test.parquet`
- rows: `0,1`
- samples per row: `4`
- `max_new_tokens`: `8192`

Results:

| model/check | row | ground truth | scores | generated tokens | boxed |
| --- | --- | --- | --- | --- | --- |
| ms1 y_o no-LoRA, AIME24 | 0 | 204 | `[1.0, 1.0, 1.0, 1.0]` | `[7605, 7605, 7605, 7605]` | all true |
| ms1 y_o no-LoRA, AIME24 | 1 | 113 | `[0.0, 0.0, 0.0, 0.0]` | `[8192, 8192, 8192, 8192]` | all false |
| ms1 y_o no-LoRA, AIME25 | 0 | 70 | `[1.0, 1.0, 1.0, 1.0]` | `[7728, 7728, 7728, 7728]` | all true |
| ms1 y_o no-LoRA, AIME25 | 1 | 588 | `[0.0, 0.0, 0.0, 0.0]` | `[8192, 8192, 8192, 8192]` | all false |
| ms40 y_r no-LoRA final, AIME24 | 0 | 204 | `[1.0, 1.0, 1.0, 1.0]` | `[5867, 5867, 5867, 5867]` | all true |
| ms40 y_r no-LoRA final, AIME24 | 1 | 113 | `[0.0, 0.0, 0.0, 0.0]` | `[8192, 8192, 8192, 8192]` | all false |

Additional result files:

- `/scratch/l/luli/jiangli/eval_diag/single_aime_check/ms1_y_o_bad/aime25_rows0_1_max8192_n4.json`
- `/scratch/l/luli/jiangli/eval_diag/single_aime_check/ms40_y_r_final/aime24_rows0_1_max8192_n4.json`

This proves the ms1 checkpoint can answer at least some AIME24 and AIME25
questions and be accepted by the local scorer when enough output budget is
available. It also shows the failure mode on harder samples: long reasoning
reaches the output limit without a boxed final answer, so the scorer returns
zero. The y_r ms40 no-LoRA final checkpoint shows the same qualitative behavior
on this two-row AIME24 probe: row 0 is correct and boxed, row 1 hits the token
limit without boxing. That is not evidence of total collapse, but it is evidence
of a shared long-output/format failure mode on some problems.

## Full vLLM Pass16 Evaluation

The requested full AIME24/AIME25 pass16 evaluation was run inside Slurm job
`303379` with `recipe/math_evaluation/benchmark_kl_model.sh`.

Node assignment:

- `tg10603`: `ms40_y_r_no_lora_final`
- `tg11103`: `ms1_y_o_no_lora_bad`

Generation settings matched the benchmark path:

- `temperature=0.6`
- `top_p=0.95`
- `top_k=-1`
- `max_tokens=38912`
- `pass_k=16`
- `n_gpus_per_node=4`

Generated files and per-run result files are under:

`eval_diag/opd_pass16_job303379/`

The final metrics were also merged into:

`results/OPD/Qwen3-1.7B.json`

Results:

| model/check | AIME24 pass@16 | AIME24 avg@16 | AIME25 pass@16 | AIME25 avg@16 |
| --- | ---: | ---: | ---: | ---: |
| ms1 y_o no-LoRA | 0.7666666666666667 | 0.5270833333333333 | 0.7 | 0.4125 |
| ms40 y_r no-LoRA final | 0.7666666666666667 | 0.38958333333333334 | 0.6 | 0.3 |
| ms40 y_o reference LoRA | 0.8 | 0.4625 | 0.6333333333333333 | 0.36875 |

This directly contradicts the external all-zero result for this exact ms1
checkpoint under the local benchmark path. The checkpoint is not globally
collapsed and is not incapable of AIME. The earlier all-zero result is now best
treated as an evaluation-environment/configuration mismatch or a failed/partial
evaluation artifact, not as model-quality evidence.

The ms1 run still showed a real long-output tail: AIME25 generation took about
56 minutes for 480 responses and frequently had only one or two active workers
near the end. That is a model behavior issue, but it did not reduce local
pass@16/avg@16 to zero.

## Main Diagnosis

The ms1 run is not a single optimizer update. It is one OPD pipeline update, but
the trainer loaded 40245 samples and saved `global_step_79`.

The practical effect is:

- full-parameter training,
- no LoRA adapter constraint,
- a large y_o batch,
- user-specified `gradient_accumulation_steps=8`,
- 79 optimizer steps from the base model in the single ms1 batch.

The ms40 runs are structurally different:

- each chunk has about 1006 samples,
- gradient accumulation is auto-set for one update per chunk,
- each batch saves at `global_step_1`.

Therefore, the strongest current explanation for the reported external
AIME24/AIME25 zero is not "the HF checkpoint is corrupt" and not "the model
cannot solve AIME". The local benchmark result for the same checkpoint is
non-zero and strong. The ms1 training setup is still risky: no-LoRA
full-parameter training plus a giant y_o chunk produced 79 optimizer updates in
one ms1 batch, and the resulting model has a visible long-output tail. But the
all-zero external result should now be treated primarily as an eval
environment/configuration/result-write issue until proven otherwise.

## Is y_r ms40 no-LoRA Also Collapsed?

Current evidence rules out simple collapse for the y_r final model.

Evidence against a simple collapse:

- checkpoint exists and loads;
- tokenizer/config/generation config are normal;
- sampled weights are finite;
- smoke generation is coherent;
- simple arithmetic is answered correctly;
- each ms40 batch uses one optimizer step, not 79 steps in one batch;
- learning rate is lower than ms1 (`2e-6` vs `5e-6`);
- full local pass16 result is strong: AIME24 pass@16 `0.7667`, AIME25 pass@16
  `0.6`.

## Reference Caveat

The good ms40 y_o result in `results/OPD/Qwen3-1.7B.json` is:

`Qwen3-1.7B_OPD_teacherQwen3-8B_y_o_kl_forward_full_vocab_clip0_vanilla_ms40_20260520-234705_step00040of00040_LORA`

Relevant AIME metrics:

- `aime24_pass16_generation_pass_16`: `0.8`
- `aime24_avg_pass1_generation_pass_16`: `0.4625`
- `aime25_pass16_generation_pass_16`: `0.6333333333333333`
- `aime25_avg_pass1_generation_pass_16`: `0.36875`

Because this is LoRA, it should not be treated as evidence that full-parameter
no-LoRA ms40 has the same quality.

## Recommended Next Checks

1. Compare the external all-zero eval environment against the local run:
   model path, tokenizer path, `top_k`, `max_tokens`, dataset parquet, scorer,
   output JSON write path, and whether generation parquet was complete.
2. Inspect ms1 generated parquet response lengths and boxed-answer rate to
   quantify the long-output tail.
3. For future ms1 no-LoRA runs, avoid user-fixed accumulation that allows many
   optimizer steps on one giant chunk. Use the `auto_one_update_per_chunk`
   behavior or explicitly cap trainer steps for ms1.
