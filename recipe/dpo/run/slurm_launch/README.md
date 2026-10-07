# PeNS Slurm jobs

`KTO.sbatch` runs KTO on the same sampled positive and negative click-history
parquets, without constructing preference pairs. It requests six hours and
four H100 80GB GPUs on `gpubase_bynode_b2` (the existing `b1` partition allows
only three hours). `../KTO.sh` delegates to the common all-data recipe: model,
LR `5e-6`, beta `0.1`, batch 128 / micro batch 4, LoRA rank 64 / alpha 128,
seed 42, 500 updates, and evaluation every 50 updates remain the same.
Experiment/checkpoint/generation names start with `kto_click_hist_all_without_weight_`.
Training and evaluation log W&B offline through the shared job runtime.

KTO uses a fixed reference policy and a detached nonnegative KL estimate from
responses rolled across the complete update batch. Reference scores for these
mismatched sequences are computed live; with LoRA, the base model with its
adapter disabled is the reference. All ranks/micro-batches share one KL
baseline per update. Desirable/undesirable loss multipliers default to 1 and
can be set with `KTO_DESIRABLE_WEIGHT` / `KTO_UNDESIRABLE_WEIGHT`; dwell/CTR
feedback weights are not used. As requested for comparison, the launcher
inherits `algorithm.average_log_prob=true`, including for the KL estimate.
This is a length-normalized KTO variant; set
`SINGLE_WISE_DPO_AVERAGE_LOG_PROB=false` for the paper's summed sequence scores.
The loss and KL estimator follow [KTO](https://arxiv.org/abs/2402.01306) and
the [TRL implementation](https://huggingface.co/docs/trl/kto_trainer).

```bash
PENS_SLURM_DRY_RUN=true bash recipe/dpo/run/slurm_launch/KTO.sbatch
sbatch recipe/dpo/run/slurm_launch/KTO.sbatch
```

`run_pens_sft_from0_500.sbatch` starts a new positive-only SFT run from the
Qwen3-4B-Instruct base model, with 4 H100 80GB GPUs and a two-hour time limit.
It uses LR `5e-6`, batch 128, micro batch 4, LoRA rank 64 / alpha 128, and seed 42.
It trains to step 500 and evaluates `eval_final.parquet` at steps 100, 200, 300,
400, and 500. The 100-step interval limits repeated full-dataset evaluation
overhead within the requested allocation; wall time is a limit, not a guarantee.
Each job has separate model, generation, metrics, and W&B outputs. Later
segments resume this job's checkpoints; the initial segment starts at zero.
After each successful evaluation, SFT keeps only the current checkpoint and
deletes earlier checkpoints. It does not retain a separate best model by
default (`SFT_KEEP_BEST_CHECKPOINT=false`). The training limit of two
checkpoints allows a previous checkpoint to remain until evaluation succeeds;
if evaluation fails, the driver stops before pruning.

```bash
PENS_SLURM_DRY_RUN=true bash recipe/dpo/run/slurm_launch/run_pens_sft_from0_500.sbatch
sbatch recipe/dpo/run/slurm_launch/run_pens_sft_from0_500.sbatch
```

Training metrics and the three evaluation metrics (`ROUGE-1`, `ROUGE-2`,
`ROUGE-L`, on the 0–100 scale) are saved as offline W&B logs under
`wandb/<experiment_name>/wandb/offline-run-*`. All Slurm launchers force
`WANDB_MODE=offline`, including when the submitter or `.env` sets online mode.
Upload the completed run from the login node after training ends; compute
nodes do not upload to W&B.

`run_single_wise_click_hist_negative_only.sbatch` uses the same Slurm
resources as `run_single_wise_click_hist_all.sbatch`: four H100 80GB GPUs,
32 CPUs, 256G RAM, and three hours. Its wrapper delegates to the all launcher,
changing only the training split to `only_negative_click_hist_train.parquet`
and using a separate experiment/output namespace. It preserves Slurm's GPU
mask. Model, loss, LR, kappa weights, batch sizes, training steps, checkpoint
and evaluation settings are inherited from all. Current defaults are LR
`5e-6`, beta `0.1`, kappa+ `3.5`, kappa- `1.5`, batch 128 / micro batch 4,
500 training steps, and evaluation every 50 steps on `eval_final.parquet`.
Training and evaluation use W&B offline through the same shared job runtime.

```bash
PENS_SLURM_DRY_RUN=true bash recipe/dpo/run/slurm_launch/run_single_wise_click_hist_negative_only.sbatch
sbatch recipe/dpo/run/slurm_launch/run_single_wise_click_hist_negative_only.sbatch
```

The complete resolved all/negative-only parameter comparison and dataset
label audit are recorded in `metrics/negative_only_vs_all_config_audit.json`.

The three `*_no_weight.sbatch` jobs are strict feedback-weight ablations:
`all_no_weight`, `negative_only_no_weight`, and `positive_only_no_weight`.
They keep the default `prospect_dpo` sigmoid objective and the same positive
and negative class reductions, setting
`algorithm.prospect_dpo_use_feedback_weights=false` so `alpha=lambda=1`.
The feedback signals and kappa values no longer affect the loss. These jobs
use the same model, optimizer, batch sizes, training/evaluation schedule,
and data processing as the default weighted all run. The positive/negative
versions select their corresponding training split. Each job requests three
hours and four H100 80GB GPUs and logs W&B offline.

```bash
sbatch recipe/dpo/run/slurm_launch/run_single_wise_click_hist_all_no_weight.sbatch
sbatch recipe/dpo/run/slurm_launch/run_single_wise_click_hist_negative_only_no_weight.sbatch
sbatch recipe/dpo/run/slurm_launch/run_single_wise_click_hist_positive_only_no_weight.sbatch
```

The older BCE `single_wise_dpo` implementation remains a separate supported
loss; these three launchers now use the strict Prospect ablation requested
here. `metrics/no_weight_config_audit.json` records the fully resolved
configuration comparison for all three jobs against weighted all.

Launcher filenames and Slurm job names use `single_wise_click_hist_*`.
The `single_wise_click_hist_*` entrypoints default to `POINTWISE_DPO_LOSS_TYPE=prospect_dpo`;
the three `*_no_weight` entrypoints force that loss and disable feedback weights.
New experiment/output names use `prospect_click_hist_<split>_<weight>_<timestamp>`.
Existing configuration keys and environment variables keep their API names.
