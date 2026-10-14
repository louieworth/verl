# PeNS Slurm jobs

Weighted PIFO launchers are `all.sbatch`, `positive_only.sbatch`, and
`negative_only.sbatch`. Their training entrypoints are `../all.sh`,
`../positive_only.sh`, and `../negative_only.sh`. The unweighted submission
files, including resume jobs, live in `ablation/` and launch the corresponding
`../ablation/*_no_weight.sh` training scripts through the shared `run_job.sh`.
The weighted resume submission files stay in this directory.

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

`negative_only.sbatch` uses the same Slurm
resources as `all.sbatch`: four H100 80GB GPUs,
32 CPUs, 256G RAM, and three hours. Its wrapper delegates to the all launcher,
changing only the training split to `only_negative_click_hist_train.parquet`
and using a separate experiment/output namespace. It preserves Slurm's GPU
mask. Model, loss, LR, kappa weights, batch sizes, training steps, checkpoint
and evaluation settings are inherited from all. Current defaults are LR
`5e-6`, beta `0.1`, kappa+ `3.5`, kappa- `1.5`, batch 128 / micro batch 4,
500 training steps, and evaluation every 50 steps on `eval_final.parquet`.
Training and evaluation use W&B offline through the same shared job runtime.

```bash
PENS_SLURM_DRY_RUN=true bash recipe/dpo/run/slurm_launch/negative_only.sbatch
sbatch recipe/dpo/run/slurm_launch/negative_only.sbatch
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
sbatch recipe/dpo/run/slurm_launch/ablation/all_no_weight.sbatch
sbatch recipe/dpo/run/slurm_launch/ablation/negative_only_no_weight.sbatch
sbatch recipe/dpo/run/slurm_launch/ablation/positive_only_no_weight.sbatch
```

The five `ablation/all_kplus3.5_kminus<N>.sbatch` jobs sweep feedback weights
on the all-data split: kappa+ is fixed at 3.5 and kappa- is 0.5, 2.5, 3.5, 4.5, or 5.5.
The existing main kappa-=1.5 run is reused rather than submitted again.
Their training entrypoints are `../ablation/all_kplus3.5_kminus<N>.sh`.
They enable feedback weights and inherit the main all recipe's model,
optimizer, and batching. They train to step 600 and evaluate every 50 steps,
for 12 evaluation boundaries. Each submission requests four H100 80GB GPUs,
32 CPUs, 256G RAM, and four hours on `gpubase_bynode_b2`, and logs W&B offline
through `run_job.sh`.
The four-hour limit is based on completed Slurm records: all-data no-weight
job 503779 ran 500 steps with evaluation in 2:33:22, and its 500-to-600 resume
job 512344 took 0:33:39. Their combined elapsed time was 3:07:01.
Weighted all-data job 511962 took 1:30:04 to resume from step 500 to 800.
The `b1` partition's maximum is three hours; `b2` allows up to 12 hours.
Default experiment names are
`prospect_click_hist_all_with_weight_kplus3.5_kminus<N>_<timestamp>` so each
weight setting has separate checkpoint, generation, and W&B directories.
Explicit experiment names and timestamps can still be supplied for resumes.

```bash
for kminus in 0.5 2.5 3.5 4.5 5.5; do
  sbatch "recipe/dpo/run/slurm_launch/ablation/all_kplus3.5_kminus${kminus}.sbatch"
done
```

The `ablation/all_kplus3.5_kminus{3.5,4.5,5.5}_resume_600.sbatch` launchers
continue interrupted jobs 513195, 513196, and 513197 in their original
`20261008_154851` experiments. They restore complete training state from
steps 550, 300, and 200, respectively, including optimizer and dataloader
state. The interleaved driver retries the missing evaluations at 550 and
200 before continuing training; kappa-=4.5 repeats unsaved updates 301--347.
They retain the original W&B identities and train to 600 with evaluation
every 50 steps. Evaluation uses eager mode to avoid CUDA graph initialization
and compiler-cache failures; sampling and scoring settings are unchanged.
All three exclude nodes `tg10803,tg11104`. The kappa-=3.5 continuation requests
two hours; the other two request four hours, with four H100 80GB GPUs each.

```bash
for kminus in 3.5 4.5 5.5; do
  sbatch "recipe/dpo/run/slurm_launch/ablation/all_kplus3.5_kminus${kminus}_resume_600.sbatch"
done
```

The older BCE `single_wise_dpo` implementation remains a separate supported
loss; these three launchers now use the strict Prospect ablation requested
here. `metrics/no_weight_config_audit.json` records the fully resolved
configuration comparison for all three jobs against weighted all.

Launcher filenames use the split names `all`, `positive_only`, and `negative_only`,
with `_no_weight` added for unweighted ablations and `_kplus3.5_kminus<N>` for
the all-data weight sweep. PIFO Slurm job names use the `pifo_` prefix.
The weighted entrypoints default to `POINTWISE_DPO_LOSS_TYPE=prospect_dpo`;
the three `ablation/*_no_weight` entrypoints force that loss and disable feedback weights.
Main experiment/output names use `prospect_click_hist_<split>_<weight>_<timestamp>`.
Existing configuration keys and environment variables keep their API names.
