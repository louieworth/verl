# Execute Step 2 (SFT warmup) + Step 3 (fixed prospect-DPO) — Qwen3-4B on Narval

## Context

`recipe/dpo/PROSPECT_DPO_IMPROVEMENT_PLAN.md` 的根因分析指出：**positive_only 胜出是因为它无意中成了 SFT；prospect_dpo 失败是因为 α∈(0,1) 压扁正梯度 + 没有 SFT warmup + logp 未归一化**。本次任务执行计划中的 **Step 2 (SFT warmup)** + **Step 3 (修复版 prospect-DPO)**，目标 R-1 ≥ 0.27（若 Step 3 结果好再考虑 Step 4 升 8B）。

**基础设施（已核实）**：
- Cluster: Narval（AlmaLinux 9.7），login node `narval2`
- 可用 GPU：A100 40GB，每 node 4 张，NVLink；141 个 bynode A100 节点
- Partitions：b2(12h)/b3(24h)/b4(72h)/b5(168h)，时间越长排队越久
- 模型：`/home/lijiang3/projects/def-y7ding/lijiang3/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`（plan 里叫 "Qwen3.5-4B" 实则是 Qwen3-4B）
- 数据：`/home/lijiang3/projects/def-y7ding/lijiang3/hf_cache/data/only_positive_click_hist_train.parquet` (683,767 rows)
- 上次 30h = 4 nodes × 4 A100 (16 GPU) × 1 epoch × 8B

**资源决定（最终采用 16 GPU，平衡排队时间和训练时长）**：
- Step 2 (SFT, 4B, 只 positive ~一半数据, **1 epoch**)：**4 nodes × 4 GPU = 16 A100**，请求 `--time=12:00:00` on `gpubase_bynode_b2`。预估 7-8h。
- Step 3 (prospect_dpo, 4B, 全量 pos+neg, **1 epoch** 先试水，好的话再多 epoch)：**4 nodes × 4 GPU = 16 A100**，请求 `--time=24:00:00` on `gpubase_bynode_b3`。预估 16-18h。
- 实测 `sbatch --test-only`：两个作业都立即启动（T+0），walltime 12h/24h/48h/72h 排队时间相同，关键是节点数而非时长。
- 选择 16 GPU 的理由：查当前 Narval 队列全体 pending = 0（4 条 held），但 8-node 作业仍比 4-node 明显更难调度（经验 2-6h vs <1h）。16 GPU 45-50h 训完 vs 32 GPU 25-30h 训完，差的 ~20h 大部分可能被多排队抵消，而且 16 GPU 省一半 GPU-hour 有助于 FairShare 回升（当前 0.150 偏低）。
- 4B 比 8B 每 step 约快 2x。总 GPU-hour = 16 GPU × (7+48)h ≈ 880 GPU-hour，与原 30h × 16 = 480 + 新 SFT 同量级，训了 4 倍工作量。

---

## Overview of Deliverables

### A. 代码改动（少量、可逆）
1. `recipe/dpo/core_algos.py`：给 `compute_prospect_dpo_alpha` 加 `alpha_max` 参数，改成对称 α ∈ [1, α_max]
2. `recipe/dpo/recipe_actor.py`：`_compute_point_policy_logps` 加 `average_log_prob` 参数；`_update_policy_prospect_dpo` 读 `meta_info["prospect_dpo_alpha_max"]` 和 `meta_info["average_log_prob"]`
3. `recipe/dpo/data/precompute_point_reference_logps.py:179`：把 `average_log_prob=False` 改成读配置
4. `recipe/dpo/config/dpo_prospect_dpo.yaml`：新增 `prospect_dpo_alpha_max: 2.0` 和 `average_log_prob: true`
5. `recipe/dpo/run/run_single_wise_dpo.sh`：新增 `SINGLE_WISE_DPO_ALPHA_MAX`、`SINGLE_WISE_DPO_AVERAGE_LOG_PROB`、`SINGLE_WISE_DPO_MODEL_DIR` default 覆盖路径；把 MODEL_DIR 默认改 4B（或通过 env 覆写）

### B. 新增文件
1. `recipe/dpo/data/prepare_pens_sft.py` — 把 `only_positive_click_hist_train.parquet` 改成 `messages` 列格式给 SFT trainer 吃
2. `recipe/dpo/run/run_pens_sft_warmup.sh` — 调用 `verl.trainer.sft_trainer` 的 launcher，LoRA r=64, lr=1e-5, 1 epoch, 只 positive
3. `recipe/dpo/run/sbatch/run_sft_warmup_16xa100.sh` — sbatch for Step 2
4. `recipe/dpo/run/sbatch/retry_multinode_sft.sh` — Ray bootstrap for Step 2（SFT 不走 Ray，用 torchrun）—— **实际上 SFT trainer 用 torch.distributed，不需要 Ray**；只需 `srun` 拉起 torchrun
5. `recipe/dpo/run/sbatch/run_prospect_dpo_fixed_16xa100.sh` — sbatch for Step 3（复用现有 `retry_multinode.sh` 的逻辑）
6. `recipe/dpo/run/sbatch/retry_multinode_prospect_fixed.sh` — Step 3 专用 Ray bootstrap + 新超参

---

## A. 代码改动详细规格

### A.1 `recipe/dpo/core_algos.py:115-131`（α 对称化）

**当前**：
```python
def compute_prospect_dpo_alpha(s_dwell, alpha_tau, alpha_k):
    s_dwell = s_dwell.float().clamp(0.0, 1.0)
    return torch.sigmoid(alpha_k * (s_dwell - alpha_tau))   # ∈ (0, 1)
```

**改为**（方案 A，新增参数 `alpha_max`）：
```python
def compute_prospect_dpo_alpha(s_dwell, alpha_tau, alpha_k, alpha_max=1.0):
    s_dwell = s_dwell.float().clamp(0.0, 1.0)
    gate = torch.sigmoid(alpha_k * (s_dwell - alpha_tau))
    return 1.0 + (alpha_max - 1.0) * gate                   # α_max=1.0 等价原行为；α_max=2.0 对称于 λ
```

### A.2 `recipe/dpo/core_algos.py:133-166`（loss 函数签名 + 调用）

在 `compute_prospect_dpo_loss` 签名加 `alpha_max: float = 1.0`，调用 `compute_prospect_dpo_alpha` 时传入。

### A.3 `recipe/dpo/recipe_actor.py:32-37`（policy logp 支持 avg）

**当前**：
```python
def _compute_point_policy_logps(self, inputs, response_mask):
    outputs = self._forward_micro_batch(inputs, temperature=1.0, calculate_entropy=False)
    token_logps = outputs["log_probs"]
    return (token_logps * response_mask.to(token_logps.dtype)).sum(dim=-1)
```

**改为**：
```python
def _compute_point_policy_logps(self, inputs, response_mask, average_log_prob=False):
    outputs = self._forward_micro_batch(inputs, temperature=1.0, calculate_entropy=False)
    token_logps = outputs["log_probs"]
    mask = response_mask.to(token_logps.dtype)
    logp_sum = (token_logps * mask).sum(dim=-1)
    if average_log_prob:
        return logp_sum / mask.sum(dim=-1).clamp(min=1)
    return logp_sum
```

### A.4 `recipe/dpo/recipe_actor.py:75-105`（_update_policy_prospect_dpo 读配置）

在 `data.meta_info.get()` 块中增加：
```python
alpha_max = data.meta_info.get("prospect_dpo_alpha_max", 1.0)
average_log_prob = data.meta_info.get("average_log_prob", False)
```
把 `alpha_max` 传给 `compute_prospect_dpo_loss`，把 `average_log_prob` 传给 `_compute_point_policy_logps`。

### A.5 `recipe/dpo/data/precompute_point_reference_logps.py:179`

**当前**：`sequence_logps = get_batch_logps(logits, batch["labels"], average_log_prob=False)`

**改为**：读 `average_log_prob` 从 worker/runner 配置（加一个 kw，默认 False，由 materializer 调用链传入）。最小改动：在 `compute_reference_logps_for_table` 和上游 `ReferenceLogpsWorker` 加一个 `average_log_prob` 字段。

### A.6 `recipe/dpo/config/dpo_prospect_dpo.yaml`

在 `algorithm:` 下加 `prospect_dpo_alpha_max: 2.0`；在顶层（或 `data:` 下）加 `average_log_prob: true`。

### A.7 `recipe/dpo/run/run_single_wise_dpo.sh`

- 新增 env（line ~46 附近）：
  - `ALPHA_MAX="${SINGLE_WISE_DPO_ALPHA_MAX:-2.0}"`（默认对称）
  - `AVERAGE_LOG_PROB="${SINGLE_WISE_DPO_AVERAGE_LOG_PROB:-true}"`（默认 avg）
- 把 MODEL_DIR 默认路径改 4B 目录（或保持 8B 默认，全靠 sbatch 里 export 覆盖）。**采纳后者：sbatch 显式 export 4B 路径**，避免影响其他 caller
- Python 调用行（line ~511+）加：
  ```
  algorithm.prospect_dpo_alpha_max=${ALPHA_MAX} \
  algorithm.average_log_prob=${AVERAGE_LOG_PROB} \
  ```

---

## B. 新增文件规格

### B.1 `recipe/dpo/data/prepare_pens_sft.py`

职责：读 `only_positive_click_hist_train.parquet`，把 `prompt` (list[{role, content}]) + `response` (str) 合并成 `messages` = prompt + `[{"role":"assistant","content":response}]`，输出到 `/home/lijiang3/projects/def-y7ding/lijiang3/hf_cache/data/pens_sft_train.parquet`。

参考 `recipe/dpo/data/prepare_pens_singlewise_dpo.py` 的 pyarrow 写法。~40 行代码即可。Schema：
```python
OUTPUT_SCHEMA = pa.schema([
    ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    ("sample_id", pa.string()),
])
```

执行命令（Step 2 前一次性运行，在 login node 或 interactive 节点即可，单进程几分钟）：
```bash
python -m recipe.dpo.data.prepare_pens_sft \
    --input /home/lijiang3/projects/def-y7ding/lijiang3/hf_cache/data/only_positive_click_hist_train.parquet \
    --output /home/lijiang3/projects/def-y7ding/lijiang3/hf_cache/data/pens_sft_train.parquet
```

### B.2 `recipe/dpo/run/run_pens_sft_warmup.sh`

调用 `python -m verl.trainer.sft_trainer`（Hydra 接 `recipe/rep_exp/config/sft_trainer.yaml` 作模板）。关键 override：
```bash
python -m verl.trainer.sft_trainer \
  --config-path=recipe/rep_exp/config --config-name=sft_trainer \
  data.train_files=[/home/lijiang3/projects/def-y7ding/lijiang3/hf_cache/data/pens_sft_train.parquet] \
  data.val_files=[/home/lijiang3/projects/def-y7ding/lijiang3/hf_cache/data/pens_sft_train.parquet] \
  data.multiturn.enable=true \
  data.multiturn.messages_key=messages \
  data.max_length=4160 \
  data.train_batch_size=256 \
  data.micro_batch_size_per_gpu=8 \
  model.partial_pretrain=/home/lijiang3/projects/def-y7ding/lijiang3/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  model.lora_rank=64 \
  model.lora_alpha=128 \
  model.target_modules=[q_proj,k_proj,v_proj,o_proj] \
  model.enable_gradient_checkpointing=true \
  optim.lr=1e-5 \
  optim.lr_warmup_steps_ratio=0.03 \
  optim.lr_scheduler=cosine \
  trainer.total_epochs=1 \
  trainer.nnodes=${NNODES:-8} \
  trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-4} \
  trainer.default_local_dir=/scratch/lijiang3/ckpt/PENS/sft_warmup_pos_qwen3-4b \
  trainer.save_freq=-1 \
  trainer.project_name=pens-sft-warmup \
  trainer.experiment_name=qwen3-4b-pos-lora-r64 \
  trainer.logger=[console,wandb] \
  trainer.checkpoint.save_contents=[model,optimizer,extra,hf_model]
```

注意 `data.max_length=4160`：4096 (prompt) + 64 (response 含 JSON 包装)。`save_contents` 包含 `hf_model` 以便 Step 3 直接把 merged HF 目录当作 MODEL_DIR。

### B.3 `recipe/dpo/run/sbatch/run_sft_warmup_16xa100.sh`

SLURM headers：
```bash
#SBATCH --job-name=sft-pens-4b-32gpu
#SBATCH --account=def-y7ding_gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=8
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --signal=B:USR1@300
```
Body：复用 `run_single_wise_dpo_16xa100.sh` 的 module load + venv + ntfy 逻辑，把最后的 `bash retry_multinode.sh` 换成 `bash retry_multinode_sft.sh`。

### B.4 `recipe/dpo/run/sbatch/retry_multinode_sft.sh`

SFT trainer 基于 `torch.distributed`（不走 Ray），用 `srun` + torchrun：
```bash
#!/bin/bash
set -euo pipefail
module load python/3.12.4 cuda/12.2 arrow/19.0.1 opencv/4.11.0 nodejs
source /project/def-y7ding/lijiang3/envs/verl/bin/activate
cd /home/lijiang3/projects/def-y7ding/lijiang3/verl
unset ROCR_VISIBLE_DEVICES

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500
export WANDB_MODE=offline
export WANDB_DIR=/scratch/lijiang3/wandb

srun --kill-on-bad-exit=1 bash -c '
  module load python/3.12.4 cuda/12.2 arrow/19.0.1 opencv/4.11.0 nodejs
  source /project/def-y7ding/lijiang3/envs/verl/bin/activate
  unset ROCR_VISIBLE_DEVICES
  export NNODES=$SLURM_JOB_NUM_NODES N_GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
  torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m verl.trainer.sft_trainer \
    --config-path=recipe/rep_exp/config --config-name=sft_trainer \
    # ... all the override args from B.2 ...
'
```

### B.5 `recipe/dpo/run/sbatch/run_prospect_dpo_fixed_16xa100.sh`

复制 `run_single_wise_dpo_16xa100.sh` 并修改：
- `--job-name=dpo-prospect-fixed-4b-32gpu`
- `--time=48:00:00`
- `--nodes=8`
- Body 中 export 覆盖：
  ```bash
  export SINGLE_WISE_DPO_NNODES=8
  export SINGLE_WISE_DPO_N_GPUS_PER_NODE=4
  export SINGLE_WISE_DPO_MICRO_BATCH_SIZE=8
  export SINGLE_WISE_DPO_TRAIN_BATCH_SIZE=256   # grad_accum = 256/(32*8) = 1
  # 使用 SFT 合并后的 HF 目录作为 base + reference
  export SINGLE_WISE_DPO_MODEL_DIR=/scratch/lijiang3/ckpt/PENS/sft_warmup_pos_qwen3-4b/global_step_xxx/hf_merged
  export SINGLE_WISE_DPO_REFERENCE_MODEL_DIR=$SINGLE_WISE_DPO_MODEL_DIR
  # 修复版 loss 超参
  export SINGLE_WISE_DPO_BETA=0.1
  export SINGLE_WISE_DPO_ALPHA_TAU=0.0
  export SINGLE_WISE_DPO_ALPHA_K=4.0
  export SINGLE_WISE_DPO_ALPHA_MAX=2.0           # 新
  export SINGLE_WISE_DPO_LAMBDA_MAX=2.0
  export SINGLE_WISE_DPO_LAMBDA_GAMMA=2.0
  export SINGLE_WISE_DPO_AVERAGE_LOG_PROB=true   # 新
  # 修复训练-推理长度 mismatch
  export SINGLE_WISE_DPO_MAX_RESPONSE_LENGTH=64
  # 多跑 2-3 epoch
  export SINGLE_WISE_DPO_TOTAL_EPOCHS=3
  # 每 1000 step 存 ckpt
  export SINGLE_WISE_DPO_SAVE_FREQ=1000
  export SINGLE_WISE_DPO_CKPT_ROOT=/scratch/lijiang3/ckpt/PENS
  # ref logps 重算（新 reference 模型）
  export SINGLE_WISE_DPO_REFERENCE_LOGPS_MATERIALIZED_DIR=/home/lijiang3/projects/def-y7ding/lijiang3/hf_cache/data/pens_sft_ref
  export SINGLE_WISE_DPO_REFERENCE_LOGPS_MAX_BATCH_SIZE=4
  export SINGLE_WISE_DPO_REFERENCE_LOGPS_MAX_BATCHED_TOKENS=16384
  export SINGLE_WISE_DPO_REFERENCE_LOGPS_ROWS_PER_TASK=512
  ```
- 最后调用 `bash recipe/dpo/run/sbatch/retry_multinode_prospect_fixed.sh`

### B.6 `recipe/dpo/run/sbatch/retry_multinode_prospect_fixed.sh`

复制现有 `retry_multinode.sh`，把 `SINGLE_WISE_DPO_*` export 块删掉（让外层 sbatch 决定），把 `assert res.get('GPU', 0) >= 16` 改 `>= 32`，最后 `bash recipe/dpo/run/run_single_wise_dpo_click_hist_all.sh`。

---

## C. 执行顺序（一次性把所有脚本/配置准备好，再依次 submit）

**阶段 0（本地，~10 分钟）**：
1. 实施 A.1–A.7 所有代码改动
2. 新建 B.1 `prepare_pens_sft.py` 并跑一次生成 `pens_sft_train.parquet`
3. 新建 B.2–B.6 脚本

**阶段 1（Step 2，~4-6h 墙钟）**：
```bash
sbatch recipe/dpo/run/sbatch/run_sft_warmup_16xa100.sh
```
监控 `squeue -u $USER`；完成后 SFT 产物在 `/scratch/lijiang3/ckpt/PENS/sft_warmup_pos_qwen3-4b/global_step_<N>/hf_merged/`。

**阶段 2（Step 2 验证，~20 分钟 on 1 GPU）**：
用 SFT 的 hf_merged 路径跑一次 eval：
```bash
bash recipe/dpo/evaluation/run_pens_personalized_eval.sh \
  MODEL_PATH=/scratch/lijiang3/ckpt/PENS/sft_warmup_pos_qwen3-4b/global_step_<N>/hf_merged
```
预期 R-1 ≥ 0.25（接近或略过 positive_only 的 0.2442）。若显著低于 0.24 说明 SFT 没生效，需先排查数据/LoRA 加载再进 Step 3。

**阶段 3（Step 3，~25-30h 墙钟）**：
在 B.5 的 sbatch 里填入 Step 2 实际产出的 `global_step_<N>` 路径，然后：
```bash
sbatch recipe/dpo/run/sbatch/run_prospect_dpo_fixed_16xa100.sh
```

**阶段 4（Step 3 每 epoch eval）**：
训练中 `SAVE_FREQ=1000` 会存 intermediate ckpt；每个完成的 ckpt 跑一次 eval，画 ROUGE vs step 曲线。

---

## D. 关键文件改动清单

| 文件 | 操作 | 说明 |
|---|---|---|
| `recipe/dpo/core_algos.py:115-166` | Edit | α 加 alpha_max 参数 + loss 签名 |
| `recipe/dpo/recipe_actor.py:32-37, 75-105` | Edit | policy logp avg + 读新 meta_info |
| `recipe/dpo/data/precompute_point_reference_logps.py:179` | Edit | ref logp avg 支持 |
| `recipe/dpo/config/dpo_prospect_dpo.yaml:33-43` | Edit | 加 alpha_max + average_log_prob |
| `recipe/dpo/run/run_single_wise_dpo.sh:42-46, 511+` | Edit | 新增 ALPHA_MAX/AVERAGE_LOG_PROB env + 透传 |
| `recipe/dpo/data/prepare_pens_sft.py` | Create | pos-only → messages 列转换 |
| `recipe/dpo/run/run_pens_sft_warmup.sh` | Create | SFT launcher |
| `recipe/dpo/run/sbatch/run_sft_warmup_16xa100.sh` | Create | SFT sbatch (8 nodes) |
| `recipe/dpo/run/sbatch/retry_multinode_sft.sh` | Create | SFT torchrun bootstrap |
| `recipe/dpo/run/sbatch/run_prospect_dpo_fixed_16xa100.sh` | Create | 修复版 DPO sbatch (8 nodes) |
| `recipe/dpo/run/sbatch/retry_multinode_prospect_fixed.sh` | Create | Step 3 Ray bootstrap（复用 retry_multinode.sh） |

---

## E. 验证（Verification）

**训练侧 sanity（W&B / 日志）**：
- Step 2：`sft/loss` 平滑下降（无 NaN），1 epoch ~5341 / (256/(32×8))×step = 确保实际走完全部 positive 数据
- Step 3：
  - `actor/prospect_dpo_loss_pos` vs `actor/prospect_dpo_loss_neg` 量级接近 1:1（alpha_max=2 后的对称性验证）
  - `actor/prospect_dpo_alpha` 均值 > 1.0（确认 alpha_max 生效）
  - `actor/dpo_loss` 平滑下降

**推理侧**（每 checkpoint 跑一次）：
```bash
bash recipe/dpo/evaluation/run_pens_personalized_eval.sh \
  MODEL_PATH=<checkpoint hf_merged 目录> \
  GEN_TEMPERATURE=0.0 GEN_TOP_P=1.0 GEN_TOP_K=-1 GEN_RESPONSE_LENGTH=64
```
指标门槛：
- `parse_status_counts.malformed_json_output + missing_structured_output < 50`（非格式化率 < 0.25%）
- `rouge_1_f1 / rouge_2_f1 / rouge_l_f1`

**目标**：
- Step 2 后：R-1 ≥ 0.25（接近 positive_only）
- Step 3 末 epoch：R-1 ≥ 0.27，超过 positive_only（0.2442）才说明修复版 loss 真正有效
- 若 ≥ 0.28：Step 4 的 8B/MLP LoRA 加成可以让它越过 NAML+IM-2 线（0.2801）

---

## F. 风险与兜底

1. **SFT trainer 不接受 list-of-dict prompt 格式**：已用 `messages` 列 + `multiturn.enable=true` 绕开。如果仍出错，fallback：直接在 `recipe/dpo/run/run_single_wise_dpo_click_hist_positive_only.sh` 跑一遍当作 implicit SFT（plan 根因分析已说明 positive_only 本质就是 SFT）。
2. **8 nodes 排队慢**：若超过 4h 仍 PENDING，降到 4 nodes（同 16 GPU 老配置），把 `--time` 延到 72h 用 b4 partition。
3. **LoRA adapter 跨 trainer 不兼容**：verl SFT 存 `hf_merged/` 完整权重（配 `save_contents: [model, optimizer, extra, hf_model]`），Step 3 直接用该 HF 目录当 MODEL_DIR，不依赖 LoRA 续训 —— 干净、稳。
4. **reference logps 缓存冲突**：Step 3 用了新 reference model（SFT-merged），必须指向新的 `REFERENCE_LOGPS_MATERIALIZED_DIR=pens_sft_ref` 子目录，避免复用旧 cache 误导训练。
5. **wall clock 超时**：Step 3 请求 48h 对 25-30h 预期留足余量；`retry_on_alloc.sh` 已有 resume 机制，SAVE_FREQ=1000 保证断点损失 ≤ 1000 步。
