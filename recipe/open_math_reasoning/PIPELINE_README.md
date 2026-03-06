# Three-Stage Policy Correction Pipeline

完整的数学推理策略校正流程，整合为单个脚本运行。

## 文件结构

```
recipe/open_math_reasoning/
├── stage1_prepare.py              # Stage 1 数据准备
├── stage2_prepare.py              # Stage 2 数据准备（按reward分割，构建校正prompt）
├── stage3_prepare.py              # Stage 3 数据准备（转换校正数据为SFT格式）
├── run_full_pipeline.sh           # 完整三阶段流程脚本
├── compute_score.py               # 评估奖���函数
└── PIPELINE_README.md             # 本文档
```

## 快速开始

### 运行完整流程

```bash
cd /home/jiangli/verl

# 使用默认配置运行
bash recipe/open_math_reasoning/run_full_pipeline.sh

# 或自定义模型路径
MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
NGPUS_PER_NODE=8 \
bash recipe/open_math_reasoning/run_full_pipeline.sh
```

### 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `Qwen/Qwen3-4B-Thinking-2507` | 基础模型路径 |
| `NGPUS_PER_NODE` | `8` | 每节点GPU数量 |
| `NNODES` | `1` | 节点数量 |
| `GEN_TP` | `1` | Tensor并行度 |
| `PASS_K` | `1` | 每个prompt生成样本数 |
| `BACKEND` | `fsdp` | SFT训练后端 |
| `DEEPSCALE_PATH` | `/data/data/jiangli/huggingface/datasets/DeepScaleR` | 数据集路径 |

## 三阶段流程详解

### Stage 1: 初始响应生成

**目标**: 使用基础模型生成初始响应并评估正确性

**步骤**:
1. `stage1_prepare.py`: 准备数据，保留 `expert_cot` 用于 Stage 2
2. `main_generation_server`: 生成初始响应
3. `main_eval`: 计算奖励 (reward=0/1)

**输入**: DeepScaleR 原始数据
**输出**:
- `evaluation_results/${MODEL_NAME}/stage1_generation.parquet`
- `evaluation_results/${MODEL_NAME}/stage1_eval_results.json`

### Stage 2: 答案校正

**目标**: 对错误答案进行最小编辑校正，并评估校正效果

**步骤**:
1. `stage2_prepare.py`: 按 reward 分割数据，为 reward=0 构建校正 prompt
2. `main_generation_server`: 生成校正后的响应
3. `main_eval`: **评估校正后的正确率**

**输入**: Stage 1 输出 (带 reward)
**输出**:
- `gen_results/stage2/deepscaleR_stage2_reward0_correction.parquet` (需校正的数据)
- `gen_results/stage2/deepscaleR_stage2_reward1.parquet` (原始正确数据，**不再使用**)
- `evaluation_results/${MODEL_NAME}/stage2_correction.parquet` (校正后的响应)
- `evaluation_results/${MODEL_NAME}/stage2_eval_results.json` (校正评估结果)

### Stage 3: 策略微调

**目标**: **只对校正后的数据进行 SFT 训练**（不管校正后是否正确）

**步骤**:
1. `stage3_prepare.py`: 将校正后的数据转换为 SFT 格式
2. `sft_trainer`: SFT 训练

**输入**:
- Stage 2 校正输出 (原 reward=0，校正后)

**输出**:
- `gen_results/stage3/deepscaleR_stage3_sft.parquet` (SFT 数据集，只包含校正数据)
- `${CKPT_HOME}/` (模型检查点)

## 单独运行各阶段

如果需要单独运行某个阶段：

### 只运行 Stage 1

```bash
# 准备数据
python3 recipe/open_math_reasoning/stage1_prepare.py \
    --input_path /path/to/DeepScaleR \
    --output_file gen_results/stage1/deepscaleR_stage1.parquet

# 生成和评估 (参考 run_full_pipeline.sh 中的 Stage 1 部分)
```

### 只运行 Stage 2

```bash
# 准备校正数据
python3 recipe/open_math_reasoning/stage2_prepare.py \
    --stage1_output evaluation_results/.../stage1_generation.parquet \
    --output_reward0 gen_results/stage2/deepscaleR_stage2_reward0_correction.parquet \
    --output_reward1 gen_results/stage2/deepscaleR_stage2_reward1.parquet

# 生成校正 (参考 run_full_pipeline.sh 中的 Stage 2 部分)
```

### 只运行 Stage 3

```bash
# 准备 SFT 数据（只使用校正数据）
python3 recipe/open_math_reasoning/stage3_prepare.py \
    --stage2_corrected evaluation_results/.../stage2_correction.parquet \
    --output_file gen_results/stage3/deepscaleR_stage3_sft.parquet

# SFT 训练 (参考 run_full_pipeline.sh 中的 Stage 3 部分)
```

## 数据流图

```
DeepScaleR 原始数据
    ↓
[stage1_prepare.py]
    ↓
Stage 1 生成 → [main_eval] → reward (0/1)
    ↓
[stage2_prepare.py]
    ↓
    ┌───────────┴───────────┐
    ↓                       ↓
reward=0 (校正)          reward=1 (不再使用)
    ↓
Stage 2 生成 → [main_eval] → 校正评估
    ↓
[stage3_prepare.py]
    ↓
只对校正数据做 SFT
    ↓
[sft_trainer]
    ↓
改进的策略 π'(a|s)
```

## 输出文件位置

| 阶段 | 文件 | 路径 |
|------|------|------|
| Stage 1 | 生成数据 | `evaluation_results/${MODEL_NAME}/stage1_generation.parquet` |
| Stage 1 | 评估结果 | `evaluation_results/${MODEL_NAME}/stage1_eval_results.json` |
| Stage 2 | 校正数据 | `evaluation_results/${MODEL_NAME}/stage2_correction.parquet` |
| Stage 2 | 评估结果 | `evaluation_results/${MODEL_NAME}/stage2_eval_results.json` |
| Stage 3 | SFT数据集 | `gen_results/stage3/deepscaleR_stage3_sft.parquet` |
| Stage 3 | 模型检查点 | `${CKPT_HOME}/` |

## 常见问题

**Q: 如何修改模型？**
A: 设置 `MODEL_PATH` 环境变量，例如：
```bash
MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B bash run_full_pipeline.sh
```

**Q: 如何调整 SFT 学习率？**
A: 修改 `run_full_pipeline.sh` 中 Stage 3 的 `optim.lr` 参数。

**Q: 如何处理更多数据集？**
A: 修改 `stage1_prepare.py` 中的 `--input_path` 和 `--data_source` 参数。

## 相关文档

- `ALGORITHM_DOCUMENTATION.md` - 详细算法文档
- `README.md` - 原始文档
