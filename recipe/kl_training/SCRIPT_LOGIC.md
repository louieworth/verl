# run_kl_training.sh 代码逻辑和文件保存说明

## 📋 脚本执行流程

### 1️⃣ 配置阶段 (第 28-104 行)

#### 输入参数
```bash
# KL 训练设置
KL_TYPE=reverse/forward          # KL 类型
KL_METHOD=monte_carlo/full_vocab # 计算方法
KL_COEF=0.1                      # KL 系数
TEMPERATURE=1.0                  # 温度
USE_INITIAL_RESPONSE=false/true  # Reverse KL 变体选择

# 模型设置
MODEL_PATH=Qwen/Qwen3-1.7B      # 学生模型
TEACHER_MODEL_PATH=""            # 教师模型（空=共享学生模型）
USE_LORA=true                    # 是否使用 LoRA
LORA_RANK=64                     # LoRA rank
LORA_ALPHA=128                   # LoRA alpha

# 训练设置
LEARNING_RATE=2e-5               # 学习率
TRAIN_BATCH_SIZE=96              # 批次大小
TOTAL_EPOCHS=1                   # 训练轮数
MAX_LENGTH=20480                 # 最大序列长度

# 数据设置
DATA_PATH=""                     # 自动设置
EXPERT_SOLUTIONS_PATH=""         # 专家解答路径
MAX_SAMPLES=""                   # 样本数量限制（测试用）
```

#### 自动路径设置 (第 78-93 行)
```bash
# 根据 KL_TYPE 自动设置数据路径
if [ "$KL_TYPE" = "forward" ]; then
    DATA_PATH="$VERL_ROOT/results/${MODEL_NAME}/stage2_correction.parquet"
else
    DATA_PATH="$VERL_ROOT/results/${MODEL_NAME}/stage1_generation.parquet"
fi

# 专家解答路径
EXPERT_SOLUTIONS_PATH="$VERL_ROOT/data/deepscaleR_expert_solutions.parquet"

# Forward KL 需要校正数据
if [ "$KL_TYPE" = "forward" ]; then
    CORRECTED_RESPONSES_PATH="$VERL_ROOT/results/${MODEL_NAME}/stage2_correction.parquet"
fi
```

#### 输出目录自动生成 (第 95-99 行)
```bash
OUTPUT_DIR="$VERL_ROOT/outputs/${MODEL_NAME}_kl_${KL_TYPE}_${KL_METHOD}_$(date +%Y%m%d_%H%M%S)"

# 示例:
# outputs/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/
```

---

### 2️⃣ 配置打印和验证 (第 107-158 行)

#### 打印配置信息
```
==========================================
KL Divergence Training Configuration
==========================================

KL Settings:
  Type:     reverse
  Method:   monte_carlo
  Coef:     0.1
  Temp:     1.0
  Use Initial Response: false

Model Settings:
  Model:    Qwen/Qwen3-1.7B
  Teacher:  Qwen/Qwen3-1.7B
  LoRA:     true (rank=64, alpha=128)

Training Settings:
  LR:       2e-5
  Batch:    96
  Epochs:   1
  Max Len:  20480

Data Settings:
  Data:     /home/jiangli/verl/results/Qwen3-1.7B/stage1_generation.parquet
  Expert:   /home/jiangli/verl/data/deepscaleR_expert_solutions.parquet
  Max Samples: all

Output:
  Dir:      /home/jiangli/verl/outputs/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015
  Wandb:    verl-kl-training / kl_reverse_monte_carlo_0306_2230
==========================================
```

#### 数据文件检查
```bash
# 检查数据文件是否存在
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found: $DATA_PATH"
    exit 1
fi

if [ ! -f "$EXPERT_SOLUTIONS_PATH" ]; then
    echo "ERROR: Expert solutions file not found: $EXPERT_SOLUTIONS_PATH"
    exit 1
fi
```

---

### 3️⃣ 创建输出目录和保存配置 (第 160-192 行)

#### 创建目录结构
```bash
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"
```

#### 📄 保存训练配置 (第一个保存的文件)
**文件**: `$OUTPUT_DIR/training_config.yaml`

```yaml
kl_type: reverse
kl_method: monte_carlo
kl_coef: 0.1
temperature: 1.0
model_path: Qwen/Qwen3-1.7B
teacher_model_path: Qwen/Qwen3-1.7B
use_lora: true
lora_rank: 64
lora_alpha: 128
learning_rate: 2e-5
train_batch_size: 96
gradient_accumulation_steps: 1
total_epochs: 1
max_length: 20480
warmup_ratio: 0.1
weight_decay: 0.01
data_path: /home/jiangli/verl/results/Qwen3-1.7B/stage1_generation.parquet
expert_solutions_path: /home/jiangli/verl/data/deepscaleR_expert_solutions.parquet
corrected_responses_path: null
max_samples: null
output_dir: /home/jiangli/verl/outputs/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015
wandb_project: verl-kl-training
wandb_run_name: kl_reverse_monte_carlo_0306_2230
```

**作用**: 记录本次训练的所有超参数，方便复现

---

### 4️⃣ 执行训练 (第 194-246 行)

#### 构建训练命令
```bash
python3 -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    $RECIPE_DIR/kl_training/run_training.py \
    --kl_type reverse \
    --kl_method monte_carlo \
    --kl_coef 0.1 \
    --temperature 1.0 \
    --student_model_path Qwen/Qwen3-1.7B \
    --use_lora true \
    --lora_rank 64 \
    --lora_alpha 128 \
    --learning_rate 2e-5 \
    --train_batch_size 96 \
    --gradient_accumulation_steps 1 \
    --total_epochs 1 \
    --max_length 20480 \
    --warmup_steps_ratio 0.1 \
    --weight_decay 0.01 \
    --data_path /path/to/stage1_generation.parquet \
    --expert_solutions_path /path/to/expert_solutions.parquet \
    --output_dir /path/to/output \
    --wandb_project verl-kl-training \
    --wandb_run_name kl_reverse_monte_carlo_0306_2230 \
    --use_initial_response false
```

#### 📄 保存训练日志
**文件**: `$OUTPUT_DIR/logs/training_YYYYMMDD_HHMMSS.log`

```
[2026-03-06 22:30:15] INFO: KL Trainer initialized:
[2026-03-06 22:30:15] INFO:   KL Type: reverse
[2026-03-06 22:30:15] INFO:   KL Method: monte_carlo
[2026-03-06 22:30:15] INFO:   Student Model: Qwen/Qwen3-1.7B
[2026-03-06 22:30:15] INFO:   Teacher Model: Same as student
[2026-03-06 22:30:15] INFO:   Use LoRA: True
[2026-03-06 22:30:20] INFO: Loading tokenizer from: Qwen/Qwen3-1.7B
[2026-03-06 22:30:25] INFO: Loading student model from: Qwen/Qwen3-1.7B
[2026-03-06 22:30:30] INFO: Applying LoRA: rank=64, alpha=128
[2026-03-06 22:30:35] INFO: Teacher model shares base model with student (memory efficient)
[2026-03-06 22:30:40] INFO: Starting training...
[2026-03-06 22:30:40] INFO: Total epochs: 1
[2026-03-06 22:30:40] INFO: Batch size: 96
[2026-03-06 22:30:40] INFO: Total steps: 1000

Epoch 0: 100%|████████| 1000/1000 [10:00<00:00, 1.67it/s, loss=0.1234, kl=0.0567]

[2026-03-06 22:40:40] INFO: Step 10: loss=0.1234, kl_loss=0.0567, kl_coef=0.1000, lr=2.00e-05
[2026-03-06 22:40:50] INFO: Step 20: loss=0.1123, kl_loss=0.0512, kl_coef=0.1000, lr=2.00e-05
...
[2026-03-06 23:30:40] INFO: Epoch 0 completed: {'loss': 0.0987, 'kl_loss': 0.0456}
[2026-03-06 23:30:40] INFO: Training completed!
[2026-03-06 23:30:45] INFO: Checkpoint saved to: /path/to/output/final
```

**作用**: 记录完整的训练过程，包括每个 step 的 loss、学习率等

---

### 5️⃣ 训练过程中保存的文件 (kl_trainer.py)

#### 📁 中间 Checkpoint (每 save_steps 保存一次)
**目录**: `$OUTPUT_DIR/checkpoint-{global_step}/`

```
checkpoint-500/
├── adapter_config.json          # LoRA 配置
├── adapter_model.bin            # LoRA 权重
└── tokenizer files              # Tokenizer 文件
```

**保存时机**:
- 每 `save_steps=500` 步保存一次
- 代码位置: `kl_trainer.py` 第 381-382 行

```python
if self.global_step % self.config.save_steps == 0:
    self.save_checkpoint()
```

#### 📁 最终 Checkpoint (训练结束时)
**目录**: `$OUTPUT_DIR/final/`

```
final/
├── adapter_config.json          # LoRA 配置
├── adapter_model.bin            # LoRA 权重 (如果 use_lora=true)
├── pytorch_model.bin            # 完整模型权重 (如果 use_lora=false)
├── config.json                  # 模型配置
├── tokenizer_config.json        # Tokenizer 配置
├── tokenizer.json               # Tokenizer
├── special_tokens_map.json      # 特殊 token 映射
└── vocab.txt                    # 词表
```

**保存时机**:
- 训练完成后
- 代码位置: `kl_trainer.py` 第 414 行

```python
self.save_checkpoint(final=True)
```

---

### 6️⃣ 完整的输出目录结构

```
outputs/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/
│
├── training_config.yaml         # ✅ 训练配置 (脚本开始时保存)
│
├── logs/
│   └── training_20260306_223015.log  # ✅ 训练日志 (训练过程中实时写入)
│
├── checkpoint-500/              # ✅ 中间 checkpoint (每 500 步)
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer files
│
├── checkpoint-1000/             # ✅ 中间 checkpoint (每 500 步)
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer files
│
└── final/                       # ✅ 最终 checkpoint (训练结束)
    ├── adapter_config.json
    ├── adapter_model.bin
    ├── config.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    ├── special_tokens_map.json
    └── vocab.txt
```

---

## 🔍 关键代码逻辑

### 1. 数据路径自动选择
```bash
# Reverse KL: 使用 Stage 1 生成结果
if [ "$KL_TYPE" = "reverse" ]; then
    DATA_PATH="results/${MODEL_NAME}/stage1_generation.parquet"
fi

# Forward KL: 使用 Stage 2 校正结果
if [ "$KL_TYPE" = "forward" ]; then
    DATA_PATH="results/${MODEL_NAME}/stage2_correction.parquet"
    CORRECTED_RESPONSES_PATH="results/${MODEL_NAME}/stage2_correction.parquet"
fi
```

### 2. Teacher Model 内存优化
```python
# kl_trainer.py 第 150-180 行
if config.use_lora:
    # Teacher 共享 Student 的 base model (节省内存)
    base_model = self.student_model.get_base_model()
    return base_model
else:
    # 加载独立的 teacher model
    model = AutoModelForCausalLM.from_pretrained(...)
    return model
```

### 3. KL Loss 计算
```python
# kl_trainer.py 第 283-318 行
if self.config.kl_method == "monte_carlo":
    # Monte Carlo: 只计算采样 token 的 KL
    student_logprobs = torch.gather(student_logprobs, ...)
    teacher_logprobs = torch.gather(teacher_logprobs, ...)
    kl_loss = compute_kl_divergence(student_logprobs, teacher_logprobs, ...)
else:
    # Full Vocabulary: 计算完整词汇表的 KL
    kl_loss = compute_kl_divergence(student_logits, teacher_logits, ...)
```

### 4. Checkpoint 保存逻辑
```python
# kl_trainer.py 第 416-445 行
def save_checkpoint(self, final: bool = False):
    if final:
        save_dir = os.path.join(self.config.output_dir, "final")
    else:
        save_dir = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")

    if self.config.use_lora:
        # 只保存 LoRA adapters
        self.student_model.save_pretrained(save_dir)
    else:
        # 保存完整模型
        self.student_model.save_pretrained(save_dir)

    # 保存 tokenizer
    self.tokenizer.save_pretrained(save_dir)
```

---

## ❌ 当前缺少的功能

### 1. 评估结果保存
**问题**: 脚本目前只保存模型 checkpoint，没有保存评估结果

**建议添加**:
```python
# 在训练结束后添加评估
def evaluate(self):
    """Evaluate on test sets."""
    results = {}
    for dataset in ["aime24", "aime25", "math500"]:
        accuracy = self.eval_on_dataset(dataset)
        results[dataset] = accuracy

    # 保存评估结果
    with open(os.path.join(self.config.output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results
```

**应该保存的文件**:
```
outputs/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/
└── eval_results.json            # ❌ 当前缺少
    {
      "aime24": 0.45,
      "aime25": 0.42,
      "math500": 0.67
    }
```

### 2. 训练曲线数据
**建议添加**:
```python
# 保存训练曲线数据
training_history = {
    "steps": [10, 20, 30, ...],
    "loss": [0.123, 0.112, 0.098, ...],
    "kl_loss": [0.056, 0.051, 0.045, ...],
    "learning_rate": [2e-5, 2e-5, 1.9e-5, ...]
}

with open(os.path.join(self.config.output_dir, "training_history.json"), "w") as f:
    json.dump(training_history, f)
```

### 3. 模型合并 (LoRA → Full Model)
**建议添加**:
```bash
# 在训练结束后自动合并 LoRA
if [ "$USE_LORA" = "true" ]; then
    echo "Merging LoRA adapters..."
    python3 -m verl.model_merger merge \
        --backend peft \
        --local_dir $OUTPUT_DIR/final \
        --target_dir $OUTPUT_DIR/hf_merged
fi
```

---

## 📊 总结

### 当前保存的文件
1. ✅ `training_config.yaml` - 训练配置
2. ✅ `logs/training_*.log` - 训练日志
3. ✅ `checkpoint-*/` - 中间 checkpoint
4. ✅ `final/` - 最终 checkpoint

### 缺少的文件
1. ❌ `eval_results.json` - 评估结果
2. ❌ `training_history.json` - 训练曲线数据
3. ❌ `hf_merged/` - 合并后的完整模型 (如果使用 LoRA)

### 建议改进
1. 添加自动评估功能
2. 保存训练曲线数据
3. 自动合并 LoRA adapters
4. 添加 wandb 集成（已有配置但未实现）
