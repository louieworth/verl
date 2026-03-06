# Wandb 训练指标记录

## 📊 Wandb 记录的所有指标

### 训练指标 (每个 logging_steps)

| 指标名称 | 说明 | 示例值 |
|---------|------|--------|
| `train/loss` | 总损失 (KL loss * kl_coef) | 0.1234 |
| `train/kl_loss` | KL 散度损失 | 0.0567 |
| `train/kl_coef` | KL 系数 | 0.1 |
| `train/student_perplexity` | 学生模型困惑度 | 15.23 |
| `train/teacher_perplexity` | 教师模型困惑度 | 12.45 |
| `train/grad_norm` | 梯度范数 | 1.2345 |
| `train/learning_rate` | 当前学习率 | 2.00e-05 |
| `train/epoch` | 当前 epoch | 0 |
| `train/global_step` | 全局步数 | 100 |

### Epoch 指标 (每个 epoch 结束)

| 指标名称 | 说明 |
|---------|------|
| `epoch/total_loss` | Epoch 平均总损失 |
| `epoch/total_kl_loss` | Epoch 平均 KL 损失 |
| `epoch/epoch` | Epoch 编号 |

### 评估指标 (如果启用评估)

| 指标名称 | 说明 |
|---------|------|
| `eval/aime24` | AIME24 准确率 |
| `eval/aime25` | AIME25 准确率 |
| `eval/math500` | MATH500 准确率 |

### Wandb Config (超参数记录)

```python
{
    "kl_type": "reverse",
    "kl_method": "monte_carlo",
    "kl_coef": 0.1,
    "temperature": 1.0,
    "use_initial_response": false,
    "model_path": "Qwen/Qwen3-1.7B",
    "use_lora": true,
    "lora_rank": 64,
    "lora_alpha": 128,
    "learning_rate": 2e-05,
    "train_batch_size": 96,
    "gradient_accumulation_steps": 1,
    "total_epochs": 1,
    "max_length": 20480,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
}
```

---

## 📁 输出文件结构 (更新后)

```
# 轻量级 - /home
/home/jiangli/verl/outputs/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/
├── training_config.yaml       # 训练配置
└── logs/
    └── training_*.log         # 训练日志 (文本)

# 重量级 - /data
/data/data/jiangli/models/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/
├── checkpoint-500/            # 中间 checkpoint
├── final/                     # 最终 checkpoint (LoRA)
└── hf_merged/                 # 合并后的完整模型

# 评估结果
/home/jiangli/verl/results/Qwen3-1.7B/
└── results.json               # 评估结果

# Wandb Dashboard (在线)
https://wandb.ai/verl-kl-training/kl_reverse_monte_carlo_0306_2230
```

**注意**: 不再保存本地 `training_history.json`，所有训练指标通过 Wandb 实时记录。

---

## 🚀 使用方法

### 启用 Wandb (默认启用)
```bash
# Wandb 默认启用，只需设置项目名称
WANDB_PROJECT=my-kl-experiments \
bash recipe/open_math_reasoning/run_kl_training.sh
```

### 禁用 Wandb
```bash
# 设置环境变量禁用 wandb
WANDB_MODE=disabled \
bash recipe/open_math_reasoning/run_kl_training.sh
```

### 自定义 Wandb Run 名称
```bash
WANDB_RUN_NAME=my_experiment_v1 \
bash recipe/open_math_reasoning/run_kl_training.sh
```

---

## 📈 Wandb Dashboard 功能

### 实时训练曲线
- Loss 曲线 (总损失, KL 损失)
- Perplexity 曲线 (学生, 教师)
- 学习率曲线
- 梯度范数曲线

### 对比实验
- 多个实验的曲线叠加对比
- 超参数对比表格

### 模型 Checkpoint
- 自动记录 checkpoint 保存时间
- 可关联 checkpoint 与训练指标

---

## 🔧 代码实现位置

### Wandb 初始化
`kl_trainer.py` 第 75-105 行

```python
def _init_wandb(self):
    """Initialize Weights & Biases logging."""
    wandb.init(
        project=self.config.wandb_project,
        name=self.config.wandb_run_name,
        config={...},
    )
```

### 训练指标记录
`kl_trainer.py` 第 470-490 行

```python
if HAS_WANDB and self.config.local_rank in [-1, 0]:
    wandb.log({
        "train/loss": loss_dict['loss'].item(),
        "train/kl_loss": loss_dict['kl_loss'].item(),
        "train/kl_coef": loss_dict['kl_coef'],
        "train/student_perplexity": loss_dict['student_perplexity'].item(),
        "train/teacher_perplexity": loss_dict['teacher_perplexity'].item(),
        "train/grad_norm": grad_norm,
        "train/learning_rate": lr,
        "train/epoch": self.epoch,
        "train/global_step": self.global_step,
    })
```

### Epoch 指标记录
`kl_trainer.py` 第 510-515 行

```python
if HAS_WANDB and self.config.local_rank in [-1, 0]:
    wandb.log({
        "epoch/total_loss": metrics["loss"],
        "epoch/total_kl_loss": metrics["kl_loss"],
        "epoch/epoch": epoch,
    })
```

### 评估结果记录
`kl_trainer.py` 第 633-635 行

```python
if HAS_WANDB and self.config.local_rank in [-1, 0]:
    wandb.log({f"eval/{dataset}": acc for dataset, acc in eval_results.items()})
```
