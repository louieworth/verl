# Wandb 训练指标记录 - 最终总结

## ✅ 已实现的 Wandb 记录指标

### 训练指标 (每 logging_steps 记录)

```python
wandb.log({
    "train/loss": loss,                    # 总损失 (KL loss * kl_coef)
    "train/kl_loss": kl_loss,              # KL 散度损失
    "train/kl_coef": kl_coef,              # KL 系数
    "train/student_perplexity": ppl,       # 学生模型困惑度
    "train/teacher_perplexity": ppl,       # 教师模型困惑度
    "train/grad_norm": grad_norm,          # 梯度范数
    "train/learning_rate": lr,             # 当前学习率
    "train/epoch": epoch,                  # 当前 epoch
    "train/global_step": step,             # 全局步数
})
```

### 评估指标 (评估完成后记录)

```python
wandb.log({
    "eval/aime24": accuracy,
    "eval/aime25": accuracy,
    "eval/math500": accuracy,
    ...
})
```

---

## 📊 控制台日志示例

```
[2026-03-06 22:30:40] INFO: KL Trainer initialized:
[2026-03-06 22:30:40] INFO:   KL Type: reverse
[2026-03-06 22:30:40] INFO:   KL Method: monte_carlo
[2026-03-06 22:30:40] INFO:   Student Model: Qwen/Qwen3-1.7B
[2026-03-06 22:30:40] INFO:   Wandb Project: verl-kl-training
[2026-03-06 22:30:40] INFO:   Wandb Run Name: kl_reverse_monte_carlo_0306_2230
[2026-03-06 22:30:45] INFO: Wandb initialized: https://wandb.ai/verl-kl-training/kl_reverse_monte_carlo_0306_2230
[2026-03-06 22:30:50] INFO: Starting training...

Epoch 0: 100%|████████| 1000/1000 [10:00<00:00]
[2026-03-06 22:31:00] INFO: Step 10: loss=0.1234, kl_loss=0.0567, kl_coef=0.10, ppl=15.23, grad_norm=1.23, lr=2.00e-05
[2026-03-06 22:31:20] INFO: Step 20: loss=0.1123, kl_loss=0.0512, kl_coef=0.10, ppl=14.87, grad_norm=1.15, lr=2.00e-05
...
[2026-03-06 22:50:40] INFO: Training completed!
[2026-03-06 22:50:45] INFO: Checkpoint saved to: /data/.../final
[2026-03-06 22:50:50] INFO: Merging LoRA adapters...
[2026-03-06 22:51:00] INFO: Merged model saved to: /data/.../hf_merged
```

---

## 📁 输出文件结构 (最终版本)

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

# Wandb Dashboard (在线，实时更新)
https://wandb.ai/verl-kl-training/kl_reverse_monte_carlo_0306_2230
```

**注意**: 不再保存本地 `training_history.json`，所有训练指标通过 Wandb 实时记录。

---

## 🚀 使用方法

### 基本训练 (Wandb 自动启用)
```bash
KL_TYPE=reverse \
USE_INITIAL_RESPONSE=false \
bash recipe/open_math_reasoning/run_kl_training.sh
```

### 禁用 Wandb
```bash
WANDB_MODE=disabled \
bash recipe/open_math_reasoning/run_kl_training.sh
```

### 自定义 Wandb 项目
```bash
WANDB_PROJECT=my-experiments \
WANDB_RUN_NAME=kl_reverse_v1 \
bash recipe/open_math_reasoning/run_kl_training.sh
```

---

## 📈 Wandb Dashboard 功能

### 实时训练曲线
- **Loss 曲线**: train/loss, train/kl_loss
- **Perplexity 曲线**: train/student_perplexity, train/teacher_perplexity
- **学习率曲线**: train/learning_rate
- **梯度范数曲线**: train/grad_norm

### 对比实验
- 多个实验的曲线叠加对比
- 超参数对比表格
- 不同 KL 类型的效果对比

### 评估结果
- eval/aime24, eval/aime25, eval/math500
- 自动记录最佳 checkpoint

---

## 🔧 代码实现位置

| 功能 | 文件 | 行号 |
|------|------|------|
| Wandb 初始化 | `kl_trainer.py` | 93-122 |
| 训练指标记录 | `kl_trainer.py` | 450-462 |
| 评估指标记录 | `kl_trainer.py` | 633-635 |
| Wandb 结束 | `kl_trainer.py` | 530-532 |
| Perplexity 计算 | `kl_trainer.py` | 378-393 |

---

## ✅ 功能检查清单

- [x] Wandb 初始化 (只在主进程)
- [x] 训练 loss 记录
- [x] KL loss 记录
- [x] KL coefficient 记录
- [x] Student perplexity 记录
- [x] Teacher perplexity 记录
- [x] Gradient norm 记录
- [x] Learning rate 记录
- [x] Epoch 记录
- [x] Global step 记录
- [x] 评估结果记录
- [x] Wandb finish (训练结束)
- [x] 移除本地 training_history.json
- [x] 配置记录到 Wandb

所有功能已实现！
