# 更新后的文件保存结构

## 📁 完整的输出目录结构

训练完成后，文件将保存在三个不同的位置：

### 1️⃣ 输出目录 (Logs & Config)
**位置**: `/home/jiangli/verl/outputs/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/`

```
outputs/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/
├── training_config.yaml              # ✅ 训练配置
└── logs/
    └── training_20260306_223015.log  # ✅ 训练日志
```

**作用**:
- 保存训练配置和日志
- 不占用 /home 太多空间（只有文本文件）

---

### 2️⃣ 模型保存目录 (Models)
**位置**: `/data/data/jiangli/models/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/`

```
/data/data/jiangli/models/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/
├── checkpoint-500/                   # ✅ 中间 checkpoint (每 500 步)
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer files
│
├── checkpoint-1000/                  # ✅ 中间 checkpoint
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer files
│
├── final/                            # ✅ 最终 checkpoint
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── config.json
│   └── tokenizer files
│
└── hf_merged/                        # ✅ 合并后的完整模型 (NEW!)
    ├── pytorch_model.bin             # 完整模型权重
    ├── config.json
    └── tokenizer files
```

**作用**:
- 保存所有模型 checkpoint
- 使用 /data 目录避免占用 /home 空间
- 自动合并 LoRA adapters 到完整模型

---

### 3️⃣ 结果目录 (Results & History)
**位置**: `/home/jiangli/verl/results/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/`

```
results/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/
├── training_history.json             # ✅ 训练曲线数据 (NEW!)
└── eval_results.json                 # ⏳ 评估结果 (待实现)
```

**training_history.json** 内容:
```json
{
  "steps": [10, 20, 30, 40, ...],
  "loss": [0.1234, 0.1123, 0.1056, 0.0987, ...],
  "kl_loss": [0.0567, 0.0512, 0.0478, 0.0456, ...],
  "learning_rate": [2e-05, 2e-05, 1.98e-05, 1.96e-05, ...]
}
```

**作用**:
- 保存训练曲线数据，方便绘图分析
- 保存评估结果（待实现）

---

## 🔄 训练流程和文件保存时机

### 训练开始时
1. ✅ 创建三个目录：`outputs/`, `/data/.../models/`, `results/`
2. ✅ 保存 `training_config.yaml` 到 `outputs/`

### 训练过程中
3. ✅ 实时写入 `logs/training_*.log` 到 `outputs/logs/`
4. ✅ 每 10 步记录到 `training_history` (内存中)
5. ✅ 每 500 步保存 checkpoint 到 `/data/.../models/checkpoint-{step}/`

### 训练结束时
6. ✅ 保存最终 checkpoint 到 `/data/.../models/final/`
7. ✅ 保存 `training_history.json` 到 `results/`
8. ✅ 合并 LoRA adapters 到 `/data/.../models/hf_merged/` (如果 use_lora=true)

---

## 🆕 新增功能

### 1. 训练历史记录
**代码位置**: `kl_trainer.py` 第 370-378 行

```python
# Record training history
self.training_history["steps"].append(self.global_step)
self.training_history["loss"].append(loss_dict['loss'].item())
self.training_history["kl_loss"].append(loss_dict['kl_loss'].item())
self.training_history["learning_rate"].append(lr)
```

**保存位置**: `results/training_history.json`

### 2. 自动合并 LoRA 模型
**代码位置**: `kl_trainer.py` 第 430-455 行

```python
def merge_lora_model(self):
    """Merge LoRA adapters into base model."""
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(...)

    # Load LoRA model
    model = PeftModel.from_pretrained(base_model, final_checkpoint)

    # Merge and unload
    merged_model = model.merge_and_unload()

    # Save merged model
    merged_model.save_pretrained(merged_dir)
```

**保存位置**: `/data/.../models/hf_merged/`

### 3. 分离的目录结构
- **outputs/**: 轻量级文件（配置、日志）
- **/data/.../models/**: 重量级文件（模型 checkpoint）
- **results/**: 结果文件（训练历史、评估结果）

---

## 📊 使用示例

### 训练命令
```bash
KL_TYPE=reverse \
USE_INITIAL_RESPONSE=false \
bash recipe/open_math_reasoning/run_kl_training.sh
```

### 训练完成后的输出
```
==========================================
Training Complete!
==========================================

Output saved to:
  Logs & Config:     /home/jiangli/verl/outputs/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015
  Model Checkpoints: /data/data/jiangli/models/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015
  Training History:  /home/jiangli/verl/results/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/training_history.json
  Merged Model:      /data/data/jiangli/models/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/hf_merged

To evaluate the trained model:
  bash recipe/open_math_reasoning/eval_model.sh --model_path /data/data/jiangli/models/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/hf_merged
```

### 查看训练历史
```python
import json

with open("results/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/training_history.json") as f:
    history = json.load(f)

import matplotlib.pyplot as plt

plt.plot(history["steps"], history["loss"], label="Total Loss")
plt.plot(history["steps"], history["kl_loss"], label="KL Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.savefig("training_curve.png")
```

---

## ⚙️ 配置选项

### 环境变量
```bash
# 模型保存位置（默认 /data/data/jiangli/models）
MODEL_SAVE_DIR=/data/data/jiangli/models

# 结果保存位置（默认自动生成）
RESULTS_DIR=/home/jiangli/verl/results/my_experiment

# 是否合并 LoRA（默认 true）
SAVE_MERGED_MODEL=true
```

### Python 参数
```bash
python3 recipe/open_math_reasoning/kl_training/run_training.py \
    --model_save_dir /data/data/jiangli/models \
    --results_dir results/my_experiment \
    --save_merged_model true \
    ...
```

---

## 🔍 文件大小估算

### 典型的 Qwen3-1.7B 模型
- **LoRA adapters**: ~50MB (checkpoint-*/final/)
- **完整模型**: ~3.4GB (hf_merged/)
- **训练日志**: ~10MB (logs/)
- **训练历史**: ~1MB (training_history.json)

### 磁盘空间使用
- **/home/jiangli/verl/outputs/**: ~10MB (日志和配置)
- **/data/data/jiangli/models/**: ~3.5GB (模型文件)
- **/home/jiangli/verl/results/**: ~1MB (结果文件)

**总计**: ~3.5GB，其中 3.4GB 在 /data 目录

---

## ✅ 改进总结

### 已实现
1. ✅ 训练历史记录 (`training_history.json`)
2. ✅ 自动合并 LoRA 模型 (`hf_merged/`)
3. ✅ 分离的目录结构（避免 /home 空间占用）
4. ✅ 模型保存到 `/data/data/jiangli/models/`
5. ✅ 结果保存到 `results/`

### 待实现
1. ⏳ 评估结果保存 (`eval_results.json`)
2. ⏳ Wandb 集成（配置已添加，功能待实现）
3. ⏳ 自动评估脚本集成
