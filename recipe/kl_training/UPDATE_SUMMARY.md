# 更新总结 - KL Training 功能增强

## 🎯 更新目标
1. ✅ 添加训练历史记录 (`training_history.json`)
2. ✅ 自动合并 LoRA 模型 (`hf_merged/`)
3. ✅ 模型保存到 `/data/data/jiangli/models/` (避免 /home 空间占用)
4. ✅ 结果保存到 `results/` 目录

---

## 📝 修改的文件

### 1. `config.py`
**新增配置项**:
```python
model_save_dir: str = "/data/data/jiangli/models"  # 模型保存位置
results_dir: str = "results/kl_training"           # 结果保存位置
save_merged_model: bool = True                     # 是否合并 LoRA
```

### 2. `kl_trainer.py`
**新增功能**:

#### a) 训练历史记录
```python
self.training_history = {
    "steps": [],
    "loss": [],
    "kl_loss": [],
    "learning_rate": [],
}

# 每 logging_steps 记录一次
self.training_history["steps"].append(self.global_step)
self.training_history["loss"].append(loss_dict['loss'].item())
self.training_history["kl_loss"].append(loss_dict['kl_loss'].item())
self.training_history["learning_rate"].append(lr)
```

#### b) 保存训练历史
```python
def save_training_history(self):
    """Save training history to JSON."""
    history_file = os.path.join(self.config.results_dir, "training_history.json")
    with open(history_file, "w") as f:
        json.dump(self.training_history, f, indent=2)
```

#### c) 合并 LoRA 模型
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

#### d) 修改 checkpoint 保存位置
```python
# 从 output_dir 改为 model_save_dir
save_dir = os.path.join(self.config.model_save_dir, "final")
```

### 3. `run_training.py`
**新增参数**:
```python
parser.add_argument("--model_save_dir", type=str, default="/data/data/jiangli/models")
parser.add_argument("--results_dir", type=str, default="")
parser.add_argument("--save_merged_model", type=lambda x: x.lower() == "true", default=True)
```

### 4. `run_kl_training.sh`
**新增环境变量**:
```bash
MODEL_SAVE_DIR=${MODEL_SAVE_DIR:-"/data/data/jiangli/models"}
RESULTS_DIR=${RESULTS_DIR:-""}
SAVE_MERGED_MODEL=${SAVE_MERGED_MODEL:-"true"}
```

**自动生成路径**:
```bash
MODEL_SAVE_DIR="$MODEL_SAVE_DIR/${MODEL_NAME}_kl_${KL_TYPE}_${KL_METHOD}_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$VERL_ROOT/results/${MODEL_NAME}_kl_${KL_TYPE}_${KL_METHOD}_$(date +%Y%m%d_%H%M%S)"
```

**创建目录**:
```bash
mkdir -p "$MODEL_SAVE_DIR"
mkdir -p "$RESULTS_DIR"
```

---

## 📁 新的文件结构

### 训练完成后的完整输出

```
# 1. 输出目录 (轻量级: ~10MB)
/home/jiangli/verl/outputs/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/
├── training_config.yaml
└── logs/
    └── training_20260306_223015.log

# 2. 模型目录 (重量级: ~3.5GB)
/data/data/jiangli/models/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/
├── checkpoint-500/          # 中间 checkpoint
├── checkpoint-1000/
├── final/                   # 最终 checkpoint (LoRA adapters)
└── hf_merged/              # ✅ 合并后的完整模型 (NEW!)

# 3. 结果目录 (轻量级: ~1MB)
/home/jiangli/verl/results/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/
└── training_history.json   # ✅ 训练曲线数据 (NEW!)
```

---

## 🔄 执行流程

### 训练开始
1. 创建三个目录: `outputs/`, `/data/.../models/`, `results/`
2. 保存 `training_config.yaml`

### 训练过程
3. 实时写入训练日志
4. 每 10 步记录训练历史 (内存)
5. 每 500 步保存 checkpoint 到 `/data/.../models/`

### 训练结束
6. 保存最终 checkpoint 到 `/data/.../models/final/`
7. 保存 `training_history.json` 到 `results/`
8. 合并 LoRA 到 `/data/.../models/hf_merged/`

---

## 📊 使用示例

### 运行训练
```bash
KL_TYPE=reverse \
USE_INITIAL_RESPONSE=false \
bash recipe/open_math_reasoning/run_kl_training.sh
```

### 训练完成输出
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
import matplotlib.pyplot as plt

# 加载训练历史
with open("results/.../training_history.json") as f:
    history = json.load(f)

# 绘制训练曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history["steps"], history["loss"], label="Total Loss")
plt.plot(history["steps"], history["kl_loss"], label="KL Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss")

plt.subplot(1, 2, 2)
plt.plot(history["steps"], history["learning_rate"])
plt.xlabel("Steps")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")

plt.tight_layout()
plt.savefig("training_curves.png")
```

---

## ✅ 功能对比

### 更新前
```
outputs/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/
├── training_config.yaml
├── logs/training_*.log
├── checkpoint-500/          # ❌ 在 outputs/ 下
├── checkpoint-1000/
└── final/                   # ❌ 在 outputs/ 下
```

**问题**:
- ❌ 所有文件都在 /home 下，占用空间大
- ❌ 没有训练历史记录
- ❌ 没有合并后的完整模型
- ❌ 没有结果目录

### 更新后
```
# 轻量级文件在 /home
outputs/.../
├── training_config.yaml
└── logs/training_*.log

# 重量级文件在 /data
/data/.../models/.../
├── checkpoint-500/          # ✅ 在 /data 下
├── checkpoint-1000/
├── final/
└── hf_merged/              # ✅ 新增

# 结果文件在 results/
results/.../
└── training_history.json   # ✅ 新增
```

**优势**:
- ✅ 模型文件保存在 /data，不占用 /home 空间
- ✅ 自动记录训练历史，方便分析
- ✅ 自动合并 LoRA，直接可用
- ✅ 清晰的目录结构

---

## 🔍 代码验证

### 语法检查
```bash
# 所有文件语法正确
python3 -m py_compile recipe/open_math_reasoning/kl_training/config.py
python3 -m py_compile recipe/open_math_reasoning/kl_training/kl_trainer.py
python3 -m py_compile recipe/open_math_reasoning/kl_training/run_training.py
bash -n recipe/open_math_reasoning/run_kl_training.sh
```

### 关键功能
1. ✅ 训练历史记录 - `kl_trainer.py:370-378`
2. ✅ 保存训练历史 - `kl_trainer.py:420-428`
3. ✅ 合并 LoRA 模型 - `kl_trainer.py:430-455`
4. ✅ 模型保存路径 - `kl_trainer.py:458-485`
5. ✅ 目录创建 - `run_kl_training.sh:164-166`

---

## 📚 相关文档

1. `CODE_REVIEW.md` - 完整的代码 review 报告
2. `SCRIPT_LOGIC.md` - 脚本逻辑和文件保存说明
3. `FILE_STRUCTURE.md` - 更新后的文件结构详解
4. `README.md` - 使用说明和 prompt 设计

---

## 🎉 总结

所有功能已成功添加：

1. ✅ **训练历史记录**: 每 10 步记录 loss、kl_loss、learning_rate
2. ✅ **自动合并模型**: 训练结束后自动合并 LoRA adapters
3. ✅ **分离目录结构**:
   - 轻量级文件 → /home (outputs/, results/)
   - 重量级文件 → /data (models/)
4. ✅ **完整的输出信息**: 训练结束后显示所有文件位置

代码已准备好进行实际训练测试！
