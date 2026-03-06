# 评估功能说明

## 📊 eval_results.json 功能

训练结束后，可以自动运行评估并保存结果到 `eval_results.json`。

---

## 🎯 功能概述

### 1. 自动评估（可选）
训练完成后自动在指定数据集上评估模型性能。

### 2. 保存评估结果
将评估结果保存为 JSON 格式，包含：
- 每个数据集的准确率
- 训练配置信息
- 时间戳

---

## 📁 文件位置

```
results/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/
├── training_history.json    # 训练曲线数据
└── eval_results.json         # ✅ 评估结果 (NEW!)
```

---

## 📄 eval_results.json 格式

```json
{
  "results": {
    "aime24": 0.45,
    "aime25": 0.42,
    "math500": 0.67,
    "hmmt24": 0.38,
    "usamo24": 0.25
  },
  "config": {
    "kl_type": "reverse",
    "kl_method": "monte_carlo",
    "kl_coef": 0.1,
    "model_path": "Qwen/Qwen3-1.7B",
    "use_lora": true,
    "lora_rank": 64,
    "learning_rate": 2e-05,
    "train_batch_size": 96,
    "total_epochs": 1
  }
}
```

---

## 🚀 使用方法

### 方法 1: 训练时自动评估

```bash
KL_TYPE=reverse \
USE_INITIAL_RESPONSE=false \
RUN_EVAL_AFTER_TRAINING=true \
EVAL_DATASETS="aime24,aime25,math500" \
bash recipe/open_math_reasoning/run_kl_training.sh
```

**输出**:
```
==========================================
Training Complete!
==========================================

Output saved to:
  Logs & Config:     /home/jiangli/verl/outputs/...
  Model Checkpoints: /data/data/jiangli/models/...
  Training History:  /home/jiangli/verl/results/.../training_history.json
  Eval Results:      /home/jiangli/verl/results/.../eval_results.json  ✅
  Merged Model:      /data/data/jiangli/models/.../hf_merged
```

### 方法 2: 训练后手动评估

```bash
# 1. 先训练（不自动评估）
KL_TYPE=reverse \
RUN_EVAL_AFTER_TRAINING=false \
bash recipe/open_math_reasoning/run_kl_training.sh

# 2. 训练完成后手动评估
bash recipe/open_math_reasoning/eval_model.sh \
    --model_path /data/data/jiangli/models/.../hf_merged \
    --eval_datasets aime24,aime25,math500 \
    --output_path results/.../eval_results.json
```

---

## ⚙️ 配置选项

### 环境变量

```bash
# 是否在训练后自动运行评估（默认 false）
RUN_EVAL_AFTER_TRAINING=true

# 评估数据集列表（逗号分隔）
EVAL_DATASETS="aime24,aime25,math500,hmmt24,usamo24"
```

### Python 参数

```python
# config.py
eval_datasets: list = ["aime24", "aime25", "math500"]
run_eval_after_training: bool = False
```

### 命令行参数

```bash
python3 recipe/open_math_reasoning/kl_training/run_training.py \
    --run_eval_after_training true \
    --eval_datasets "aime24,aime25,math500" \
    ...
```

---

## 📊 支持的评估数据集

### 默认数据集
- `aime24` - AIME 2024 (30题)
- `aime25` - AIME 2025 (30题)
- `math500` - MATH500 (500题)

### 可选数据集
- `amc23` - AMC 2023
- `hmmt24` - HMMT 2024 (30题)
- `hmmt25` - HMMT 2025 (30题)
- `usamo24` - USAMO 2024 (证明题)
- `usamo25` - USAMO 2025 (证明题)

---

## 🔧 实现细节

### 1. 评估流程

```python
# kl_trainer.py
def run_evaluation(self):
    """Run evaluation on specified datasets and save results."""

    # 1. 确定评估模型路径
    if self.config.use_lora and self.config.save_merged_model:
        eval_model_path = os.path.join(self.config.model_save_dir, "hf_merged")
    else:
        eval_model_path = os.path.join(self.config.model_save_dir, "final")

    # 2. 对每个数据集运行评估
    eval_results = {}
    for dataset in self.config.eval_datasets:
        logger.info(f"Evaluating on {dataset}...")
        # 调用评估函数
        accuracy = evaluate_on_dataset(eval_model_path, dataset)
        eval_results[dataset] = accuracy

    # 3. 保存评估结果
    save_eval_results(
        results=eval_results,
        output_path=os.path.join(self.config.results_dir, "eval_results.json"),
        config=training_config_dict,
    )
```

### 2. 评估工具函数

```python
# eval_utils.py
def save_eval_results(results, output_path, config=None):
    """Save evaluation results to JSON file."""
    output = {
        "results": results,
        "config": config,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
```

---

## 📈 使用评估结果

### 查看结果

```python
import json

# 加载评估结果
with open("results/.../eval_results.json") as f:
    eval_data = json.load(f)

print("Evaluation Results:")
for dataset, accuracy in eval_data["results"].items():
    print(f"  {dataset}: {accuracy:.2%}")
```

### 对比不同实验

```python
import json
import pandas as pd

# 加载多个实验的结果
experiments = [
    "results/exp1/eval_results.json",
    "results/exp2/eval_results.json",
    "results/exp3/eval_results.json",
]

data = []
for exp_path in experiments:
    with open(exp_path) as f:
        exp_data = json.load(f)

    row = {
        "experiment": exp_path.split("/")[1],
        "kl_type": exp_data["config"]["kl_type"],
        "kl_method": exp_data["config"]["kl_method"],
        **exp_data["results"]
    }
    data.append(row)

df = pd.DataFrame(data)
print(df)
```

### 绘制对比图

```python
import matplotlib.pyplot as plt

# 对比不同 KL 变体的效果
fig, ax = plt.subplots(figsize=(10, 6))

experiments = ["Reverse KL V1", "Reverse KL V2", "Forward KL"]
datasets = ["aime24", "aime25", "math500"]

# 假设数据
data = {
    "Reverse KL V1": [0.45, 0.42, 0.67],
    "Reverse KL V2": [0.48, 0.44, 0.69],
    "Forward KL": [0.43, 0.40, 0.65],
}

x = range(len(datasets))
width = 0.25

for i, (exp, scores) in enumerate(data.items()):
    ax.bar([xi + i*width for xi in x], scores, width, label=exp)

ax.set_xlabel("Dataset")
ax.set_ylabel("Accuracy")
ax.set_title("KL Training Variants Comparison")
ax.set_xticks([xi + width for xi in x])
ax.set_xticklabels(datasets)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("kl_variants_comparison.png")
```

---

## ⚠️ 注意事项

### 1. 评估时间
- 自动评估会增加训练脚本的总运行时间
- 建议在小规模测试时关闭自动评估
- 完整训练时可以开启自动评估

### 2. 评估实现
当前实现是**占位符**，需要根据实际评估 pipeline 进行适配：

```python
# 需要实现的部分
def evaluate_on_dataset(model_path, dataset):
    """
    实际的评估逻辑，需要：
    1. 加载模型
    2. 在数据集上生成回答
    3. 计算准确率
    4. 返回结果
    """
    # TODO: 实现实际的评估逻辑
    pass
```

### 3. 集成现有评估脚本
可以调用现有的 `run_eval.sh`:

```python
import subprocess

def evaluate_on_dataset(model_path, dataset):
    """Call existing evaluation script."""
    cmd = [
        "bash",
        "recipe/open_math_reasoning/run_eval.sh",
        "--model_path", model_path,
        "--dataset", dataset,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # 解析输出获取准确率
    accuracy = parse_accuracy_from_output(result.stdout)
    return accuracy
```

---

## ✅ 完整的输出文件

训练完成后（开启自动评估）：

```
# 1. 输出目录
outputs/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/
├── training_config.yaml
└── logs/training_*.log

# 2. 模型目录
/data/data/jiangli/models/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/
├── checkpoint-500/
├── final/
└── hf_merged/

# 3. 结果目录
results/Qwen3-1.7B_kl_reverse_monte_carlo_20260306_223015/
├── training_history.json    # ✅ 训练曲线
└── eval_results.json         # ✅ 评估结果
```

---

## 🎯 总结

### 已实现
1. ✅ 评估配置选项 (`run_eval_after_training`, `eval_datasets`)
2. ✅ 评估结果保存 (`eval_results.json`)
3. ✅ 训练配置记录（包含在评估结果中）
4. ✅ Shell 脚本集成

### 待完善
1. ⏳ 实际的评估逻辑（需要根据现有 pipeline 适配）
2. ⏳ 与现有评估脚本的集成
3. ⏳ 更详细的评估指标（不仅是准确率）

### 使用建议
1. 小规模测试时关闭自动评估 (`RUN_EVAL_AFTER_TRAINING=false`)
2. 完整训练时开启自动评估 (`RUN_EVAL_AFTER_TRAINING=true`)
3. 根据实际需求选择评估数据集
