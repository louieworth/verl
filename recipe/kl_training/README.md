# KL Divergence Training for Math Reasoning

## 概述

本模块实现了 Token-Level KL Divergence 训练，支持 4 种变体：

| 变体 | KL 方向 | 计算方式 | 特点 |
|------|---------|----------|------|
| 1 | Reverse KL | Monte Carlo | 模式寻求，内存高效 |
| 2 | Reverse KL | Full Vocabulary | 模式寻求，更精确 |
| 3 | Forward KL | Monte Carlo | 分布覆盖，内存高效 |
| 4 | Forward KL | Full Vocabulary | 分布覆盖，更精确 |

---

## 核心算法差异

### Reverse KL vs Forward KL 的本质区别

#### Reverse KL (模式寻求)
```
目标: D_KL[Student || Teacher]
含义: 让 Student 学习 Teacher 的"主要模式"
特点:
- Student 会专注于 Teacher 概率高的区域
- 适合学习"如何生成好的推理"
- 不需要错误的 initial response
```

#### Forward KL (分布覆盖)
```
目标: D_KL[Teacher || Student]
含义: 让 Student 覆盖 Teacher 的"整个分布"
特点:
- Student 会尝试覆盖 Teacher 的所有可能输出
- 适合学习"如何校正错误"
- 需要错误的 initial response 作为上下文
```

---

## Reverse KL Prompt 的两个变体

### Variant 1: Teacher 不看 Initial Response (默认)

**目标**: 学习在只有专家解答指导下如何推理

**Student 看到**:
```
Problem: Find the derivative of f(x)=3x^2+2x-5 at x=2

Please reason step by step, and put your final answer within \boxed{}.
```

**Teacher 看到**:
```
Problem: Find the derivative of f(x)=3x^2+2x-5 at x=2

Here is a reference solution:
First find f'(x) = 6x+2, then evaluate at x=2: f'(2) = 6(2) + 2 = 14

After understanding the reference solution, please try to solve this problem using your own approach below:
Answer:

Please reason step by step, and put your final answer within \boxed{}.
```

**使用场景**:
- 更干净的学习信号
- Teacher 只关注专家解答的推理模式
- 适合学习"如何直接生成高质量推理"

---

### Variant 2: Teacher 看到 Initial Response

**目标**: 学习在看到自己的初始尝试和专家解答后如何改进

**Student 看到**:
```
Problem: Find the derivative of f(x)=3x^2+2x-5 at x=2

Please reason step by step, and put your final answer within \boxed{}.
```

**Teacher 看到**:
```
Problem: Find the derivative of f(x)=3x^2+2x-5 at x=2

Your initial attempt:
f'(x) = 6x + 2, so f'(2) = 12 + 2 = 14... wait, let me recalculate...

Here is a reference solution:
First find f'(x) = 6x+2, then evaluate at x=2: f'(2) = 6(2) + 2 = 14

After understanding the reference solution, please try to solve this problem using your own approach below:
Answer:

Please reason step by step, and put your final answer within \boxed{}.
```

**使用场景**:
- Teacher 能看到模型的初始推理过程
- 学习"如何在看到自己错误后改进"
- 可能提供更丰富的上下文信息

---

## Prompt 设计差异总结

### Reverse KL Variant 1 (无 Initial Response)
- Student 看到: 只有问题
- Teacher 看到: 问题 + 专家解答
- **不包含 initial response**
- 目标: 学习专家的推理模式

### Reverse KL Variant 2 (有 Initial Response)
- Student 看到: 只有问题
- Teacher 看到: 问题 + 初始尝试 + 专家解答
- **包含 initial response**
- 目标: 学习如何从初始尝试改进到专家水平

### Forward KL (校正任务)
- Student 看到: 只有问题
- Teacher 看到: 问题 + 错误回答 + 专家解答 + 校正策略
- **必须包含错误的 initial response**
- 目标: 学习如何校正错误

---

### Forward KL Prompt

**目标**: 学习如何从错误回答校正到正确回答

**Student 看到**:
```
{PROBLEM}

Please reason step by step, and put your final answer within \boxed{}.
```

**Teacher 看到** (完全按照 `stage2_prepare_v2.py`):
```
Your task is to correct your wrong mathematical solution using the expert solution as reference.

**Problem:**
{PROBLEM}

**Your Initial Solution (Wrong):**
{INITIAL_RESPONSE}

**Expert Solution (Correct):**
{EXPERT_SOLUTION}

**Correction Strategy:**
1. First, try to MINIMALLY EDIT your initial solution:
   - Keep your original structure, style, and flow
   - Only change specific wrong steps/numbers/equations
   - Preserve your original wording and explanations where correct

2. If minimal editing is NOT feasible (e.g., fundamental approach error):
   - Then rewrite using the expert solution's approach
   - But still try to maintain your original style and format

**Key Principles:**
- Prefer MINIMAL EDITS over complete rewrites
- Stay as close as possible to your original solution style
- Only use the expert solution to identify and fix specific errors
- Output ONLY the corrected solution, no meta-commentary

Please provide your corrected solution:

Please reason step by step, and put your final answer within \boxed{}.
```

**关键点**:
- Student 只看到问题（学习从头生成）
- Teacher 看到完整的校正上下文（问题 + 错误回答 + 专家解答）
- 包含错误的 initial response
- 学习目标：如何校正错误推理

---

## 为什么 Prompt 不同？

### Reverse KL 的逻辑
1. **训练目标**: 让模型学会"在有专家指导下如何推理"
2. **Teacher 的角色**: 展示"看到专家解答后应该如何生成"
3. **Student 的角色**: 学习模仿 Teacher 的输出分布
4. **不需要错误回答**: 因为目标不是校正，而是直接生成

### Forward KL 的逻辑
1. **训练目标**: 让模型学会"如何校正错误推理"
2. **Teacher 的角色**: 展示"如何从错误校正到正确"
3. **Student 的角色**: 学习覆盖 Teacher 的校正能力
4. **必须有错误回答**: 因为校正任务的输入就是错误回答

---

## 数据流程

### Reverse KL 数据流程
```
Stage 1 生成结果 (stage1_generation.parquet)
    ↓
提取: problem, expert_solution
    ↓
构造 Reverse KL 训练数据:
  - Student input: problem
  - Teacher input: problem + expert_solution
    ↓
训练: Student 学习 Teacher 的输出分布
```

### Forward KL 数据流程
```
Stage 1 生成结果 (stage1_generation.parquet)
    ↓
Stage 2 校正结果 (stage2_correction.parquet)
    ↓
提取: problem, initial_response, expert_solution, corrected_response
    ↓
构造 Forward KL 训练数据:
  - Student input: problem
  - Teacher input: problem + initial_response + expert_solution
  - Target: corrected_response
    ↓
训练: Student 学习生成 corrected_response
```

---

## 使用方法

### Reverse KL Variant 1 训练 (不包含 Initial Response)
```bash
KL_TYPE=reverse \
KL_METHOD=monte_carlo \
USE_INITIAL_RESPONSE=false \
bash recipe/open_math_reasoning/run_kl_training.sh
```

### Reverse KL Variant 2 训练 (包含 Initial Response)
```bash
KL_TYPE=reverse \
KL_METHOD=monte_carlo \
USE_INITIAL_RESPONSE=true \
bash recipe/open_math_reasoning/run_kl_training.sh
```

### Forward KL 训练
```bash
KL_TYPE=forward \
KL_METHOD=monte_carlo \
bash recipe/open_math_reasoning/run_kl_training.sh
```

### 快速测试
```bash
MAX_SAMPLES=100 \
KL_TYPE=reverse \
USE_INITIAL_RESPONSE=false \
bash recipe/open_math_reasoning/run_kl_training.sh
```

---

## 文件结构

```
recipe/open_math_reasoning/kl_training/
├── __init__.py              # 模块初始化
├── kl_utils.py              # KL 计算函数（4种变体）
├── data_utils.py            # 数据准备（Reverse/Forward prompt）
├── config.py                # 配置类
├── kl_trainer.py            # 训练器
├── run_training.py          # 训练入口
├── test_kl_utils.py         # 单元测试
└── README.md                # 本文档
```

---

## 实验建议

### 先测试 Reverse KL
- 更简单，不需要 Stage 2 数据
- 内存效率更高（Monte Carlo）
- 适合快速验证

### 再测试 Forward KL
- 需要先运行 Stage 2 生成校正数据
- 学习"校正"能力
- 可能需要更多训练步数

### 对比实验
```bash
# Reverse KL
KL_TYPE=reverse bash run_kl_training.sh

# Forward KL
KL_TYPE=forward bash run_kl_training.sh

# 评估两者在 AIME24/25 上的表现
```

---

## 参考文献

- **miniLLM**: `LMOps/minillm/minillm/utils.py:get_rev_kl()`
- **Stage 1 Prompt**: `recipe/open_math_reasoning/stage1_prepare.py`
- **Stage 2 Prompt**: `recipe/open_math_reasoning/stage2_prepare_v2.py`
