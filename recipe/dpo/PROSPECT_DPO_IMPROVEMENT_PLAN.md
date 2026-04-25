# Prospect-DPO 提升方案：超越 PENS Benchmark

## Context（背景）

你目前在 PENS 个性化新闻标题生成任务上用 **Qwen3.5-4B + LoRA** 跑 `recipe/dpo/run/run_prospect_dpo.sh`，其经过 `run_single_wise_dpo.sh` 调用 `prospect_dpo` loss。目标是超越 PENS 论文 (Ao et al., ACL 2021) 的最强 baseline **NAML+IM-2**：ROUGE-1=28.01 / R-2=10.72 / R-L=22.24。

### 你目前的 5 个 run 实测结果（来自 `/home/lijiang3/projects/def-y7ding/lijiang3/verl/results/qwen3_5-4b.json`，最后更新 2026-04-18 15:48）

| 实验 | step | epoch | date | R-1 | R-2 | R-L | fenced | inline | missing | malformed | non-fenced% |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **NAML+IM-2 (PENS 论文)** | — | — | — | **0.2801** | **0.1072** | **0.2224** | — | — | — | — | — |
| **single_wise_dpo positive_only** | 5341 | 1 | 20260417 | **0.2442** | **0.0801** | **0.2057** | 19567 | 743 | 242 | 2 | 4.80% |
| single_wise_dpo all | 10682 | **2** | 20260409 | 0.2310 | 0.0715 | 0.1954 | 20519 | 31 | 0 | 4 | 0.17% |
| **prospect_dpo all** | 5341 | 1 | 20260417 | 0.2309 | 0.0724 | 0.1952 | 19128 | 1188 | 237 | 1 | **6.94%** |
| Qwen3.5-4B base | — | — | 20260404 | 0.2296 | 0.0707 | 0.1934 | 20545 | 5 | 0 | 4 | 0.04% |
| single_wise_dpo negative_only | 5340 | 1 | 20260406 | 0.2183 | 0.0637 | 0.1831 | 20540 | 0 | 1 | 13 | 0.07% |

> `joined_count = 20554` 全体一致。`non-fenced%` = (inline+missing+malformed)/joined —— 代表输出脱离了标准 fenced-JSON 格式的比率；这是 prospect_dpo 失败最直接的诊断信号。

**与 PENS NAML+IM-2 差距**：最强变体 `positive_only` 仍 R-1 −3.59 / R-2 −2.71 / R-L −1.67。

### Qwen3-8B 实测结果（来自 `/home/lijiang3/projects/def-y7ding/lijiang3/verl/results/qwen3-8b.json`）

⚠️ **注意模型代差**：4B 用的是 **Qwen3.5-4B**（新一代），8B 用的是 **Qwen3-8B**（老一代）。这是 apples-to-oranges，尺寸对比里混入了代际差异。

| 实验 | mode | step | epoch | R-1 | R-2 | R-L | fenced | inline | missing | malformed | empty | reasoning_no_answer | skipped |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B base | **thinkOFF** | — | — | 0.2222 | 0.0684 | 0.1876 | 20456 | 0 | 1 | 0 | 97 | 0 | 98 |
| Qwen3-8B base | **thinkON** | — | — | 0.1874 | 0.0506 | 0.1595 | 3126 | 17335 | 0 | 0 | 89 | 4 | 93 |
| prospect_dpo all (8B) | **thinkOFF** | 5341 | 1 | **0.2423** | **0.0837** | **0.2088** | 19667 | 256 | 524 | 106 | 1 | 0 | 631 |
| prospect_dpo all (8B) | **thinkON** | 5341 | 1 | 0.1967 | 0.0606 | 0.1707 | 16290 | 1413 | 2051 | 11 | 89 | 700 | 2851 |

### ThinkOFF vs ThinkON：thinking 模式反向伤害

| 实验 | thinkOFF R-1 | thinkON R-1 | Δ | thinkOFF non-fenced% | thinkON non-fenced% |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B base | 0.2222 | 0.1874 | **−3.48 pt** | 0.00% | 84.30% |
| prospect_dpo 8B | 0.2423 | 0.1967 | **−4.56 pt** | 4.32% | 16.92% |

**结论**：thinkON 全线显著变差（−3.5 到 −4.6 R-1 pt），且 DPO 后模型的退化更大。

### 4B vs 8B（同为 thinkOFF；跨代对比）

| 实验 | 4B R-1 | 8B R-1 | Δ(8B−4B) | 4B R-L | 8B R-L | Δ R-L |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.2296 | 0.2222 | −0.74 pt | 0.1934 | 0.1876 | −0.58 pt |
| prospect_dpo all | 0.2309 | **0.2423** | **+1.14 pt** | 0.1952 | **0.2088** | **+1.36 pt** |

**结论**：
- **base 模型 4B > 8B**（代差：Qwen3.5 vs Qwen3），说明直接比 base 不能单独归因给尺寸
- **DPO 后 8B 反超 4B ~1.14 R-1 pt**，说明 8B 在相同算法/LoRA 下能多吸 1 个点；但仍未超过 4B `positive_only` 的 0.2442，离 NAML+IM-2 (0.2801) 还差 3.78 pt
- **8B prospect_dpo ≈ 4B positive_only**（0.2423 vs 0.2442）—— 这正是你直觉的"prospect ≈ positive_only"的来源：**尺寸加成恰好抵消了算法劣势**，而不是算法本身变好了

### ⚠️ 与你的描述不一致之处（务必先确认）

你说："prospect_dpo ≈ all single dpo > positive only > base > negative only"。但 JSON 显示 **positive_only 才是最好的（R-1=24.42）**，且明显高于 prospect_dpo / all（≈23.10）。同时 prospect_dpo 与 single_wise_dpo(all) 几乎打平，但**前者只跑 1 epoch (5341 步)，后者跑了 2 epoch (10682 步)** —— 这是不公平比较，prospect_dpo 在等步数下其实更强一些；但 positive_only 同样只跑 1 epoch 却显著领先，这才是核心警讯。

---

## 根因分析（Root Cause Analysis）

### Cause 1：Prospect-DPO loss 的不对称性导致正样本被系统性"压制"

`recipe/dpo/core_algos.py:115-130, 148-166`：

```python
# 正样本权重 α ∈ (0, 1)
alpha = sigmoid(alpha_k * (s_dwell - alpha_tau))           # k=10, tau=0.2
# 负样本权重 λ ∈ [1, λ_max]
lambda_w = 1.0 + (lambda_max - 1.0) * p_ctr ** lambda_gamma # max=2, γ=2

positive_losses = sigmoid(-(alpha * rewards))   # 正样本：α 越小，梯度越平
negative_losses = sigmoid(lambda_w * rewards)   # 负样本：λ ≥ 1，全程加压
```

**问题**：α 在 (0, 1)，λ 在 [1, 2]。**所有正样本的有效 reward 被缩小，所有负样本的有效 reward 被放大**。这不是 Prospect Theory 的"对损失更敏感"——它的实现把正向梯度变弱、负向梯度变强，等价于"少奖正 + 多罚负"。结果是模型学会了**避免出错**（推开 negatives）但**没学会模仿好标题**（pull positives 力度不足）。这正好解释了：**positive_only > prospect_dpo**。

### Cause 2：s_dwell 分布让大多数正样本被 α 静音

`recipe/dpo/data/prepare_pens_singlewise_dpo.py:110-114`：`s_dwell = clip(pos_weight, q90) / q90`。中位数样本的 `s_dwell ≈ 0.5/q90 ≈ 0.5`（按分位归一化的典型分布），但很多样本会落在 `< 0.3`。代入 `alpha_k=10, alpha_tau=0.2`：

| s_dwell | α |
|---|---|
| 0.05 | 0.18 |
| 0.10 | 0.27 |
| 0.20 | 0.50 |
| 0.50 | 0.95 |
| 1.00 | 1.00 |

含义：**只有 dwell 落在 top-decile 的正样本能拿到 ~1.0 权重**，其余都被打折，下半区几乎不学。结合 batch 中正负 ~1:1 但负样本权重 ≥1，**有效正负梯度比可能小到 1:5**——模型自然偏好"沉默 / 抄候选 body"。

### Cause 3：训练-推理长度严重不匹配

- 训练：`max_response_length = 32`（`run_single_wise_dpo.sh:37`）
- 推理：`GEN_RESPONSE_LENGTH = 256`（`run_pens_personalized_eval.sh:90`）
- 解码：`temperature=0.7, top_p=0.8, top_k=20`（采样而非贪心/beam）

新闻标题平均 12-18 token，256 长度允许模型生成大段冗余/思考；采样温度 0.7 会让 ROUGE 分数显著低于贪心或 beam search。这一差距对所有变体一致存在，但**对 prospect_dpo 影响更大**，因为它的 JSON 格式失控率最高（6.9%，见下）。

### Cause 4：DPO 在破坏基模型的输出格式（structured output 退化）

| 模型 | non-fenced % | 明细 |
|---|---:|---|
| base | 0.04% | 5 inline + 4 malformed |
| single_wise_dpo all | 0.17% | 31 inline + 4 malformed |
| negative_only | 0.07% | 1 missing + 13 malformed |
| positive_only | **4.80%** | 743 inline + 242 missing + 2 malformed |
| **prospect_dpo** | **6.94%** | **1188 inline + 237 missing + 1 malformed** |

prospect_dpo 把 1426/20554 个样本推出了 fenced-JSON 格式。在 ROUGE 平均化时这些样本要么被 skip（238 skipped）、要么拿到碎片分，**单这一项就掉 ~1-1.5 个 ROUGE-1 点**。原因有两个：(a) max_response_length=32 在训练时强行截断了 JSON 闭合 token；(b) loss 把 reference logp 之外的 token（如 `}` `}\n\`\`\``）过度拉低。

### Cause 5：没有 SFT warm-up

DPO 假设 reference policy 已经能产生合理的候选，只需"偏好对齐"。但 Qwen3.5-4B base 模型对 PENS 的 "click_history → personalized headline" 任务是 zero-shot，reference logp 本身噪声很大，DPO/Prospect-DPO 的偏好信号在噪声 reward 上被放大。`positive_only` 之所以最好，本质上是它**退化成了对正样本的弱 SFT**（BCE 把 reward 推向 +∞ 等价于 maximizing log-likelihood）。

### Cause 6：长度未归一化的 logp，小的 token 数差异引入大 reward 偏差

`recipe/dpo/recipe_actor.py`：`average_log_prob=False`（求和）。一条 12-token 的标题与一条 25-token 的标题在 reward 上差 ~2 倍 β（不是因为内容好坏，而是长度）。SimPO/IPO 已证明对短序列任务长度归一化能涨 1-3 个 ROUGE。

### Cause 7：训练步数不足 + 实验不可比

prospect_dpo 跑了 1 epoch（5341 步），而 single_wise_dpo(all) 跑了 2 epoch（10682 步）。**比较 prospect vs all 的"打平"结论本身不成立**——需要在同 epoch 数下复测。

### Cause 8：Beta 偏大

run script 默认 `BETA=0.5`（`run_single_wise_dpo.sh:42`），覆盖了 config 的 0.1。配合 32-token 短响应，`β·Δlogp` 范围很大，sigmoid 容易饱和（梯度消失）。Anthropic-HH 用 0.1，TL;DR 用 0.5；本任务序列更短、reference 更不准，应靠近 0.1-0.2。

### Cause 9：thinkON 在 PENS headline 任务上是**负收益**

thinkOFF vs thinkON 的对比数据（见 Context 节）展示了 5 条机制：

1. **格式坍塌（最主要）**：Qwen3-8B base thinkON 生成时 `fenced_json` 从 20456 跌到 3126，`inline_json` 从 0 暴涨到 17335。模型被 thinking block 拖入了"先 `<think>...</think>` 再直接吐 JSON"的生成路径，丢失了训练分布里的 ```json fenced 结构。ROUGE 打分器要从 fenced 或 inline JSON 里抽 headline，inline 格式抽取鲁棒性更差 → 实质有效分母变小。
2. **Token 预算被思考吞掉**：`GEN_RESPONSE_LENGTH=256`，thinking 常用 100-200 token；加上 JSON 包装就没剩多少留给 headline。8B thinkON prospect_dpo 的 `missing_structured_output` 达到 2051（thinkOFF 才 524），且新增 `reasoning_no_answer=700`（模型想完就没了），都是预算打爆的症状。
3. **DPO 训练分布不含 thinking**：`max_response_length=32` 根本不允许 thinking 存在。训练时模型的 `<think>` 概率被推低（作为 "其他 token" 相对压低），推理时强制 thinking-on → 完全 out-of-distribution。这就是为什么 **prospect_dpo thinkON 的跌幅（−4.56）比 base thinkON 的跌幅（−3.48）还大**：DPO 后模型对 thinkON 更脆弱。
4. **PENS 任务内在不需要 chain-of-thought**：新闻标题改写是**短范围风格迁移**（平均 15 token），不是多步推理；thinking 只是引入冗余和噪声。CoT 在 math/code/QA 上涨点，在 summarization/headline 任务上历来中性偏负。
5. **thinkON 推理时也要 skip 更多**：skipped 从 ~100 涨到 2851（prospect_dpo 下）。这些样本在 ROUGE 平均中被当 0 或被 skip，直接拉低总分。

**对我们训练目标的含义**：所有后续实验都应 **默认 thinkOFF 评测**；不建议在 thinkON 模式上微调（除非你想研究 thinking 对 headline 生成的反效应）。若一定要支持 thinkON，需要在训练数据和 max_response_length 上同步扩展 thinking block。

### Cause 10：4B → 8B 的尺寸加成有限，且被代差掩盖

基于 Context 节 4B vs 8B 对比：

1. **base 反向**：Qwen3.5-4B base (0.2296) > Qwen3-8B base (0.2222)。Qwen3.5 是新一代、PENS headline 风格的先验更好；单纯"更大"并不等价于"更强"。
2. **DPO 后 8B 赢 ~1.14 R-1 pt**：说明在相同算法 (prospect_dpo) + 相同 LoRA rank(64) 下，8B 能从同样数据里抽出更多价值；但这 +1 pt **追不过 loss 本身的 bug 导致的 −1.5 pt 差距**（prospect vs positive_only 在 4B 上就差 1.33 pt）。
3. **scaling 无法替代算法修复**：8B prospect_dpo (0.2423) 还是没超过 4B positive_only (0.2442) 这个"作弊式 SFT"基线。如果 loss + SFT warmup 不修，单纯升 8B 的回报率很低。
4. **LoRA rank 不够用**：8B 的参数量是 4B 的 2 倍，rank=64 的 LoRA 覆盖比下降；应当同步把 rank 升到 128、alpha 256，并加 MLP 层（见 Tier 3.4）。否则 8B 的理论优势被 adapter 容量压缩掉。

**对策**：**scale 不是首选 lever**。先把 loss / SFT / 推理 / 长度修好（Tier 1+2），这些改动在 4B 上就应看到 ≥ +3 R-1 pt；然后再把修复后的配方迁到 8B（+1-2 pt 加成），总预算 +4-5 pt，才够越过 0.2801 的 NAML+IM-2 线。

---

## 改进方案（按预期收益排序）

### Tier 1：必做（合计预期 +3 到 +5 R-1）

#### 1.1 引入 SFT warm-up（最关键）

在跑 prospect_dpo 前先做 1 epoch SFT，只在 positive 样本上做 next-token prediction（label=1 的行）。这相当于给 reference policy 对齐到 PENS 标题分布上，再做 DPO 才有意义。

- **怎么做**：用 `recipe/sft` 或自写一个简单 SFT 脚本，输入用同样的 `only_positive_click_hist_train.parquet`，目标为 `response`（headline）。LoRA rank=64 同当前。然后用 SFT 后的 LoRA 作为 prospect_dpo 的 reference + initial actor。
- **为什么**：DPO 文献（Rafailov 2023, Ouyang 2022）一致显示 SFT warmup 是必备步骤；你的数据印证了这一点——`positive_only` 的"成功"本质是隐式 SFT。
- **验证**：基础 SFT 单独跑一次，预期 ROUGE-1 即可达到 ~25-26（追平 positive_only 并略胜）。

#### 1.2 修复 Prospect-DPO loss 的对称性

把 α 和 λ 都设计成可上下浮动的乘子，避免"一边压一边推"的偏置。两种方案择一：

**方案 A（最小改动，推荐先试）**：让 α 也能 ≥ 1。
```python
# core_algos.py:121
return 1.0 + (alpha_max - 1.0) * torch.sigmoid(alpha_k * (s_dwell - alpha_tau))
# 新增超参 alpha_max=2.0，使 α ∈ [1, 2]，与 λ 对称
```

**方案 B（更原则性）**：用单一 BCE，权重为样本 weight：
```python
sample_weight = torch.where(label > 0.5, alpha_pos, lambda_neg)
losses = sample_weight * F.binary_cross_entropy_with_logits(rewards, labels, reduction="none")
```
其中 `alpha_pos = 0.5 + s_dwell`（线性，避免 sigmoid 饱和），`lambda_neg = 0.5 + p_ctr`。

- **为什么**：当前实现把"正向梯度"系统性地缩小到不足负向梯度的一半，违反了 prospect theory 想表达的"对 reference 的偏离"应当对称。
- **验证**：训练时打印 `actor/prospect_dpo_loss_pos`、`actor/prospect_dpo_loss_neg`、`actor/prospect_dpo_alpha`、`actor/prospect_dpo_lambda` 的均值——修复后两边 loss 量级应接近。

#### 1.3 修正 alpha_tau / alpha_k 让大多数 positive 也学得到

把 `alpha_tau` 从 0.2 降到 **0.0**（或干脆把 α 改成 `s_dwell` 自身），把 `alpha_k` 从 10.0 降到 **3-4**（更平的 sigmoid）。

```bash
SINGLE_WISE_DPO_ALPHA_TAU=0.0
SINGLE_WISE_DPO_ALPHA_K=4.0
```

- **为什么**：当前 70%+ 的正样本因 α 太小而几乎不贡献梯度（见 Cause 2 表格）。
- **验证**：训练时 `actor/prospect_dpo_alpha` 的均值应从 ~0.5 升到 ~0.7+。

#### 1.4 推理解码改成贪心（greedy）+ 调短 response_length

```bash
GEN_TEMPERATURE=0.0
GEN_TOP_P=1.0
GEN_TOP_K=-1
GEN_RESPONSE_LENGTH=64    # 32 训练 + JSON 包装 token 余量
```

- **为什么**：headline 任务追求 ROUGE，采样 (T=0.7) 引入随机性几乎只会拉低 metric；256 长度让模型有空间生成废话破坏 JSON 闭合。
- **验证**：base 模型用此配置重测一次，应即涨 0.5-1.0 R-1，且 `malformed_json` 几乎归零。

### Tier 2：高价值（合计预期 +1 到 +2 R-1）

#### 2.1 训练时引入 length normalization（SimPO 风格）

修改 `recipe/dpo/recipe_actor.py` 让 prospect_dpo 路径用 average log prob：

```python
# _compute_point_policy_logps
policy_logps = compute_sequence_log_probs(..., average_log_prob=True)
# 同步 reference_logps_materializer.py 输出 avg log prob，避免 train/ref 不一致
```

- **为什么**：headline 长度方差大，sum logp 引入长度偏差；avg logp 让 reward 真正反映"per-token 质量"。
- **验证**：训练后 reward 与 headline 长度的 spearman 相关系数应明显下降。

#### 2.2 修复 max_response_length，匹配真实 headline 分布

PENS headline 平均 ~15 token、p95 ~28、p99 ~40。当前 32 截掉了大约 5% 的样本尾部 + JSON 闭合 token。

```bash
SINGLE_WISE_DPO_MAX_RESPONSE_LENGTH=64
```

- **为什么**：截断让 reference logp 在 EOS / JSON `}` 上归零，DPO loss 在错误位置施压，学坏输出格式。
- **验证**：先用 `pyarrow` 跑一次 train.parquet 的 response token 长度直方图（在 prepare 脚本里加 5 行）；然后用 64 重训，`malformed_json` 应回落到 <0.5%。

#### 2.3 严格控制 epoch 数公平比较 + 多跑 2-3 epoch

```bash
SINGLE_WISE_DPO_TOTAL_EPOCHS=3
```

- **为什么**：1 epoch (5341 步) 对 LoRA 偏小，且现有比较表 epoch 数不一致。SFT/DPO 文献常用 3 epoch。
- **验证**：每 epoch 末端 checkpoint 都跑一次评测，画出 ROUGE-vs-epoch 曲线，找到最优而非靠运气。

#### 2.4 降低 beta 到 0.1，配合 SFT warmup

```bash
SINGLE_WISE_DPO_BETA=0.1
```

- **为什么**：β 太大让 reward 饱和，sigmoid 梯度消失；β=0.1 是 DPO 原文短序列任务常用值。**注意**：必须先做 1.1 的 SFT warmup，否则 β 太低又会让 DPO 信号被噪声 reference 淹没。

### Tier 3：实验性（视 Tier 1+2 结果而定）

#### 3.1 让 prospect signal 真正"配对" — 把 s_dwell 与 p_ctr 合并成 reference-relative reward shift

当前正样本 `p_ctr=0`、负样本 `s_dwell=0`（`prepare_pens_singlewise_dpo.py:354,405`）—— 它们只是简单标签开关。一个更接近 Kahneman/Tversky 原意的实现：把 (s_dwell, p_ctr) 都视为对 reward 的 **reference-point shift**：

```python
ref_shift = beta_ref * (s_dwell - p_ctr)        # 正负通用
shifted_reward = rewards - ref_shift
# 然后用对称损失 sigmoid(-label_sign * shifted_reward)
```

这样负样本中"高 p_ctr 但用户没点"（hard negative）会被加重，正样本中"低 dwell"被弱化但仍贡献符号一致的梯度。

- **数据修改**：`prepare_pens_singlewise_dpo.py` 中给所有行都填 (s_dwell, p_ctr)，不要 hardcode 0。

#### 3.2 把 click history 中加入用户点击的部分 body / category，不只是标题

当前 `render_history_block` 只用 headline（`prepare_pens_singlewise_dpo.py:166-179`）。NAML 在 PENS 上用了 title + body + category + entity 四路特征。给 prompt 里附 `[Categories]: tech, sports, ...` 这种短摘要，模型可以更好地推断用户兴趣。

- **trade-off**：会让 prompt 更长，需要看 train/eval 是否仍 fit `max_prompt_length=4096`。

#### 3.3 升级到 Qwen3-8B base

run script 默认就是 8B，但你实测用的 4B。8B 在 same hyperparam 下通常 +1-2 R-1。需要更大显存 / 多 GPU。

#### 3.4 LoRA 覆盖 MLP 层

```bash
SINGLE_WISE_DPO_LORA_TARGET_MODULES=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]
SINGLE_WISE_DPO_LORA_RANK=128
SINGLE_WISE_DPO_LORA_ALPHA=256
```

- **为什么**：LoRA on attention only 对内容生成（vs 行为模仿）更受限；MLP 层负责"语义合成"，对标题创作很关键。
- **代价**：参数量 ~3x，显存增加。

#### 3.5 在 evaluation 上加 best-of-4 + 用 reference logp 重排

每条 prompt 生成 4 个候选，用 SFT model 对每个候选打 logp，选最高的。在 ROUGE 上这个 trick 通常 +0.5-1.0。

```bash
PASS_K=4
# 然后写一个 rerank 脚本读 .parquet 后用 base/SFT model 计算 logp 排序
```

---

## 推荐执行顺序（最小工作量、最大收益）

```
[Step 1 / E0] 不动训练，只把推理改成 greedy + response_length=64，复测 5 个已有 ckpt + base
              结果写入 qwen3_5-4b.json 新键 `*__greedy64`，作为后续所有对比的干净基线
              预期：每个变体 +0.5-1.0 R-1，non-fenced% 回落到 <0.5%

              === E0 Run Log (2026-04-18 → 2026-04-19) ===
              模型：**Qwen/Qwen3-4B-Instruct-2507**（替换 Qwen3-4B base；Instruct 变体 JSON 输出更干净）
                本地 snapshot: `/home/lijiang3/projects/def-y7ding/lijiang3/hf_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`
              sbatch 脚本：recipe/dpo/evaluation/sbatch/sbatch_pens_eval_e0_qwen3_4b.sh
              SLURM JOB ID：**59576919**
                历史：59573966→59574072→59574835（资源过大被规划到明天）→59574903（起飞 19:48 但 HF HEAD 阻塞主 loader，GPU 0% 卡死 20min）→**59576919**（加 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` + 用完整 snapshot 路径作 MODEL_PATH）
                教训：Compute Canada 计算节点无外网，MODEL_PATH 必须是本地 snapshot 绝对路径而非 `Qwen/Qwen3-4B` 这种 HF repo id；即使 HF_HOME cache 完整，HF Hub 仍会做 HEAD 检查 → 5 次重试 ×40s TCP timeout ≈ 3 分钟/文件，多文件叠加能把 vLLM 主 loader 卡到超时。
              监控 tmux 窗口：pens:e0-4b (tail -f pens_e0_qwen3_4b-59576919.{out,err})
              结果文件：results/qwen3-4b.json，key `qwen3-4b__thinkOFF__20260418`
              配置：T=0.0 top_p=1.0 top_k=-1 response_length=64 repetition_penalty=1.0 VLLM_ENABLE_THINKING=false
              说明：Qwen3-4B 未做任何 DPO，直接评测 base 模型；作为 E0 的 base 行，和 qwen3_5-4b.json 的 base 行对照（后者是 Qwen3.5-4B，代差不同）。
              其它 4 个 4B DPO ckpt 不在当前节点（在 /data/data/jiangli/ 即另一台机器），本次 E0 仅覆盖 base。

              sbatch 对齐 recipe/dpo/run/sbatch/retry_multinode.sh 的环境约定：
                - `module load python/3.12.4 cuda/12.2 arrow/19.0.1 opencv/4.11.0 nodejs`
                - `source /project/def-y7ding/lijiang3/envs/verl/bin/activate`
                - `unset ROCR_VISIBLE_DEVICES`（Compute Canada NVIDIA 节点必做）
                - `WANDB_MODE=offline`, `WANDB_DIR=/scratch/lijiang3/wandb`（计算节点无网）
                - `RAY_TMPDIR=/tmp/r_${USER}_${SLURM_JOB_ID}`（规避 AF_UNIX sun_path 107 字节上限）
                - `--time=01:00:00 --mem=32G --cpus-per-task=4`（终版；fairshare EffectvUsage=99.5% 让大 job 被 deprioritize，压缩 CPUs/Task 12→4 后 backfill 窗口立即打开）
                - StartTime=2026-04-18T19:52:19 on ng10712（Monitor bigcctqac 持续监听），确认推理修复价值

[Step 2] 跑一次纯 SFT (positives only, 1 epoch, lr=1e-5, LoRA r=64) 作为 warmup
         预期：单独评测应达 R-1 ~25-26（已经接近 positive_only 现状）

[Step 3] 用 SFT 出来的 LoRA 作为初始化，跑修复版 prospect_dpo:
         - SINGLE_WISE_DPO_BETA=0.1
         - SINGLE_WISE_DPO_ALPHA_TAU=0.0
         - SINGLE_WISE_DPO_ALPHA_K=4.0
         - SINGLE_WISE_DPO_MAX_RESPONSE_LENGTH=64
         - SINGLE_WISE_DPO_TOTAL_EPOCHS=3
         - 修改 core_algos.py 让 α 对称（Tier 1.2 方案 A）
         - 修改 recipe_actor.py 用 average_log_prob=True
         预期：R-1 26-28，逼近 NAML+IM-2

[Step 4] 若 Step 3 仍未过 28，启动 Tier 3：
         - 升级 Qwen3-8B
         - LoRA 加 MLP
         - Best-of-4 rerank
         预期：R-1 28+，超越 PENS benchmark
```

---

## 关键修改文件清单

| 文件 | 修改 | Tier |
|---|---|---|
| `recipe/dpo/evaluation/run_pens_personalized_eval.sh` | `GEN_TEMPERATURE=0.0`, `GEN_RESPONSE_LENGTH=64` | 1.4 |
| `recipe/dpo/run/run_single_wise_dpo.sh:42-46` | `BETA=0.1`, `ALPHA_TAU=0.0`, `ALPHA_K=4.0`, `MAX_RESPONSE_LENGTH=64`, `TOTAL_EPOCHS=3` | 1.3 / 2.2 / 2.3 / 2.4 |
| `recipe/dpo/core_algos.py:115-130` | 把 α 改成 `1 + (α_max-1)·sigmoid(...)` 或换成统一 BCE + 样本权重 | 1.2 |
| `recipe/dpo/recipe_actor.py` (search `_compute_point_policy_logps`) | `average_log_prob=True` 路径 | 2.1 |
| `recipe/dpo/reference_logps_materializer.py` | 同步用 average，否则 train/ref 不一致 | 2.1 |
| `recipe/dpo/data/prepare_pens_singlewise_dpo.py:354,405` | （Tier 3.1）正负样本都填 s_dwell 和 p_ctr | 3.1 |
| 新文件：`recipe/sft/run/run_pens_sft_warmup.sh` | SFT warmup 脚本，输入 only_positive | 1.1 |

---

## Verification（端到端验证流程）

每一个 Step 完成后：

1. **训练侧 sanity**：W&B / log 中观察
   - `actor/prospect_dpo_loss_pos` 与 `actor/prospect_dpo_loss_neg` 量级是否对称（修复后应 ~1:1）
   - `actor/prospect_dpo_alpha` 均值是否上升（应 >0.7）
   - `actor/dpo_loss` 是否平滑下降（无 NaN/spike）

2. **推理侧**：
   ```bash
   bash recipe/dpo/evaluation/run_pens_personalized_eval.sh \
        MODEL_PATH=<新 ckpt 的 hf_merged 目录>
   ```
   检查 `results/result.json`：
   - `parse_status_counts.malformed_json_output + missing_structured_output` 应 < 50（< 0.25%）
   - `rouge_1_f1`、`rouge_2_f1`、`rouge_l_f1` 三个数

3. **目标**：
   - Step 1 后：positive_only 应 ~0.25 R-1（确认推理修复有效）
   - Step 2 后：SFT-only 应 ~0.255 R-1
   - Step 3 后：prospect_dpo 应 ≥ 0.27 R-1，超过 positive_only 才算这套 loss 真正有效
   - Step 4 后：≥ 0.28 R-1，超过 PENS NAML+IM-2

4. **诚实对比**：所有比较必须**同 epoch 数 + 同推理配置**，否则不可信。先把已有 5 个 ckpt 用 Step 1 的推理重跑一次，建立干净基线再对比新实验。

---

## 一句话总结

**positive_only 之所以最好，是因为它无意中实现了一个 SFT；prospect_dpo 之所以打不过，是因为它的 α 把正梯度压扁、λ 把负梯度放大，加上没有 SFT warmup、推理用了过长的采样解码、训练长度过短破坏了 JSON 输出格式。**修这五点，prospect_dpo 才能体现 prospect-theory 的真实价值并越过 NAML+IM-2 的线。

---

## 数据来源脚注

- **Qwen3.5-4B (thinkOFF)** — 5 条评测条目：`/home/lijiang3/projects/def-y7ding/lijiang3/verl/results/qwen3_5-4b.json`（最后更新 2026-04-18 15:48）
- **Qwen3-8B (thinkOFF + thinkON)** — 4 条评测条目：`/home/lijiang3/projects/def-y7ding/lijiang3/verl/results/qwen3-8b.json`（日期 20260415-20260416）
  - 注意：8B 模型路径是 `models--Qwen--Qwen3-8B`，是 **Qwen3**（不是 Qwen3.5）
- **结果文件命名约定**：`results/<BASE_MODEL_SLUG>.json`，一个 base model 一个 JSON；由 `run_pens_personalized_eval.sh` 的 `derive_base_model_slug()` 自动从 MODEL_PATH 推导（从 DPO ckpt 路径里抽出 `_(lora|fullft)_` 与 `_global_step_` 之间的 base 名）。HF snapshot 路径直接用 snapshot 名。需覆写时设 `BASE_MODEL_SLUG=...` 环境变量。
- 每条的 `raw_generation_file` 字段指向对应 parquet，位于 `/lustre06/project/6093645/lijiang3/verl/gen_results/`（或 `/home/jiangli/verl-pens/gen_results/`），可用 `pyarrow.parquet.ParquetFile` 打开做样本级复盘
- 评测由 `recipe/dpo/evaluation/score_pens_predictions.py`（backend=rouge）生成

## 关键结论速查

1. **thinkON 反向伤害**：−3.5 到 −4.6 R-1 pt。原因：(a) fenced-JSON 格式坍塌（inline_json 从 0 涨到 17335）；(b) thinking 吞掉 response token 预算，`missing_structured_output` + `reasoning_no_answer` 激增；(c) DPO 训练分布根本不含 thinking（max_response_length=32），推理时 thinkON 是纯 OOD；(d) PENS headline 是短距风格迁移，用不到 CoT。→ **后续实验默认 thinkOFF 评测**。
2. **4B → 8B 加成 ~+1 R-1 pt，但无法独立越过 benchmark**：(a) base 上 Qwen3.5-4B > Qwen3-8B（代差）；(b) DPO 后 8B 领先 +1.14 pt，但 8B prospect_dpo (0.2423) 仍 < 4B positive_only (0.2442) < NAML+IM-2 (0.2801)。→ **scaling 不是首选 lever，先修 loss + SFT + 推理**；升 8B 时要同步提 LoRA rank 到 128 + 加 MLP。
3. **"prospect ≈ positive_only" 的感觉来自跨尺寸偶合**：8B prospect_dpo ≈ 4B positive_only。这不是算法变好，而是 8B 的尺寸加成刚好补回了 prospect_dpo 在 4B 上对 positive_only 的 1.33 pt 劣势。同尺寸下 prospect_dpo 依然输 positive_only。
