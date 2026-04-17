# 2026-04-16 更新日志: Forward KL 切换到 full-form + full_vocab + per-token clipping

## 本次要改的内容

三个耦合的改动,核心目标: 让 `forward KL + correction` 真正跑一次 soft-label distillation,而不是只拿到 hard-label SFT 的梯度信号。

1. **Loss 改回 forward KL 完整形式** (MC 路径): `kl = log q - log p`,而不是只保留 `-log p`。
2. **打开 per-token clipping**: `KL_TOKEN_CLIP` 从 `0` 改成 `0.1`。
3. **切换到 full_vocab**: `KL_METHOD` 默认从 `monte_carlo` 改成 `full_vocab`,真正利用 teacher 的整个分布做蒸馏。
4. **Temperature 一个字都不改**: stage1/stage2 采样保持 0.6,training 保持 1.0,eval 保持 0.6。不把 temperature 当"需要对齐"的参数。详见 §3.5 —— 这是今天讨论后修正的理解,之前版本写过"四处统一到 1.0",已作废。

下面分别写要改哪些文件、为什么改、改完之后数值/梯度行为分别是什么。

---

## 1. Loss 改成 forward KL 完整形式

### 1.1 要改的文件

`recipe/kl_training/kl_utils.py:57-95` → `compute_forward_kl_monte_carlo`

### 1.2 现在的实现

```python
def compute_forward_kl_monte_carlo(teacher_logprobs, student_logprobs, mask, reduction="mean"):
    del teacher_logprobs              # 被丢掉
    kl = -student_logprobs            # 只剩 NLL
    ...
```

### 1.3 改成

```python
def compute_forward_kl_monte_carlo(teacher_logprobs, student_logprobs, mask, reduction="mean"):
    # teacher_logprobs 来自 forward_only 的 teacher pass,已经 detached,
    # 保留它不会改变 student 参数的梯度(d/dθ 的 teacher 项 = 0),
    # 但能得到每个 token 真实的 KL 值,用于 metric 和 per-token clip。
    kl = (teacher_logprobs - student_logprobs) * mask
    if reduction == "mean":
        return kl.sum() / mask.sum()
    elif reduction == "sum":
        return kl.sum()
    else:  # none
        return kl
```

### 1.4 梯度等价性验证(为什么这样改不会改变训练结果,只会改变可观测量)

$$
\mathcal{L}_\text{MC} = \mathbb{E}_{y\sim q}[\log q(y) - \log p_\theta(y)]
$$

$$
\nabla_\theta \mathcal{L}_\text{MC} = -\mathbb{E}_{y\sim q}[\nabla_\theta \log p_\theta(y)]
$$

teacher_logprobs 是常数(forward_only, `kl_trainer.py:563-567` 的 teacher engine 已 detach) → 梯度与 `-student_logprobs` 完全一致。

**改动前后的区别**:
- 数值 (loss 显示值): 之前是 NLL,之后是真实 KL 分布值
- Metric: `kl_p50/p95/p99/max` (`kl_trainer.py:372-384`) 从"NLL 分布"变成"真实 KL 分布" → 才能用来调 `kl_token_clip`
- 梯度: 不变
- `clip` 语义变化: 见 §2

### 1.5 副作用 / 注意

- `_summarize_step_output` (`kl_trainer.py:423-447`) 现在计算的 `kl_loss = kl_num / response_tokens` 会变成真实 KL 平均值,不再是 NLL。以前 wandb 里 `train/kl_loss` 大致 ≈ `log(student_perplexity)`,改完不再相等。**这个是我们想要的新行为**,但看 wandb 曲线时要注意 y 轴尺度会跳变。
- `student_perplexity` 独立于 kl_loss 计算(`kl_trainer.py:438-440`),不受影响,仍然是 SFT 意义上的 perplexity。

---

## 2. Per-token KL clipping: `KL_TOKEN_CLIP=0.1`

### 2.1 要改的文件

`recipe/kl_training/run_correction_kl_training.sh:36`

```bash
# 现在
KL_TOKEN_CLIP=${KL_TOKEN_CLIP:-0}       # Per-token KL clip. 0 for forward KL MC (clipping NLL kills SFT signal); ~0.1 for reverse KL.
# 改成
KL_TOKEN_CLIP=${KL_TOKEN_CLIP:-0.1}     # Per-token KL clip (OPSD jsd_token_clip 风格). 现在 forward KL 返回真实 KL 值,clip 作用在 "teacher 比 student 高出异常多" 的 style tokens 上,保护 math token 的梯度。
```

### 2.2 Clip 的语义变化(很关键)

之前(`kl = -student_logprobs`),clamp(max=0.1) 会把 `-log p_student > 0.1` 的位置截掉,也就是**截掉 student 最不确定、最需要学习**的 token → 毁掉 SFT 信号。这就是之前注释里说的 "clipping NLL kills SFT signal",完全正确。

改完之后(`kl = log p_teacher - log p_student`),clamp(max=0.1) 截的是**"teacher 比 student 高出 0.1 nats 以上"**的 token。这些是:
- 异常的 style tokens ("wait", "think", "let me" 之类),teacher 在这些位置很 peaked 而 student 分布平坦 → 单 token KL 爆表
- OOV / rare token 的偶发 spike

OPSD README 指出 style tokens 的 per-token KL 比 math tokens 高 **6-15×**,不 clip 就是这些无信息量的 token 主导梯度。clip 它们后,math token 的梯度才能被看到。

### 2.3 Clip 位置

已经实现在 `kl_trainer.py:386-387`:
```python
if self.config.kl_token_clip and self.config.kl_token_clip > 0:
    kl_per_position = kl_per_position.clamp(max=self.config.kl_token_clip)
```

**工作机理**: `clamp(kl_per_position, max=0.1)`:
- 对 `teacher_logp - student_logp ≤ 0.1` 的位置: loss 保持为 `(teacher_logp - student_logp)`,梯度 = `-∇log p_student`
- 对 `teacher_logp - student_logp > 0.1` 的位置: loss 被 clamp 成常数 `0.1`,梯度 = 0

等价于对极端 KL token 做 0/1 mask,和 OPSD 的 `jsd_token_clip` 设计一致。

### 2.4 阈值选择

- OPSD 默认: `0.05` (full_vocab JSD 下观测到的 math vs style 分水岭)
- 本次先试 **`0.1`** (用户指定,比 OPSD 宽松一倍,保留更多中等幅度的 KL token 进入梯度)
- 如果 wandb 里 `train/clip_frac > 0.2` → 阈值过紧,太多 token 被 mask,考虑调到 `0.15` 或 `0.2`
- 如果 `train/clip_frac < 0.02` → 阈值过松,没 clip 到什么,`kl_p95` / `kl_max` 会很大,考虑收紧到 `0.05`
- 目标区间: `clip_frac ≈ 0.05 - 0.15`

---

## 3. 切换到 `KL_METHOD=full_vocab`

### 3.1 要改的文件

`recipe/kl_training/run_correction_kl_training.sh:34`

```bash
# 现在
KL_METHOD=${KL_METHOD:-"monte_carlo"}  # monte_carlo or full_vocab
# 改成
KL_METHOD=${KL_METHOD:-"full_vocab"}   # full_vocab 提供 soft-label 蒸馏信号,对 reasoning task 收益更大
```

### 3.2 为什么切 full_vocab 是最大的单点提升

MC 和 full_vocab 梯度的信息量差距:

- **MC**: 每个 token 位置只传递 "teacher 采到了 token `y_t`" 这 1 bit 信息。student 梯度 = `-∇log p(y_t)`,其他 152k 词的概率 student 怎么分配都不管。
- **full_vocab**: 每个 token 位置传递 teacher 在整个 vocab 上的分布。student 梯度 = `∇ Σ_v q(v) · (log q(v) - log p(v))`,强制 student 把概率质量分配成 `q(·)` 的形状。

对于数学推理,典型 next-token 分布往往 "top-3 各占 30-50%" (比如 `[", ", " so", " which"]`),MC 只能学到 "top1 是对的,其他是错的" 的 hard label;full_vocab 学到 "top-3 都合理,应该是这个相对比例"。**soft label 的信息带宽大约是 `log2(vocab_size) / 1 ≈ 17×`**,对 reasoning 特别重要。

这也是 OPSD 默认走 full_vocab JSD,而不是 MC 的核心理由。

### 3.3 已经就绪的基础设施(不需要改)

切到 `full_vocab` 触发 `kl_trainer.py` 里几个条件分支:

1. `_common_meta:245`: `return_logits = (kl_method == "full_vocab")` → teacher/student engine 返回 logits 而非 logprobs。✓
2. `_make_student_batch:269-270`: 当 full_vocab 时把 `teacher_logits` 写入 tensordict。✓
3. `_compute_kl_loss:343-355`: 调用 `compute_forward_kl_full_vocab` 路径。✓
4. `compute_forward_kl_full_vocab` (`kl_utils.py:146-190`): 完整实现。已经是 `sum_v q(v) [log q(v) - log p(v)]`,也是 fp32 softmax,数值稳定。✓

### 3.4 显存与吞吐影响 (关键)

从返回 logprobs (per-token 标量) 切到返回 logits (per-token 152k-dim fp32/bf16 向量),teacher/student 的 forward 输出都会暴涨。以 Qwen3-8B (vocab ≈ 151936) + `max_token_len_per_gpu=40960` 估算:

- Teacher logits: `40960 × 151936 × 2 bytes ≈ 12 GB` (bf16)
- Student logits (forward + 保留到 backward): 同量级
- 加上 softmax 中间状态 (fp32): 约 `40960 × 151936 × 4 bytes ≈ 24 GB` per term,虽然可以 fused kernel 省掉

**大概率第一次跑会 OOM**。可选缓解(按顺序试):
1. `MAX_TOKEN_LEN_PER_GPU` 从 40960 降到 20480 (直接减半 activation)
2. 打开 `SP_SIZE=4` 或 `8` (Ulysses SP,把 seq 维切分,logits 被分到多卡) —— 现在是 `SP_SIZE=2`
3. `param_offload=true, optimizer_offload=true` 腾 HBM
4. `use_torch_compile=false` 暂时关掉,编译图在 full_vocab 下可能更激进,不稳

如果这些都不行,再考虑 chunked vocab softmax (`F.cross_entropy` 内核会自动做,但用户代码里是手写 softmax,需要改 `compute_forward_kl_full_vocab` 做 vocab chunking)。先不做,看实测。

### 3.5 Temperature 的角色:不是"对齐采样 T",而是"决定 soft-label 有多 soft"

这一节重新写过(2026-04-16 讨论)。之前的版本写成"要把 training / sampling / eval 四处统一到 T=1.0",这个直觉是错的 —— training temperature 的作用和 sampling temperature 不是同一件事。

#### 3.5.1 Training T 实际在做什么

`compute_forward_kl_full_vocab` (`kl_utils.py:171-180`):
```python
teacher_logits = teacher_logits / temperature
student_logits = student_logits / temperature
teacher_probs = softmax(teacher_logits, dim=-1)
kl = sum_v teacher_probs * (teacher_logprobs - student_logprobs)
```

两边同时除以 T 后再 softmax。T 越小,teacher 分布越 peaked (接近 one-hot);T 越大越 flat。所以 training T 定义的是 **"你希望 student 去拟合的 teacher 分布形状"**,跟"采样时用的 T"在数学上不是同一个对象:
- 采样 T 作用在**一次**采样产生**一个离散 token** `y_1`
- 训练 T 作用在**每个位置**的**整个 vocab 上的连续分布**

`y_1` 永远是 teacher 在某个联合约束 (T=0.6, top_p=0.95, top_k=20) 下的 **合法 sample**,不管 training T 取什么,这个 sample 都在 teacher 支持集里。"对齐 T" 并不能让两个对象变成同一个。

#### 3.5.2 T→0 退化方向

Qwen3-8B 在 reasoning token 上 T=1.0 的 top-10 大致像 `[0.25, 0.18, 0.12, 0.08, 0.06, ...]`,T=0.6 会变成 `[0.48, 0.23, 0.11, 0.05, 0.03, ...]`,top1 概率接近翻倍,长尾被压平。

- **T → 0**: teacher 分布趋于 one-hot,`sum_v q(v)(log q - log p)` 退化成对 top1 token 的 `-log p_student(top1)` → **等价于 hard-label cross entropy = teacher-forced NLL = 之前 MC 实现的形式**
- **T = 0.6**: 已经是"偏 hard-label"的中间态,top1 占到近 50%
- **T = 1.0**: 原生分布,top-10 token 都能贡献有效的 soft 信号
- **T > 1.0** (e.g. 2.0, Hinton KD 传统): 进一步平滑,让 teacher 在 rank 10-50 的 token 上的"暗知识"也能传递

#### 3.5.3 为什么本次更新里 training T 应该 = 1.0 (而不是 0.6)

Full_vocab 切换的核心价值在于 soft-label bandwidth (`~17× over MC`, 见 §3.2)。**如果同时把 training T 设到 0.6,Qwen3-8B 的 teacher 分布会被压成接近 hard-label,full_vocab 的 bandwidth 优势被砍掉大半 —— 等于花 10+ GB 显存买了半个 MC**。性价比很差。

Hinton KD 的传统是"训练 T 大,推理 T 小",OPSD 的 full_vocab JSD 也是训练 T=1.0。本次更新 first run 就跟这个传统走。

#### 3.5.4 结论

| 位置 | 现在 | 本次改动 | 说明 |
|---|---|---|---|
| Stage1 采样 | T=0.6 | **不动** | 采样 T 跟 training T 解耦,保持 0.6 对 rollout 质量更稳 |
| Stage2 采样 | T=0.6 | **不动** | 同上,`y_1` 质量依赖 top_p/top_k/T 联合约束 |
| Training loss | T=1.0 (config 默认) | **不动,仍然 T=1.0** | full_vocab 下 soft-label bandwidth 最大化 |
| Eval | T=0.6 | **不动** | 推理时果断出答案 |

所以本次更新 **temperature 一个字都不改**。之前说的"留作 P2 follow-up 把四处统一"取消,改成: **training T 留作独立的 sweep hyperparameter**,合理的 sweep 区间是 `[0.5, 1.0, 2.0]` 而不是 `[0.6, 1.0]`。

#### 3.5.5 潜在 double-apply 风险

`kl_trainer.py:253-254 / 264-265` 把 `temperature` 塞进 tensordict 传给 verl engine。需要验证的细节:

- **MC 路径**: engine 内部用 T 计算 `log_softmax(logits/T)`,返回 logprobs。`compute_forward_kl_monte_carlo` 不再 apply T。**只除一次,OK**。
- **full_vocab 路径**: engine 返回的是**原始 logits** 还是 **logits/T**? 这决定了 `compute_forward_kl_full_vocab:171-174` 里的 `/temperature` 会不会变成 **double apply** (最终等价于 `logits / T^2`)。

**smoke test 时的验证步骤**:
1. 先用 `TEMPERATURE=1.0` 跑 10 步,记下 `train/kl_loss` 的数量级
2. 换 `TEMPERATURE=2.0` 跑 10 步,如果 kl_loss 是 T=1.0 的 `~1/4` (正确:每个 logit 除以 2,softmax 变更 flat,KL 变小),说明单次 apply
3. 如果 kl_loss 是 `~1/16` 或类似平方级变化,说明 engine 和 `compute_*_full_vocab` 各 apply 了一次 → 需要在 `compute_forward_kl_full_vocab` 里删掉 `/temperature` 或在 engine 那边关掉

如果发现 double apply,修法: 把 `kl_utils.py:compute_forward_kl_full_vocab` 里的 `teacher_logits / temperature, student_logits / temperature` 两行删掉(因为 engine 已经处理了),同时 `reverse_kl_full_vocab` 也要一起改以保持对称。

**本次更新不预先改**,先验证再决定。smoke test 这一步是强制的,不跳过。

---

## 4. 改动 summary 表

| # | 文件 | 行 | 改动 |
|---|---|---|---|
| 1 | `recipe/kl_training/kl_utils.py` | 57-95 | `compute_forward_kl_monte_carlo` 改成返回 `(teacher_logp - student_logp) * mask`,删掉 `del teacher_logprobs` 和 NLL 退化说明 |
| 2 | `recipe/kl_training/run_correction_kl_training.sh` | 36 | `KL_TOKEN_CLIP` 默认 `0` → `0.1` |
| 3 | `recipe/kl_training/run_correction_kl_training.sh` | 34 | `KL_METHOD` 默认 `monte_carlo` → `full_vocab` |

不需要改的文件 (已经支持):
- `recipe/kl_training/kl_trainer.py`: `_common_meta` / `_make_student_batch` / `_compute_kl_loss` 已有 `full_vocab` 条件分支
- `recipe/kl_training/kl_utils.py:compute_forward_kl_full_vocab`: 实现已在,reduction="none" 已支持

---

## 5. 训练前的 sanity checklist

改完之后第一次启动 `run_correction_kl_training.sh` 前要在 wandb 上重点盯以下指标:

### Step 1-10 (最前面的 warmup)
- `train/kl_loss` > 0 且数量级合理(预期 0.5-5 nats 区间;远大于说明 teacher/student 从一开始就偏离很多,大概率是 prompt alignment bug;远小于说明 teacher ≈ student,correction 没给信号)
- `train/kl_p50` < `train/kl_p95` < `train/kl_p99` < `train/kl_max`,且 `p50 < 0.1 < p95` 左右(意味着 clip 作用在分布尾部,不是中位)
- `train/clip_frac` 落在 `0.05 - 0.15` (见 §2.4)
- `train/student_perplexity` 有限值,不是 NaN/Inf
- `train/response_tokens` > 0 (truncation 没把 response 吃光)

### Step 50-100 (稳定之后)
- `train/kl_loss` 缓慢单调下降
- `train/student_perplexity` 缓慢下降
- `train/grad_norm` < `max_grad_norm` (=1.0) 的多数时间 (否则 LR 太大)
- GPU memory 稳定,没有 OOM 或 OOM-adjacent 的警告

### 红灯信号
- `train/kl_loss` 不动或上升 → teacher/student 分布没对齐,检查 §3.5 temperature 或 prompt 构造
- `train/clip_frac > 0.5` → 阈值太紧,上调 `KL_TOKEN_CLIP` 到 0.2-0.3
- `train/clip_frac < 0.01` 且 `kl_p99 > 5` → 阈值太松,下调到 0.05
- OOM → 按 §3.4 顺序降 `max_token_len_per_gpu` / 加 SP / 加 offload

---

## 6. 本次改动不涉及的内容(记录一下,留给下次)

以下项在之前的分析里(`ANALYSIS_forward_correction.md`)提到了,本次**不动**:

- Stage 2 post-rewrite reward filter(需要新增一个 filter 步骤)
- `FORWARD_STAGE2_MODE` 默认切到 `reward0_only`
- Best-of-N rewrite (`rollout.n` 调大)
- Teacher prompt 限长 / truncation 监控
- 多 epoch pipeline curriculum (`TOTAL_EPOCHS`)
- LoRA rank / LR 微调
- Temperature 三处统一

本次只跑 loss / KL 形式这一条"纯蒸馏信号"的改动,方便和之前的 `forward+correction` baseline 做干净对比。如果这一改有明显提升,再叠加上面的 data-quality 改动看增量收益。

---

## 7. 回滚策略

如果切 full_vocab 之后 OOM 实在压不住,或者 loss 行为异常,可以快速回退:
- `KL_METHOD=monte_carlo bash run_correction_kl_training.sh` — 仍然跑 MC,但因为 `kl_utils.py` 已经改,MC 下 loss 值是真实 KL,clip 也有意义,等于只回退到 "MC with real KL metrics + clip",比原来的 NLL 强,而显存和原来一样。

这给了一个中间状态作为 fallback: 真实 KL loss + MC 采样 + per-token clip,全部在原来的显存预算内。

---

## 8. 下一步(文档写完后)

按顺序执行:
1. 修改 `kl_utils.py:compute_forward_kl_monte_carlo` (§1.3)
2. 修改 `run_correction_kl_training.sh:34, 36` (§2.1, §3.1)
3. 启一次短跑 (`MAX_SAMPLES=200`) 做 smoke test,确认 loss / clip / memory 正常
4. 确认 OK 后全量跑一次,结果写入 `results/Qwen3-8B/results.json`,命名 tag 建议: `forward_full_vocab_clip0.1_<date>_epoch1`
5. 对比 `forward_monte_carlo_correction_0414` baseline

---

## 9. TL;DR — 本次改动精简总结

**一句话**: forward KL MC 从"退化成 NLL"恢复成完整形式,同时默认切 `full_vocab` + 打开 per-token clip,让 `forward+correction` 第一次真正跑一次 soft-label distillation。

**实际代码改动(3 处,已全部落地)**:

| # | 文件 | 改动 | 行为变化 |
|---|---|---|---|
| 1 | `recipe/kl_training/kl_utils.py` | `compute_forward_kl_monte_carlo` 从 `kl = -student_logp` 改回 `kl = (teacher_logp - student_logp) * mask` | 梯度**不变**(teacher 项 detached);`train/kl_loss` 从 NLL 变成真实 KL 值;`kl_p50/p95/p99` 变成真实 KL 分位数,`kl_token_clip` 开始有物理意义 |
| 2 | `recipe/kl_training/run_correction_kl_training.sh:34` | `KL_METHOD` 默认 `monte_carlo` → `full_vocab` | teacher/student engine 返回 logits 而不是 logprobs,`compute_forward_kl_full_vocab` 走 `sum_v q(v)(log q - log p)`;soft-label bandwidth 从每 token 1 bit 提到 `~log2(V)` bit;代价:teacher/student logits 各 ~12 GB,可能需要降 `max_token_len_per_gpu` 或加 SP |
| 3 | `recipe/kl_training/run_correction_kl_training.sh:36` | `KL_TOKEN_CLIP` 默认 `0` → `0.1` | 改动 1 之后 clip 的是"teacher 比 student 高出 >0.1 nats"的 style tokens (OPSD 观察 style tokens KL 是 math tokens 的 6-15 倍),保护 math token 梯度不被吞 |

**显式没有改的东西(避免和 distillation signal 改动混淆归因)**:
- Temperature (training=1.0 / sampling=0.6 / eval=0.6 都保留,**不"对齐"**,详见 §3.5)
- Stage 2 reward filter / best-of-N / `reward0_only` 默认
- Teacher prompt 长度 / truncation 策略
- LoRA rank / LR / epoch 数

**下一步**: 先 `MAX_SAMPLES=200` smoke test 验证 kl_loss / clip_frac / 显存 / temperature 是否 double-apply(§3.5.5),通过后全量跑,tag `forward_full_vocab_clip0.1_20260416_epoch1`,对比 `forward_monte_carlo_correction_0414` baseline。

**回滚**: `KL_METHOD=monte_carlo` 一行环境变量即可退回"MC + 真实 KL metrics + clip"的中间态,显存和原 baseline 相同,比原始 NLL 实现严格更强。

# Future
1. 只是用correction 对的，也就是wrong - > correction 这个阶段的数量。