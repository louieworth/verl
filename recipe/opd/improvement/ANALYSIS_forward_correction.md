# 在保留 two-stage correction 方法的前提下,提升 `forward KL + correction` 的效果

## Context

`results/Qwen3-8B/results.json` 显示 `kl_forward_monte_carlo_correction` 相对 Qwen3-8B base 在多数 benchmark 上持平或回退:

| benchmark | base | reverse+rewrite | forward+correction(0414) | forward+correction(20260415) |
|---|---|---|---|---|
| aime24 | 0.700 | 0.700 | 0.700 | 0.700 |
| aime25 | 0.533 | 0.533 | 0.533 | 0.533 |
| math500 | 0.914 | 0.912 | 0.906 | 0.906 |
| hmmt25 | 0.367 | 0.467 | 0.533 | 0.367 |
| beyondaime | 0.470 | 0.470 | 0.430 | 0.410 |
| amobench | 0.205 | 0.154 | 0.179 | 0.154 |
| gsm8k | 0.318 | 0.328 | 0.302 | — |

用户的方法设计(固定,不动):
1. **Stage 1**: student 对 bare problem 采样初稿 `y_0`。
2. **Stage 2 (correction)**: teacher (= student with LoRA 关闭 / 同一份权重) 读到 `problem + y_0 + expert_cot`,产出"修正版" `y_1`。
3. **Training**: student 在 bare `problem` 条件下对 `y_1` 做分布匹配 / SFT,目标是把"在参考答案引导下能产出好答案的能力"蒸馏回"没有参考答案时也能独立产出好答案"。

这个设计本身是合理的 —— 比单纯 rewrite(没有 initial_response)理论上更有优势,因为 teacher 看到了 student 自己的 failure mode,"修正"比"重写"更贴近 student 当前分布。下面分析**为什么这个方法在当前实现下没有跑出优势**,并给出**保留 two-stage correction 结构**的改进方案。

---

## 1. 根本问题:`forward KL + monte_carlo` 当前实现根本没有在蒸馏

`recipe/opd/kl_utils.py:85-95`:

```python
def compute_forward_kl_monte_carlo(teacher_logprobs, student_logprobs, mask, ...):
    del teacher_logprobs        # 被丢掉
    kl = -student_logprobs      # 只剩 NLL
```

当前 `KL_TYPE=forward, KL_METHOD=monte_carlo` 的 loss 数学上就是:

$$
L = \frac{1}{\sum m_t} \sum_t m_t \cdot (-\log p_\text{student}(y^{(1)}_t \mid y^{(1)}_{<t}, \text{problem}))
$$

也就是 **纯 teacher-forced NLL**。`kl_trainer.py:562-567` 里 teacher forward pass 跑了,产出的 `teacher_log_probs` 被读进 `_compute_kl_loss` (`kl_trainer.py:321`),然后在 `compute_forward_kl_monte_carlo` 里 **`del teacher_logprobs`** 丢掉。**teacher 完全没有进入 loss,只是白白烧 FLOPs。**

**这意味着:**
- 你以为的"forward KL 的 mode-covering 特性"根本没在生效 —— mode-covering 只存在于 `full_vocab` 路径 (`kl_utils.py:146-190`),你没走。
- 你以为的"teacher 的 correction 信号帮助 student",实际上只有 target tokens `y_1` 被保留,teacher 在这些 token 上的**分布**(那才是真正的 knowledge)完全没传给 student。
- reverse KL MC 相反 —— `compute_reverse_kl_monte_carlo` 真实地用了 `teacher_logprobs`(`kl_utils.py:42-47`),是一个真正的 off-policy distillation loss。讽刺的是 `forward+correction` 反而退化成 baseline SFT。

**这是 #1 优先级要修的 bug**,而且修起来不改动方法的任何假设。

---

## 2. Stage 2 目标 tokens 的质量没被控制

`run_correction_kl_training.sh:38` 默认 `FORWARD_STAGE2_MODE="rewrite_all"` → `stage2_prepare_rewrite_all.py` 不做任何 reward 过滤 (只 normalize reward 但不 filter),对所有 stage1 sample 都生成 rewrite 并全部送去训练。

结果: student 学的 target `y_1` 里混着:
- teacher 在 `expert_cot` 帮助下依然答错的(即使读了参考答案也没抄对)
- rewrite 过程中"退化"的(stage1 本来答对,rewrite 后被改错)
- 风格上向参考答案靠拢但语义错的

你在把这些错答案 SFT 进去,模型被教歪。`amobench 0.205 → 0.179, gsm8k 0.318 → 0.302, beyondaime 0.470 → 0.430` 这种轻度回退和"一部分 target 是错的"完全一致。

**注意**: pipeline 里 `stage1_prepare.py` / `stage2_prepare_rewrite_all.py` 都已经把 reward 字段写入 `extra_info.reward`,并且 eval 路径有 `amobench_parser_reward` 这类 reward 解析器。**过滤所需要的信息都已经在 parquet 里了**,只差加一步 `.filter(lambda x: x["extra_info"]["reward"] == 1)`。

---

## 3. Stage 2 target 只采一次,n=1,没有 best-of-N

`run_correction_kl_training.sh:383`: `actor_rollout_ref.rollout.n=1`。每个问题只生成一条 rewrite。对难题,一次 rewrite 的正确率本身就不高 —— 想象 stage1 在该题上失败,teacher = 同一个 student 在更丰富的 context 下只做一次 rewrite,这次 rewrite 也未必对。

**后果**: 训练 target 的正确率 ≈ teacher 一次 pass@1。没有 best-of-N 过滤,你实际上在用一个噪声很大的 teacher signal。

---

## 4. `rewrite_all` vs `reward0_only` 的选择

你 pipeline 里已经实现了两种模式:
- `rewrite_all`: 对所有题生成 rewrite (`stage2_prepare_rewrite_all.py`)
- `reward0_only`: 只对 stage1 答错的题生成 rewrite (`stage2_prepare_rewrite_reward0.py`)

**默认用的是 `rewrite_all`**。但从方法论上,correction 的信号价值应该主要来自 stage1 失败的样本 —— stage1 本来就做对了的题,rewrite 不会教给模型任何新东西(除非 rewrite 比原答案更优雅,但那是风格学习,不是推理学习),反而有可能把对的搞成错的。

---

## 5. Prompt 长度压垮 training signal

`run_correction_kl_training.sh:74`: `MAX_LENGTH=34816`

`data_utils.py:119-126`:
```python
full_ids = (prompt_ids + response_ids)[: self.max_length]
```

- `student_prompt` = bare `problem` (~200-500 tokens)
- `teacher_prompt` = `problem + initial_response + expert_cot` + instruction = 轻松 5-15k tokens
- `response y_1` = 数千 tokens (math reasoning trace)

如果 `teacher_prompt + y_1 > max_length`,teacher 侧的 `y_1` 会被从尾巴截短。`_prepare_item:143-151` 里 `common_len = min(student_resp, teacher_resp)` + `torch.equal` check 会**静默把 response 缩到 teacher 那一端的长度**,student 侧的 response 也跟着缩。

**结果**: 长且硬的题被"静默 truncate"掉大量 response tokens,loss mask 里剩不了几个有效 token,loss 梯度基本靠短题撑。这解释了为什么 beyondaime/amobench (长 reasoning) 回退,而 math500 (较短) 只是轻微回退。

---

## 6. 训练时 teacher prompt 和 stage2 生成时的 prompt 不一致?

`data_utils.py:221-226` 构建 training 时的 `teacher_prompt` 用的是 `build_teacher_prompt(..., use_initial_response=self.use_initial_response)`,会按 `correction` 模式拼 `problem + initial_response + expert_cot`。这**和 stage2 生成 `y_1` 时的 prompt 一致**(`run_correction_kl_training.sh:361-364` → `stage2_prepare_rewrite_all.py:73-78`),所以 teacher logprobs 是在生成分布下 well-defined 的。这里没错。

**但是 `initial_response` 要求被写回 parquet 的 `extra_info.initial_response` 字段**(`data_utils.py:203`),如果 stage2 parquet 里这个字段丢了或者是空,training 会 raise (`data_utils.py:215-218`)。需要确认: `stage2_prepare_rewrite_all.py:66-68` 确实写了 `extra_info["initial_response"] = initial_response`,OK 没问题。但**如果 pipeline 多 epoch 运行,epoch N+1 的 stage2 parquet 的 `initial_response` 字段必须是 epoch N+1 的 stage1 输出**,不能重用旧 epoch 的。看 `run_correction_kl_training.sh:154-189` 每个 epoch 都重新写入自己的 `epoch${N}/` 目录 → 没问题,但`DATA_PATH` override 存在时要小心。

---

## 7. 其他次要问题

### 7.1 `kl_token_clip=0`

`run_correction_kl_training.sh:36`: 注释说 "clipping NLL kills SFT signal,所以 forward MC 下设 0"。
这个 reasoning 在当前 MC=NLL 的实现下是对的,但**一旦切到 full_vocab forward KL (见 §1 修复),必须重新打开 clip ≈ 0.05**。否则 OPSD 观察到的现象会复现: style tokens ("wait", "think", "let me") 的 KL 比 math token 高 6-15 倍,loss 完全被 style token 主导,math 正确性学不到。

### 7.2 Temperature 不一致

- 生成阶段 T=0.6 (`run_correction_kl_training.sh:328,374`)
- Training loss T=1.0 (`kl_trainer.py:254, 265`)
- Eval T=0.6

MC 路径下无所谓 (因为 teacher dist 根本不参与 loss)。切到 full_vocab 后,teacher 在 T=1 下的分布 ≠ 采样时 T=0.6 下的分布 —— 你蒸馏的是一个和生成过程不对齐的 target distribution。**建议统一到 T=1.0**,对所有阶段。

### 7.3 teacher = student + LoRA?

`kl_trainer.py:231-233`: 当 `teacher_model_path` 为空时 teacher 读 student_model_path。**重要**: 如果 LoRA 开着,trainable=False 的 teacher worker 还会不会 load LoRA adapter? 看 `_build_model_config:164-167`: `lora_rank=self.config.lora_rank if trainable and self.config.use_lora else 0`,teacher 的 `trainable=False` → lora_rank=0。**OK,teacher 是 base 权重,student 是 base + LoRA**。所以训练过程中 teacher logits 是固定的 base 分布,student 偏离 base 越多,两者 gap 越大。这正是 distillation 想要的 —— 只是需要 teacher logits 真的进 loss (见 §1)。

### 7.4 LoRA rank / LR 太保守

rank 64 / alpha 128 / lr 2e-5 对 8B 模型 + 只跑 1 个 outer epoch 的情况偏保守,配合上述信号质量问题 → 很可能模型几乎没移动(aime24/25 三个 run 分数一字不差,0.700 / 0.533,这基本就是 "LoRA 梯度太小或 target 太噪,权重没变多少")。

---

## 8. 改进方案(按收益 / 工程量 排序,全部在 two-stage correction 框架内)

### P0 — 让 `forward + correction` 真的在做蒸馏

**改动**: `recipe/opd/kl_utils.py:85-95` 和 shell 默认值。

**路线 A(最小改动,推荐先跑)**: 让 `compute_forward_kl_monte_carlo` 真正用 teacher_logprobs。数学上,当 target tokens 来自 teacher(你这里是 correction rewrite),forward KL MC 的无偏估计是:

$$
\hat L_\text{fwd-KL} = \frac{1}{\sum m} \sum_t m_t \cdot (\log p_\text{teacher}(y_t) - \log p_\text{student}(y_t))
$$

teacher 项对 student 参数梯度是 0,所以优化上等价于 `-log p_student`。**但是作为 metric 和 per-token clip 的基础,必须保留 teacher 项**,否则你看到的 "kl loss" 就是 NLL,无法区分"模型学不动"和"teacher 本身就很难"。并且有 teacher 项后,`kl_token_clip` 才有意义 —— 可以 clip 住那些 teacher 比 student 高很多的异常 token。

修改后:
```python
def compute_forward_kl_monte_carlo(teacher_logprobs, student_logprobs, mask, reduction="mean"):
    kl = (teacher_logprobs - student_logprobs) * mask  # shape [B, T]
    # 注意: teacher 项对梯度无贡献,但参与数值/clip
    if reduction == "none":
        return kl
    ...
```

**路线 B(真正的 mode-covering,推荐最终用)**: 直接切 `KL_METHOD=full_vocab`,走 `compute_forward_kl_full_vocab` (`kl_utils.py:146-190`)。teacher logits 在每个位置对所有 vocab 计算 `sum_v q(v) [log q(v) - log p(v)]`,这才是真正的 mode-covering forward KL,也是你方法论上想要的东西。代价: 显存 + 时间 × vocab_size / 1,但有 `max_token_len_per_gpu` dynamic batching,实测可以跑。

**配合**: `KL_TOKEN_CLIP=0.05` 必须打开(见 §7.1)。`TEMPERATURE=1.0` 三处统一。

这一项是**单点收益最大的修复**。

### P1 — 清理 Stage 2 target 信号质量

**P1.1 Reward filter stage2 targets**

`stage2_prepare_rewrite_all.py` 末尾,map 之后 filter:

```python
ds_rewrite = ds_rewrite.filter(
    lambda x: _normalize_reward_value((x.get("extra_info") or {}).get("reward", 0)) == 1
)
```

**但是**: stage2 prepare 是在 rewrite 生成**之前**跑的,它只有 stage1 的 reward。你需要的是 **rewrite 之后对 `y_1` 重新评分,然后 filter**。也就是在 `run_correction_kl_training.sh:368-386` 的 stage2 生成之后、training 之前,插入一个 scoring + filter 步骤:

1. 用 `amobench_parser_reward` / `compute_score.py` 对 stage2 输出的 `y_1` 评分。
2. 过滤出 reward==1 的那部分,写成新的 parquet (e.g. `deepscaleR_stage2_${PROMPT_MODE_TAG}_responses_filtered.parquet`)。
3. 把 training `DATA_PATH` 指向 filtered 版本。

关键代码: `recipe/open_math_reasoning/build_reward_filtered_dataset.py` 已经存在(从 `ls` 看到),大概率就是做这件事的,需要读一下直接复用。

**P1.2 切 `FORWARD_STAGE2_MODE=reward0_only` 作为默认**

`run_correction_kl_training.sh:38` 把默认从 `rewrite_all` 改成 `reward0_only`。理由见 §4。配合 P1.1 的 post-rewrite filter,最终 training data 是:

> stage1 答错的题 ∩ stage2 correction rewrite 答对的题

这才是真正"有教学价值"的 correction sample —— 模型原来不会,看了 expert_cot 之后会了。

**P1.3 Best-of-N rewrite**

`run_correction_kl_training.sh:383` 把 `actor_rollout_ref.rollout.n=1` 调到 `n=4` 或 `n=8`。配合 P1.1 filter,只保留每道题 reward==1 的 rewrite 中 log-likelihood 最高的一条(或随机一条)。这能显著提升 stage2 target 的 pass@1,但会放大 stage2 生成耗时 4-8 倍。

工程上: 参考 `verl.trainer.main_generation_server` 的 `rollout.n` 参数,已经支持多采样。后续 best-of-N 筛选在 filter 脚本里做。

### P2 — 修掉 length truncation 导致的 training signal 损失

**P2.1 Teacher prompt 瘦身**

`teacher_prompt` 里最贵的是 `initial_response` (可以几千 token),但它的信息量很低 —— teacher 已经在 stage2 生成 `y_1` 的时候 condition 过 `initial_response` 了,training 时 teacher 再重新 score `y_1` 其实不需要 `initial_response`,**只需要 `problem + expert_cot`**,因为 teacher 的作用此时是给 `y_1` 打分数(计算 logprobs / logits),scoring prompt 和 generation prompt 保持一致才能得到"正确"的 teacher distribution。

**权衡**:
- 如果 scoring prompt 去掉 `initial_response`,teacher logits 就不再是"产生 `y_1` 时的 teacher 分布",数学上不匹配。
- 如果保留,就会被截断,大量 token 被吃掉。

**推荐**: 保留数学一致性,把 `initial_response` 留在 teacher prompt,但**限长**: 在 `data_utils.py` 构造 teacher prompt 时把 `initial_response` 截到 ≤ 2048 token,`expert_cot` 截到 ≤ 3072 token。stage2 生成时也保持同样的截断策略一致。修改点:

- `data_utils.py:build_teacher_prompt` 里加 `max_initial_len, max_expert_len` 参数
- `stage2_prepare_rewrite_all.py` / `stage2_prepare_rewrite_reward0.py` 里相应截断
- 两处用同一个 helper,保证 scoring 和 generation 看到的 teacher prompt 完全一致

**P2.2 选择性放宽 `MAX_LENGTH`**

现在 `MAX_LENGTH=34816`,`max_token_len_per_gpu=40960`。够,但问题是 student 侧 `max_length` 也是 34816,而 student prompt 只有几百 token,所以 student 可以容纳很长的 `y_1`。真正的瓶颈是 teacher 侧。如果 P2.1 做完 teacher 侧 prompt ≤ 5k,那 `y_1` 能在 teacher 侧保留 ~30k token,不会被截 —— 这就够了,不需要进一步改 `MAX_LENGTH`。

**P2.3 对 length 做监控**

在 `data_utils.py:_prepare_item` 加 metric: 记录 `common_len / student_response_len` 的比例,wandb 报 p50/p95。如果 p95 < 1.0,说明很多样本被截,立刻回到 P2.1 调参。

### P3 — 多轮 pipeline epoch,构造 curriculum

`run_correction_kl_training.sh:51` `TOTAL_EPOCHS=1` 改 2-3。配合 `reward0_only` + filter:

- Epoch 1: 基于 base Qwen3-8B 的 stage1 失败题目训 correction
- Epoch 2: 基于 epoch1 微调后模型重跑 stage1 → 失败题目会变 "更难" (容易的已经学会了)→ stage2 rewrite on harder problems → 更高价值的 correction signal
- Epoch 3: 同理

这是你 pipeline 框架原生支持的 (`resolve_epoch_model_path` 已经实现了 epoch N+1 读 epoch N merged model)。理论上应该比 1 epoch 强很多,但前提是 epoch 1 的单轮信号是**干净的** —— 所以必须先做完 P0 + P1,不然 epoch 1 的错误信号会被 epoch 2 放大。

### P4 — 超参小调

- `LORA_RANK=128, LORA_ALPHA=256`: 给模型更多容量吸收 correction signal。配合 cleaner signal,不会过拟合。
- `LEARNING_RATE=1e-5`: 更稳,避免 style-shift 导致的回退。
- `TRAIN_EPOCHS_PER_ROUND=2`: 每个 pipeline epoch 的 inner epoch 从 1 提到 2,配合 clean signal 下安全。
- `WARMUP_RATIO=0.05`: 现在 0.1 偏高,corpus 小时 warmup 占比太多。
- `kl_p95` / `kl_max` 已经在 wandb 里记录了 (`kl_trainer.py:599-614`),切 full_vocab 后重点盯这两个指标调 `kl_token_clip`。

### P5 — 方法论级的小扩展(可选,不破坏 two-stage 结构)

这些是在"保持 stage1 → stage2 correction → forward KL distill"整体不变的前提下,可选的增量改进:

**P5.1 在 loss 里加一个 student prompt consistency term**

核心 gap: target `y_1` 来自 teacher 读 `problem + init + expert_cot` 条件,但 student training 时读 bare `problem`。这个 gap 是方法的**内在特征**,无法消除 —— 整个 two-stage correction 的动机就是"把在 hint 条件下的能力蒸馏到 bare 条件下"。你能做的是让蒸馏信号更 robust:

在 loss 里加一个 auxiliary SFT loss 用 **teacher 自己的 prompt (带 hint)** 作为 student 的 input:

$$
L_\text{total} = L_\text{fwd-KL}(\text{student}|\text{bare}, y_1) + \lambda \cdot L_\text{SFT}(\text{student}|\text{hint}, y_1)
$$

第二项让 student 在"我看到了 hint"条件下也能复现 `y_1`(这比第一项容易),起到 regularizer 的作用: 保证 student 不会因为过度压缩 hint→bare 的 gap 而破坏掉它自己读 hint 时的能力。λ ≈ 0.1-0.3。

工程上: 扩展 `_make_student_batch` 产生两组 `input_ids`,loss 加权求和。改动中等。

**P5.2 Token-level curriculum masking**

stage2 `y_1` 里的 token 分两类:
- "reasoning 类"(数学推导,答案)→ 高价值
- "style 类"(Let me reason..., Okay so..., Let's see) → 低价值,容易 memorize,容易过拟合

可以用 teacher entropy 做 soft mask: teacher 在该位置熵越低(越确定),权重越高。这给真正"教学点"更大梯度。
实现: 在 `_compute_kl_loss` 里 per-token 再乘一个 `weight = f(teacher_entropy)`。teacher_entropy 从 full_vocab 路径直接算出来。配合 P0 路线 B。

**P5.3 Teacher 打分是否需要"无 hint" 作 baseline**

如果想要更强的信号,stage2 后可以额外计算 `logp_teacher_no_hint(y_1 | problem)` —— 即 teacher 在 bare prompt 下对 `y_1` 的 logprob。这给你一个衡量 "该 token 有多依赖 hint" 的分数: 当 `logp_teacher_with_hint - logp_teacher_no_hint` 很大时,这个 token 就是"hint 的价值所在"。可以用它给 loss 加权,把学习重点放在真正由 expert_cot 带来的 token 上。

工程上这比 P0-P4 贵得多,放在 P5 最后作为 follow-up。

---

## 9. 建议实验顺序

1. **Step 1 (~1 天)**: 修 `kl_utils.compute_forward_kl_monte_carlo` 把 teacher 项加回去(路线 A) + 把 `KL_TOKEN_CLIP=0.05` 打开 + temperature 三处统一到 1.0。重跑一轮,对比 baseline。
   - 预期: 如果 correction target 质量 OK,应该能看到 0.5-1% 级别的提升。如果看不到,说明问题不在 loss 而在 data 质量,进 Step 2。
2. **Step 2 (~半天)**: 加 stage2 post-rewrite reward filter (复用 `build_reward_filtered_dataset.py`),默认切到 `reward0_only`。重跑。
   - 预期: 这一步应该是最大的单点提升,因为 signal 从 "50% 是错的" 变成 "接近 100% 是对的"。
3. **Step 3 (~半天)**: 把 `rollout.n` 调到 4 做 best-of-N rewrite,再 filter。重跑。
   - 预期: 进一步 +0.5-1%,代价是 stage2 生成时间 4x。
4. **Step 4 (~1 天)**: 切到 `KL_METHOD=full_vocab` (路线 B),调 `kl_token_clip` 到让 `kl_p95 / kl_max < 2.0` 左右。重跑。
   - 预期: mode-covering 的真正收益在这一步才显现,尤其对 "teacher 分布比较 peaked" 的 math token。
5. **Step 5 (~1 天)**: 开 2-3 个 pipeline epoch (`TOTAL_EPOCHS=3`)。重跑。
   - 预期: curriculum 带来的提升,不过对 LoRA 训练要监控是否 drift 过多。
6. **可选 Step 6**: P4 超参,P5 方法论扩展,看 1-5 之后瓶颈在哪决定。

每一步跑完都记得 append 到 `results/Qwen3-8B/results.json` 以便对比,并重点看 **math500 / beyondaime / amobench / gsm8k** (aime24/aime25 对小变化不敏感,不要只看这两个)。

---

## 10. 关键文件 & 修改点 summary

| 改动 | 文件 | 行 |
|---|---|---|
| **[P0]** forward MC 加回 teacher 项 | `recipe/opd/kl_utils.py` | 57-95 |
| **[P0]** shell 默认 `KL_TOKEN_CLIP=0.05, TEMPERATURE=1` | `recipe/opd/run_correction_kl_training.sh` | 33-36 |
| **[P0 alt]** 切 full_vocab | `run_correction_kl_training.sh` env + pass through | 34 |
| **[P1.1]** stage2 post-rewrite reward filter | 新脚本或复用 `recipe/open_math_reasoning/build_reward_filtered_dataset.py`;shell 里插入 filter 步骤 | `run_correction_kl_training.sh:368-386` 之后 |
| **[P1.2]** 默认 reward0_only | `run_correction_kl_training.sh` | 38 |
| **[P1.3]** best-of-N rewrite | `run_correction_kl_training.sh` | 383 (`rollout.n`) |
| **[P2.1]** teacher prompt 限长 (initial_response / expert_cot) | `recipe/opd/data_utils.py:build_teacher_prompt` 以及 `stage2_prepare_rewrite_all.py` 保持一致 | 32-73 / 16-44 |
| **[P2.3]** length truncation 监控 | `recipe/opd/data_utils.py:_prepare_item` | 137-171 |
| **[P3]** 多 epoch curriculum | `run_correction_kl_training.sh` | 51 |
| **[P4]** 超参 | `run_correction_kl_training.sh` | 44-54 |
| **[P5.1]** 双 student prompt aux loss (可选) | `recipe/opd/kl_trainer.py:_make_student_batch / _compute_kl_loss` | 259-421 |

## 11. 验证

- **Unit**: `recipe/opd/test_kl_utils.py` 补充 test case: 验证修复后 forward MC 的数值 = `(teacher_logp - student_logp).mean(mask)`,且梯度 = `-student_logp` 梯度 (teacher 项 detach 后梯度为 0)。
- **Data sanity** (每次改完 stage2 filter / best-of-N): 读 filtered parquet,抽样 10 条 target `y_1`,人工确认答案正确 + 推理连贯。再跑 `amobench_parser_reward` 在 filtered set 上,reward=1 占比应该 ≈ 1.0。
- **Training sanity** (每次 loss 改动):
  - `train/kl_loss` 应该从初始 > 0 单调下降
  - `train/kl_p95 / kl_max` 切到 full_vocab 后应该稳定在 clip 附近,`train/clip_frac` < 0.1
  - `train/student_perplexity` 对 target 缓慢下降(不应该爆炸)
  - `train/response_tokens` 应该接近 dataset 均值(如果远小于,说明 truncation 问题还没解决)
- **End-to-end**: 每个 Step 跑完 `run_correction_kl_training.sh` → `run_eval_suite.py` → 读 `results/Qwen3-8B/results.json`,对比上一步。重点看非饱和 benchmark: math500 / beyondaime / amobench / gsm8k / hmmt25。
