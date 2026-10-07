# PIPO：后续实验与论文任务

## 2026-10-07：Topic 分析与后续验证

- [x] 在唯一指定的 eval_final 上比较 15 个类别的 PIFO (all)、PIFO (+)、SFT 与 Base；计算三项 ROUGE、配对 crossed bootstrap、类别等权汇总、文章等权及逐篇删除敏感性。
- [x] 核查 Music、Kids、Travel 和 Video 的已保存措辞；使用 vivid-figures-skill 和 pastel-girl 绘制横向双柱图，正文 subsection 放到 Ablation Studies 前，完整分析放入 appendix。代码位于 `paper_plots/code/analyze_topic_performance.py` 和 `paper_plots/code/generate_topic_performance.py`；本次未编译论文。
- [ ] 在下方既有多 seed 受控矩阵完成后，用共同评分协议检验 topic × method 差异，尤其核验 Music 的较大增益是否跨 seed 保持。当前区间仅反映固定模型的 user/article 重采样。
- [ ] 后续确认性评估增加 Kids/Video 的独立文章与读者覆盖，并冻结类别划分；当前 Kids 的增益主要来自单篇文章，Video 只有四篇文章。此项需要额外预留数据，不纳入本次唯一 eval_final 的结果。
- [ ] 对 Travel/Video 的低增益和损失进行独立事实支持、内容重点与参考代表性盲评，区分真实生成问题与词面 ROUGE 对改写的敏感性；复用下方既有盲评任务，不将措辞诊断当作因果机制。

## 2026-10-07：个性化案例展示待补

- [ ] 在 `Validating Personalization Through Reader Matching` 中补充同篇文章的不同读者案例，展示共享文章、各自点击历史、人工参考标题、模型输出与 reader matching margins；重建对应附录，说明案例选择、实际 prompt 和额外案例。使用唯一指定的 `eval_final.parquet` 和已保存结果，并披露案例的筛选规则。

当前正文的具体案例、案例图片和后续讨论已替换为 TODO；对应的 `Same-Article Case Construction and Prompt Details` 附录已删除。已有素材和复现代码仍位于 [personalization_cases](paper_plots/personalization_cases/README.md)，供后续整理。下方 2026-10-06 条目保留为历史完成记录。

## 2026-10-06：同篇文章 A/B 案例与改写距离分析

- [x] 明确 Vanilla 是共享的文章原标题；Base 是使用各自历史生成标题的未适配模型。
- [x] 从指定 eval_final 的保存输出筛选三篇优势案例：正文 N20015，附录 N83082、N49846。两位用户均提高 own-reference 分数，且 joint headline 均更接近自己的作者；披露结果筛选，不将案例增益当作总体效果。
- [x] 用实际 canonical prompt 生成六条 GPT-5.6 Luna 案例标题，保存模型 ID、原始输出、prompt hash 和 CLI receipts；不调用 API key，不根据分数重写或重新生成。
- [x] GPT-6.1 Sol 仅看 30 条点击历史描述主题并选两个支撑标题；图中 history 最多两行，完整历史和 prompt 另存。
- [x] 安装并使用 FigGenie skill 输出三张左右 A/B 图，沿用 pastel-girl；使用 vivid 完整 dot-CI 源码绘制改写距离诊断。单独图件导出 PDF/SVG/PNG，当前编辑任务未编译论文。
- [x] 全部 3,655 篇多用户文章的 Vanilla/Base 改写距离分组、类别调整斜率和 2,000 次 crossed bootstrap；目前不支持“改写差距越大，joint matching 增益越大”。
- [ ] 完成下述既有 E1 history-swap / empty-history 推理，检验案例差异是否依赖正确用户历史。当前交换 reference 的 matching 不能替代这项实验。
- [ ] 独立核验历史主题描述、参考标题代表性、案例事实支持和语义 matching；Sol 主题描述不是偏好构念验证。
- [ ] 当前六条 GPT-5.6 Luna 只用于定性展示。若扩展总体比较，先冻结全 eval 的生成/解析/失败重试与评分协议，并计入全部 18,635 contexts，不能使用三篇筛选案例宣称总体领先。
- [ ] 已启动的 GPT-6 Luna 全量生成完成后，按原保存协议核对覆盖、空输出和 prompt hashes，再做统一总体评分；本轮未启动额外全量生成或提交 sbatch。

本轮源码、输入、输出、图件与复现说明位于 [personalization_cases](paper_plots/personalization_cases/README.md)。以下清单保留此前分析及待跑实验，历史编译记录不表示当前论文 PDF 已更新。

更新：2026-10-06。此前的离线分析只使用已保存输出，没有训练或提交 sbatch。新增 GPT-6 Luna 标题生成由用户另行授权，通过 ChatGPT 登录的 Codex CLI 执行；完整评分尚未完成，其余后续实验仍为待办。

## 新增：GPT-6 Luna 外部模型基线

- [x] 确认 CLI 已通过 ChatGPT 登录，模型目录支持 `gpt-6-luna`；没有使用 API key。
- [x] 完成前四条盲生成验证，保留原始标题、输入 hash、CLI 事件与 token usage；第 4 条曾遇临时 model-capacity 错误，重试成功。
- [x] 导出唯一 eval_final 的全部 18,635 个原始 prompts，生成输入不包含 gold titles 或 PIPO 输出。
- [x] 实现逐条独立会话、断点续跑、失败停止与全量完成后自动评分；不提交 sbatch。
- [ ] 完成全量生成。实时进度与停止原因以 [status.json](paper_plots/gpt6_luna_eval/status.json) 为准；仅临时 model-capacity 错误最多重试 5 次，每次等待 30 秒（含首次请求最多尝试 6 次），其他额度、认证或推理错误立即停止。失败请求不会被当作空标题。
- [ ] 完成与 Base、SFT、PIPO (+)、PIPO (all) 的配对 ROUGE、E1 matching、交叉 bootstrap 区间与描述性 category 分析。
- [ ] 全量结果完成后，再决定是否写入论文及按既有 pastel-girl 色系画图。当前没有修改或编译论文。

流程说明与恢复命令：[paper_plots/gpt6_luna_eval/README.md](paper_plots/gpt6_luna_eval/README.md)。使用名称 “GPT-6 Luna (Codex, zero-shot)”；CLI 系统指令和未开放的解码设置使其不能被表述为与 PIPO 完全匹配的推理协议。它是外部模型对比，不能单独识别 PIPO 训练机制。

## 已完成的分析

- [x] 仅使用 /scratch/l/luli/data/pens/pens_eval/eval_final.parquet，按 user_id/news_id 对齐 Base、SFT、PIPO (+)、PIPO (all)。
- [x] 验证 canonical prompt/reference；排除额外行，保留四模型共有的 95 个空输出；重新计算全分母 ROUGE。
- [x] E1：同篇文章的双向 correct/swapped matching，3,655 篇多用户文章、46,053 对用户；2,000 次交叉 user/article bootstrap。
- [x] 用户增益与损失分布、15 个 category 的区间与 FDR、相似历史敏感性。
- [x] 30 条实际展示历史的 lexical concentration、distinctiveness、history/body cosine；不把完整 metadata 历史长度当作 prompt 长度。
- [x] 七项语言特征的整体均值与 article-centered matching；重复标题、distinct-2、词汇集中度、数字支持筛查。
- [x] 1,266 篇文章的 body-supported numeric-focus proxy；该指标没有语义或事实性验证。
- [x] 实际 sampled training article 的正/负分支覆盖，以及供给分数与权重的动态范围。
- [x] 25 个真实保存 checkpoint 的 matching 轨迹；5 个缺失 SFT checkpoint 不插值。
- [x] 改善、无增量、损失三组中位案例，以及带明确筛选规则的内容焦点案例；对比全部四模型。
- [x] 准备 100 contexts × 4 models 的盲评包，尚未评分。
- [x] 用 vivid-figures-skill 的完整源码配方与既有 pastel-girl 色系生成 6 张图，保留来源与差异记录。
- [x] 更新 paper/src/result.tex 和 paper/src/appendix.tex。
- [x] 编译论文并检查新增图表、表格和案例页；编译缓存与 PDF 位于 paper/build。

分析脚本、CSV、manifest、案例、盲评文件和图件位于 [paper_plots/marketing_analysis](paper_plots/marketing_analysis/README.md)。

目前最清楚的发现是：PIPO (all) 相比 PIPO (+) 的 article-average ROUGE-L matching 增量为 0.474，95% crossed bootstrap interval 为 [0.250, 0.661]；98/103 用户的估计 matching 增量为正。但七项风格匹配没有一致改善，数字支持 flags 也没有改善。这些证据仅针对保存模型与固定样本。

训练权重审计还有一个关键发现：negative p_ctr 在同一 training article 的不同用户之间恒定，因此其权重本身不包含同篇文章内的用户差异。需要下面的归属安慰剂来判断个人匹配增量是否利用了 history/non-click 的关联。

## P0：先解决论文与实验口径

- [ ] 保持当前实验对应的 reference-free signed sigmoid recipe，统一 method、prelim、theory 的对象或明确理论只是另一线性目标的分析。当前正文结果与附录已声明差异，不能把保存运行当成线性目标定理的实证检验。
- [ ] 更新 main.tex 中尚保留的旧 abstract/introduction/conclusion 数字和配置叙述，检查 data.tex 与当前 sampled training pool 的关系。本轮重点修改 result.tex 与 appendix.tex，未重写其他章节。
- [ ] 补齐 s_dwell 与 p_ctr 的 upstream 定义、分母、时间窗、归一化、缺失处理、曝光/检查机制与 leakage 检查。现有审计只读取已供给的归一化列。
- [ ] 从训练 lineage 核实文章级 negative score 的计算窗口，以及目标 exposure 是否参与它的构造。不能把文章级高分非点击直接解释为个人明确拒绝。
- [ ] 冻结下一轮问题、训练条件、评分规则、step 500 和分析代码，记录本次 eval 的使用情况。继续用指定 eval_final 做诊断，不声称它已成为未见测试；确认性结论需要真正预留样本或新参与者。
- [ ] 论文提交前替换旧 positive-only unweighted ablation 中 step 400/450/500 的 step-350 display repeats。优先读取已完成 run 的真实输出；若确实没有输出，再另行安排实验。附录已披露占位口径。

## P0：E2 + E4 受控训练矩阵（20 条件 × 3 seeds = 60 runs）

所有条件从同一 Base 初始化；不用全量正样本训练得到的 warm start。使用同一 tokenization、历史选择、最大长度、LoRA 配置、learning rate、更新数和 scorer。建议固定 learning rate 4.7e-6、beta 0.1、weighted kappa+=3.5/kappa-=1.5，positive NLL coefficient=0，negative_scale=1，reference_free=true，average_log_prob=true。

未来 runs 启用 prospect_dpo_global_class_mean=true，正负混合时每次 update 使用 64 positives + 64 negatives；positive-only 使用 128 positives；固定 500 updates。显式分层抽样保证分支比例，满 update 的 class mean 在所有条件一致。这个 reduction 与 legacy runs 有区别，不能把 legacy all/+ 直接充作新矩阵的 seed 42。

negative ratio 指不同负样本训练池大小 N-/N+，不是 update 内的正负比例。改变池大小同时保持每次 update 64/64；报告池内 unique rows、unique users、重复抽样、每条记录有效曝光次数。由于两种方法的正 token 数不同，500 updates 是 matched-update 分析；同时报告实际 tokens/GPU-hours，不能称完全 matched-token。

| 条件组 | 正目标记录预算 | 方法/对照 | 负训练池比例 | 条件数 |
|---|---|---|---|---:|
| A：稀缺性 | 10%、25%、50%、100% | weighted positive-only；unweighted joint；weighted joint | joint 为 1，positive-only 为 0 | 12 |
| B：负记录边界 | 100% | unweighted joint；weighted joint | 分别新增 0.25 和 2 | 4 |
| C：信息与强度控制 | 100% | 以下四个 controls | 1 | 4 |
| 合计 | | | | 20 |

三个训练 seeds 为 42、43、44，共 60 runs。数据子集由一个固定 data seed 构造；训练 seed 与 data seed 分开记录。10/25/50/100% positives 使用嵌套子集，并报告实际取整后的数量。主正训练池为当前 156,823 条 sampled positives，而不是把全库 683,767 条误称为当前 100%。

当前 full negative pool 有 683,570 条记录；限制在这 156,823 条 positives 的用户集合后有 326,954 条 eligible negatives，最大可行比例约为 2.085。因此 ratio=2 在当前 100% 正预算可实现。负池从 eligible unique sample_id 中不放回抽取，pool=.25/1/2 尽量嵌套。低正预算条件按自己的正用户集合重新定义 eligible pool，报告重叠与覆盖，不默认全用户 negatives 可用。

四个 C controls：

1. **Negative history association shuffle**：保留 negative article、response、label、score、category、频次，交换 negative record 对应的用户历史；优先在相同 article、相近实际 history 长度的 strata 内置换，禁止原归属固定点。不能置换的 strata 明确报告。同步重建 prompt 与 history metadata，保证除了历史归属没有其他修改。对照真实 weighted joint，检验 non-click/history 关联的信息价值。文章级 p_ctr 不会因置换变成个体 score。
2. **Branch-mean constant weights**：分别把正/负 multiplier 固定为对应新训练池原权重均值，并仍置于 sigmoid 内。保持平均权重尺度，检验观测层异质性；这不完全匹配 sigmoid 饱和后的有效梯度，需要记录梯度贡献。
3. **Positive-weight permutation**：保留正记录和正 multiplier 分布，只打乱对应关系；negative 分支保持真实值。检验 positive score 与目标的对应关系。
4. **Negative-weight permutation**：在文章层打乱 negative score/weight，保留每篇文章内部一致性。用 article-frequency 分层，尽量保持 record-weighted 分布和均值；报告误差。若按 record 随机置换，会创造原数据没有的同篇用户权重差异，不能当成严格 placebo。

- [ ] 先确认抽样池与置换 invariants，再创建训练配置；本轮没有生成/提交 sbatch。
- [ ] 按 A 的 10%、100% 六条件组织首批计算；B/C 作为解释与边界检验。全部三个 seeds 的结果都报告，不能只保留最好的 seed。
- [ ] 每个 saved output 必须对齐同一 eval_final，空输出进分母，保存 run config/hash/seed/sample IDs；不按该 eval 选 step。
- [ ] 主 outcome：E1 paired matching contrast；辅助 outcome：own-reference ROUGE、受损用户比例、输出差异、事实性筛查。主 family 与多个对照的 multiplicity 规则在看结果前确定。
- [ ] 用训练 seeds 的差异与 user/article 的采样区间分开报告不确定性；不要把 2,000 次输出 bootstrap 当作三个训练 seeds。
- [ ] 记录 GPU-hours、peak memory、processed input/response tokens、数据处理和审核时间。达到同一 matching threshold 的更新数需使用冻结开发规则；没有 runtime 日志时不编造成本估计。
- [ ] 预先增加一个 equal-positive-exposure slice：positive-only 的 update 250 对比 joint 的 update 500，两者各处理 32,000 个正记录 exposures；它是辅助剂量对齐比较，不替代所有模型 step-500 的主比较，也不声称 token/GPU 成本相同。未来保存并评估这两个冻结 checkpoints，不根据结果选择更新数。
- [ ] 可选补充：同预算 SFT，帮助区分 positive-only sigmoid 与普通 supervised learning。此项额外 runs 不包含在 60 个核心 runs 中。

正样本稀缺性解释限于“生成器可用的正目标记录减少”。若 histories、score generator 或其他预训练仍使用额外正反馈，不能声称只用了 10% 的全部正反馈信息。

## P0：E1 的推理与语义稳健性（本轮未执行）

- [ ] 对 Base、SFT、PIPO (+)、PIPO (all) 的冻结 step 500/0，补 history length 0、5、10、20；已有 length 30 作为第五条件。仅使用指定 eval_final，不新造 eval。保留同一 article body、同一最大 token budget、同一 decoding/解析规则；历史顺序和截断方式事先固定。
- [ ] 每个 history 条件报告 full-denominator ROUGE、matching、空输出、截断率与 tokens。空历史也可以受 body/style 影响，不预设它的 matching 恰好为零。
- [ ] 用同篇文章中另一个用户的 history 生成；与正确 history 形成 method × history assignment 检验。现有 E1 交换的是参考作者匹配，不等同于重新运行 history-swap 推理。
- [ ] BERTScore matching：先固定独立 encoder 的 checkpoint、revision、layer、language、normalization 与 scorer 版本；然后才运行语义 encoder。本文尚未计算 BERTScore。与 ROUGE 使用同一 pair/sample、同一 mean-within-author reference rule；ROUGE 不能替代语义检验。
- [ ] 95 个共同空输出：先检查是否因 input/context budget 或保存规则产生。必要时在统一预设预算下对四模型共同修复这些 contexts；原始全分母结果仍保留，修复结果作为明确标记的 sensitivity。不得只给最好模型重试。

## P0：E3/E7 独立内容审核与构念验证

- [ ] 对 data/human_review_blind.csv 完成 100 contexts/400 headlines 的双人独立审核，评审看 article body 与匿名标题，不看 model key 或 outcome。
- [ ] 先用约 10 contexts pilot 定义 factual support、unsupported emphasis、clarity 的 rubric；锁定后重新独立标注全包。保存原始评分、分歧与 adjudication，报告 agreement；pilot 若修改 rubric，不能假装已预注册。
- [ ] 从审核中标记可核对的 article propositions/entities/number facts，区分事实选择与表达风格。用 reference 与 headline 的独立事实集算 own/swapped focus F1；当前 digit-string Dice 只算 pilot proxy。
- [ ] 验证小型 emotion/hyperbole/source lexicons 的 precision/coverage；扩展到 NRC、concreteness 或 readability 前核对词典版本、使用条件与独立 construct validation。不要把自建 cue 自动叫做 LIWC/情绪强度/clickbait。
- [ ] 可选预冻结 micro-edit：仅改变一个语言维度，保持事实、长度与原意，交给独立读者随机比较；检验读者偏好 × 表达版本。
- [ ] 补全 30 条展示历史的 article-category metadata，再考虑类别兴趣熵。当前仅有少量 history IDs 可匹配；不能由缺失 lookup 生成可靠 entropy。

## P0：E5 真正的读者个性化实验

- [ ] 用新参与者测量事前阅读兴趣/语言偏好并独立生成历史表示；原 NT 参与者不能仅凭 ID 被重联系，也不能把其他人的历史当成新参与者的正确历史。
- [ ] 先 pilot，再基于 method × assignment interaction 做功效分析与样本量计算；不套用普通均值比较的样本量。
- [ ] 冻结 PIPO (all)/PIPO (+) 的 headline 集合。2 × 2：method × 正确接收者/错配接收者；同篇固定内容、交换标题接收者，随机化、遮蔽方法身份。
- [ ] 区分点击/选择、读后评价、事实信任或后悔；事先确定一个主 outcome。报告真正匹配 interaction，而不只说平均点击更高。
- [ ] 在适用的研究审查与知情同意程序完成后再招募。此项研究尚未开始，不能在论文里填预期 click lift。

## P1：管理应用与外推

- [ ] E6 selective personalization：用训练/开发数据学习配置规则，held-out folds 比较统一生成、cluster、individual；报告 coverage、matching、损失、审核和推理成本。不能在同一 eval 上挑受益用户再估计“选择性部署增益”。
- [ ] 在独立文章/新参与者或另一 backbone 上复现方向；只有用户今后提供或授权新评价数据后再做，不替换本轮唯一 eval。
- [ ] 随机线上曝光或预留真实响应验证后，再计算信息投资/商业 break-even；目前离线 overlap 不能换算收入、福利或 causal lift。

## 相关论文如何对应这些待办

- [Causal Alignment](https://leekevin.io/caLM-revised.pdf)：借鉴用随机实验验证真实响应的逻辑；本项目的离线匹配仍不能给出 causal click lift。
- [Balancing Engagement and Polarization](https://arxiv.org/abs/2504.13444)：借鉴内容特征、可控制的配置与非目标后果检查；不假设新闻情绪/具体性有通用最优方向。
- [From What Works to Why It Works](https://doi.org/10.2139/ssrn.5437276)：既有计划中记录了本地论文的人工审核、竞争性消融与数据可用性启发；本项目尚未完成相应机制验证。
- [Generative Personalization](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5329729)：借鉴个体语言偏好和参考匹配；现有 feature 均值不一致提升，需事实焦点和独立读者实验。
