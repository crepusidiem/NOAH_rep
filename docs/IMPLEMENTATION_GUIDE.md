# NOAH 技术复现操作指南

核对日期：2026-10-14。用途：把论文要求转成后续实施步骤，尤其是预处理、特征提取、数据窗口、模型和训练的衔接。当前完成情况见 [PROJECT_STATUS.md](PROJECT_STATUS.md)。

## 1. 先明确已经知道什么

论文规定了 ECG 的输入规格、patch 和最终均值池化，ECHO 的预处理步骤与全局 CLS 提取，以及文本和胸片的编码器。待明确的主要是具体权重版本、部分输入处理参数、边界策略和若干模型/训练设置。

本指南补充“怎样实施、怎样验收、哪些选择仍要记录”，不替代已有文件：

- [REPRODUCTION_SPEC.md](REPRODUCTION_SPEC.md)：完整架构、公式、张量、目标与超参数。
- [PAPER_TRACEABILITY.md](PAPER_TRACEABILITY.md)：要求对应的页码、章节、公式位置。
- [ASSUMPTIONS.md](ASSUMPTIONS.md)：每项未公开设置及独立配置项。
- [TEST_PLAN.md](TEST_PLAN.md)：单元、集成与科学正确性测试。
- [REPRODUCTION_PLAN.md](REPRODUCTION_PLAN.md)：阶段依赖和扩展计划。

主证据是 [NOAH.pdf](../paper/NOAH.pdf)，arXiv:2609.09140v1，共 50 页；SHA-256 为 `0f3a7bb049d6f7eb6cdf61c2aa9692d0695037851f95b73711a26ed803576730`。技术细节依据 pp.15-21、29、49-50；PDF 页码与印刷页码一致。论文显示公式没有编号，已有文档的 E01-E16 是定位标签，不能称为论文的 Equation 1-16。

### 决策分类与证据来源分开记录

| 分类 | 含义 | 实施要求 |
|---|---|---|
| EXPLICIT | NOAH 论文直接规定 | 按其实现；改变时登记 DEVIATION |
| DERIVED | 可由论文无歧义推导 | 写出推导，不额外添参数 |
| ASSUMED | 为未规定部分选择一个具体方案 | 独立配置、记录理由和影响 |
| DEVIATION | 改变明确机制、规模或信息条件 | 单独命名实验并解释变化 |

另设 `evidence_source`，区分 NOAH 论文、NOAH 作者代码、编码器上游代码、当前仓库。**“上游代码可核实”不等于“NOAH 作者一定用了这个版本”。** 从上游补入 NOAH 未写明的设置，仍是 ASSUMED，直到取得对应作者证据。实施时将候选方案登记为独立配置，并按研究者当前任务推进。

## 2. 当前仓库起点

代码基线：`30dd669325f99e4ce9fe554cae2934376f35f879`，本节为 2026-10-14 的静态检查，不是服务器运行报告。

| 文件/环节 | 已有内容 | 后续接通内容 |
|---|---|---|
| `data/pilot_cohort_manifest.csv` | 1,000 个独立 subject ID；700/150/150 | 从源数据验收事件量和模态内容；人数不代表可用候选池上限 |
| `src/data/subcohort.py` | 成年、住院、ECG、CXR 交集；seed=42；取前 1,000 人 | 记录这是多模态富集 pilot，与论文自然人群分开 |
| `src/data/extract_events.py` | 入/出院与死亡、数值化验、ECG/CXR 索引；时间排序 | 临床项目名称/单位、其余事件读取、真实媒体解码、文本读取 |
| `src/data/schema.py` | 31 类/6 模态枚举，事件与前缀字段 | 稳定来源映射、完整窗口和 loss ownership |
| `numeric.py` / `vitals.py` | 候选 44-D Fourier 编码；min/max 聚合 helper | 登记数值 convention，连接 vitals 读取与聚合 |
| `src/models/temporal.py` / `tests/test_temporal_oracle.py` | 时间特征 helper 和测试定义 | 完整因果注意力、变分模块、层级解码、训练与 rollout；执行并记录测试 |
| `scripts/test_pilot_extraction.py` | 对清单前两人调用四个读取函数 | 不可据此宣称 1,000 人已完成 ETL/编码/训练 |
| `sbatch/run_pilot_gpu.sh` | 环境与资源申请框架，训练入口仍占位 | 在授权实施后接入实际编码/训练命令 |

现有代码的小时单位、出生年 1 月 1 日代理、同刻排序、最后 gap=0、2h/1h 聚合和 Fourier 相位=1，都不能自动升级为论文事实。`schema.py` 的 numeric=44 注释也不是 Table 15 的原始数值维度声明；Table 15 没有给这个维度。

O2 GPU 平台可用。实际卡型、配额、显存和排队时间应在运行时记录。运行环境与作业结果记录在服务器报告中。

## 3. 完整数据路径和实施顺序

```text
版本固定的源表、文本、媒体
  -> 患者/就诊/检查关联和事件映射
  -> 时间、单位、缺失和密集 vitals 处理
  -> 各模态预处理及冻结特征提取
  -> 内容 embedding 缓存和来源索引
  -> 人口学前缀、窗口、attention mask、loss mask
  -> NOAH 可训练投影、时间注意力、prior/posterior、层级解码
  -> 双路径训练和 KL 退火
  -> 仅历史/先验可用的自回归 rollout
```

在后续获得实施授权后，先把源字段/编码器/时间约定写成运行配置；数据工作与数学参考测试可独立推进。先验收几条真实内容及合成机制，再扩大到两名患者和完整 1,000 人。不要先跑完整训练再反推预处理规则。

“编码器资产”在实施中具体指：模型权重、结构配置、tokenizer/processor、相关预处理代码和版本。公开模型的发布页已找到；未核对到本地副本不等于资产不可获得。

## 4. 各模态的论文特征提取要求

`N` 为当前批次中该模态的有效记录数，不是患者人数；一个患者可有多条记录。内容空间与 NOAH 隐藏空间是两个不同空间。

| 模态 | NOAH 明确要求 | 内容表示 | 证据 |
|---|---|---|---|
| 文本/事件 specifics | 非训练外部 BioLORD；预计算文本和 specifics | 文本 `[N,768]`；specifics=768 是同编码器空间推导 | pp.15、17；Table 15 p.49；ref.39 p.29 |
| CXR | RAD-DINO 图像编码 | `[N,768]` | p.15；Table 15 p.49 |
| ECG | 12 导联、10s、500Hz；1x24 非重叠 patch；最终 patch 均值 | patch `[N,2496,192]`；全局 `[N,192]` | p.15 |
| ECHO | 指定超声预处理；EchoPrime 全局 CLS | `[N,512]` | pp.15-16；Table 15 p.49 |
| 数值 | Fourier features，22 个 dyadic 尺度 `2^-7 ... 2^14` | 完整 sin/cos 对给出候选 44-D；完整 convention 未公开 | pp.17、20；Table 16 p.50 |
| 类别 | 预训练时学习事件类别 embedding | 论文完整模型类别表示 `[N,768]` | p.17；Table 16 p.50 |

外部编码器固定参数。NOAH 的投影层与模型联合训练，将不同维度的内容映射到共享 `d=768`（论文 profile）。禁止为了统一维度，先把所有外部编码器的输出人为截断/补零到 768。

### 4.1 文本与 specifics：BioLORD

**论文明确：**文本和类型细节通过 BioLORD 编码，外部编码器不训练；没有详细给出长笔记分段、池化、输出归一化和准确权重 revision。

**上游已经核实：**[BioLORD-2023 模型卡](https://huggingface.co/FremyCompany/BioLORD-2023)、[pooling 配置](https://huggingface.co/FremyCompany/BioLORD-2023/blob/main/1_Pooling/config.json)、[sentence 配置](https://huggingface.co/FremyCompany/BioLORD-2023/blob/main/sentence_bert_config.json)提供具体使用方法。当前 sentence 配置为 `max_seq_length=128`，pooling 配置为有效词 token 的均值、不是 CLS。[modules.json](https://huggingface.co/FremyCompany/BioLORD-2023/raw/main/modules.json)列出 Transformer 和 Pooling；模型卡手写 Transformers 示例额外进行了 L2 归一化，不能未经比较就把两种调用路径视作数值相同。

**后续步骤：**

1. 指定文本事件粒度及原文来源，区分正文与“项目/药物名称”等 specifics。
2. 将 `FremyCompany/BioLORD-2023` 作为公开候选资产，固定 revision、tokenizer 和 pooling 配置；该候选不证明 NOAH 的准确 revision。
3. 显式设置 `text_max_length`、`text_long_policy`、`text_pooling`、`text_l2_normalize`。可选择截断或分块后聚合，但 NOAH 未规定的选择均需登记 A-09。
4. 每条输入输出一个定义清楚的 768-D 表示，保存截断/分块统计。不可将超长报告默默截断而称“完整报告已消费”。
5. 用短句、长报告、空文本和不同批次验证输出。若比对两套 API，必须统一长度、pooling 和归一化后再比较。

当前 `item_<id>` 只是化验项目占位串。应先关联项目字典和单位，才能作为有临床含义的 specifics；不能把可产生一个向量等同于语义正确。

### 4.2 胸片：RAD-DINO

**论文明确：**编码器为 RAD-DINO，内容向量 768-D；没有写明 NOAH 的图像 resize/crop、各视角处理和输出 pooling 接口。

**上游已经核实：**[官方模型卡](https://huggingface.co/microsoft/rad-dino)提供 CLS 和 patch 特征接口，CLS 为 768-D。[发布的 processor 配置](https://huggingface.co/microsoft/rad-dino/raw/main/preprocessor_config.json)设置 RGB、短边 518、518x518 中心裁剪、`1/255` rescale、每通道 mean=0.5307/std=0.2583。它是公开候选输入方案，不是 NOAH 已确认配置。模型卡还区分公开权重和 RAD-DINO 自己原论文中的权重；这不能用来断言 NOAH 选了哪一个。

**后续步骤：**

1. 从 metadata 定位实际 image ID/文件；明确“每张图一个事件”还是检查内合并多图。NOAH 的准确 image/study 聚合规则需单独登记。
2. 核对 `mimic-cxr-resized` 副本的原始来源、分辨率和已执行变换。将低分辨率图再次放大到 518 不会恢复丢失的信息。
3. 固定 checkpoint 与匹配 processor，检查灰度转 RGB、pixel range、插值、裁剪及是否已经归一化，避免重复 rescale。
4. 可将公开 CLS 输出作为 A-09 的 pilot 候选；不要把 patch 网格、分类器输出或 DINO 自监督训练 head 当成同一种 768-D 表示。
5. 保存 image ID、处理器 revision、输出选择和缓存对应关系；对多视角、极端宽高比和解码失败做验收。

不能把 RAD-DINO 自己预训练 DICOM 的处理步骤不加区分地重复应用到 NOAH 所用的现成 CXR-JPG。

### 4.3 心电波形：OTIS

**论文明确：**每条波形 12 导联、10 秒、500Hz，可推导原始数组 `[N,12,5000]`。每导联 208 个宽 24 的 patch，共 2,496 个 192-D token；最终 patch 取均值得到 `[N,192]`。池化不包含 CLS 或分类 head。

**上游已经核实：**[官方仓库](https://github.com/oetu/otis)描述样本为 `(channels,variates,time)`，可对应适配器输入 `[N,1,12,5000]`。其 [main_linprobe.py](https://github.com/oetu/otis/blob/main/main_linprobe.py)示例默认 patch 为 `(1,100)`，而 NOAH 要求 `(1,24)`；脚本也存在 patch 不匹配时丢弃旧 patch 权重的处理。**不能直接照搬这种重初始化流程来声称使用了 NOAH 的冻结编码器。**

[models_vit.py](https://github.com/oetu/otis/blob/main/models_vit.py)含不同 pooling 分支、分类 head 和域/变量位置表示；函数名 `forward_features` 本身不保证返回完整 patch 序列。准确 checkpoint、最终 norm 位置和域设置仍需固定。

**后续步骤：**

1. 读取 WFDB header 与 waveform，核对采样率、样本数、导联名称/顺序、物理单位、缺失和异常值。
2. 独立配置 `ecg_lead_order`、`ecg_normalization`、`ecg_missing_lead_policy`、`ecg_domain_policy`。不要自动加入滤波、截断、重采样或零补导联而不登记。
3. 找到兼容 192-D、1x24 patch 的权重并检查加载键/形状。不要用任意宽 100 的 checkpoint 加 `strict=False` 掩盖不匹配。
4. 提取最终层 **patch**，确认 2,496 个有效 token，按论文作均值。随机 patch 丢弃不属于该完整记录提取方案；记录输出 norm 的选择。
5. `208*24=4992`，还剩每导联 8 个样本；裁掉、补齐或其他尾部策略需 `ecg_tail_policy`，不是已知作者规则。
6. 验收导联重排、长度异常、常量/非有限信号、patch 均值与手算参考，以及重复编码稳定性。

机器测量表或文字 ECG 报告可以形成另一个事件来源，但不能替代论文的真实波形内容路径。

### 4.4 超声视频：EchoPrime

**NOAH 已明确的顺序（pp.15-16）：**

1. 根据 DICOM `SequenceOfUltrasoundRegions` 提取超声区域。
2. 去掉 ECG 叠加及烧录文字等标注。
3. resize 到 224x224，存成 AVI。
4. 加载时额外作 10% zoom crop，再 resize 到 224x224。
5. 用 EchoPrime 训练集 mean/std 归一化。
6. 以 stride=2 采样 16 帧，较短视频补零。
7. 冻结 EchoPrime，提取论文所称全局 CLS，512-D；这里是一条视频的内容表示。

**上游可以补充的具体候选：**当前 [echo_prime/model.py](https://github.com/echonet/EchoPrime/blob/main/echo_prime/model.py)使用 `[N,3,16,224,224]`；在原始 0-255 float 空间按 RGB mean `[29.110628,28.076836,29.096405]`、std `[47.989223,46.456997,47.20083]` 归一化，再补零并从起点 0、跨度 32、步长 2 切片，实际输出 16 帧。`embed_videos` 输出 512-D；`encode_study` 额外拼 11-D 视角信息，输出 523-D。**NOAH 不能直接使用后者。**公开接口的 512-D 与 NOAH 所称 CLS 的准确对应仍需核对，不应另取内部任意 CLS。

[utils/utils.py](https://github.com/echonet/EchoPrime/blob/main/utils/utils.py)给出裁剪/插值函数，其中 zoom 按各边比例裁；这不证明 NOAH 的 crop convention 相同。其像素掩膜函数没有使用 DICOM 区域元数据，不能替代 NOAH 明确要求的元数据区域提取，也不自动满足 NOAH 的全部去标注步骤。

**后续步骤与独立配置：**

- `echo_checkpoint_revision`、`echo_output_selection`：锁定视频编码器及 512-D 输出路径，不调用报告检索或追加 view one-hot。
- `echo_region_policy`、`echo_missing_region_policy`：多 region 如何选择，tag 缺失如何处理；替代明确的 metadata 提取时登记 DEVIATION。
- `echo_overlay_removal`：保存裁剪/掩膜位置，抽样查看是否仍有文字或 ECG，以及是否误伤超声区域。
- `echo_zoom_convention`：每边 10% 与总计 10% 不是同一个裁剪；整数边界和插值需记录。
- `echo_color_policy`、`echo_pixel_range`、`echo_normalization`：解码色彩/灰度通道与常数相匹配；不要先除 255 再套针对 0-255 的常数。
- `echo_frame_start`、`echo_sampling_policy`：可选择上游起点 0 作为 ASSUMED pilot 候选；多片段均值或均匀采完整视频不是已给出的 NOAH 方法。
- `echo_padding_stage`：原始空间黑帧补零与归一化后补零不同，论文未明确这一顺序。上游后者只能作为登记后的候选。

验收原视频长度和实际 frame indices。起点 0 的长视频候选是 `0,2,...,30`，不是 32 个输出帧。检查短视频的输出帧数、补零位置、224x224、通道/数值范围、去标注效果及有限 512-D 输出。仅有超声测量表不能完成视频路径。

ECHO 当前库存报告缺失；五模态 pilot 可以作为提议先行，但六模态完整复现仍需 ECHO。是否先行由项目范围决定；不能默认补出一个零视频或零 embedding。

### 4.5 数值和类别

数值没有待下载的“数值预训练模型”。输入标量先保留临床类型/单位，采用 Table 16 的 22 个 dyadic 尺度。当前 `numeric.py` 的 sin-bank 后 cos-bank、相位 1、raw float32 是候选 convention；`normalize` 参数虽存在，当前函数没有实施归一化。后续须独立设置数值单位、变换、相位、顺序和精度（A-10），用高精度参考验收大相位，不能混用时间特征的 base=10000 几何频率。

类别 lookup 可训练。现有 enum 的整数顺序是仓库约定，不是作者公布的原始 ID。类别值 payload、缺失值和前缀编码仍见 A-08；不能因缺值就训练一个伪零数值目标。

## 5. 冻结提取、缓存和 provenance

以下是工程验收要求，不是论文公布的缓存格式：

1. 固定外部参数，设置评估模式，在不建立外部编码器梯度的模式下提取；检查同输入重跑差异、dropout/随机 crop 和种子。
2. 对原始内容按记录 ID 编码，保存 `record_id -> embedding`；不把检查、视频、患者级平均混为一谈。
3. 缓存 key 覆盖内容标识/校验、权重 revision、processor、output selection、预处理配置和版本。改其中一项时不能复用旧特征。
4. 记录输入数、成功数、失败原因、过滤数、输出维度/范数/非有限数及患者覆盖；失败不能默默变为零向量。
5. trainable NOAH 投影应在训练时运行，不能预先缓存一个随机投影到 768-D 的结果并冻结它。
6. 从样本自拟合的 scaler、阈值、class weights 和类别-模态 support，应显式声明拟合范围。train-only 是避免跨 split 信息的工程候选；论文未明确所有统计量的拟合范围，见对应 A-ID。

每个运行的 manifest 至少记录：数据版本/根目录、cohort/split 哈希、各编码器 model ID 与 revision、权重与配置哈希、tokenizer/processor、norm/pooling、样本范围、随机性、缓存精度、active modalities、事件集合、失败策略，以及每个未明确选择的分类与证据。公开网页的 `main` 会变化；运行时需锁定具体 revision。

## 6. 时间线、窗口与损失掩码

实施 DATA-01..07 前先声明 A-01..07、A-12..14、A-62 的所用选择。来源表 join 后保留 source table/row 和记录粒度，防止一对多复制事件。缺失不是零；过程结束信息在实际终止时才产生结束事件。

当前代码用出生年 1 月 1 日作为 origin、小时计算时间、以年表示年龄；这些是已写入代码的 ASSUMED 约定。vitals helper 的 2h span/1h stride 也不是论文明确数值。最后一个事件 `delta_next=0` 不能同时代表“未知后继”和真实同刻后继；后续用独立可用性标志/目标 mask 处理边界。

论文明确：70/15/15 患者划分；至少 64 scored tokens；训练/验证最多 2048 tokens、重叠 256；六项人口学前缀；前缀/重叠不计损失。实施时：

- 保留同一患者的所有窗口在同一 split；先冻结所用 cohort/split，再建窗口。
- 显式决定 2048 是否包含前缀、首窗如何处理、尾窗和 test chunk 如何处理；这些不是库的通用默认。
- 对**移位后的目标事件**判定 loss ownership。每个可计分临床事件恰好计一次，首窗不能无理由删除前 256 个目标。
- attention mask 和 loss mask 分开；前缀和重叠事件可作为上下文。
- posterior reconstruction、prior reconstruction 和 KL 同受 ownership/padding mask；specifics、value、positive gap 再加各自有效性 mask。
- 论文人口学窗口末端取值、聚合均值时间和真 forward gap 可能有前瞻信息。literal paper 与 strict as-of 信息条件必须分开配置和报告；不要将严格因果“修复”悄悄当成原论文。

**小对齐例子：**事件时间为 `[0,2,5]`。预测第二事件时历史内容是第一事件，第二事件的 incoming gap=2，而第二事件的 outgoing gap=3。论文解码器文字称输出 outgoing gap；不能直接把 3 当成第二事件的发生时间偏移。采用哪个索引/调度方案、首步 free rollout 如何获得起始 gap，见 A-14，必须在配置、图解和测试中一致。

## 7. NOAH 模型连接的最低要求

按 [specification](REPRODUCTION_SPEC.md) 的 TIME/ALIGN/VAR/DEC/LOSS/OPT/ROLL 要求施工，不用标准 Transformer/VAE 默认替代论文机制。以下为实施检查点：

| 模块 | 必须保持的机制 | 仍需配置的例子 |
|---|---|---|
| 输入/投影 | 类别、specifics、value 与时间信息组成共享 token；外部固定、投影可训练 | 缺失处理、输入时间融合、线性 bias/norm |
| 时间注意力 | Q 加 position/age/forward；K 加 position/age/backward；`log1p` gap；因果内容 mask | 单位、边界/索引、layer 0-3 的解释、backbone 细节 |
| prior/posterior | `p(z_t|h_(t-1))` 与 `q(z_t|h_(t-1),e_t)`；对角 Gaussian；共享 latent | MLP 深度/激活，h0、posterior embedding 阶段 |
| 随机性 | 每事件、每解码 pass 一个共享 z；先验 sigma 平滑限制到 0.05-2 | 平滑映射具体形式；不独立给各 head 加噪声 |
| 层级解码 | category features，specifics features；modality/value 分支；time 读 z | routing、head 结构、类别权重/compatibility support |
| 门控 | 是否有 specifics、是否有正 gap 用硬判定；正 gap 的 log 回归只对正目标 | 判定阈值、正值数值保护 |
| 数值 rollout | 输出与输入同一 Fourier basis 的 coefficients，直接复用 | 不能擅自标量反演再编码；诊断可另外实现 |

论文完整模型：d=768、12 层、12 heads、latent=384、FFN expansion=8、150,322,970 个 trainable parameters（p.50）。缩小模型/上下文是 pilot 的 DEVIATION，要命名和记录；保留完整 profile 用于参数和结构验收，不能宣称 tiny model 已复现论文容量。

### 7.1 训练目标和优化

论文总目标为 posterior 重建 + KL + 辅助 prior 重建（pp.20-21）：

`L_t = 0.3 * L_rec(posterior sample) + beta * KL(q || p) + 0.1 * L_rec(prior sample)`。

beta 从 0 线性退火到 1，10 epochs；LR warmup 为 6 epochs，是另一套时钟。峰值 LR=6e-4、cosine、weight decay=0.01、clip norm=10。Muon 用于隐藏 2D 矩阵，AdamW 用于 embeddings、heads 和非 2D 参数；不能仅按 tensor rank 把所有 head 矩阵分给 Muon。dropout/drop path/attention dropout 均为 0.1（p.50）。

连续 embedding 的概率/回归损失、variance、维度 reduction、组件权重、batch reduction 未完全公开，不能把自己选择的 MSE/cosine 命名为作者原始损失。按 A-27..30 分别记录。总 epochs、global batch、optimizer internals 与 checkpoint selection 等也不是已知值。

先用合成数据的独立数学参考验证，再 tiny-batch overfit，随后真实 pilot。记录 prior/posterior reconstruction、原始/加权 KL、beta、sigma、无效样本、时间和各模态目标；只看 total loss 不能证明因果和变分实现正确。短 pilot 若改变 10-epoch KL/6-epoch LR 时钟，要记录变化，不能默默压缩。

### 7.2 推理

free rollout 只能使用观察/生成历史与 conditional prior；posterior 所见的真实目标 embedding 在预测时不存在。预测并解码后，把内容表示重新投影、与事件/时间信息组成输入并追加；不要调用外部编码器把预测 embedding 当作原始图像/文本再次编码。

time-controlled 模式输入指定时间 gap，论文评估用真实 gap，是另一个信息条件；不得当作 free forecasting。同刻事件可有真实 zero gap，不用加 epsilon 强迫时间前进。固定 token budget、上下文截取和停止条件必须记录。

## 8. 依赖顺序与完成证据

下表给出实施顺序和验收要求。完整测试清单见 [TEST_PLAN.md](TEST_PLAN.md)。

| 阶段 | 工作 | 必须能展示的结果 |
|---|---|---|
| A | 固定 run profile、数据根目录、字段映射与编码器候选 | 每个未明确设置有 config key、分类、理由；公开资产与作者证据区分 |
| B | 数学/边界参考与几条真实内容 | temporal/KL/Fourier/ownership 手算一致；实际内容解码/编码维度及 provenance |
| C | 合成模型与完整训练目标/rollout | 移位、因果、双重建、共享 latent、硬门控、prior-only 测试通过 |
| D | 两名患者的真实全路径 | 不只是打印 media ID；有事件、真实 embedding、窗口和 mask |
| E | 1,000 人小规模训练 | 可重启 checkpoint、验证、prior rollout、组件日志和显存/吞吐 |
| F | 扩展决策与论文规模 | 核对全部模态/事件、profile、数据规模和评价协议，再估算资源 |

O2 已有可用条件；先测小模型峰值显存、token 数、编码和训练吞吐，再估算扩展成本。8x A100 80GB 是论文预训练环境（p.50），不是 1,000 人 pilot 的最低硬件要求。

1,000 人五模态试验仍是提议，不自动等同完整六模态复现。active modalities、是否保留六模态全局 vocabulary、未覆盖的类别和采样富集必须可配置并进入记录。缩小任务不能删除 prior、posterior 或时间机制而仍使用原模型名称。

## 9. 需要保留的独立选择

已有 A-ID 提供全部细节；至少不要把以下设置揉成一个 `faithful=True`：

| 范围 | 分开的设置 | 影响 |
|---|---|---|
| 文本 | checkpoint、长度、长文本策略、pooling、L2 | 内容截断和表示空间变化 |
| CXR | 文件来源/resize、processor、视角聚合、output selection | 可见结构、记录粒度与特征变化 |
| ECG | checkpoint/patch、lead order、norm、tail、domain、final norm | 信号含义及是否真正使用冻结 patch 表示 |
| ECHO | region、cleanup、zoom、color/range、norm、frame start、padding stage、output | 运动片段、像素分布和 512/523 接口变化 |
| 数值/时间 | 两套 basis、单位/变换、相位、boundary、gap alignment | aliasing、因果信息和目标错位 |
| 训练/rollout | likelihood/reduction、weights、anneal clock、budget、time profile | loss 尺度、学习行为和预测信息条件 |

在获授权的 pilot 中可以选择并记录一个合理候选继续工作，无需等每个未来评价 A-ID 都解决。不能据此把尚未解决的选择隐藏在 paper-scale 配置中。

## 10. 外部核对来源与边界

公开来源核对日期：2026-10-14。运行时固定权重、处理器与代码版本，并记录加载验收结果。

- BioLORD：[model card](https://huggingface.co/FremyCompany/BioLORD-2023)、[pooling](https://huggingface.co/FremyCompany/BioLORD-2023/blob/main/1_Pooling/config.json)、[sentence length](https://huggingface.co/FremyCompany/BioLORD-2023/blob/main/sentence_bert_config.json)、[modules](https://huggingface.co/FremyCompany/BioLORD-2023/raw/main/modules.json)。
- RAD-DINO：[model card](https://huggingface.co/microsoft/rad-dino)、[processor](https://huggingface.co/microsoft/rad-dino/raw/main/preprocessor_config.json)。
- OTIS：[official repository](https://github.com/oetu/otis)、[models_vit.py](https://github.com/oetu/otis/blob/main/models_vit.py)、[main_linprobe.py](https://github.com/oetu/otis/blob/main/main_linprobe.py)。README 指向 `util/preprocess.py` 和权重入口，具体数值归一化公式待确认。
- EchoPrime：[official repository](https://github.com/echonet/EchoPrime)、[model.py](https://github.com/echonet/EchoPrime/blob/main/echo_prime/model.py)、[utils.py](https://github.com/echonet/EchoPrime/blob/main/utils/utils.py)。

公开候选设置只有在运行 revision/哈希已固定、shape/接口检查通过、所用假设已记录后，才成为该次运行的具体技术方案。它们仍不自动证明与 NOAH 原作者的全部设置一致。
