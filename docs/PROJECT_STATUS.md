# NOAH 项目进度与当前行动清单

更新日期：2026-10-14（America/New_York）。代码基线：`30dd669325f99e4ce9fe554cae2934376f35f879`，另有本地文档更新。依据是当前本地文件、论文核对和研究者提供的信息；服务器执行结果待报告补齐。

## 1. 目标和当前位置

长期目标是科学上忠实、可测试、可追溯的 NOAH 复现，包括论文规模的数据、模型和评价。近期目标是 **固定 1,000 人候选清单，完成一次小规模真实数据训练、验证、重启和先验 rollout**。首先验证论文特有机制，而非要求小样本达到论文指标。

当前阶段：**论文审计与部分基础代码已具备，完整数据-训练路径尚未接通。**

| 层次 | 当前状态 | 判断依据 |
|---|---|---|
| 论文架构/追溯/假设/测试计划 | 文档已完成，并补充操作指南 | `docs/REPRODUCTION_SPEC.md` 等及 `IMPLEMENTATION_GUIDE.md` |
| Pilot 候选患者及划分 | 清单已生成；独立统计 1,000 unique IDs，700/150/150 | `data/pilot_cohort_manifest.csv` |
| 患者筛选代码 | 已写；成年人、住院、ECG、CXR 交集，seed=42 | `src/data/subcohort.py`；富集工程子集 |
| 部分事件提取 | 代码已写：入出院/死亡、数值化验、ECG/CXR 索引 | `src/data/extract_events.py` |
| 数值、vitals、时间模块和部分测试 | helper 和测试定义已写；执行结果待记录 | `numeric.py`、`vitals.py`、`temporal.py`、`tests/test_temporal_oracle.py` |
| 文本、媒体解码及冻结 embedding | 完整接入/缓存产物未在当前仓库找到 | 索引/path 读取不等于内容提取 |
| 窗口、前缀与 loss masks | 有 schema 字段；完整构建/验收仍待完成 | 当前没有完整 window builder |
| 完整 NOAH、双路径训练和 rollout | 尚待实现/集成 | 当前模型目录仅时间 helper |
| O2 算力平台 | 研究者已确认可用 | 实际分配和吞吐待运行测量 |
| 首次完整服务器运行 | 还没有对应完成证据 | GPU 脚本仍占位，CPU 入口只调用两名患者读取示例 |

本地目录中有文档和演示材料，但不能据此推断它们均已提交到 GitHub。以上患者数是候选清单统计，尚不是完成全部事件资格筛选后的最终训练人数；服务器候选池总人数也没有由这份清单确定。

## 2. 已讨论的计划与仍需决定的范围

| 项目 | 当前约定 |
|---|---|
| 近期规模 | 用现有 1,000 人候选清单，不先扩大到 5,000 人 |
| 时间估计 | 研究者接受暂按一名主要开发者、必要资产/数据与 GPU 可用、4-6 周的工作假设；不是已开始的倒计时或实测训练时长 |
| Pilot 模态 | 提议先用 categorical、numeric、text、CXR、ECG；ECHO 的先后安排仍需范围确认 |
| 模型规模 | 提议缩小宽度/层数/上下文以验证流程；具体配置未冻结 |
| 必须保留的机制 | 方向性时间注意力、因果内容 mask、conditional prior/informed posterior、共享 latent、层级解码、双重建与 KL、prior-only rollout |
| 后续完整目标 | 六模态、完整事件覆盖、论文架构和评价；ECHO 和更大算力在此阶段安排 |
| 阶段执行范围 | 启动 pilot 时从 N-01/N-02/N-03 开始，按研究者当前任务推进；论文规模训练另定范围和资源 |

## 3. 接下来优先做什么

这里的状态均为待办，不代表已经运行。实施细节见 [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)。

| 优先级 / ID | 下一步 | 完成证据 | 依赖与建议执行位置 |
|---|---|---|---|
| P0 / N-01 | 生成一份服务器当前状态报告 | 代码 commit/dirty 状态、实际 repo/interpreter 路径、数据根目录/文件格式、汇总读样本结果、现有 checkpoint 与作业状态 | 服务器 CLI 只读检查；按 [报告约定](../reports/README.md) |
| P0 / N-02 | 冻结第一版 pilot 范围与最少必要设置 | active modalities、事件集合、tiny 参数、时间/目标对齐、数值基与 loss 选择；逐项分类 | 本地文档/配置设计；关键范围由研究者/PI 确定 |
| P0 / N-03 | 核对冻结编码器与源数据的具体接口 | BioLORD 长文策略、CXR 副本与 processor、兼容 OTIS 1x24 的权重/池化；先验收各模态少量真实样本 | 服务器有资产后；不能用宽 100 的默认 OTIS 直接代替 |
| P1 / N-04 | 补齐并执行核心独立参考测试 | 时间/移位、mask ownership、Gaussian KL、共享 latent/层级解码、数值基、推理信息边界的结果 | 数学与合成数据工作不依赖 ECHO 或完整真实数据 |
| P1 / N-05 | 完成选定事件读取、语义清洗、embedding 和窗口 | 两人小样本的真实内容路径和 mask 验收；随后 1,000 候选者的各阶段汇总 | N-02/N-03；不要在不完整 readers 上提前以 64 events 排除患者 |
| P1 / N-06 | 实现并连接完整 tiny NOAH | 合成 forward/backward、tiny-batch overfit、双路径损失与 prior-only rollout | 实施授权、N-02/N-04；可与 N-05 独立推进 |
| P1 / N-07 | 首次完整真实数据运行 | 训练/验证、checkpoint 保存/恢复、prior rollout、组件损失和资源统计 | N-05/N-06 验收通过 |
| P2 / N-08 | 决定扩大范围与资源申请 | 更自然的人群设计、ECHO 路径、完整事件/模型设置、基于实测的时间/存储/GPU 预算 | N-07；不凭人数或脚本时限估算完整成本 |

当前技术缺口主要是 N-03/N-05/N-06 的连接与实现。具体 encoder 文件、真实目录和环境属于待核对依赖，不应直接写成“不可获得”。ECHO 缺失只阻塞完整六模态目标，是否阻塞首个 pilot 取决于范围选择。

## 4. 首次完整运行怎样判定

- 完成真实内容读取/编码、按患者划分、窗口与 ownership；排除和失败有计数及原因。
- 核心数学/时间/因果测试通过；未知最后 gap 与真实零 gap 分开。
- 同一运行执行 posterior/prior 重建、KL 与验证；记录各分量及退火时钟。
- 保存后能恢复模型、优化器和运行进度，并说明随机状态与数据顺序恢复策略。
- rollout 在没有目标 embedding 的条件下用 prior/history 生成事件，包含时间和内容表示。
- 留下代码/配置/数据索引与缓存版本、作业 ID/退出状态、资源和测试汇总。

“作业 COMPLETED”“模型能 forward”“总 loss 下降”单独都不构成上述里程碑。较短 pilot 的配置和指标要与论文规模目标分开。

## 5. 如何更新本文件

每次服务器工作完成，先保存对应 [report bundle](../reports/README.md)，再更新本表的日期、证据路径、状态和下一动作。用“已写代码 / 已生成产物 / 已通过测试 / 已完成运行 / 待办 / 待确认”描述，避免一个不透明的完成百分比。

将 `REPRODUCTION_PLAN.md` 作为行动计划、`IMPLEMENTATION_GUIDE.md` 作为技术做法、`ASSUMPTIONS.md` 作为科学选择登记；PPT 是这些信息的汇报摘要，不是项目状态的唯一记录。

远程协作与 SSH 操作见 [REMOTE_WORKFLOW.md](REMOTE_WORKFLOW.md)。目前可以继续“服务器 CLI 执行 + GitHub 同步代码和汇总报告 + 本地核对”方式，不必先切换连接架构。
