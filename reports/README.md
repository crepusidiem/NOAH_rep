# 服务器执行报告约定

这里定义服务器执行报告的保存方式。当前尚无真实 report bundle。发布内容遵循 [仓库规范](../docs/REPOSITORY_POLICY.md)。

## 保存位置

```text
reports/
  README.md
  server/
    YYYY-MM-DD_<run_id>/
      REPORT.md                 命令、范围、结果、失败和下一步
      environment.json         筛选后的版本/环境信息（可选）
      summary.json             汇总计数、形状、测试/资源结果（可选）
      sanitized_job_log.txt     已清理且可同步的日志（可选）
```

先用一个 `REPORT.md` 就能工作，无需先建设复杂报告系统。文件名和格式是工程建议，不是 NOAH 论文要求。代码/schema/config 与报告在同一分支对齐；报告注明**执行时的** source commit，而非提交报告之后的新 commit。

## 每份报告填写这些信息

1. 检查日期和时区、run ID、stage（只读检查/测试/ETL/编码/训练/推理）、执行者。
2. 实际 repo 绝对路径、branch、source commit、执行时是否有未提交更改；如有，附经过检查的代码 diff 或对应版本说明。
3. 实际解释器/环境和所需包版本；CPU/GPU 节点类型、Slurm job ID、状态、退出码、申请和实际资源（若适用）。
4. 数据根目录及解析后的实际位置、源版本、文件格式（如 `.csv` 或 `.csv.gz`）、所检查的表/媒体范围。目录存在与读样本成功分开写。
5. 原始命令或调用入口、配置路径/哈希、随机种子、样本量/抽样方式。不要只写“执行成功”。
6. 结果汇总：候选/保留患者、split、事件/模态、成功/失败数、embedding shape、窗口/scored token、测试数/失败原因等，仅填实际测量的部分。
7. 输出产物保存在服务器的哪个位置；小型可同步汇总的相对链接；数据/缓存/配置版本。
8. 实际耗时、显存/内存、吞吐（若测量）；排队时间、编码时间和训练时间分别写。
9. 问题及影响到哪一里程碑；下一步及对应 `PROJECT_STATUS.md` 的 N-ID。
10. 未运行的检查标记 `NOT_RUN`，不可用的测量标记 `NOT_MEASURED`，未确认的事实标记 `UNKNOWN`。不得补造数字或把申请的资源当成已获得资源。

报告应同步代码和汇总，不上传临床原文、原始媒体、embedding 缓存或新增患者级细表。日志先清理患者内容和凭据；环境信息用选定版本字段，不把完整 `env` 输出提交到仓库。

## 最小 REPORT.md 模板

```markdown
# Server report: <run_id>

- Checked at / timezone: <actual time>
- Stage: <inspection | tests | ETL | encoding | training | rollout>
- Repo / branch / source commit: <actual values>
- Dirty tree at execution: <yes/no and explanation>
- Interpreter / environment: <actual path and versions>
- Data root / scope: <actual paths, versions, formats, sample scope>
- Slurm job / state / exit code: <actual values or NOT_RUN>
- Command / config / seed: <exact sanitized values>

## Results
<Measured aggregate values. Use NOT_MEASURED for unmeasured fields.>

## Outputs
<Server artifact locations and links to safe summary files.>

## Failures and limitations
<Observed failures and untested scope.>

## Next action
<Action, dependency and PROJECT_STATUS N-ID.>
```

更新流程：[REMOTE_WORKFLOW.md](../docs/REMOTE_WORKFLOW.md)。当前进度：[PROJECT_STATUS.md](../docs/PROJECT_STATUS.md)。

Repository contents, ignored outputs and publishing checks follow [REPOSITORY_POLICY.md](../docs/REPOSITORY_POLICY.md). Documentation describes the current state; Git retains version history.
