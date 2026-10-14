# O2 远程执行、本地核对与 SSH 协作

更新日期：2026-10-14。当前会话的工作目录是本地 Windows `E:\python_projects\NOAH_rep`。SSH、scp 客户端已在本机检测到，不代表 O2 已认证或当前会话已切到远端。

## 1. 目前推荐的方式

继续使用你已经在服务器运行的 CLI agent：服务器执行检查/授权后的实施与作业，本地核对代码、论文约束和报告。这种方式足够支持目前的阶段，没有必须立即建立 SSH 直连的依赖。

关键是将“报告 prose”与“具体执行证据”对齐：按 [reports/README.md](../reports/README.md)保存一个小型 report bundle，注明 source commit、实际路径、命令、范围、输出、退出状态和汇总数。先做最小 REPORT.md 即可。

需要频繁查看目录、诊断报错、确认 checkpoint/processor 或迭代同一作业时，直接 SSH/远端项目会更方便。**你在另一个终端执行 ssh，不会自动让本地这段会话获得那个终端的远端执行环境。**

## 2. GitHub 同步的最小工作流程

### 服务器端

1. 在实际 clone 中读取 `AGENTS.md`、当前进度、实施指南和行动计划。
2. 先检查 `git status --short`、当前 branch 和 `git rev-parse HEAD`；工作树不干净时不要盲目 pull/reset。
3. 对已同步的计划执行你明确指定的阶段；不要把“生成一份报告”自动扩展成训练模型。
4. 写 `reports/server/<date>_<run_id>/REPORT.md`，必要时附可安全同步的 JSON 和已清理日志。服务器缓存与临床内容留在服务器。
5. 在报告中写执行时 commit/dirty 状态，再提交本次实际修改的代码、配置和报告。使用明确文件路径选择性提交；不要直接 `git add .` 混入数据、缓存和旧临时文件。
6. 将已审查的修改推到约定分支；把 branch、commit、report 路径告诉本地审阅者。提交前遵循 [仓库发布规范](REPOSITORY_POLICY.md)。

### 本地端

在 PowerShell、仓库目录中，先取远端信息并查看状态：

```powershell
git fetch origin
git status --short
```

工作树干净且分支允许 fast-forward 时，才使用：

```powershell
git pull --ff-only
```

当前本地有未提交文档和未跟踪文件，不能假设直接 pull 一定合适。先保存/提交需要的文档，再同步；不为同步删除用户修改。不需要合并时，也可以在 fetch 后用 `git show origin/<branch>:reports/server/<bundle>/REPORT.md` 读取明确的远端文件。

让服务器端执行者和本地审阅者使用同一 source commit；报告提交本身改变 HEAD，所以报告内仍保留原执行 commit。如果报告对应未提交代码，另存可审查的 diff。不要为了对齐覆盖另一端正在使用的分支。

## 3. 可直接交给服务器 CLI 的检查任务

以下是可复制的**只读检查提示**：

> 先读 AGENTS.md、docs/PROJECT_STATUS.md、docs/IMPLEMENTATION_GUIDE.md、docs/REPRODUCTION_PLAN.md 和 reports/README.md。完成 N-01：仅检查当前环境和数据/资产布局，写 reports/server/<实际日期>_<run_id>/REPORT.md。不要开始新模型实施、全量 ETL、下载模型或训练。
>
> 记录实际 repo/branch/commit/dirty 状态和解释器。检查 /home/wey371/rajpurkarlab/datasets 的实际位置、顶层/少量必要子目录、相关 CSV 或 CSV.GZ、ECG hea/dat 和 CXR 图像索引/样本路径。检查已有 encoder checkpoint/processor 的文件与版本；特别确认 OTIS 的 1x24 兼容性，以及 resized CXR 的来源。若尚无资产则写 UNKNOWN 或未发现，不擅自替换。
>
> 仅对合理小样本做读取/形状检查，保留抽样方法；全目录计数、表扫描、模型推理和测试等计算工作在适当 Slurm 资源中运行。若未申请/未执行，写 NOT_RUN，不把范围扩成全部数据。报告只包含可同步的汇总、命令和已清理日志，不写患者原文/媒体/新患者细表。列出 N-02/N-03 的可行动结果和下一步。不要仅给“数据完整/98.5%可用”的结论，也不要推断 GPU 脚本已经训练。

首次报告最重要的是路径/格式/解释器/现有资产。后续按阶段追加编码成功数、窗口/mask 和训练结果，不要求第一份检查把全部 559M 事件扫描一遍。

## 4. 基本 SSH：先验证你能登录

O2 的 SSH hostname 是 `o2.hms.harvard.edu`，用户名是你自己的小写 HMS ID。你本机已有 OpenSSH，因此可直接在 PowerShell 执行下面的命令（先替换占位符）：

```powershell
ssh <your_hms_id>@o2.hms.harvard.edu
```

密码与 Duo 认证由你在本地完成，不发送到聊天或报告。HMS 外部网络的登录通常有两因素认证；是否需 VPN 取决于实际网络路径，不是所有场景一律先开 VPN。[HMS 官方登录说明](https://harvardmed.atlassian.net/wiki/spaces/O2/pages/1601700123/How%2Bto%2Blogin%2Bto%2BO2)、[两因素说明](https://harvardmed.atlassian.net/wiki/spaces/O2/pages/1605009747)

登录后做轻量定位：

```bash
hostname
pwd
git -C <remote_repo_absolute_path> rev-parse HEAD
```

远端 repo 路径需要你实际确认；不能把 dataset root 当成 repo root，也不能把脚本里的 venv 路径当成已经存在且正确。

## 5. 计算与服务器 CLI 在 Slurm 作业中运行

O2 登录节点用于轻量操作、编辑和提交作业；表扫描、编码、训练等不要直接运行在登录节点。按 [HMS Slurm 官方说明](https://harvardmed.atlassian.net/wiki/spaces/O2/pages/1586793632)申请 batch 或 interactive 资源。

一个 CPU 交互会话的**资源示例**如下；分区/时限/内存应匹配你的账户和本次工作，不能把例子当成已验证配额：

```bash
srun --pty -p interactive -c 2 --mem=8G -t 02:00:00 /bin/bash
```

分配成功后，确认计算节点与仓库，再启动已经安装和认证的 CLI 工具：

```bash
hostname
cd <remote_repo_absolute_path>
git rev-parse HEAD
```

如果使用 Codex CLI，之后运行 `codex`；如果用其他已配置工具，使用它自己的启动命令。Codex 在启动它的机器/项目目录中读写和运行工具。[OpenAI 官方 CLI 说明](https://learn.chatgpt.com/docs/codex/cli)

需要编码器 GPU 推理或训练时，另申请 GPU 作业；当前 `run_pilot_gpu.sh` 没有训练命令，提交它只能准备环境，不能完成计划中的首轮训练。暂时无需在本指南中替你提交任何 GPU 作业。

## 6. 让桌面应用使用远端项目（可选）

按 [OpenAI 官方 SSH 远端项目说明](https://learn.chatgpt.com/docs/remote-connections#connect-to-an-ssh-host)，桌面应用可以读取具体 SSH host alias，远端需要已安装/认证、在 login-shell PATH 中可用的 Codex。应用设置入口为 `Settings > Connections`，添加 SSH host 并选择远端项目目录。是否显示该入口以你当前应用/工作区可用功能为准。

对于 O2，建议连接已分配的计算节点运行长期 agent，而非把远端 app server 当作常驻进程放在登录节点。可在本机 `%USERPROFILE%\.ssh\config` 中**追加而非覆盖**以下示例；用户名和计算节点名来自你实际的 Slurm 会话：

```sshconfig
Host o2login
    HostName o2.hms.harvard.edu
    User <your_hms_id>

Host o2job
    HostName <allocated_compute_node>
    User <your_hms_id>
    ProxyJump o2login
```

已有可用 SSH key 时才添加匹配的 `IdentityFile`；不要为省事关闭主机校验或绕过 HMS 认证。先在本机验证 `ssh o2job` 能进入**仍在分配中的**节点，再将其作为应用的 SSH 项目 host。HMS 官方也说明了通过登录节点连接作业节点的方式。[HMS 远程开发说明](https://harvardmed.atlassian.net/wiki/spaces/O2/pages/2051211265)

节点分配到期/断线/认证问题会使这种连接失效；长训练应由可恢复的 batch job 执行，不依赖桌面连接一直在线。若远端 CLI 尚不可用、app 不支持该连接或认证不能完成，就继续第 2-3 节的报告流程，不需要为了当前计划强行配置新连接。

若之后希望本地审阅者直接查看远端，请提供：已验证的 host alias、远端 repo 绝对路径、所用 branch/commit、当前可做的范围（只读/具体阶段）、以及 MFA 是否需要你交互处理。不要提供密码、私钥或 token。普通终端 SSH 成功也不等于当前本地任务已迁移到服务器。

## 7. 参考资料与连接验收

连接步骤参考上述 OpenAI 与 HMS 官方资料。账户认证、Slurm 配额与连接功能以当前环境为准。

本机已检测到 ssh/scp 客户端；没有验证 O2 登录、用户 SSH config、节点分配、远端 Codex 可执行性或目录读取。服务器执行报告补齐的是这些具体信息，而不是重复确认是否存在 O2 平台。
