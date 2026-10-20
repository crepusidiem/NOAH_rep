# O2 execution, local review and SSH collaboration

Updated: 2026-10-14. Local review uses Windows at `E:\python_projects\NOAH_rep`. OpenSSH/scp clients are available locally; O2 authentication, node allocation and remote tool execution require environment-specific verification.

## 1. Recommended workflow

Use the configured server CLI agent for inspection and explicitly requested implementation/jobs. Review code, paper constraints and reports locally. Direct SSH is not a prerequisite for this stage.

Align reports with execution evidence using [reports/README.md](../reports/README.md): source commit, actual paths, commands, scope, outputs, exit status and measured aggregates. A minimal `REPORT.md` is sufficient.

Direct SSH or a remote project is useful for frequent directory inspection, error diagnosis, checkpoint/processor verification and iterative job work. Opening SSH in another terminal does not automatically give a local agent that terminal's remote execution environment.

## 2. Minimal GitHub synchronization workflow

### Server

1. Read `AGENTS.md`, current status, the implementation guide and reproduction plan in the actual clone.
2. Check `git status --short`, branch and `git rev-parse HEAD`. Do not blindly pull/reset a dirty tree.
3. Execute the requested phase. A report-only request does not authorize model training.
4. Write `reports/server/<date>_<run_id>/REPORT.md`; add safe JSON or sanitized logs if needed. Keep caches and clinical content on the server.
5. Record execution-time commit/dirty state, then commit reviewed code, configuration and reports using explicit paths. Avoid blanket `git add .`.
6. Push reviewed changes to the agreed branch and give the reviewer the branch, commit and report path. Follow [REPOSITORY_POLICY.md](REPOSITORY_POLICY.md).

### Local

In the repository's PowerShell terminal:

```powershell
git fetch origin
git status --short
```

For a clean tree and a branch permitting fast-forward synchronization:

```powershell
git pull --ff-only
```

If local work is uncommitted, preserve/commit it before synchronization; do not delete another contributor's edits. When no merge is needed, read a specific remote report after fetching with `git show origin/<branch>:reports/server/<bundle>/REPORT.md`.

Use the same source commit for execution and review. Committing a report changes HEAD, so retain its original execution commit. For dirty runs, retain a reviewable code diff. Do not overwrite a branch actively used by the other environment.

## 3. Read-only task for the server CLI

Copyable N-01 prompt:

> Read AGENTS.md, docs/PROJECT_STATUS.md, docs/IMPLEMENTATION_GUIDE.md, docs/REPRODUCTION_PLAN.md and reports/README.md. Complete N-01 by inspecting the current environment and data/asset layout and writing reports/server/<actual_date>_<run_id>/REPORT.md in English. Do not start model implementation, full ETL, model downloads or training.
>
> Record the actual repo, branch, commit, dirty state and interpreter. Inspect the resolved /home/wey371/rajpurkarlab/datasets location, top-level and necessary child directories, relevant CSV/CSV.GZ files, ECG hea/dat pairs, and CXR indices/sample paths. Inventory existing encoder checkpoint/processor files and versions, particularly OTIS 1x24 compatibility and resized-CXR provenance. Mark absent or unconfirmed assets explicitly; do not substitute silently.
>
> Use reasonably small samples for read/shape checks and record the sampling method. Run full-directory counting, table scans, model inference and tests in suitable Slurm allocations. Mark unexecuted checks NOT_RUN and preserve the inspected scope. Include only aggregate summaries, sanitized commands and logs, without patient text/media/new patient-level tables. Provide actionable findings for N-02/N-03 and the next step. Establish coverage from measurements, and distinguish GPU script contents from executed training.

Prioritize paths, formats, interpreter and existing assets in the first report. Add encoding, window/mask and training results as stages run; N-01 does not require scanning all 559M events.

## 4. Basic SSH login

O2 hostname: `o2.hms.harvard.edu`. Use your lowercase HMS ID. In a local OpenSSH terminal, replace the placeholder:

```powershell
ssh <your_hms_id>@o2.hms.harvard.edu
```

Complete password/Duo authentication interactively; do not send credentials to chat or reports. External login typically uses two-factor authentication; VPN requirements depend on the network path. See [HMS login instructions](https://harvardmed.atlassian.net/wiki/spaces/O2/pages/1601700123/How%2Bto%2Blogin%2Bto%2BO2) and [two-factor instructions](https://harvardmed.atlassian.net/wiki/spaces/O2/pages/1605009747).

After login, perform lightweight location checks:

```bash
hostname
pwd
git -C <remote_repo_absolute_path> rev-parse HEAD
```

Confirm the actual repository path. The dataset root is not the repository root; a venv path appearing in a script does not establish that it exists or is correct.

## 5. Compute and CLI work in Slurm allocations

Login nodes support lightweight operations, editing and job submission. Run table scans, encoding and training in batch or interactive allocations using [HMS Slurm guidance](https://harvardmed.atlassian.net/wiki/spaces/O2/pages/1586793632).

Example CPU interactive request; adapt partition/time/memory to the account and task. This is not a verified quota:

```bash
srun --pty -p interactive -c 2 --mem=8G -t 02:00:00 /bin/bash
```

After allocation, verify the compute node and repository, then start the installed/authenticated CLI:

```bash
hostname
cd <remote_repo_absolute_path>
git rev-parse HEAD
```

For Codex CLI, run `codex`; otherwise use the configured tool's command. The CLI operates in the machine/project environment where it starts. See [OpenAI CLI documentation](https://learn.chatgpt.com/docs/codex/cli).

Request GPU allocations for encoder inference/training. Training entry points and earlier run reports exist. Inspect the actual batch script and current fidelity gates before submission; a script alone is not evidence of completed training.

## 6. Optional desktop remote project

The [OpenAI SSH remote-project guide](https://learn.chatgpt.com/docs/remote-connections#connect-to-an-ssh-host) describes SSH host aliases. The remote environment needs installed/authenticated Codex in login-shell PATH. Where available, use `Settings > Connections`, add the SSH host and select the remote repository. Feature availability depends on the current application/workspace.

For O2, run a long-lived agent on an allocated compute node rather than as a persistent login-node process. Append, rather than overwrite, this example in local `%USERPROFILE%\.ssh\config`; use the actual HMS ID and allocated node:

```sshconfig
Host o2login
    HostName o2.hms.harvard.edu
    User <your_hms_id>

Host o2job
    HostName <allocated_compute_node>
    User <your_hms_id>
    ProxyJump o2login
```

Add `IdentityFile` only for an existing usable key. Preserve host verification and HMS authentication. Verify that `ssh o2job` reaches a still-allocated node before using it as the app's project host. See [HMS remote development](https://harvardmed.atlassian.net/wiki/spaces/O2/pages/2051211265).

Allocation expiry, disconnection or authentication failures invalidate this connection. Run long training as resumable batch jobs independent of desktop connectivity. If the remote CLI/app connection/authentication is unavailable, continue the report workflow in sections 2-3.

For direct remote review, provide a verified host alias, absolute repo path, branch/commit, allowed task scope and whether interactive MFA is required. Do not provide passwords, private keys or tokens. Terminal SSH success does not automatically migrate a local task to the server.

## 7. Connection acceptance

Follow current OpenAI/HMS guidance and actual account behavior for authentication, Slurm quotas and connection features. Verify O2 login, SSH configuration, node allocation, remote tool executability and directory reads in server reports. O2 platform access is already confirmed; these checks establish the usable execution environment.
