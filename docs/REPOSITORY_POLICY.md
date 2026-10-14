# Repository content and publishing policy

## What belongs in Git

Keep reproduction source, tests, configuration, dependency definitions, current documentation, the reference paper, and reviewed aggregate execution reports. Small synthetic fixtures and necessary manifests require an explicit content review before publication. The existing cohort manifest is tracked; this policy does not establish permission to redistribute patient-level records.

## What stays local or on O2

Presentation generation scripts, PPT files, slide renders and intermediate inspection files stay in `.data-audit-ppt-build/` or `output/`. Raw clinical tables, notes, images, waveforms, videos, extracted patient events, embeddings, model weights and checkpoints stay outside Git. Use ignored `data/raw/`, `data/processed/`, `data/cache/`, `data/embeddings/`, `cache/`, `embeddings/`, `checkpoints/`, `runs/` and `logs/` when local directories are needed. Keep unrelated scratch work in ignored `tmp/` or `temp/`.

Reviewed server summaries belong in `reports/server/`. Include commands, source commit, configuration hashes, aggregate counts and job outcomes; exclude patient content, identifying record listings, credentials and raw job output containing clinical records. Presentation outputs remain available locally.

## Before committing or pushing

Inspect the working tree and tracked files, then stage only explicit reviewed paths:

```sh
git status --short
git status --short --ignored
git ls-files
git ls-files -ci --exclude-standard
```

The last command detects tracked files that match ignore rules. `.gitignore` prevents adding untracked files; it does not remove tracked files or erase Git history.

Review the exact staged content before committing, and review outgoing commits before pushing:

```sh
git diff --cached --name-status
git diff --cached --stat
git diff --cached
git log --oneline @{upstream}..HEAD
git diff --stat @{upstream} HEAD
```

The outgoing checks require a configured upstream and refreshed remote refs. Do not use blanket `git add .` or force-add ignored artifacts. If an unwanted file is already tracked, preserve its local copy and arrange a reviewed index removal; handle any history cleanup separately.

## Documentation style

English is the repository language for collaborators. Write documentation, reports, reusable agent prompts, comments/docstrings, test descriptions, user-facing messages and commit/PR descriptions in English; use English filenames. Private conversation may be in Chinese. Preserve technical identifiers, source titles and original data values. Clinical data is not translated or uploaded to satisfy this policy. Keep private Chinese explanations outside versioned repository content.

Documentation describes current behavior, current progress, unresolved technical choices, acceptance criteria and next actions. Update it in place. Keep historical comparisons, superseded reports and narrative responses to earlier versions in Git history. Keep source provenance and dated execution evidence where they support scientific reproducibility. Preserve distinctions between paper requirements, implementation assumptions and measured results without repeating historical corrections.
