# Speech Enhancement Papers Canary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a validated pull request containing the first standalone canary generated from `will-rice/papers-template` release `v0.1.0`.

**Architecture:** Render the released Copier template into the existing repository, then specialize only declarative discovery policy, bounded conversion limits, and the nightly trigger. Preserve generated pipeline code and the repository-owned corpus/state boundary, while adapting workflow tests and documentation to the canary's intentionally manual launch.

**Tech Stack:** Copier 9.10.2, Python 3.12, uv, Pydantic, PyYAML, pytest, pre-commit, GitHub Actions

## Global Constraints

- Generate from `https://github.com/will-rice/papers-template.git` at immutable release `v0.1.0` with Copier `9.10.2`.
- Use repository name `Speech Enhancement Papers`, slug `speech-enhancement-papers`, and description `Research papers on speech enhancement, denoising, dereverberation, and restoration`.
- Enable only arXiv and require no secrets.
- Keep `papers/`, `papers.csv`, `.papers-state.yml`, and caches repository-owned across Copier updates.
- Keep the nightly workflow manual-only until a successful reviewed manual run.
- Do not dispatch the nightly workflow, merge the pull request, add a custom topic plugin, or broaden the approved topic.

---

### Task 1: Render the released template

**Files:**
- Create: `.copier-answers.yml`
- Create: generated repository files from `template/`
- Preserve: `README.md` repository identity while replacing it with generated documentation

**Interfaces:**
- Consumes: Copier source `https://github.com/will-rice/papers-template.git`, release `v0.1.0`
- Produces: a generated Python package, workflows, tests, lockfile, and Copier provenance

- [x] **Step 1: Verify the immutable release**

Run:

```bash
git ls-remote --tags https://github.com/will-rice/papers-template.git refs/tags/v0.1.0
```

Expected: `refs/tags/v0.1.0` resolves to
`eecf1d519dbb146ecd5ad89cc05e6c636e249eef`.

- [x] **Step 2: Render with explicit answers**

Run:

```bash
uvx --from copier==9.10.2 copier copy \
  --trust \
  --overwrite \
  --vcs-ref v0.1.0 \
  --data project_name='Speech Enhancement Papers' \
  --data project_slug='speech-enhancement-papers' \
  --data topic_description='Research papers on speech enhancement, denoising, dereverberation, and restoration' \
  --data template_version='0.1.0' \
  https://github.com/will-rice/papers-template.git .
```

Expected: Copier reports `Copying from template version 0.1.0` and writes
`.copier-answers.yml`.

- [x] **Step 3: Verify recorded provenance**

Run:

```bash
uv run python - <<'PY'
from pathlib import Path
import yaml

answers = yaml.safe_load(Path(".copier-answers.yml").read_text())
assert answers["_src_path"] == "https://github.com/will-rice/papers-template.git"
assert answers["_commit"] == "v0.1.0"
assert answers["template_version"] == "0.1.0"
assert answers["project_slug"] == "speech-enhancement-papers"
PY
```

Expected: exit code 0 with no output.

### Task 2: Configure the bounded canary

**Files:**
- Modify: `papers.yml`
- Modify: `.github/workflows/nightly.yml`
- Modify: `tests/test_workflows.py`
- Modify: `README.md`
- Delete: `topic_plugin.py`

**Interfaces:**
- Consumes: generated `PipelineConfig`, arXiv adapter `filters.search_query`, and workflow loader tests
- Produces: declarative topic policy, bounded conversion behavior, and a manual-only nightly workflow

- [x] **Step 1: Configure arXiv and topic gates**

Replace `papers.yml` with:

```yaml
repository:
  name: "Speech Enhancement Papers"
  slug: "speech-enhancement-papers"
  description: "Research papers on speech enhancement, denoising, dereverberation, and restoration"
adapters:
  - name: arxiv
    enabled: true
    secret_env: null
    lookback_days: 30
    page_size: 50
    max_pages: 3
    max_results: 100
    filters:
      search_query: '(cat:eess.AS OR cat:cs.SD) AND (all:"speech enhancement" OR all:"speech denoising" OR all:"noise suppression" OR all:dereverberation OR all:"speech restoration" OR all:"speech super-resolution")'
topic:
  include_any:
    - "speech enhancement"
    - "speech denoising"
    - "noise suppression"
    - "dereverberation"
    - "speech restoration"
    - "speech super-resolution"
  include_all: []
  exclude_any: []
  categories:
    - "eess.AS"
    - "cs.SD"
  plugin: null
fetch:
  request_timeout_seconds: 30
  retries: 3
  backoff_seconds: 1
  total_deadline_seconds: 900
conversion:
  max_batches_per_run: 1
  max_papers: 5
  max_cost: 50
  html_cost: 2
  latex_cost: 4
  pdf_cost: 20
concurrency:
  html: 2
  latex: 1
  pdf: 1
```

- [x] **Step 2: Make nightly dispatch-only**

Set the trigger in `.github/workflows/nightly.yml` to:

```yaml
on:
  workflow_dispatch:
```

Update `test_nightly_has_non_overlapping_mutation_concurrency` in
`tests/test_workflows.py` to assert:

```python
assert set(data["on"]) == {"workflow_dispatch"}
```

- [x] **Step 3: Remove the unused custom plugin**

Delete `topic_plugin.py`. Keep `topic.plugin: null` so all acceptance decisions
come from the declarative phrases and categories.

- [x] **Step 4: Align operator documentation**

State in `README.md` that the nightly workflow is initially dispatch-only and
that its schedule is enabled only after a successful manual run is reviewed.
Keep the generated ownership-boundary documentation intact.

- [x] **Step 5: Run focused configuration and workflow tests**

Run:

```bash
UV_OFFLINE=true uv run pytest tests/test_config.py tests/test_workflows.py
```

Expected: all selected tests pass.

- [x] **Step 6: Commit the canary configuration**

Run:

```bash
git add . ':!docs/superpowers/plans/2026-09-24-speech-enhancement-canary.md'
git commit -m "feat: generate speech enhancement papers canary" \
  -m "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
```

Expected: a conventional feature commit containing generated files and
canary-specific configuration.

### Task 3: Validate the complete repository

**Files:**
- Modify only if validation identifies a generation or configuration defect

**Interfaces:**
- Consumes: locked dependency graph, generated CLI, offline fixtures, hooks, and workflows
- Produces: exact validation evidence suitable for the pull request and parent report

- [x] **Step 1: Sync the locked environment**

Run:

```bash
uv sync --locked --extra dev
```

Expected: exit code 0 and a Python 3.12 development environment matching
`uv.lock`.

- [x] **Step 2: Validate configuration without converter installations**

Run:

```bash
uv run papers-pipeline validate --config papers.yml --config-only
```

Expected: `valid: papers.yml`.

- [x] **Step 3: Run the complete offline suite**

Run:

```bash
UV_OFFLINE=true uv run pytest
```

Expected: exit code 0 with every collected test passing.

- [x] **Step 4: Run repository hooks**

Run:

```bash
uv run pre-commit run --all-files
```

Expected: `ruff check`, `ruff format`, and `mypy` all pass.

- [x] **Step 5: Run smoke assertions**

Run:

```bash
git diff --check
uv run python - <<'PY'
from pathlib import Path
import yaml
import papers_pipeline

answers = yaml.safe_load(Path(".copier-answers.yml").read_text())
config = yaml.safe_load(Path("papers.yml").read_text())
workflow = yaml.load(
    Path(".github/workflows/nightly.yml").read_text(),
    Loader=yaml.BaseLoader,
)
assert answers["_src_path"] == "https://github.com/will-rice/papers-template.git"
assert answers["_commit"] == "v0.1.0"
assert answers["template_version"] == "0.1.0"
assert [adapter["name"] for adapter in config["adapters"]] == ["arxiv"]
assert config["topic"]["plugin"] is None
assert set(workflow["on"]) == {"workflow_dispatch"}
assert papers_pipeline.__file__
PY
```

Expected: both commands exit 0 with no assertion failures.

### Task 4: Publish without running automation

**Files:**
- Modify: `docs/superpowers/plans/2026-09-24-speech-enhancement-canary.md`

**Interfaces:**
- Consumes: validated commits on the feature branch
- Produces: a pushed branch, an open pull request to `main`, and a parent-session report

- [x] **Step 1: Mark completed plan steps and commit the plan**

Run:

```bash
git add docs/superpowers/plans/2026-09-24-speech-enhancement-canary.md
git commit -m "docs: add speech enhancement canary plan" \
  -m "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
```

Expected: a conventional documentation commit.

- [ ] **Step 2: Push the feature branch**

Run:

```bash
git push --set-upstream origin will-rice-speech-enhancement-canary
```

Expected: the remote branch is created without dispatching any workflow.

- [ ] **Step 3: Open the pull request**

Create a non-draft pull request targeting `main` whose body records template
provenance, the manual-only nightly gate, configuration limits, and exact
validation results.

Expected: an open pull request URL in `will-rice/speech-enhancement-papers`.

- [ ] **Step 4: Verify publication state**

Run:

```bash
gh pr view --json url,state,isDraft,baseRefName,headRefName,commits
gh run list --workflow nightly.yml --branch will-rice-speech-enhancement-canary --limit 1
```

Expected: the pull request is open, non-draft, targets `main`, and no nightly
workflow run was dispatched for the feature branch.

- [ ] **Step 5: Report to the parent session**

Send the pull request URL, every commit SHA, exact command results, and any
blockers to project session `8cd59274-557d-49a8-9304-120c87989540`.
