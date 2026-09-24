# alphaXiv Collection Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an authenticated GitHub Action that idempotently adds every arXiv paper in `papers.csv` to a configured alphaXiv collection.

**Architecture:** A focused `papers_pipeline` module parses inventory and drives the pinned `alphaxiv` CLI, while a separate read-only workflow invokes it after every completed nightly run or on manual dispatch. The module compares canonical arXiv IDs before writing, never removes remote entries, and fails explicitly on configuration, payload, or CLI errors.

**Tech Stack:** Python 3.12 standard library, `alphaxiv-py==0.7.0`, GitHub Actions, pytest, PyYAML

## Global Constraints

- Use GitHub secret `ALPHAXIV_API_KEY` with folder-write authorization; never
  print or persist it.
- Use GitHub variable `ALPHAXIV_COLLECTION` as the exact folder name or ID.
- Pin the external CLI to `alphaxiv-py==0.7.0`.
- Keep synchronization additive; never remove alphaXiv collection entries.
- Trigger after completed `Nightly papers` runs on `main`, regardless of conclusion, and support `workflow_dispatch`.
- Keep repository permissions read-only and serialize collection updates.
- Do not merge or dispatch the new workflow while preparing the pull request.

---

### Task 1: Implement the idempotent sync script

**Files:**
- Create: `src/papers_pipeline/alphaxiv_sync.py`
- Create: `tests/test_sync_alphaxiv.py`

**Interfaces:**
- Consumes: CSV columns `source` and `arxiv_id`; normalized JSON from `alphaxiv folders show <collection> --json`
- Produces: `read_arxiv_ids(path: Path) -> list[str]`, `canonical_arxiv_id(value: str) -> str`, `sync_collection(inventory: Path, collection: str, runner: Runner = subprocess.run) -> SyncResult`, and CLI exit status

- [x] **Step 1: Write parser and normalization tests**

Create `tests/test_sync_alphaxiv.py` with temporary inventories proving that
`read_arxiv_ids`:

```python
def test_read_arxiv_ids_filters_deduplicates_and_preserves_order(tmp_path: Path) -> None:
    inventory = tmp_path / "papers.csv"
    inventory.write_text(
        "source,arxiv_id\n"
        "arxiv,2608.26403v1\n"
        "dblp,\n"
        "arxiv,2608.26403v1\n"
        "arxiv,2608.28493v2\n",
        encoding="utf-8",
    )

    assert read_arxiv_ids(inventory) == ["2608.26403v1", "2608.28493v2"]
```

Also assert `canonical_arxiv_id("2608.26403v3") == "2608.26403"` and that a
missing `arxiv_id` header raises `ValueError("papers.csv is missing arxiv_id")`.

- [x] **Step 2: Run parser tests and verify failure**

Run:

```bash
UV_OFFLINE=true uv run pytest tests/test_sync_alphaxiv.py -q
```

Expected: collection fails because `papers_pipeline.alphaxiv_sync` does not exist.

- [x] **Step 3: Implement CSV parsing**

In `src/papers_pipeline/alphaxiv_sync.py`, use `csv.DictReader`, preserve
first-seen order, ignore rows whose `source` is not `arxiv`, ignore empty IDs,
deduplicate canonical IDs, and remove only a terminal `v` plus digits when
canonicalizing.

- [x] **Step 4: Run parser tests**

Run:

```bash
UV_OFFLINE=true uv run pytest tests/test_sync_alphaxiv.py -q
```

Expected: parser and canonicalization tests pass.

- [x] **Step 5: Write synchronization tests**

Add a fake subprocess runner and tests proving:

```python
result = sync_collection(inventory, "Speech Enhancement", runner=fake_runner)
assert result.inventory_count == 2
assert result.existing_count == 1
assert result.added_count == 1
assert fake_runner.commands == [
    ["alphaxiv", "folders", "show", "Speech Enhancement", "--json"],
    [
        "alphaxiv",
        "paper",
        "folders",
        "add",
        "2608.28493v2",
        "Speech Enhancement",
        "--yes",
    ],
]
```

The folder fixture must contain:

```json
{
  "papers": [
    {"preferred_id": "2608.26403v2"}
  ]
}
```

Also test that malformed JSON, a non-object payload, a missing `papers` list,
and a failed CLI command raise explicit exceptions rather than returning a
success-shaped result.

- [x] **Step 6: Implement synchronization**

Define:

```python
@dataclass(frozen=True)
class SyncResult:
    inventory_count: int
    existing_count: int
    added_count: int


Runner = Callable[..., subprocess.CompletedProcess[str]]
```

Run the folder lookup with `capture_output=True`, `text=True`, and
`check=True`; validate its JSON shape; compare canonical IDs; then invoke one
non-interactive add command per missing ID with `check=True`. Return counts and
print:

```text
alphaXiv sync complete: inventory=<n> existing=<n> added=<n>
```

The CLI entry point requires non-empty `ALPHAXIV_API_KEY` and
`ALPHAXIV_COLLECTION`, accepts `--inventory` defaulting to `papers.csv`, and
returns non-zero for any exception.

- [x] **Step 7: Run sync tests**

Run:

```bash
UV_OFFLINE=true uv run pytest tests/test_sync_alphaxiv.py -q
```

Expected: all sync tests pass without network access.

### Task 2: Add the GitHub Action

**Files:**
- Create: `.github/workflows/alphaxiv.yml`
- Modify: `tests/test_workflows.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `ALPHAXIV_API_KEY`, `ALPHAXIV_COLLECTION`, `papers.csv`, and `papers_pipeline.alphaxiv_sync`
- Produces: serialized additive collection updates after nightly completion or manual dispatch

- [x] **Step 1: Extend workflow tests**

Update the expected workflow filenames to include `alphaxiv.yml`, then add a
test asserting:

```python
data["on"] == {
    "workflow_run": {
        "workflows": ["Nightly papers"],
        "types": ["completed"],
        "branches": ["main"],
    },
    "workflow_dispatch": None,
}
assert data["permissions"] == {"contents": "read"}
assert data["concurrency"] == {
    "group": "alphaxiv-collection",
    "cancel-in-progress": False,
}
```

Also assert every job and step has a timeout, actions are SHA-pinned, the
workflow installs `alphaxiv-py==0.7.0`, checks out `main`, passes
`${{ secrets.ALPHAXIV_API_KEY }}` and `${{ vars.ALPHAXIV_COLLECTION }}`, and
runs `uv run python -m papers_pipeline.alphaxiv_sync`.

- [x] **Step 2: Run workflow tests and verify failure**

Run:

```bash
UV_OFFLINE=true uv run pytest tests/test_workflows.py -q
```

Expected: failure because `alphaxiv.yml` does not exist.

- [x] **Step 3: Create the workflow**

Create `.github/workflows/alphaxiv.yml` with:

```yaml
name: Sync alphaXiv collection

on:
  workflow_run:
    workflows: ["Nightly papers"]
    types: [completed]
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: alphaxiv-collection
  cancel-in-progress: false

jobs:
  sync:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - name: Check out main
        uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262
        timeout-minutes: 2
        with:
          ref: main
          persist-credentials: false
      - name: Set up Python
        uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065
        timeout-minutes: 2
        with:
          python-version: "3.12"
      - name: Install pinned alphaXiv CLI
        timeout-minutes: 3
        run: |
          python -m pip install uv==0.8.17
          uv sync --locked
          uv tool install alphaxiv-py==0.7.0
          echo "$HOME/.local/bin" >> "$GITHUB_PATH"
      - name: Sync collection
        timeout-minutes: 20
        env:
          ALPHAXIV_API_KEY: ${{ secrets.ALPHAXIV_API_KEY }}
          ALPHAXIV_COLLECTION: ${{ vars.ALPHAXIV_COLLECTION }}
        run: uv run python -m papers_pipeline.alphaxiv_sync
```

- [x] **Step 4: Document setup and semantics**

Add an `alphaXiv collection` section to `README.md` that names the required
secret and variable, explains additive behavior, notes that nightly completion
includes safe partial runs, and gives the manual workflow as the initial
population path.

- [x] **Step 5: Run focused tests**

Run:

```bash
UV_OFFLINE=true uv run pytest tests/test_sync_alphaxiv.py tests/test_workflows.py -q
```

Expected: all selected tests pass.

- [x] **Step 6: Commit the implementation**

Run:

```bash
git add src/papers_pipeline/alphaxiv_sync.py tests/test_sync_alphaxiv.py tests/test_workflows.py \
  .github/workflows/alphaxiv.yml README.md
git commit -m "feat: sync papers to alphaxiv collection" \
  -m "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
```

### Task 3: Validate and publish

**Files:**
- Modify: `docs/superpowers/plans/2026-09-24-alphaxiv-collection-sync.md`

**Interfaces:**
- Consumes: completed sync script, workflow, tests, and documentation
- Produces: validation evidence and an unmerged pull request

- [x] **Step 1: Run complete validation**

Run:

```bash
uv run papers-pipeline validate --config papers.yml --config-only
UV_OFFLINE=true uv run pytest
uv run pre-commit run --all-files
git diff --check
```

Expected: configuration valid, all tests pass, Ruff and mypy pass, and no
whitespace errors.

- [x] **Step 2: Run workflow and script smoke checks**

Parse `.github/workflows/alphaxiv.yml` with the repository's `WorkflowLoader`
and assert both triggers, read-only permissions, pinned dependency, main
checkout, secret/variable wiring, and script command. Run the script against a
temporary fake `alphaxiv` executable and inventory to prove one missing paper
is added without a network request.

- [x] **Step 3: Commit the completed plan**

Mark all implementation and validation steps complete, then run:

```bash
git add docs/superpowers/plans/2026-09-24-alphaxiv-collection-sync.md
git commit -m "docs: add alphaxiv sync implementation plan" \
  -m "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
```

- [ ] **Step 4: Push and open the pull request**

Push `will-rice-alphaxiv-collection-sync`, open a non-draft pull request to
`main`, and include required repository setup and exact validation results.
Do not dispatch the alphaXiv workflow or merge the pull request.
