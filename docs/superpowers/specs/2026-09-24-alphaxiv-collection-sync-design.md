# alphaXiv Collection Sync Design

## Purpose

Keep an existing alphaXiv collection aligned with the arXiv papers recorded in
`papers.csv` without coupling alphaXiv availability to paper discovery or
conversion. The integration is additive: it adds missing papers and never
removes collection entries.

## Considered approaches

1. **Separate workflow with a tested repository script (selected).** Trigger
   after every completed `Nightly papers` workflow and allow manual dispatch.
   This still syncs inventory commits when a later conversion step fails, keeps
   alphaXiv failures isolated from nightly execution, and makes parsing and
   idempotence testable offline.
2. **A final job in `nightly.yml`.** This gives one workflow view, but a failed
   conversion would prevent collection sync even when the inventory commit was
   already pushed. It also couples an external collection service to the
   template-owned pipeline workflow.
3. **A `push` trigger on `papers.csv`.** This is simple, but GitHub does not
   recursively trigger workflows for commits pushed with the repository
   `GITHUB_TOKEN`, so nightly-generated inventory commits may not launch it.

## Workflow

Add `.github/workflows/alphaxiv.yml` with two triggers:

- `workflow_run` after `Nightly papers` completes on `main`, regardless of the
  nightly conclusion because inventory commits can precede a safe conversion
  failure.
- `workflow_dispatch` for initial population and controlled retries.

The job checks out `main`, syncs the locked repository environment, installs
pinned `alphaxiv-py==0.7.0`, and invokes
`python -m papers_pipeline.alphaxiv_sync`. It has read-only repository
permissions, a bounded timeout, and non-cancelling concurrency so two
collection writes cannot race. The workflow does not modify repository
content.

Configuration is supplied by:

- GitHub Actions secret `ALPHAXIV_API_KEY`, containing a key authorized for
  folder writes.
- GitHub Actions variable `ALPHAXIV_COLLECTION`, containing an exact alphaXiv
  folder name or folder ID.

Missing configuration fails before any API request. No credentials are written
to disk, command output, artifacts, or repository files.

## Synchronization behavior

`papers_pipeline.alphaxiv_sync` reads unique, non-empty `arxiv_id` values from
`papers.csv`. It asks the alphaXiv CLI for the configured folder as normalized
JSON, canonicalizes both inventory and collection IDs by removing a trailing
arXiv version suffix, and adds only missing papers with:

```text
alphaxiv paper folders add <arxiv-id> <collection> --yes
```

The script is deliberately additive. Papers manually placed in the collection
are preserved, and repository removals do not delete remote entries. A failed
addition stops the run with a non-zero exit status. Earlier successful
additions remain safe; a retry rereads the collection and skips them.

## Validation

Offline unit tests cover CSV parsing, duplicate/version normalization,
idempotent skipping, missing configuration, malformed alphaXiv JSON, and
propagation of CLI failures. Workflow tests cover pinned actions, timeouts,
read-only permissions, triggers, concurrency, the pinned SDK version, and
secret/variable wiring.

Before publication, run configuration validation, the complete offline test
suite, pre-commit, `git diff --check`, and YAML/synchronization smoke checks.
Open a pull request to `main`; do not merge it or dispatch the alphaXiv
workflow.
