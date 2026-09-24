# Speech Enhancement Papers Canary Design

## Purpose

Create `will-rice/speech-enhancement-papers` as the first public,
release-backed canary generated from `will-rice/papers-template`. The canary
proves that template release `v0.1.0` can produce a standalone repository with
safe, bounded paper discovery before scheduled operation is enabled.

## Template provenance

Generate the repository with Copier `9.10.2` from
`https://github.com/will-rice/papers-template.git` at immutable release
`v0.1.0`. Keep `.copier-answers.yml` committed with the remote `_src_path`,
`_commit: v0.1.0`, `template_version: 0.1.0`, and the repository identity:

- Name: `Speech Enhancement Papers`
- Slug: `speech-enhancement-papers`
- Description: `Research papers on speech enhancement, denoising, dereverberation, and restoration`

Copier continues to exclude repository-owned corpus and runtime state:
`papers/`, `papers.csv`, `.papers-state.yml`, and cache directories.

## Discovery and topic policy

Enable only the arXiv adapter. It requires no secret and uses a 30-day
lookback, page size 50, at most 3 pages, and at most 100 results. Its server
query requires category `eess.AS` or `cs.SD` and at least one of these concepts:
speech enhancement, speech denoising, noise suppression, dereverberation,
speech restoration, or speech super-resolution.

Apply the same focused concepts as declarative `include_any` topic gates and
allow only categories `eess.AS` and `cs.SD`. Start with no exclusion phrases
and no topic plugin. Exclusions may be added only in response to a concrete
false-positive fixture.

## Operational bounds

Each run may process one conversion batch containing at most 5 papers and cost
at most 50 units. HTML, LaTeX, and PDF cost 2, 4, and 20 units respectively.
Concurrency is 2 for HTML and 1 each for LaTeX and PDF.

Fetches retain the template defaults of a 30-second request timeout, 3 retries,
1-second backoff, and a 900-second total deadline.

The initial nightly workflow is manual-only through `workflow_dispatch`.
Scheduling remains disabled until a successful manual run has been reviewed.
No workflow is triggered as part of repository creation.

## Validation and release gate

Before opening the pull request:

1. Sync the locked development environment with
   `uv sync --locked --extra dev`.
2. Validate configuration with
   `uv run papers-pipeline validate --config papers.yml --config-only`.
3. Run the complete offline suite with `UV_OFFLINE=true uv run pytest`.
4. Run all repository hooks with `uv run pre-commit run --all-files`.
5. Run repository smoke checks with `git diff --check`, verify the Copier
   answers and nightly trigger, and confirm the generated CLI is importable.

Commit the generated repository and canary-specific configuration with
conventional commit messages, push the feature branch, and open a pull request
to `main`. Do not merge the pull request or dispatch the nightly workflow.
