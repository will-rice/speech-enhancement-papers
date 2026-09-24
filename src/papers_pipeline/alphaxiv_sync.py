"""Add papers from the repository inventory to an alphaXiv collection."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Runner = Callable[..., subprocess.CompletedProcess[str]]
API_KEY_PREFIX = "axv1_"


@dataclass(frozen=True)
class SyncResult:
    inventory_count: int
    existing_count: int
    added_count: int


def canonical_arxiv_id(value: str) -> str:
    return re.sub(r"v\d+$", "", value.strip(), flags=re.IGNORECASE)


def read_arxiv_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as inventory_file:
        reader = csv.DictReader(inventory_file)
        if reader.fieldnames is None or "arxiv_id" not in reader.fieldnames:
            raise ValueError("papers.csv is missing arxiv_id")

        identifiers: list[str] = []
        seen: set[str] = set()
        for row in reader:
            if row.get("source") != "arxiv":
                continue
            identifier = (row.get("arxiv_id") or "").strip()
            canonical_id = canonical_arxiv_id(identifier)
            if canonical_id and canonical_id not in seen:
                identifiers.append(identifier)
                seen.add(canonical_id)
        return identifiers


def _folder_paper_ids(payload: object) -> set[str]:
    if not isinstance(payload, dict):
        raise ValueError("invalid alphaXiv folder payload: expected object")
    papers = payload.get("papers")
    if not isinstance(papers, list):
        raise ValueError("invalid alphaXiv folder payload: expected papers list")

    identifiers: set[str] = set()
    for paper in papers:
        if not isinstance(paper, dict):
            raise ValueError("invalid alphaXiv folder payload: expected paper object")
        preferred_id = paper.get("preferred_id")
        if not isinstance(preferred_id, str) or not preferred_id.strip():
            raise ValueError(
                "invalid alphaXiv folder payload: expected paper preferred_id"
            )
        identifiers.add(canonical_arxiv_id(preferred_id))
    return identifiers


def _run(
    runner: Runner,
    command: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    return runner(
        command,
        capture_output=True,
        text=True,
        check=True,
    )


def sync_collection(
    inventory: Path,
    collection: str,
    *,
    runner: Runner = subprocess.run,
) -> SyncResult:
    identifiers = read_arxiv_ids(inventory)
    folder_result = _run(
        runner,
        ["alphaxiv", "folders", "show", collection, "--json"],
    )
    try:
        folder_payload: Any = json.loads(folder_result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("alphaXiv folders show returned invalid JSON") from error

    existing_ids = _folder_paper_ids(folder_payload)
    existing_count = 0
    added_count = 0
    for identifier in identifiers:
        if canonical_arxiv_id(identifier) in existing_ids:
            existing_count += 1
            continue
        _run(
            runner,
            [
                "alphaxiv",
                "paper",
                "folders",
                "add",
                identifier,
                collection,
                "--yes",
            ],
        )
        existing_ids.add(canonical_arxiv_id(identifier))
        added_count += 1

    return SyncResult(
        inventory_count=len(identifiers),
        existing_count=existing_count,
        added_count=added_count,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=Path("papers.csv"))
    args = parser.parse_args(argv)

    api_key = os.environ.get("ALPHAXIV_API_KEY", "").strip()
    if not api_key:
        print("error: ALPHAXIV_API_KEY must be set", file=sys.stderr)
        return 2
    # alphaxiv-py silently ignores env keys without this prefix.
    if not api_key.startswith(API_KEY_PREFIX):
        print(
            f"error: ALPHAXIV_API_KEY must be an alphaXiv API key ({API_KEY_PREFIX}...)",
            file=sys.stderr,
        )
        return 2
    collection = os.environ.get("ALPHAXIV_COLLECTION", "").strip()
    if not collection:
        print("error: ALPHAXIV_COLLECTION must be set", file=sys.stderr)
        return 2

    try:
        result = sync_collection(args.inventory, collection)
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() if error.stderr else str(error)
        print(f"error: alphaXiv sync failed: {detail}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"error: alphaXiv sync failed: {error}", file=sys.stderr)
        return 1

    print(
        "alphaXiv sync complete: "
        f"inventory={result.inventory_count} "
        f"existing={result.existing_count} "
        f"added={result.added_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
