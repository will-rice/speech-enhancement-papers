"""Add papers from the repository inventory to an alphaXiv collection."""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from alphaxiv import AlphaXivClient
from alphaxiv.exceptions import AlphaXivError


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


async def sync_collection(
    inventory: Path,
    collection: str,
    client: AlphaXivClient,
) -> SyncResult:
    identifiers = read_arxiv_ids(inventory)
    folder = await client.folders.get(collection)
    existing_ids = {canonical_arxiv_id(paper.preferred_id) for paper in folder.papers}
    missing = [
        identifier
        for identifier in identifiers
        if canonical_arxiv_id(identifier) not in existing_ids
    ]
    resolved = await asyncio.gather(
        *(client.papers.resolve(identifier) for identifier in missing)
    )
    group_ids: list[str] = []
    for paper in resolved:
        if not paper.group_id:
            raise ValueError(f"alphaXiv has no paper group for {paper.input_id}")
        group_ids.append(paper.group_id)
    if group_ids:
        await client.folders.add_papers(folder.id, group_ids)

    return SyncResult(
        inventory_count=len(identifiers),
        existing_count=len(identifiers) - len(missing),
        added_count=len(group_ids),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=Path("papers.csv"))
    args = parser.parse_args(argv)

    api_key = os.environ.get("ALPHAXIV_API_KEY", "").strip()
    if not api_key:
        print("error: ALPHAXIV_API_KEY must be set", file=sys.stderr)
        return 2
    collection = os.environ.get("ALPHAXIV_COLLECTION", "").strip()
    if not collection:
        print("error: ALPHAXIV_COLLECTION must be set", file=sys.stderr)
        return 2

    try:
        result = asyncio.run(_sync(args.inventory, collection, api_key))
    except (AlphaXivError, OSError, ValueError) as error:
        print(f"error: alphaXiv sync failed: {error}", file=sys.stderr)
        return 1

    print(
        "alphaXiv sync complete: "
        f"inventory={result.inventory_count} "
        f"existing={result.existing_count} "
        f"added={result.added_count}"
    )
    return 0


async def _sync(inventory: Path, collection: str, api_key: str) -> SyncResult:
    # Pass the key explicitly: alphaxiv-py 0.7.0 ignores ALPHAXIV_API_KEY values
    # without the legacy axv1_ prefix, but the client accepts any bearer key.
    async with AlphaXivClient(api_key=api_key) as client:
        return await sync_collection(inventory, collection, client)


if __name__ == "__main__":
    raise SystemExit(main())
