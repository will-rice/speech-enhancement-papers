"""Add papers from the repository inventory to an alphaXiv collection."""

from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import json
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

MCP_URL = "https://api.alphaxiv.org/mcp/v1"
# save_papers_to_folder accepts at most 50 papers per call.
SAVE_BATCH_SIZE = 50


@dataclass(frozen=True)
class SyncResult:
    inventory_count: int
    before_count: int
    after_count: int


def canonical_arxiv_id(value: str) -> str:
    return re.sub(r"v\d+$", "", value.strip(), flags=re.IGNORECASE)


def read_arxiv_ids(path: Path) -> list[str]:
    """Return unique versionless arXiv IDs from the inventory, in order."""
    with path.open(newline="", encoding="utf-8") as inventory_file:
        reader = csv.DictReader(inventory_file)
        if reader.fieldnames is None or "arxiv_id" not in reader.fieldnames:
            raise ValueError("papers.csv is missing arxiv_id")
        identifiers = (
            canonical_arxiv_id(row.get("arxiv_id") or "")
            for row in reader
            if row.get("source") == "arxiv"
        )
        return list(dict.fromkeys(filter(None, identifiers)))


async def sync_collection(
    inventory: Path,
    collection: str,
    client: Client,
) -> SyncResult:
    """Save every inventory paper to the named folder.

    save_papers_to_folder is idempotent and never removes papers, so the
    whole inventory is sent on every run.
    """
    identifiers = read_arxiv_ids(inventory)
    before = await find_folder(client, collection)
    for batch in itertools.batched(identifiers, SAVE_BATCH_SIZE):
        await call_tool(
            client,
            "save_papers_to_folder",
            {"folder_id": before["folder_id"], "paper_ids_or_urls": list(batch)},
        )
    after = await find_folder(client, collection)
    return SyncResult(
        inventory_count=len(identifiers),
        before_count=before["paper_count"],
        after_count=after["paper_count"],
    )


async def find_folder(client: Client, name: str) -> dict[str, Any]:
    library = await call_tool(client, "list_library", {})
    matches: list[dict[str, Any]] = [
        folder for folder in library["folders"] if folder["name"] == name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one alphaXiv folder named {name!r}, found {len(matches)}"
        )
    return matches[0]


async def call_tool(client: Client, name: str, arguments: dict[str, Any]) -> Any:
    result = await client.call_tool(name, arguments)
    text = "".join(block.text for block in result.content if block.type == "text")
    if result.is_error:
        raise ValueError(f"alphaXiv {name} failed: {text}")
    return json.loads(text)


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

    result = asyncio.run(sync(args.inventory, collection, api_key))
    print(
        "alphaXiv sync complete: "
        f"inventory={result.inventory_count} "
        f"folder_papers={result.before_count}->{result.after_count}"
    )
    return 0


async def sync(inventory: Path, collection: str, api_key: str) -> SyncResult:
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx2.AsyncClient(headers=headers) as http_client:
        transport = streamable_http_client(MCP_URL, http_client=http_client)
        async with Client(transport) as client:
            return await sync_collection(inventory, collection, client)


if __name__ == "__main__":
    raise SystemExit(main())
