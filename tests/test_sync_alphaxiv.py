from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcp import Client
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from papers_pipeline.alphaxiv_sync import (
    canonical_arxiv_id,
    main,
    read_arxiv_ids,
    sync_collection,
)

COLLECTION = "Speech Enhancement"


def fake_alphaxiv(
    folder_names: list[str],
    *,
    save_error: str | None = None,
) -> tuple[MCPServer, list[dict[str, object]]]:
    """Return an in-process alphaXiv MCP server and the saves it receives."""
    server = MCPServer("alphaxiv")
    folders: dict[str, set[str]] = {
        f"folder-{index}": set() for index in range(len(folder_names))
    }
    names = dict(zip(folders, folder_names, strict=True))
    saves: list[dict[str, object]] = []

    @server.tool(structured_output=False)
    def list_library() -> str:
        return json.dumps(
            {
                "folders": [
                    {
                        "folder_id": folder_id,
                        "name": names[folder_id],
                        "paper_count": len(papers),
                    }
                    for folder_id, papers in folders.items()
                ]
            }
        )

    @server.tool(structured_output=False)
    def save_papers_to_folder(folder_id: str, paper_ids_or_urls: list[str]) -> str:
        if save_error is not None:
            raise ToolError(save_error)
        saves.append({"folder_id": folder_id, "paper_ids_or_urls": paper_ids_or_urls})
        folders[folder_id].update(paper_ids_or_urls)
        return json.dumps({"saved": len(paper_ids_or_urls)})

    return server, saves


def write_inventory(path: Path, *arxiv_ids: str) -> None:
    rows = "".join(f"arxiv,{arxiv_id}\n" for arxiv_id in arxiv_ids)
    path.write_text(f"source,arxiv_id\ndblp,\n{rows}", encoding="utf-8")


def test_read_arxiv_ids_filters_deduplicates_and_drops_versions(
    tmp_path: Path,
) -> None:
    inventory = tmp_path / "papers.csv"
    write_inventory(
        inventory, "2608.26403v1", "2608.26403v1", "2608.26403v2", "2608.28493v2"
    )

    assert read_arxiv_ids(inventory) == ["2608.26403", "2608.28493"]


def test_read_arxiv_ids_requires_header(tmp_path: Path) -> None:
    inventory = tmp_path / "papers.csv"
    inventory.write_text("source,title\narxiv,Example\n", encoding="utf-8")

    with pytest.raises(ValueError, match="papers.csv is missing arxiv_id"):
        read_arxiv_ids(inventory)


def test_canonical_arxiv_id_removes_version() -> None:
    assert canonical_arxiv_id("2608.26403v3") == "2608.26403"
    assert canonical_arxiv_id("hep-th/9901001v2") == "hep-th/9901001"


async def test_sync_collection_saves_inventory_in_batches(tmp_path: Path) -> None:
    inventory = tmp_path / "papers.csv"
    arxiv_ids = [f"2609.{index:05d}v1" for index in range(51)]
    write_inventory(inventory, *arxiv_ids)
    server, saves = fake_alphaxiv(["Other", COLLECTION])

    async with Client(server, raise_exceptions=True) as client:
        result = await sync_collection(inventory, COLLECTION, client)

    expected = [canonical_arxiv_id(arxiv_id) for arxiv_id in arxiv_ids]
    assert saves == [
        {"folder_id": "folder-1", "paper_ids_or_urls": expected[:50]},
        {"folder_id": "folder-1", "paper_ids_or_urls": expected[50:]},
    ]
    assert (result.inventory_count, result.before_count, result.after_count) == (
        51,
        0,
        51,
    )


@pytest.mark.parametrize("folder_names", [["Other"], [COLLECTION, COLLECTION]])
async def test_sync_collection_requires_exactly_one_folder(
    tmp_path: Path,
    folder_names: list[str],
) -> None:
    inventory = tmp_path / "papers.csv"
    write_inventory(inventory, "2608.26403v1")
    server, saves = fake_alphaxiv(folder_names)

    async with Client(server) as client:
        with pytest.raises(ValueError, match="expected one alphaXiv folder"):
            await sync_collection(inventory, COLLECTION, client)
    assert saves == []


async def test_sync_collection_reports_tool_errors(tmp_path: Path) -> None:
    inventory = tmp_path / "papers.csv"
    write_inventory(inventory, "2608.26403v1")
    server, _ = fake_alphaxiv([COLLECTION], save_error="write failed")

    async with Client(server) as client:
        with pytest.raises(ValueError, match="save_papers_to_folder failed: .*write"):
            await sync_collection(inventory, COLLECTION, client)


def test_main_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("ALPHAXIV_API_KEY", raising=False)
    monkeypatch.setenv("ALPHAXIV_COLLECTION", COLLECTION)

    assert main([]) == 2
    assert capsys.readouterr().err == "error: ALPHAXIV_API_KEY must be set\n"


def test_main_requires_collection(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("ALPHAXIV_API_KEY", "axv2_test-key")
    monkeypatch.delenv("ALPHAXIV_COLLECTION", raising=False)

    assert main([]) == 2
    assert capsys.readouterr().err == "error: ALPHAXIV_COLLECTION must be set\n"
