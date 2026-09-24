from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import pytest

from papers_pipeline.alphaxiv_sync import (
    canonical_arxiv_id,
    main,
    read_arxiv_ids,
    sync_collection,
)


class FakeRunner:
    def __init__(
        self,
        folder_payload: object,
        *,
        fail_on_add: bool = False,
    ) -> None:
        self.folder_payload = folder_payload
        self.fail_on_add = fail_on_add
        self.commands: list[list[str]] = []

    def __call__(
        self,
        argv: Sequence[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        command = list(argv)
        self.commands.append(command)
        assert kwargs == {
            "capture_output": True,
            "text": True,
            "check": True,
        }
        if command[1:3] == ["folders", "show"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(self.folder_payload),
                stderr="",
            )
        if self.fail_on_add:
            raise subprocess.CalledProcessError(1, command, stderr="write failed")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def write_inventory(path: Path) -> None:
    path.write_text(
        "source,arxiv_id\n"
        "arxiv,2608.26403v1\n"
        "dblp,\n"
        "arxiv,2608.26403v1\n"
        "arxiv,2608.26403v2\n"
        "arxiv,2608.28493v2\n",
        encoding="utf-8",
    )


def test_read_arxiv_ids_filters_deduplicates_and_preserves_order(
    tmp_path: Path,
) -> None:
    inventory = tmp_path / "papers.csv"
    write_inventory(inventory)

    assert read_arxiv_ids(inventory) == ["2608.26403v1", "2608.28493v2"]


def test_read_arxiv_ids_requires_header(tmp_path: Path) -> None:
    inventory = tmp_path / "papers.csv"
    inventory.write_text("source,title\narxiv,Example\n", encoding="utf-8")

    with pytest.raises(ValueError, match="papers.csv is missing arxiv_id"):
        read_arxiv_ids(inventory)


def test_canonical_arxiv_id_removes_version() -> None:
    assert canonical_arxiv_id("2608.26403v3") == "2608.26403"
    assert canonical_arxiv_id("hep-th/9901001v2") == "hep-th/9901001"


def test_sync_collection_adds_only_missing_papers(tmp_path: Path) -> None:
    inventory = tmp_path / "papers.csv"
    write_inventory(inventory)
    runner = FakeRunner(
        {
            "papers": [
                {"preferred_id": "2608.26403v2"},
                {"preferred_id": "unrelated"},
            ]
        }
    )

    result = sync_collection(
        inventory,
        "Speech Enhancement",
        runner=runner,
    )

    assert result.inventory_count == 2
    assert result.existing_count == 1
    assert result.added_count == 1
    assert runner.commands == [
        [
            "alphaxiv",
            "folders",
            "show",
            "Speech Enhancement",
            "--json",
        ],
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


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"papers": "not-a-list"},
        {"papers": ["not-an-object"]},
        {"papers": [{"title": "missing preferred id"}]},
    ],
)
def test_sync_collection_rejects_invalid_folder_payload(
    tmp_path: Path,
    payload: object,
) -> None:
    inventory = tmp_path / "papers.csv"
    write_inventory(inventory)

    with pytest.raises(ValueError, match="invalid alphaXiv folder payload"):
        sync_collection(inventory, "Speech Enhancement", runner=FakeRunner(payload))


def test_sync_collection_rejects_malformed_json(tmp_path: Path) -> None:
    inventory = tmp_path / "papers.csv"
    write_inventory(inventory)

    def malformed_runner(
        argv: Sequence[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(argv), 0, stdout="{", stderr="")

    with pytest.raises(ValueError, match="alphaXiv folders show returned invalid JSON"):
        sync_collection(inventory, "Speech Enhancement", runner=malformed_runner)


def test_sync_collection_propagates_cli_failure(tmp_path: Path) -> None:
    inventory = tmp_path / "papers.csv"
    write_inventory(inventory)
    runner = FakeRunner({"papers": []}, fail_on_add=True)

    with pytest.raises(subprocess.CalledProcessError):
        sync_collection(inventory, "Speech Enhancement", runner=runner)


def test_main_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("ALPHAXIV_API_KEY", raising=False)
    monkeypatch.setenv("ALPHAXIV_COLLECTION", "Speech Enhancement")

    assert main([]) == 2
    assert capsys.readouterr().err == "error: ALPHAXIV_API_KEY must be set\n"


def test_main_rejects_non_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("ALPHAXIV_API_KEY", "session-token")
    monkeypatch.setenv("ALPHAXIV_COLLECTION", "Speech Enhancement")

    assert main([]) == 2
    assert capsys.readouterr().err == (
        "error: ALPHAXIV_API_KEY must be an alphaXiv API key (axv1_...)\n"
    )


def test_main_requires_collection(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("ALPHAXIV_API_KEY", "axv1_test-key")
    monkeypatch.delenv("ALPHAXIV_COLLECTION", raising=False)

    assert main([]) == 2
    assert capsys.readouterr().err == "error: ALPHAXIV_COLLECTION must be set\n"
