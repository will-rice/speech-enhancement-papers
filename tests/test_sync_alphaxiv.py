from __future__ import annotations

from pathlib import Path

import pytest
from pytest_httpx import HTTPXMock

from papers_pipeline.alphaxiv_sync import canonical_arxiv_id, main, read_arxiv_ids

API_KEY = "axv2_test-key"
COLLECTION = "Speech Enhancement"
FOLDERS_URL = "https://api.alphaxiv.org/folders/v3"


def folder_payload(*canonical_ids: str) -> list[dict[str, object]]:
    return [
        {"id": "folder-2", "name": "Other", "papers": []},
        {
            "id": "folder-1",
            "name": COLLECTION,
            "papers": [
                {"paperGroupId": f"group-{index}", "canonicalId": canonical_id}
                for index, canonical_id in enumerate(canonical_ids)
            ],
        },
    ]


def legacy_payload(group_id: str | None) -> dict[str, object]:
    return {
        "paper": {
            "paper_version": {"id": "version-1", "version_label": "v2"},
            "paper_group": {"id": group_id, "universal_paper_id": "2608.28493"},
        }
    }


def run_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> int:
    inventory = tmp_path / "papers.csv"
    write_inventory(inventory)
    monkeypatch.setenv("ALPHAXIV_API_KEY", API_KEY)
    monkeypatch.setenv("ALPHAXIV_COLLECTION", COLLECTION)
    return main(["--inventory", str(inventory)])


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


def test_main_adds_only_missing_papers_with_explicit_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        url=FOLDERS_URL,
        json=folder_payload("2608.26403v2", "unrelated"),
        is_reusable=True,
    )
    httpx_mock.add_response(
        url="https://api.alphaxiv.org/papers/v3/legacy/2608.28493v2",
        json=legacy_payload("group-new"),
    )
    httpx_mock.add_response(
        method="POST",
        url=f"{FOLDERS_URL}/folder-1/add-papers",
        match_json={"paperGroupIds": ["group-new"]},
        json={},
    )

    assert run_main(tmp_path, monkeypatch) == 0
    assert capsys.readouterr().out == (
        "alphaXiv sync complete: inventory=2 existing=1 added=1\n"
    )
    assert {
        request.headers["Authorization"] for request in httpx_mock.get_requests()
    } == {f"Bearer {API_KEY}"}


def test_main_skips_writes_when_collection_is_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        url=FOLDERS_URL,
        json=folder_payload("2608.26403v1", "2608.28493v1"),
    )

    assert run_main(tmp_path, monkeypatch) == 0
    assert capsys.readouterr().out == (
        "alphaXiv sync complete: inventory=2 existing=2 added=0\n"
    )


def test_main_rejects_papers_without_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(url=FOLDERS_URL, json=folder_payload("2608.26403v1"))
    httpx_mock.add_response(
        url="https://api.alphaxiv.org/papers/v3/legacy/2608.28493v2",
        json=legacy_payload(None),
    )

    assert run_main(tmp_path, monkeypatch) == 1
    assert capsys.readouterr().err == (
        "error: alphaXiv sync failed: alphaXiv has no paper group for 2608.28493v2\n"
    )


def test_main_reports_api_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(url=FOLDERS_URL, status_code=401, is_reusable=True)

    assert run_main(tmp_path, monkeypatch) == 1
    assert capsys.readouterr().err.startswith("error: alphaXiv sync failed: ")


def test_main_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("ALPHAXIV_API_KEY", raising=False)
    monkeypatch.setenv("ALPHAXIV_COLLECTION", "Speech Enhancement")

    assert main([]) == 2
    assert capsys.readouterr().err == "error: ALPHAXIV_API_KEY must be set\n"


def test_main_requires_collection(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("ALPHAXIV_API_KEY", API_KEY)
    monkeypatch.delenv("ALPHAXIV_COLLECTION", raising=False)

    assert main([]) == 2
    assert capsys.readouterr().err == "error: ALPHAXIV_COLLECTION must be set\n"
