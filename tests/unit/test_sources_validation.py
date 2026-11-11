import io
import stat
import zipfile
from pathlib import Path

import pytest

from secknow.sources.validation import (
    ArchiveLimits,
    UnsafeInputError,
    extract_archive,
    safe_relative_path,
    sniff_format,
    validate_magic,
)


def test_safe_paths_and_magic_validation(tmp_path: Path) -> None:
    assert safe_relative_path("dir/file.json").as_posix() == "dir/file.json"
    for unsafe in ("../escape", "/absolute", "C:\\escape", "dir/../escape", ""):
        with pytest.raises(UnsafeInputError):
            safe_relative_path(unsafe)

    document = tmp_path / "record.json"
    document.write_text('{"id": "CVE-2026-0001"}', encoding="utf-8")
    validate_magic(document, "json")
    with pytest.raises(UnsafeInputError):
        validate_magic(document, "xml")
    assert sniff_format(b"PK\x03\x04payload") == "zip"


def test_zip_extraction_rejects_traversal_and_links(tmp_path: Path) -> None:
    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    with pytest.raises(UnsafeInputError):
        extract_archive(traversal, tmp_path / "out-traversal")
    assert not (tmp_path / "escape.txt").exists()

    link = tmp_path / "link.zip"
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(link, "w") as archive:
        archive.writestr(info, "target")
    with pytest.raises(UnsafeInputError):
        extract_archive(link, tmp_path / "out-link")


def test_zip_extraction_enforces_limits(tmp_path: Path) -> None:
    archive_path = tmp_path / "records.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("records/one.json", io.BytesIO(b"{}" * 10).getvalue())

    with pytest.raises(UnsafeInputError):
        extract_archive(
            archive_path,
            tmp_path / "too-small",
            limits=ArchiveLimits(max_member_bytes=8, max_total_bytes=8),
        )

    output = extract_archive(archive_path, tmp_path / "valid")
    assert output[0].read_bytes() == b"{}" * 10
