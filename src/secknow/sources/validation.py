from __future__ import annotations

import gzip
import shutil
import stat
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, Protocol


class BinaryReader(Protocol):
    def read(self, size: int = -1) -> bytes: ...


class UnsafeInputError(ValueError):
    """Raised when an input cannot be handled without escaping safety limits."""


@dataclass(frozen=True, slots=True)
class ArchiveLimits:
    max_entries: int = 10_000
    max_member_bytes: int = 256 * 1024 * 1024
    max_total_bytes: int = 2 * 1024 * 1024 * 1024
    max_compression_ratio: int = 200


DEFAULT_ARCHIVE_LIMITS = ArchiveLimits()


def safe_relative_path(value: str) -> PurePosixPath:
    if not value or "\x00" in value:
        raise UnsafeInputError("empty paths and NUL bytes are forbidden")
    normalized = value.replace("\\", "/")
    if normalized.startswith("/") or (len(normalized) >= 2 and normalized[1] == ":"):
        raise UnsafeInputError(f"absolute path is forbidden: {value!r}")
    if ":" in normalized or any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise UnsafeInputError(f"non-canonical relative path: {value!r}")
    path = PurePosixPath(normalized)
    return path


def confined_path(root: Path, relative: str | PurePosixPath) -> Path:
    rel = safe_relative_path(str(relative))
    root_resolved = root.resolve()
    candidate = root_resolved.joinpath(*rel.parts).resolve()
    if not candidate.is_relative_to(root_resolved):
        raise UnsafeInputError(f"path escapes destination: {relative!s}")
    return candidate


def sniff_format(data: bytes) -> str:
    if data.startswith(b"PK\x03\x04") or data.startswith(b"PK\x05\x06"):
        return "zip"
    if data.startswith(b"\x1f\x8b"):
        return "gzip"
    if len(data) > 265 and data[257:262] == b"ustar":
        return "tar"
    stripped = data.lstrip()
    if stripped.startswith((b"{", b"[")):
        return "json"
    if stripped.startswith(b"<"):
        return "xml"
    if b"\x00" not in data:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            return "text"
    return "binary"


def validate_magic(path: Path, expected: str) -> None:
    with path.open("rb") as stream:
        sample = stream.read(4096)
    actual = sniff_format(sample)
    acceptable = {
        "yaml": {"text"},
        "markdown": {"text"},
        "csv": {"text"},
        "stix": {"json"},
    }.get(expected, {expected})
    if actual not in acceptable:
        raise UnsafeInputError(f"expected {expected}, detected {actual} in {path.name}")


def _validate_size(name: str, size: int, compressed: int, limits: ArchiveLimits) -> None:
    if size < 0 or size > limits.max_member_bytes:
        raise UnsafeInputError(f"archive member is too large: {name!r}")
    if compressed == 0 and size > 0:
        raise UnsafeInputError(f"invalid compressed size for {name!r}")
    if compressed > 0 and size / compressed > limits.max_compression_ratio:
        raise UnsafeInputError(f"excessive compression ratio for {name!r}")


def _copy_limited(source: BinaryReader, destination: IO[bytes], limit: int) -> int:
    copied = 0
    while block := source.read(min(1024 * 1024, limit - copied + 1)):
        copied += len(block)
        if copied > limit:
            raise UnsafeInputError("decompressed member exceeded its declared safety limit")
        destination.write(block)
    return copied


def _safe_destination(root: Path, name: str) -> Path:
    destination = confined_path(root, name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _extract_zip(path: Path, destination: Path, limits: ArchiveLimits) -> tuple[Path, ...]:
    extracted: list[Path] = []
    total = 0
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > limits.max_entries:
            raise UnsafeInputError("archive has too many entries")
        for info in entries:
            safe_relative_path(info.filename)
            mode = info.external_attr >> 16
            file_type = stat.S_IFMT(mode)
            if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise UnsafeInputError(
                    f"links and special archive members are forbidden: {info.filename}"
                )
            if info.flag_bits & 0x1:
                raise UnsafeInputError(f"encrypted archive member is forbidden: {info.filename}")
            if info.is_dir():
                confined_path(destination, info.filename).mkdir(parents=True, exist_ok=True)
                continue
            _validate_size(info.filename, info.file_size, info.compress_size, limits)
            total += info.file_size
            if total > limits.max_total_bytes:
                raise UnsafeInputError("archive expands beyond the total size limit")
            output = _safe_destination(destination, info.filename)
            with archive.open(info) as source, output.open("xb") as target:
                copied = _copy_limited(source, target, info.file_size)
            if copied != info.file_size:
                raise UnsafeInputError(f"archive size mismatch for {info.filename!r}")
            extracted.append(output)
    return tuple(extracted)


def _extract_tar(path: Path, destination: Path, limits: ArchiveLimits) -> tuple[Path, ...]:
    extracted: list[Path] = []
    total = 0
    with tarfile.open(path, mode="r:*") as archive:
        entries = archive.getmembers()
        if len(entries) > limits.max_entries:
            raise UnsafeInputError("archive has too many entries")
        for member in entries:
            safe_relative_path(member.name)
            if member.isdir():
                confined_path(destination, member.name).mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise UnsafeInputError(
                    f"links and special archive members are forbidden: {member.name}"
                )
            _validate_size(member.name, member.size, member.size, limits)
            total += member.size
            if total > limits.max_total_bytes:
                raise UnsafeInputError("archive expands beyond the total size limit")
            source = archive.extractfile(member)
            if source is None:
                raise UnsafeInputError(f"cannot read archive member: {member.name}")
            output = _safe_destination(destination, member.name)
            with source, output.open("xb") as target:
                copied = _copy_limited(source, target, member.size)
            if copied != member.size:
                raise UnsafeInputError(f"archive size mismatch for {member.name!r}")
            extracted.append(output)
    return tuple(extracted)


def _extract_gzip(path: Path, destination: Path, limits: ArchiveLimits) -> tuple[Path, ...]:
    output_name = path.name[:-3] if path.name.lower().endswith(".gz") else f"{path.name}.out"
    output = _safe_destination(destination, output_name)
    with gzip.open(path, "rb") as source, output.open("xb") as target:
        _copy_limited(source, target, limits.max_member_bytes)
    return (output,)


def extract_archive(
    path: Path,
    destination: Path,
    *,
    archive_format: str | None = None,
    limits: ArchiveLimits = DEFAULT_ARCHIVE_LIMITS,
) -> tuple[Path, ...]:
    """Extract data without invoking external tools or trusting member paths."""
    if destination.exists() and any(destination.iterdir()):
        raise UnsafeInputError("archive destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    with path.open("rb") as stream:
        detected = sniff_format(stream.read(4096))
    selected = archive_format or detected
    if selected != detected:
        raise UnsafeInputError(f"archive format mismatch: expected {selected}, detected {detected}")
    try:
        if selected == "zip":
            return _extract_zip(path, destination, limits)
        if selected == "tar":
            return _extract_tar(path, destination, limits)
        if selected == "gzip":
            return _extract_gzip(path, destination, limits)
        raise UnsafeInputError(f"unsupported archive format: {selected}")
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
