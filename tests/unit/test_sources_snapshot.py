from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from secknow.sources.manifest import LicenseInfo, RiskState
from secknow.sources.snapshot import (
    BlobStore,
    SnapshotCatalog,
    SnapshotFile,
    SnapshotMetadata,
    SnapshotState,
)


def license_info() -> LicenseInfo:
    return LicenseInfo(
        expression="CC0-1.0",
        name="CC0",
        url="https://creativecommons.org/publicdomain/zero/1.0/",
        attribution_required=False,
        redistribution_allowed=True,
    )


def metadata(
    store: BlobStore,
    revision: str,
    content: bytes,
    risk: RiskState = RiskState.DEFAULT,
) -> SnapshotMetadata:
    digest = store.put_bytes(content)
    return SnapshotMetadata.create(
        source_id="sample",
        source_revision=revision,
        risk_state=risk,
        license=license_info(),
        files=(
            SnapshotFile(
                relative_path="records/data.json",
                blob_sha256=digest,
                size_bytes=len(content),
                media_type="application/json",
            ),
        ),
        provenance_urls=("https://example.com/data.json",),
        created_at=datetime(2026, 8, 5, tzinfo=UTC),
    )


def test_blob_store_snapshot_publish_and_rollback(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    catalog = SnapshotCatalog(tmp_path, store)
    first = metadata(store, "one", b'{"revision":1}')
    second = metadata(store, "two", b'{"revision":2}')

    first_path = catalog.stage(first)
    assert first_path.parent.parent.name == SnapshotState.NORMALIZED
    assert first_path.joinpath("payload/records/data.json").read_bytes() == b'{"revision":1}'
    catalog.publish(first.snapshot_id)
    assert catalog.current("sample") == first

    catalog.stage(second)
    catalog.publish(second.snapshot_id)
    assert catalog.current("sample") == second
    assert catalog.rollback("sample") == first
    assert catalog.current("sample") == first
    assert store.put_bytes(b'{"revision":1}') == first.files[0].blob_sha256


def test_quarantine_requires_explicit_publication(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    catalog = SnapshotCatalog(tmp_path, store)
    quarantined = metadata(store, "risky", b"payload", RiskState.QUARANTINE)

    staged = catalog.stage(quarantined)
    assert staged.parent.parent.name == SnapshotState.QUARANTINE
    with pytest.raises(PermissionError):
        catalog.publish(quarantined.snapshot_id)
    catalog.publish(quarantined.snapshot_id, allow_quarantine=True)
    assert catalog.current("sample") == quarantined


def test_snapshot_identity_excludes_observation_time(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    first = metadata(store, "one", b"same")
    second = first.model_copy(update={"created_at": datetime(2026, 8, 6, tzinfo=UTC)})
    assert second.snapshot_id == first.snapshot_id
    with pytest.raises(ValidationError):
        first.source_revision = "changed"  # type: ignore[misc]
